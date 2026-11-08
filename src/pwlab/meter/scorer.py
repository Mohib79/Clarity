"""
Unified Password Scorer (Defender's View).
Combines Monte Carlo guess estimation, statistical model log-probabilities,
pattern analysis tags, plain-language explanations, and baseline comparisons.
"""

from dataclasses import dataclass, asdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from pwlab.config import AppConfig, load_config
from pwlab.model.facade import PasswordModel
from pwlab.meter.estimator import MonteCarloEstimator
from pwlab.meter.explain import generate_explanations
from pwlab.meter.baselines import evaluate_luds, evaluate_zxcvbn
from pwlab.patterns.taggers import analyze, Analysis


@dataclass
class ScoreResult:
    password: str
    label: str  # 'Weak', 'Medium', 'Strong'
    log10_guesses: float
    estimated_guesses: float
    logprob: float
    reasons: List[str]
    suggestions: List[str]
    structure: str
    segments: List[Dict[str, Any]]
    tags: List[Dict[str, Any]]
    baselines: Dict[str, Any]

    def to_dict(self) -> dict:
        d = asdict(self)
        # Avoid echoing full plain password if needed for sanitization
        return d


class PasswordScorer:
    """
    Evaluates password strength and produces explainable feedback.
    """

    def __init__(
        self,
        model: Optional[PasswordModel] = None,
        estimator: Optional[MonteCarloEstimator] = None,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or load_config()

        # Load or initialize model
        if model is not None:
            self.model = model
        else:
            model_path = Path(self.config.paths.model_file)
            if model_path.exists():
                self.model = PasswordModel.load(model_path)
            else:
                self.model = PasswordModel(ngram_order=self.config.model.ngram_order)

        # Load or initialize estimator
        if estimator is not None:
            self.estimator = estimator
        else:
            mc_path = Path(self.config.paths.mc_table)
            if mc_path.exists():
                self.estimator = MonteCarloEstimator.load(mc_path)
            else:
                self.estimator = MonteCarloEstimator()

    def score(self, password: str, include_baselines: bool = True) -> ScoreResult:
        """
        Evaluate strength of a single password.
        Returns detailed ScoreResult with label, guess metrics, reasons, suggestions, and baselines.
        """
        if not password:
            return ScoreResult(
                password="",
                label="Weak",
                log10_guesses=0.0,
                estimated_guesses=1.0,
                logprob=-100.0,
                reasons=["Password is empty."],
                suggestions=["Please enter a password."],
                structure="",
                segments=[],
                tags=[],
                baselines={},
            )

        # 1. Structural pattern analysis
        analysis: Analysis = analyze(password)

        # 2. Probability calculation from hybrid model
        logp = self.model.logprob(password, kind="hybrid")

        # 3. Guess-count estimation via Monte Carlo lookup
        est_guesses, log10_g = self.estimator.estimate_guesses(logp)

        # 4. Categorization: Weak / Medium / Strong
        weak_thresh = self.config.meter.weak_threshold  # default 6.0 (< 10^6)
        strong_thresh = self.config.meter.strong_threshold  # default 10.0 (>= 10^10)

        if log10_g < weak_thresh:
            label = "Weak"
        elif log10_g < strong_thresh:
            label = "Medium"
        else:
            label = "Strong"

        # 5. Pattern Overrides & Security Safeguards
        tag_types = {t.tag_type for t in analysis.tags}

        # Override 1: Common breached password hit is always Weak
        if "common_password" in tag_types:
            label = "Weak"
            log10_g = min(log10_g, 2.0)
            est_guesses = min(est_guesses, 100.0)

        # Override 2: Very short passwords (< 6 chars) are always Weak
        if len(password) < 6:
            label = "Weak"
            log10_g = min(log10_g, 4.0)
            est_guesses = min(est_guesses, 10000.0)

        # Override 3: Cap at Medium if it relies on simple word+year or keyboard walk
        if ("keyboard_walk" in tag_types or ("dictionary" in tag_types and "year" in tag_types)) and label == "Strong":
            label = "Medium"
            log10_g = min(log10_g, 8.5)
            est_guesses = min(est_guesses, 10.0 ** 8.5)

        # 6. Generate plain-language explanations & actionable suggestions
        reasons, suggestions = generate_explanations(analysis, log10_guesses=log10_g, label=label)

        # 7. Evaluate comparative baselines (LUDS checklist & zxcvbn)
        baselines_dict = {}
        if include_baselines:
            luds_res = evaluate_luds(password)
            zxcvbn_res = evaluate_zxcvbn(password)
            baselines_dict["luds"] = {
                "label": luds_res.label,
                "score": luds_res.score_raw,
                "passed_rules": luds_res.details["passed_count"],
            }
            baselines_dict["zxcvbn"] = {
                "label": zxcvbn_res.label,
                "score": zxcvbn_res.score_raw,
                "log10_guesses": zxcvbn_res.log10_guesses,
            }

        analysis_dict = analysis.to_dict()

        return ScoreResult(
            password=password,
            label=label,
            log10_guesses=round(log10_g, 2),
            estimated_guesses=est_guesses,
            logprob=round(logp, 2),
            reasons=reasons,
            suggestions=suggestions,
            structure=analysis.structure,
            segments=analysis_dict["segments"],
            tags=analysis_dict["tags"],
            baselines=baselines_dict,
        )


# Global shared instance
_SHARED_SCORER: Optional[PasswordScorer] = None


def score_password(password: str) -> ScoreResult:
    """Convenience function using shared PasswordScorer instance."""
    global _SHARED_SCORER
    if _SHARED_SCORER is None:
        _SHARED_SCORER = PasswordScorer()
    return _SHARED_SCORER.score(password)

