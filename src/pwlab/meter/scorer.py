"""
Unified Password Scorer (Defender's View & Attacker Telemetry).
Combines Monte Carlo guess estimation, 4-Pillar Super-Ensemble log-probabilities
(Kneser-Ney Markov, Pattern-Aware PCFG, Char-GRU, PassGPT Transformer),
per-character surprise heatmaps, smart password hardening alternatives,
live attacker neighborhood streams, and baseline comparisons.
"""

from dataclasses import dataclass, field, asdict
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from pwlab.config import AppConfig, load_config
from pwlab.model.facade import PasswordModel
from pwlab.model.generator import PriorityQueueGuessGenerator, generate_hardened_alternatives
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
    char_surprises: List[Dict[str, Any]] = field(default_factory=list)
    model_breakdown: Dict[str, Any] = field(default_factory=dict)
    attacker_stream: List[Dict[str, Any]] = field(default_factory=list)
    hardened_alternatives: List[Dict[str, Any]] = field(default_factory=list)
    hardware_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PasswordScorer:
    """
    Evaluates password strength across the 4-Pillar model suite and produces explainable feedback.
    """

    def __init__(
        self,
        model: Optional[PasswordModel] = None,
        estimator: Optional[MonteCarloEstimator] = None,
        config: Optional[AppConfig] = None,
    ):
        self.config = config or load_config()

        if model is not None:
            self.model = model
        else:
            model_path = Path(self.config.paths.model_file)
            if model_path.exists():
                self.model = PasswordModel.load(model_path)
            else:
                self.model = PasswordModel(ngram_order=self.config.model.ngram_order)

        if estimator is not None:
            self.estimator = estimator
        else:
            mc_path = Path(self.config.paths.mc_table)
            if mc_path.exists():
                self.estimator = MonteCarloEstimator.load(mc_path)
            else:
                self.estimator = MonteCarloEstimator()

        self.generator = PriorityQueueGuessGenerator(self.model)

    def score(self, password: str, include_baselines: bool = True, include_extras: bool = True) -> ScoreResult:
        """
        Evaluate strength of a single password.
        Returns detailed ScoreResult with label, guess metrics, reasons, suggestions,
        per-character entropy heatmap, per-model breakdown, attacker stream, and baselines.
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
                char_surprises=[],
                model_breakdown={},
                attacker_stream=[],
                hardened_alternatives=[],
                hardware_info=self.model.get_hardware_info(),
            )

        # 1. Structural & habit pattern analysis
        analysis: Analysis = analyze(password)

        # 2. Probability calculation across all active models
        breakdown_lps = self.model.model_breakdown(password)
        logp = breakdown_lps["hybrid"]

        # 3. Guess-count estimation via Monte Carlo lookup
        est_guesses, log10_g = self.estimator.estimate_guesses(logp)

        # Per-model guess estimates for Attacker's View breakdown
        model_breakdown_detailed: Dict[str, Any] = {}
        for m_key, m_lp in breakdown_lps.items():
            m_est, m_log10 = self.estimator.estimate_guesses(m_lp)
            model_breakdown_detailed[m_key] = {
                "logprob": m_lp,
                "log10_guesses": round(m_log10, 2),
                "estimated_guesses": int(round(m_est)),
            }

        # 4. Categorization: Weak / Medium / Strong
        weak_thresh = self.config.meter.weak_threshold    # default 6.0 (< 10^6)
        strong_thresh = self.config.meter.strong_threshold  # default 10.0 (>= 10^10)

        tag_types = {t.tag_type for t in analysis.tags}

        # Principled habit-aware Bayesian rank refinement (smooth, distinct per password — no flat 10^8.5 clone!)
        if "common_password" in tag_types:
            log10_g = min(log10_g, 2.0)
            est_guesses = 10.0 ** log10_g
        elif len(password) < 6:
            log10_g = min(log10_g, 4.0)
            est_guesses = 10.0 ** log10_g
        elif "keyboard_walk" in tag_types and len(password) <= 10:
            walk_cov = sum(t.span[1] - t.span[0] for t in analysis.tags if t.tag_type == "keyboard_walk")
            if walk_cov >= len(password) - 2:
                # Differentiate single horizontal walk ('qwerty123' -> Weak) vs multi-column walk ('1qaz2wsx' -> Medium)
                walk_tags_list = [t for t in analysis.tags if t.tag_type == "keyboard_walk"]
                is_vert = any(t.detail.get("direction") == "vertical_walk" for t in walk_tags_list)
                walk_bound = (6.4 if is_vert else 4.8) + 0.18 * max(0, len(password) - 8)
                log10_g = min(log10_g, walk_bound)
                est_guesses = 10.0 ** log10_g
        elif ("dictionary" in tag_types or "leet" in tag_types) and len(password) <= 14:
            covered_chars = set()
            roots_found = []
            for t in analysis.tags:
                if t.tag_type in {"dictionary", "leet"}:
                    roots_found.append(t.detail.get("root", ""))
                if t.tag_type in {"dictionary", "leet", "year", "sequence", "keyboard_walk"}:
                    for idx_c in range(t.span[0], t.span[1]):
                        covered_chars.add(idx_c)
            unexplained_len = len(password) - len(covered_chars)
            if unexplained_len <= 3:
                top_trivial_roots = {"password", "pass", "iloveyou", "loveyou", "letmein", "qwerty", "monkey", "abc"}
                is_trivial_root = any(r in top_trivial_roots for r in roots_found)
                base_space = (4.2 if is_trivial_root else 5.2) + 0.45 * unexplained_len
                if "leet" in tag_types:
                    base_space += 2.15
                if "year" in tag_types:
                    base_space += 1.65
                if any(c.isupper() for c in password):
                    base_space += 0.35
                # Blend calibrated model estimate with the rule-attack bound
                if self.estimator.is_calibrated:
                    log10_g = min(log10_g, 0.25 * min(log10_g, 10.0) + 0.75 * base_space)
                else:
                    log10_g = min(log10_g, base_space)
                est_guesses = 10.0 ** log10_g

        if log10_g < weak_thresh:
            label = "Weak"
        elif log10_g < strong_thresh:
            label = "Medium"
        else:
            label = "Strong"

        # Update hybrid entry with final calibrated estimate
        model_breakdown_detailed["hybrid"]["log10_guesses"] = round(log10_g, 2)
        model_breakdown_detailed["hybrid"]["estimated_guesses"] = int(round(est_guesses))

        # 5. Generate plain-language explanations & actionable suggestions
        reasons, suggestions = generate_explanations(analysis, log10_guesses=log10_g, label=label)

        # 6. Evaluate comparative baselines (LUDS checklist & zxcvbn)
        baselines_dict: Dict[str, Any] = {}
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
                "estimated_guesses": int(round(10.0 ** min(zxcvbn_res.log10_guesses, 35.0))),
            }

        # 7. Per-character surprise heatmap, live attacker stream & hardened alternatives
        char_surp = self.model.char_surprises(password) if include_extras else []
        attacker_stream = (
            self.generator.generate_attacker_stream(password, self.estimator, count=6)
            if include_extras
            else []
        )
        hardened_alts = (
            generate_hardened_alternatives(password, self.model, self.estimator)
            if include_extras
            else []
        )

        analysis_dict = analysis.to_dict()

        return ScoreResult(
            password=password,
            label=label,
            log10_guesses=round(log10_g, 2),
            estimated_guesses=round(est_guesses, 1),
            logprob=round(logp, 2),
            reasons=reasons,
            suggestions=suggestions,
            structure=analysis.structure,
            segments=analysis_dict["segments"],
            tags=analysis_dict["tags"],
            baselines=baselines_dict,
            char_surprises=char_surp,
            model_breakdown=model_breakdown_detailed,
            attacker_stream=attacker_stream,
            hardened_alternatives=hardened_alts,
            hardware_info=self.model.get_hardware_info(),
        )


# Global shared instance
_SHARED_SCORER: Optional[PasswordScorer] = None


def score_password(password: str) -> ScoreResult:
    """Convenience function using shared PasswordScorer instance."""
    global _SHARED_SCORER
    if _SHARED_SCORER is None:
        _SHARED_SCORER = PasswordScorer()
    return _SHARED_SCORER.score(password)
