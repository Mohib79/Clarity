"""
Unified PasswordModel facade.
Combines character-level Markov model and probabilistic structure model into a single API.
Supports training, hybrid scoring via numerically stable log-sum-exp mixture, sampling, and persistence.
"""

import math
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pwlab.model.markov import MarkovModel
from pwlab.model.structure import StructureModel


def log_sum_exp_pair(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -float("inf") and b == -float("inf"):
        return -float("inf")
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


class PasswordModel:
    """
    Unified password probability engine.
    Integrates MarkovModel (statistical transitions) and StructureModel (structural patterns).
    """

    def __init__(
        self,
        ngram_order: int = 4,
        smoothing: str = "backoff",
        additive_delta: float = 0.01,
        backoff_discount: float = 0.75,
        mixture_lambda: float = 0.5,  # Weight for Markov vs Structure
    ):
        self.ngram_order = ngram_order
        self.smoothing = smoothing
        self.additive_delta = additive_delta
        self.backoff_discount = backoff_discount
        self.mixture_lambda = mixture_lambda

        self.markov_model = MarkovModel(
            order=ngram_order,
            smoothing=smoothing,
            additive_delta=additive_delta,
            backoff_discount=backoff_discount,
        )
        self.structure_model = StructureModel(letter_ngram_order=min(ngram_order, 3))
        self.is_trained = False

    def train(
        self,
        train_items: List[Tuple[str, int]],
        val_items: Optional[List[Tuple[str, int]]] = None,
        tune_lambda: bool = False,
    ) -> Dict[str, float]:
        """
        Train both Markov and Structure models on training split.
        Optionally tune mixture lambda using held-out validation split.
        """
        self.markov_model.train(train_items)
        self.structure_model.train(train_items)
        self.is_trained = True

        metrics = {}
        if val_items:
            val_ppl_markov = self.markov_model.perplexity(val_items)
            metrics["val_perplexity_markov"] = val_ppl_markov

            if tune_lambda:
                best_lambda = 0.5
                best_ppl = float("inf")
                # Grid search lambda from 0.1 to 0.9
                for candidate_lambda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                    self.mixture_lambda = candidate_lambda
                    ppl = self.perplexity(val_items, kind="hybrid")
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_lambda = candidate_lambda

                self.mixture_lambda = best_lambda
                metrics["tuned_lambda"] = best_lambda
                metrics["val_perplexity_hybrid"] = best_ppl

        return metrics

    def logprob(self, pw: str, kind: str = "hybrid") -> float:
        """
        Compute natural log-probability ln P(pw).
        kind: 'markov', 'structure', or 'hybrid'
        """
        if not pw:
            return -100.0

        if kind == "markov":
            return self.markov_model.logprob(pw)
        elif kind == "structure":
            return self.structure_model.logprob(pw)
        elif kind == "hybrid":
            # Log-mixture: ln (lambda * P_markov + (1 - lambda) * P_struct)
            logp_m = self.markov_model.logprob(pw)
            logp_s = self.structure_model.logprob(pw)

            # a = log(lambda) + logp_m
            # b = log(1 - lambda) + logp_s
            a = math.log(max(self.mixture_lambda, 1e-9)) + logp_m
            b = math.log(max(1.0 - self.mixture_lambda, 1e-9)) + logp_s
            return log_sum_exp_pair(a, b)
        else:
            raise ValueError(f"Unknown model kind: {kind}. Expected 'markov', 'structure', or 'hybrid'.")

    def prob(self, pw: str, kind: str = "hybrid") -> float:
        """Compute probability P(pw)."""
        return math.exp(self.logprob(pw, kind=kind))

    def sample(self, rng: Optional[random.Random] = None, kind: str = "hybrid") -> str:
        """Sample a password string according to the model."""
        if rng is None:
            rng = random.Random()

        if kind == "markov":
            return self.markov_model.sample(rng)
        elif kind == "structure":
            return self.structure_model.sample(rng)
        elif kind == "hybrid":
            if rng.random() < self.mixture_lambda:
                return self.markov_model.sample(rng)
            else:
                return self.structure_model.sample(rng)
        else:
            raise ValueError(f"Unknown model kind: {kind}")

    def perplexity(self, items: List[Tuple[str, int]], kind: str = "hybrid") -> float:
        """Compute perplexity on a dataset under specified model."""
        total_logp = 0.0
        total_tokens = 0

        for pw, weight in items:
            lp = self.logprob(pw, kind=kind)
            total_logp += weight * lp
            total_tokens += weight * (len(pw) + 1)

        if total_tokens == 0:
            return float("inf")

        cross_entropy = -total_logp / total_tokens
        return math.exp(cross_entropy)

    def save(self, path: str | Path) -> None:
        """Save complete PasswordModel state to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "ngram_order": self.ngram_order,
            "smoothing": self.smoothing,
            "additive_delta": self.additive_delta,
            "backoff_discount": self.backoff_discount,
            "mixture_lambda": self.mixture_lambda,
            "is_trained": self.is_trained,
        }

        # Save markov weights
        markov_bytes_path = p.with_suffix(".markov.pkl")
        self.markov_model.save(str(markov_bytes_path))
        data["markov_path"] = markov_bytes_path.name

        # Save structure weights
        struct_bytes_path = p.with_suffix(".struct.pkl")
        self.structure_model.save(str(struct_bytes_path))
        data["struct_path"] = struct_bytes_path.name

        with open(p, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "PasswordModel":
        """Load complete PasswordModel state from disk."""
        p = Path(path)
        with open(p, "rb") as f:
            data = pickle.load(f)

        model = cls(
            ngram_order=data["ngram_order"],
            smoothing=data["smoothing"],
            additive_delta=data["additive_delta"],
            backoff_discount=data["backoff_discount"],
            mixture_lambda=data["mixture_lambda"],
        )
        model.is_trained = data.get("is_trained", True)

        markov_path = p.parent / data["markov_path"]
        model.markov_model = MarkovModel.load(str(markov_path))

        struct_path = p.parent / data["struct_path"]
        model.structure_model = StructureModel.load(str(struct_path))

        return model

