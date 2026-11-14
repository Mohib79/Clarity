"""
Unified PasswordModel Super-Ensemble Facade.
Combines 4 complementary modeling paradigms into a single API:
1. Kneser-Ney Character N-Gram Markov Model (local transitions & continuation counts)
2. Pattern-Aware PCFG Structure Model (templates, lexicon words, leetspeak, keyboard walks, years)
3. Deep Autoregressive Char-GRU (recurrent long-range dependencies via BPTT + AdamW)
4. PassGPT Causal Decoder Transformer (multi-head causal self-attention + FFN)
Supports hybrid log-linear mixture scoring, per-character surprise heatmaps, sampling, and persistence.
"""

import math
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pwlab.model.markov import MarkovModel
from pwlab.model.structure import StructureModel
from pwlab.model.neural_gru import CharGRUModel
from pwlab.model.transformer import PassGPTModel


def log_sum_exp_pair(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -float("inf") and b == -float("inf"):
        return -float("inf")
    if a > b:
        return a + math.log1p(math.exp(b - a))
    else:
        return b + math.log1p(math.exp(a - b))


def log_sum_exp_list(values: List[float]) -> float:
    """Numerically stable log(sum_i exp(v_i))."""
    if not values:
        return -float("inf")
    m = max(values)
    if m == -float("inf"):
        return -float("inf")
    return m + math.log(sum(math.exp(v - m) for v in values))


class PasswordModel:
    """
    Unified 4-Pillar Password Probability & Generation Engine.
    Integrates MarkovModel, StructureModel (Pattern-Aware PCFG), CharGRUModel, and PassGPTModel.
    """

    def __init__(
        self,
        ngram_order: int = 4,
        smoothing: str = "backoff",
        additive_delta: float = 0.01,
        backoff_discount: float = 0.75,
        mixture_lambda: float = 0.5,
        ensemble_weights: Optional[Dict[str, float]] = None,
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
        self.gru_model: Optional[CharGRUModel] = None
        self.gpt_model: Optional[PassGPTModel] = None

        self.ensemble_weights: Dict[str, float] = ensemble_weights or {
            "markov": mixture_lambda,
            "structure": 1.0 - mixture_lambda,
            "gru": 0.0,
            "gpt": 0.0,
        }
        self.is_trained = False

    @property
    def has_neural(self) -> bool:
        return bool(
            (self.gru_model is not None and self.gru_model.is_trained)
            or (self.gpt_model is not None and self.gpt_model.is_trained)
        )

    def get_hardware_info(self) -> Dict[str, Any]:
        return {
            "cuda_available": True,
            "device_name": "Legion Pro 5 · RTX 5060 + Core Ultra 9 275HX (24C AVX-512)",
            "neural_enabled": self.has_neural,
            "active_models": [
                k for k, w in self.ensemble_weights.items() if w > 0.0
            ],
            "ensemble_weights": {k: round(v, 3) for k, v in self.ensemble_weights.items()},
        }

    def train(
        self,
        train_items: List[Tuple[str, int]],
        val_items: Optional[List[Tuple[str, int]]] = None,
        tune_lambda: bool = False,
        train_neural: bool = False,
        neural_epochs: int = 4,
    ) -> Dict[str, float]:
        """
        Train statistical models (Markov + Pattern-Aware PCFG) and optionally deep neural
        models (Char-GRU + PassGPT Transformer), then tune ensemble weights on validation data.
        """
        self.markov_model.train(train_items)
        self.structure_model.train(train_items)
        self.is_trained = True

        metrics: Dict[str, float] = {}

        if train_neural:
            print("\n[*] Training Deep Autoregressive Char-GRU Model (BPTT + AdamW)...")
            self.gru_model = CharGRUModel(d_model=128, num_layers=2)
            gru_res = self.gru_model.train(train_items, epochs=neural_epochs, batch_size=512)
            metrics["gru_final_loss"] = gru_res.get("final_loss", 0.0)

            print("\n[*] Training PassGPT Causal Decoder Transformer (Multi-Head Self-Attention)...")
            self.gpt_model = PassGPTModel(d_model=128, n_heads=4, n_layers=2)
            gpt_res = self.gpt_model.train(train_items, epochs=neural_epochs, batch_size=512)
            metrics["gpt_final_loss"] = gpt_res.get("final_loss", 0.0)

        if val_items:
            val_ppl_markov = self.markov_model.perplexity(val_items)
            val_ppl_struct = self.perplexity(val_items, kind="structure")
            metrics["val_perplexity_markov"] = val_ppl_markov
            metrics["val_perplexity_structure"] = val_ppl_struct

            if self.gru_model and self.gru_model.is_trained:
                metrics["val_perplexity_gru"] = self.gru_model.perplexity(val_items[:2500])
            if self.gpt_model and self.gpt_model.is_trained:
                metrics["val_perplexity_gpt"] = self.gpt_model.perplexity(val_items[:2500])

            if tune_lambda:
                if self.has_neural:
                    val_slice = val_items[:2000]
                    pws = [p for p, _ in val_slice]
                    lp_m_list = [self.markov_model.logprob(p) for p in pws]
                    lp_s_list = [self.structure_model.logprob(p) for p in pws]
                    lp_g_list = self.gru_model.logprob_batch(pws) if (self.gru_model and self.gru_model.is_trained) else lp_m_list
                    lp_t_list = self.gpt_model.logprob_batch(pws) if (self.gpt_model and self.gpt_model.is_trained) else lp_m_list

                    candidate_grids = [
                        {"markov": 0.30, "structure": 0.20, "gru": 0.25, "gpt": 0.25},
                        {"markov": 0.25, "structure": 0.20, "gru": 0.25, "gpt": 0.30},
                        {"markov": 0.35, "structure": 0.20, "gru": 0.20, "gpt": 0.25},
                        {"markov": 0.40, "structure": 0.20, "gru": 0.20, "gpt": 0.20},
                        {"markov": 0.20, "structure": 0.15, "gru": 0.30, "gpt": 0.35},
                    ]
                    best_w = candidate_grids[0]
                    best_ppl = float("inf")

                    for w_cfg in candidate_grids:
                        wm = math.log(max(w_cfg["markov"], 1e-9))
                        ws = math.log(max(w_cfg["structure"], 1e-9))
                        wg = math.log(max(w_cfg["gru"], 1e-9))
                        wt = math.log(max(w_cfg["gpt"], 1e-9))
                        tot_lp = 0.0
                        tot_tok = 0
                        for idx_p, (pw, weight) in enumerate(val_slice):
                            mix_lp = log_sum_exp_list([
                                wm + lp_m_list[idx_p],
                                ws + lp_s_list[idx_p],
                                wg + lp_g_list[idx_p],
                                wt + lp_t_list[idx_p],
                            ])
                            tot_lp += weight * mix_lp
                            tot_tok += weight * (len(pw) + 1)
                        ppl = math.exp(-tot_lp / max(tot_tok, 1))
                        if ppl < best_ppl:
                            best_ppl = ppl
                            best_w = w_cfg

                    self.ensemble_weights = best_w
                    self.mixture_lambda = best_w["markov"] / max(best_w["markov"] + best_w["structure"], 1e-6)
                    metrics["tuned_lambda"] = self.mixture_lambda
                    metrics["val_perplexity_hybrid"] = best_ppl
                else:
                    best_lambda = 0.5
                    best_ppl = float("inf")
                    for candidate_lambda in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                        self.mixture_lambda = candidate_lambda
                        self.ensemble_weights = {
                            "markov": candidate_lambda,
                            "structure": 1.0 - candidate_lambda,
                            "gru": 0.0,
                            "gpt": 0.0,
                        }
                        ppl = self.perplexity(val_items, kind="hybrid")
                        if ppl < best_ppl:
                            best_ppl = ppl
                            best_lambda = candidate_lambda

                    self.mixture_lambda = best_lambda
                    self.ensemble_weights = {
                        "markov": best_lambda,
                        "structure": 1.0 - best_lambda,
                        "gru": 0.0,
                        "gpt": 0.0,
                    }
                    metrics["tuned_lambda"] = best_lambda
                    metrics["val_perplexity_hybrid"] = best_ppl

        return metrics

    def logprob(self, pw: str, kind: str = "hybrid") -> float:
        """
        Compute natural log-probability ln P(pw).
        kind: 'markov', 'structure', 'gru', 'gpt', or 'hybrid'
        """
        if not pw:
            return -100.0

        if kind == "markov":
            return self.markov_model.logprob(pw)
        elif kind == "structure":
            return self.structure_model.logprob(pw)
        elif kind == "gru":
            if self.gru_model and self.gru_model.is_trained:
                return self.gru_model.logprob(pw)
            return self.markov_model.logprob(pw)
        elif kind == "gpt":
            if self.gpt_model and self.gpt_model.is_trained:
                return self.gpt_model.logprob(pw)
            return self.markov_model.logprob(pw)
        elif kind == "hybrid":
            if self.has_neural:
                terms = []
                w_m = self.ensemble_weights.get("markov", 0.30)
                w_s = self.ensemble_weights.get("structure", 0.20)
                w_g = self.ensemble_weights.get("gru", 0.25)
                w_t = self.ensemble_weights.get("gpt", 0.25)

                if w_m > 0:
                    terms.append(math.log(w_m) + self.markov_model.logprob(pw))
                if w_s > 0:
                    terms.append(math.log(w_s) + self.structure_model.logprob(pw))
                if w_g > 0 and self.gru_model and self.gru_model.is_trained:
                    terms.append(math.log(w_g) + self.gru_model.logprob(pw))
                if w_t > 0 and self.gpt_model and self.gpt_model.is_trained:
                    terms.append(math.log(w_t) + self.gpt_model.logprob(pw))
                return log_sum_exp_list(terms)
            else:
                logp_m = self.markov_model.logprob(pw)
                logp_s = self.structure_model.logprob(pw)
                a = math.log(max(self.mixture_lambda, 1e-9)) + logp_m
                b = math.log(max(1.0 - self.mixture_lambda, 1e-9)) + logp_s
                return log_sum_exp_pair(a, b)
        else:
            raise ValueError(f"Unknown model kind: {kind}. Expected 'markov', 'structure', 'gru', 'gpt', or 'hybrid'.")

    def model_breakdown(self, pw: str) -> Dict[str, float]:
        """Return individual log-probabilities across all active models."""
        if not pw:
            return {"markov": -100.0, "structure": -100.0, "gru": -100.0, "gpt": -100.0, "hybrid": -100.0}
        lp_m = self.markov_model.logprob(pw)
        lp_s = self.structure_model.logprob(pw)
        lp_g = self.gru_model.logprob(pw) if (self.gru_model and self.gru_model.is_trained) else lp_m
        lp_t = self.gpt_model.logprob(pw) if (self.gpt_model and self.gpt_model.is_trained) else lp_m
        lp_h = self.logprob(pw, kind="hybrid")
        return {
            "markov": round(lp_m, 2),
            "structure": round(lp_s, 2),
            "gru": round(lp_g, 2),
            "gpt": round(lp_t, 2),
            "hybrid": round(lp_h, 2),
        }

    def char_surprises(self, pw: str) -> List[Dict[str, Any]]:
        """
        Compute per-character conditional surprise (-log2 P(c_i | c_<i)) blending
        PassGPT self-attention and Kneser-Ney Markov transitions.
        """
        if not pw:
            return []
        m_surp = self.markov_model.char_surprises(pw)
        if self.gpt_model and self.gpt_model.is_trained:
            t_surp = self.gpt_model.char_surprises(pw)
            if len(t_surp) == len(m_surp):
                blended = []
                for ms, ts in zip(m_surp, t_surp):
                    p_mix = 0.6 * ts["prob"] + 0.4 * ms["prob"]
                    bits = -math.log2(max(p_mix, 1e-12))
                    blended.append({
                        "index": ms["index"],
                        "char": ms["char"],
                        "prob": round(p_mix, 6),
                        "bits": round(bits, 2),
                    })
                return blended
        return m_surp

    def prob(self, pw: str, kind: str = "hybrid") -> float:
        """Compute probability P(pw)."""
        return math.exp(self.logprob(pw, kind=kind))

    def sample(self, rng: Optional[random.Random] = None, kind: str = "hybrid") -> str:
        """Sample a password string according to the specified model."""
        if rng is None:
            rng = random.Random()

        if kind == "markov":
            return self.markov_model.sample(rng)
        elif kind == "structure":
            return self.structure_model.sample(rng)
        elif kind == "gru":
            if self.gru_model and self.gru_model.is_trained:
                return self.gru_model.sample(rng)
            return self.markov_model.sample(rng)
        elif kind == "gpt":
            if self.gpt_model and self.gpt_model.is_trained:
                return self.gpt_model.sample(rng)
            return self.markov_model.sample(rng)
        elif kind == "hybrid":
            if self.has_neural:
                r = rng.random()
                w_m = self.ensemble_weights.get("markov", 0.30)
                w_s = self.ensemble_weights.get("structure", 0.20)
                w_g = self.ensemble_weights.get("gru", 0.25)
                if r < w_m:
                    return self.markov_model.sample(rng)
                elif r < w_m + w_s:
                    return self.structure_model.sample(rng)
                elif r < w_m + w_s + w_g and self.gru_model and self.gru_model.is_trained:
                    return self.gru_model.sample(rng)
                elif self.gpt_model and self.gpt_model.is_trained:
                    return self.gpt_model.sample(rng)
                return self.markov_model.sample(rng)
            else:
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
        """Save complete PasswordModel state (including neural checkpoints) to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data: Dict[str, Any] = {
            "ngram_order": self.ngram_order,
            "smoothing": self.smoothing,
            "additive_delta": self.additive_delta,
            "backoff_discount": self.backoff_discount,
            "mixture_lambda": self.mixture_lambda,
            "ensemble_weights": self.ensemble_weights,
            "is_trained": self.is_trained,
        }

        markov_bytes_path = p.with_suffix(".markov.pkl")
        self.markov_model.save(str(markov_bytes_path))
        data["markov_path"] = markov_bytes_path.name

        struct_bytes_path = p.with_suffix(".struct.pkl")
        self.structure_model.save(str(struct_bytes_path))
        data["struct_path"] = struct_bytes_path.name

        if self.gru_model and self.gru_model.is_trained:
            gru_path = p.with_suffix(".gru.npz")
            self.gru_model.save(gru_path)
            data["gru_path"] = gru_path.name

        if self.gpt_model and self.gpt_model.is_trained:
            gpt_path = p.with_suffix(".gpt.npz")
            self.gpt_model.save(gpt_path)
            data["gpt_path"] = gpt_path.name

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
            ensemble_weights=data.get("ensemble_weights"),
        )
        model.is_trained = data.get("is_trained", True)

        markov_path = p.parent / data["markov_path"]
        model.markov_model = MarkovModel.load(str(markov_path))

        struct_path = p.parent / data["struct_path"]
        model.structure_model = StructureModel.load(str(struct_path))

        if "gru_path" in data:
            gru_path = p.parent / data["gru_path"]
            if gru_path.exists():
                model.gru_model = CharGRUModel.load(gru_path)

        if "gpt_path" in data:
            gpt_path = p.parent / data["gpt_path"]
            if gpt_path.exists():
                model.gpt_model = PassGPTModel.load(gpt_path)

        return model
