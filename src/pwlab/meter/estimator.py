"""
Monte Carlo Password Guessability Estimator (GPU-Batched & Importance-Sampled).
Implements the Dell'Amico & Filippone (ACM CCS 2015) Monte Carlo method
to estimate the number of guesses required to crack a password under a statistical
or deep neural password model.
"""

import math
from pathlib import Path
import random
from typing import List, Optional, Tuple
import numpy as np

from pwlab.model.facade import PasswordModel, log_sum_exp_list


class MonteCarloEstimator:
    """
    Estimates the rank / guess budget required to guess a password using Monte Carlo sampling.
    Precomputes a calibrated lookup table:
      - N samples drawn from P_M(w)
      - Sorted by log-probability
      - Cumulative weights: C_k = sum_{j=1}^k 1 / (N * p_j)
    """

    def __init__(self, log_probs: Optional[np.ndarray] = None, cum_ranks: Optional[np.ndarray] = None):
        self.log_probs = log_probs  # Sorted descending
        self.cum_ranks = cum_ranks  # Monotonically increasing estimated ranks
        self.is_calibrated = log_probs is not None and cum_ranks is not None

    @classmethod
    def calibrate(
        cls,
        model: PasswordModel,
        n_samples: int = 50000,
        seed: int = 42,
        progress: bool = True,
    ) -> "MonteCarloEstimator":
        """
        Draw N samples from the model, compute their probabilities under the hybrid ensemble,
        and build the cumulative rank table. Uses fast GPU batching when neural models are active.
        """
        rng = random.Random(seed)
        if progress:
            print(f"[*] Calibrating Monte Carlo estimator with {n_samples:,} samples...")

        sampled_passwords: List[str] = []

        if model.has_neural:
            # Draw proportional slices from each active pillar using GPU batching for neural models
            w_m = model.ensemble_weights.get("markov", 0.25)
            w_s = model.ensemble_weights.get("structure", 0.20)
            w_g = model.ensemble_weights.get("gru", 0.25)
            w_t = model.ensemble_weights.get("gpt", 0.30)

            n_m = int(round(n_samples * w_m))
            n_s = int(round(n_samples * w_s))
            n_g = int(round(n_samples * w_g))
            n_t = max(0, n_samples - n_m - n_s - n_g)

            for _ in range(n_m):
                pw = model.markov_model.sample(rng)
                if pw:
                    sampled_passwords.append(pw)
            for _ in range(n_s):
                pw = model.structure_model.sample(rng)
                if pw:
                    sampled_passwords.append(pw)
            if n_g > 0 and model.gru_model and model.gru_model.is_trained:
                gru_pairs = model.gru_model.sample_batch(n_g, batch_size=512, seed=seed)
                sampled_passwords.extend(pw for pw, _ in gru_pairs if pw)
            if n_t > 0 and model.gpt_model and model.gpt_model.is_trained:
                gpt_pairs = model.gpt_model.sample_batch(n_t, batch_size=512, seed=seed + 1)
                sampled_passwords.extend(pw for pw, _ in gpt_pairs if pw)

            # Score all sampled passwords in batch under the hybrid ensemble
            lp_m_all = [model.markov_model.logprob(p) for p in sampled_passwords]
            lp_s_all = [model.structure_model.logprob(p) for p in sampled_passwords]
            lp_g_all = (
                model.gru_model.logprob_batch(sampled_passwords, batch_size=1024)
                if (model.gru_model and model.gru_model.is_trained)
                else lp_m_all
            )
            lp_t_all = (
                model.gpt_model.logprob_batch(sampled_passwords, batch_size=1024)
                if (model.gpt_model and model.gpt_model.is_trained)
                else lp_m_all
            )

            lw_m = math.log(max(w_m, 1e-9))
            lw_s = math.log(max(w_s, 1e-9))
            lw_g = math.log(max(w_g, 1e-9))
            lw_t = math.log(max(w_t, 1e-9))

            sample_logps = [
                log_sum_exp_list([
                    lw_m + lm,
                    lw_s + ls,
                    lw_g + lg,
                    lw_t + lt,
                ])
                for lm, ls, lg, lt in zip(lp_m_all, lp_s_all, lp_g_all, lp_t_all)
            ]
        else:
            sample_logps = []
            for _ in range(n_samples):
                pw = model.sample(rng=rng, kind="hybrid")
                if pw:
                    lp = model.logprob(pw, kind="hybrid")
                    sample_logps.append(lp)

        if not sample_logps:
            raise RuntimeError("Failed to generate samples from model during MC calibration.")

        # 2. Sort log-probabilities descending (highest probability first)
        sample_logps_arr = np.array(sample_logps, dtype=np.float64)
        sample_logps_arr.sort()
        sample_logps_arr = sample_logps_arr[::-1]

        # 3. Compute weights: w_i = 1 / (N * p_i) = exp(-log(N) - logp_i)
        log_n = math.log(len(sample_logps_arr))
        log_weights = -log_n - sample_logps_arr

        # Numerical stabilization
        log_weights_clipped = np.clip(log_weights, -50.0, 700.0)
        weights = np.exp(log_weights_clipped)

        # 4. Cumulative sum of weights gives estimated rank
        cum_ranks = np.cumsum(weights)
        cum_ranks = np.maximum.accumulate(cum_ranks)

        if progress:
            print(f"[+] Calibration complete! Log-prob range: [{sample_logps_arr[-1]:.2f}, {sample_logps_arr[0]:.2f}]")
            print(f"    Estimated rank range: [1, 10^{math.log10(max(cum_ranks[-1], 1)):.2f}]")

        return cls(log_probs=sample_logps_arr, cum_ranks=cum_ranks)

    def estimate_guesses(self, logprob: float) -> Tuple[float, float]:
        """
        Estimate guess count for a password with given log-probability.
        Returns: (estimated_guesses, log10_guesses)
        """
        if not self.is_calibrated:
            entropy = -logprob / math.log(2.0)
            log10_est = max(entropy * math.log10(2.0) * 0.6, 0.0)
            return 10.0 ** min(log10_est, 30.0), log10_est

        idx = int(np.searchsorted(-self.log_probs, -logprob))

        if idx <= 0:
            guesses = 1.0
            log10_g = 0.0
        elif idx >= len(self.log_probs):
            last_rank = self.cum_ranks[-1]
            last_logp = self.log_probs[-1]
            diff_lp = min(last_logp - logprob, 50.0)
            extrapolated_guesses = last_rank * math.exp(diff_lp)
            guesses = extrapolated_guesses
            log10_g = math.log10(max(guesses, 1.0))
        else:
            guesses = float(self.cum_ranks[idx - 1])
            log10_g = math.log10(max(guesses, 1.0))

        return guesses, log10_g

    def save(self, path: str | Path) -> None:
        """Save calibrated arrays to compressed .npz file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, log_probs=self.log_probs, cum_ranks=self.cum_ranks)

    @classmethod
    def load(cls, path: str | Path) -> "MonteCarloEstimator":
        """Load calibrated estimator from .npz file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"MC table file not found: {path}")
        data = np.load(p)
        return cls(log_probs=data["log_probs"], cum_ranks=data["cum_ranks"])
