"""
Monte Carlo Password Guessability Estimator.
Implements the Dell'Amico & Filippone (ACM CCS 2015) Monte Carlo method
to estimate the number of guesses required to crack a password under a statistical model.
"""

import math
from pathlib import Path
import random
from typing import Optional, Tuple
import numpy as np

from pwlab.model.facade import PasswordModel


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
        Draw N samples from the model, compute their probabilities,
        and build the cumulative rank table.
        """
        rng = random.Random(seed)
        if progress:
            print(f"[*] Calibrating Monte Carlo estimator with {n_samples:,} samples...")

        # 1. Sample N passwords from model and score their log-probabilities
        sample_logps = []
        for i in range(n_samples):
            # Sample from hybrid model
            pw = model.sample(rng=rng, kind="hybrid")
            if pw:
                lp = model.logprob(pw, kind="hybrid")
                sample_logps.append(lp)

        if not sample_logps:
            raise RuntimeError("Failed to generate samples from model during MC calibration.")

        # 2. Sort log-probabilities descending (highest probability first)
        sample_logps = np.array(sample_logps, dtype=np.float64)
        sample_logps.sort()
        sample_logps = sample_logps[::-1]  # descending

        # 3. Compute weights: w_i = 1 / (N * p_i) = exp(-log(N) - logp_i)
        # To avoid overflow for very rare samples, clip minimum logp
        log_n = math.log(len(sample_logps))
        log_weights = -log_n - sample_logps

        # Numerical stabilization: weights in linear space
        # Clip max log_weight to avoid 64-bit float overflow (exp(709) max)
        log_weights_clipped = np.clip(log_weights, -50.0, 700.0)
        weights = np.exp(log_weights_clipped)

        # 4. Cumulative sum of weights gives estimated rank
        cum_ranks = np.cumsum(weights)

        # Ensure strict monotonicity
        cum_ranks = np.maximum.accumulate(cum_ranks)

        if progress:
            print(f"[+] Calibration complete! Log-prob range: [{sample_logps[-1]:.2f}, {sample_logps[0]:.2f}]")
            print(f"    Estimated rank range: [1, 10^{math.log10(max(cum_ranks[-1], 1)):.2f}]")

        return cls(log_probs=sample_logps, cum_ranks=cum_ranks)

    def estimate_guesses(self, logprob: float) -> Tuple[float, float]:
        """
        Estimate guess count for a password with given log-probability.
        Returns: (estimated_guesses, log10_guesses)
        """
        if not self.is_calibrated:
            # Fallback heuristic if not yet calibrated:
            # Shannon entropy estimate: log2(1/p) / log2(10)
            entropy = -logprob / math.log(2.0)
            log10_est = max(entropy * math.log10(2.0) * 0.6, 0.0)
            return 10.0 ** min(log10_est, 30.0), log10_est

        # Binary search: find index where sample_logps <= logprob
        # Since log_probs is sorted descending:
        # We search with negated values or use np.searchsorted on reversed
        idx = int(np.searchsorted(-self.log_probs, -logprob))

        if idx <= 0:
            # Probability is higher than or equal to the highest sampled password
            guesses = 1.0
            log10_g = 0.0
        elif idx >= len(self.log_probs):
            # Probability is lower than the lowest sample
            # Smooth analytical extrapolation beyond sample tail
            last_rank = self.cum_ranks[-1]
            last_logp = self.log_probs[-1]
            # Extra guesses proportional to probability ratio: exp(last_logp - logprob)
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

