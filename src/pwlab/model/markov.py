"""
Character-level n-gram Markov model with smoothing.
Supports additive (Laplace/delta) and interpolated backoff smoothing,
perplexity evaluation, next-character distribution generation, and sampling.
"""

from collections import Counter, defaultdict
import math
import pickle
import random
import string
from typing import Dict, List, Optional, Set, Tuple


START_TOKEN = "\x02"  # STX
END_TOKEN = "\x03"    # ETX

# Printable ASCII characters (0x20 to 0x7E) + END_TOKEN
PRINTABLE_ASCII = [chr(i) for i in range(32, 127)]
VOCABULARY = PRINTABLE_ASCII + [END_TOKEN]
VOCAB_SIZE = len(VOCABULARY)  # 95 + 1 = 96
CHAR_TO_IDX = {c: i for i, c in enumerate(VOCABULARY)}


class MarkovModel:
    """
    Character-level n-gram Markov model.
    Learns conditional character transition probabilities P(c_i | c_{i-n+1}...c_{i-1}).
    """

    def __init__(
        self,
        order: int = 4,
        smoothing: str = "backoff",  # 'additive' or 'backoff'
        additive_delta: float = 0.01,
        backoff_discount: float = 0.75,
    ):
        if order < 1:
            raise ValueError("Markov model order must be at least 1")
        self.order = order
        self.smoothing = smoothing
        self.additive_delta = additive_delta
        self.backoff_discount = backoff_discount

        # Context counts:
        # For order K, we store counts for orders 1 through K
        # contexts[k][context_str][next_char] = count
        self.counts: Dict[int, Dict[str, Counter]] = {k: defaultdict(Counter) for k in range(1, order + 1)}
        self.context_totals: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in range(1, order + 1)}
        self.unigram_counts: Counter = Counter()
        self.total_tokens = 0

    def _pad_password(self, pw: str) -> str:
        """Pad password with (order - 1) start tokens and 1 end token."""
        pad_len = self.order - 1
        return (START_TOKEN * pad_len) + pw + END_TOKEN

    def train(self, items: List[Tuple[str, int]]) -> None:
        """
        Train n-gram model from a list of (password, count) tuples.
        """
        self.counts = {k: defaultdict(Counter) for k in range(1, self.order + 1)}
        self.context_totals = {k: defaultdict(int) for k in range(1, self.order + 1)}
        self.unigram_counts = Counter()
        self.total_tokens = 0

        for pw, weight in items:
            padded = self._pad_password(pw)
            n_tokens = len(padded)

            for i in range(self.order - 1, n_tokens):
                target_char = padded[i]
                self.unigram_counts[target_char] += weight
                self.total_tokens += weight

                # Record counts for all orders from 1 to self.order
                for k in range(1, self.order + 1):
                    ctx = padded[i - (k - 1):i]
                    self.counts[k][ctx][target_char] += weight
                    self.context_totals[k][ctx] += weight

    def char_prob(self, context: str, char: str) -> float:
        """
        Compute P(char | context) with configured smoothing.
        """
        if self.smoothing == "additive":
            return self._char_prob_additive(context, char)
        else:
            return self._char_prob_backoff(context, char)

    def _char_prob_additive(self, context: str, char: str) -> float:
        """Additive (Laplace / delta) smoothing."""
        ctx = context[-(self.order - 1):] if self.order > 1 else ""
        c_counts = self.counts[self.order].get(ctx, None)
        c_count = c_counts[char] if c_counts else 0
        total = self.context_totals[self.order].get(ctx, 0)

        # P = (count + delta) / (total + delta * vocab_size)
        return (c_count + self.additive_delta) / (total + self.additive_delta * VOCAB_SIZE)

    def _char_prob_backoff(self, context: str, char: str) -> float:
        """
        Interpolated backoff smoothing (Jelinek-Mercer style interpolation across orders).
        Combines order k...1 smoothly down to uniform prior.
        """
        prob = 1.0 / VOCAB_SIZE  # Base uniform distribution

        # Interpolate from unigram up to order k
        for k in range(1, self.order + 1):
            ctx = context[-(k - 1):] if k > 1 else ""
            c_counts = self.counts[k].get(ctx, None)
            total = self.context_totals[k].get(ctx, 0)

            if total > 0 and c_counts:
                # Count of this char in this context
                c_count = c_counts.get(char, 0)
                mle = c_count / total
                # Interpolation weight lambda based on context frequency (Witten-Bell heuristic)
                num_distinct = len(c_counts)
                lambda_weight = total / (total + num_distinct)
                prob = lambda_weight * mle + (1.0 - lambda_weight) * prob
            else:
                # If context unseen at order k, fall back to lower order prob
                pass

        return prob

    def logprob(self, pw: str) -> float:
        """
        Compute natural log-probability ln P(pw) of the complete password (including END_TOKEN).
        """
        padded = self._pad_password(pw)
        logp = 0.0

        for i in range(self.order - 1, len(padded)):
            ctx = padded[i - (self.order - 1):i]
            target_char = padded[i]
            p = self.char_prob(ctx, target_char)
            logp += math.log(max(p, 1e-15))

        return logp

    def next_char_dist(self, context: str) -> List[Tuple[str, float]]:
        """
        Get the next-character probability distribution given a context, sorted descending by probability.
        Returns: list of (char, prob)
        """
        dist = []
        for c in VOCABULARY:
            p = self.char_prob(context, c)
            dist.append((c, p))

        dist.sort(key=lambda x: x[1], reverse=True)
        return dist

    def sample(self, rng: Optional[random.Random] = None, max_len: int = 32) -> str:
        """
        Sample a password string autoregressively from the model.
        Used for Monte Carlo rank and guess estimation.
        """
        if rng is None:
            rng = random.Random()

        current = START_TOKEN * (self.order - 1)
        output_chars = []

        for _ in range(max_len):
            ctx = current[-(self.order - 1):] if self.order > 1 else ""
            dist = [(c, self.char_prob(ctx, c)) for c in VOCABULARY]
            chars, weights = zip(*dist)

            chosen = rng.choices(chars, weights=weights, k=1)[0]
            if chosen == END_TOKEN:
                break

            output_chars.append(chosen)
            current += chosen

        return "".join(output_chars)

    def perplexity(self, items: List[Tuple[str, int]]) -> float:
        """
        Compute model perplexity on a held-out dataset (e.g. validation split).
        Perplexity = exp(- (1 / N) * sum(weight * logprob(pw)))
        """
        total_logp = 0.0
        total_tokens = 0

        for pw, weight in items:
            lp = self.logprob(pw)
            total_logp += weight * lp
            total_tokens += weight * (len(pw) + 1)  # +1 for END_TOKEN

        if total_tokens == 0:
            return float("inf")

        cross_entropy = -total_logp / total_tokens
        return math.exp(cross_entropy)

    def save(self, path: str) -> None:
        """Save model to disk using pickle."""
        data = {
            "order": self.order,
            "smoothing": self.smoothing,
            "additive_delta": self.additive_delta,
            "backoff_discount": self.backoff_discount,
            "counts": {k: {ctx: dict(counter) for ctx, counter in self.counts[k].items()} for k in self.counts},
            "context_totals": {k: dict(totals) for k, totals in self.context_totals.items()},
            "unigram_counts": dict(self.unigram_counts),
            "total_tokens": self.total_tokens,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "MarkovModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        m = cls(
            order=data["order"],
            smoothing=data["smoothing"],
            additive_delta=data["additive_delta"],
            backoff_discount=data["backoff_discount"],
        )
        m.total_tokens = data["total_tokens"]
        m.unigram_counts = Counter(data["unigram_counts"])

        m.counts = {}
        m.context_totals = {}
        for k in range(1, m.order + 1):
            m.counts[k] = defaultdict(Counter)
            if k in data["counts"]:
                for ctx, cdict in data["counts"][k].items():
                    m.counts[k][ctx] = Counter(cdict)

            m.context_totals[k] = defaultdict(int)
            if k in data["context_totals"]:
                m.context_totals[k].update(data["context_totals"][k])

        return m

