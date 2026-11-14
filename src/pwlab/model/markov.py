"""
Character-level n-gram Markov model with Kneser-Ney / Interpolated Backoff smoothing
and positional awareness.
Supports additive (Laplace/delta), Witten-Bell interpolated backoff, and Interpolated
Kneser-Ney discounting, perplexity evaluation, per-character surprise analysis,
next-character distribution generation, and fast cached sampling.
"""

from collections import Counter, defaultdict
import math
import pickle
import random
from typing import Any, Dict, List, Optional, Set, Tuple


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
    Learns conditional character transition probabilities P(c_i | c_{i-n+1}...c_{i-1})
    with Interpolated Kneser-Ney / Backoff discounting and continuation probabilities.
    """

    def __init__(
        self,
        order: int = 4,
        smoothing: str = "backoff",  # 'additive', 'backoff', or 'kneser_ney'
        additive_delta: float = 0.01,
        backoff_discount: float = 0.75,
    ):
        if order < 1:
            raise ValueError("Markov model order must be at least 1")
        self.order = order
        self.smoothing = smoothing
        self.additive_delta = additive_delta
        self.backoff_discount = backoff_discount

        self.counts: Dict[int, Dict[str, Counter]] = {k: defaultdict(Counter) for k in range(1, order + 1)}
        self.context_totals: Dict[int, Dict[str, int]] = {k: defaultdict(int) for k in range(1, order + 1)}
        self.unigram_counts: Counter = Counter()
        self.continuation_counts: Counter = Counter()
        self.total_continuation_types: int = 0
        self.total_tokens = 0
        self._dist_cache: Dict[str, Tuple[List[str], List[float]]] = {}

    def _pad_password(self, pw: str) -> str:
        """Pad password with (order - 1) start tokens and 1 end token."""
        pad_len = self.order - 1
        return (START_TOKEN * pad_len) + pw + END_TOKEN

    def train(self, items: List[Tuple[str, int]]) -> None:
        """
        Train n-gram model from a list of (password, count) tuples.
        Computes both frequency counts and Kneser-Ney continuation counts.
        """
        self.counts = {k: defaultdict(Counter) for k in range(1, self.order + 1)}
        self.context_totals = {k: defaultdict(int) for k in range(1, self.order + 1)}
        self.unigram_counts = Counter()
        self.continuation_counts = Counter()
        self.total_tokens = 0
        self._dist_cache.clear()

        for pw, weight in items:
            padded = self._pad_password(pw)
            n_tokens = len(padded)

            for i in range(self.order - 1, n_tokens):
                target_char = padded[i]
                self.unigram_counts[target_char] += weight
                self.total_tokens += weight

                for k in range(1, self.order + 1):
                    ctx = padded[i - (k - 1):i]
                    self.counts[k][ctx][target_char] += weight
                    self.context_totals[k][ctx] += weight

        # Compute Kneser-Ney continuation counts: number of distinct 1-char histories preceding each char
        if self.order >= 2 and 2 in self.counts:
            for ctx, char_counter in self.counts[2].items():
                for ch in char_counter:
                    self.continuation_counts[ch] += 1
        self.total_continuation_types = sum(self.continuation_counts.values())

    def char_prob(self, context: str, char: str) -> float:
        """Compute P(char | context) with configured smoothing."""
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
        return (c_count + self.additive_delta) / (total + self.additive_delta * VOCAB_SIZE)

    def _char_prob_backoff(self, context: str, char: str) -> float:
        """
        Interpolated Kneser-Ney / Witten-Bell Backoff smoothing.
        Guarantees exact probability normalization sum_{c in VOCABULARY} P(c | context) == 1.0.
        """
        # Base distribution: blend uniform prior with Kneser-Ney continuation probability
        prob = 1.0 / VOCAB_SIZE
        if self.total_continuation_types > 0:
            cont_p = (self.continuation_counts.get(char, 0) + 0.5) / (
                self.total_continuation_types + 0.5 * VOCAB_SIZE
            )
            prob = 0.5 * cont_p + 0.5 * prob

        discount = max(0.0, min(self.backoff_discount, 0.95))

        for k in range(1, self.order + 1):
            ctx = context[-(k - 1):] if k > 1 else ""
            c_counts = self.counts[k].get(ctx, None)
            total = self.context_totals[k].get(ctx, 0)

            if total > 0 and c_counts:
                c_count = c_counts.get(char, 0)
                num_distinct = len(c_counts)

                if self.smoothing == "kneser_ney":
                    # Absolute discounting (Interpolated Kneser-Ney)
                    d_eff = min(discount, 0.9)
                    discounted_mle = max(c_count - d_eff, 0.0) / total
                    bow = (d_eff * num_distinct) / total
                    prob = discounted_mle + bow * prob
                else:
                    # Hybrid Witten-Bell + Kneser-Ney discounting
                    d_eff = discount * 0.25
                    discounted_mle = max(c_count - d_eff, 0.0) / total
                    rem_mass = (d_eff * num_distinct) / total if c_count > 0 else 0.0
                    # Exact normalization across all c in VOCABULARY:
                    # Sum_c max(c_count - d_eff, 0)/total = (total - d_eff * num_distinct) / total
                    wb_lambda = total / (total + num_distinct)
                    kn_stay = 1.0 - (d_eff * num_distinct) / total
                    stay_mass = wb_lambda * kn_stay
                    prob = wb_lambda * discounted_mle + (1.0 - stay_mass) * prob

        return prob

    def logprob(self, pw: str) -> float:
        """Compute natural log-probability ln P(pw)."""
        padded = self._pad_password(pw)
        logp = 0.0

        for i in range(self.order - 1, len(padded)):
            ctx = padded[i - (self.order - 1):i]
            target_char = padded[i]
            p = self.char_prob(ctx, target_char)
            logp += math.log(max(p, 1e-15))

        return logp

    def char_surprises(self, pw: str) -> List[Dict[str, Any]]:
        """
        Compute per-character conditional probability and surprise (-log2 P(c_i | c_<i))
        for explainable character-level heatmap visualization.
        """
        if not pw:
            return []
        padded = self._pad_password(pw)
        out: List[Dict[str, Any]] = []
        for idx, ch in enumerate(pw):
            pos = (self.order - 1) + idx
            ctx = padded[pos - (self.order - 1):pos]
            p = max(self.char_prob(ctx, ch), 1e-12)
            bits = -math.log2(p)
            out.append({
                "index": idx,
                "char": ch,
                "prob": round(p, 6),
                "bits": round(bits, 2),
            })
        return out

    def get_sampling_dist(self, context: str) -> Tuple[List[str], List[float]]:
        """Fast cached distribution for sampling."""
        ctx = context[-(self.order - 1):] if self.order > 1 else ""
        if ctx in self._dist_cache:
            return self._dist_cache[ctx]

        weights = [self.char_prob(ctx, c) for c in VOCABULARY]
        self._dist_cache[ctx] = (VOCABULARY, weights)
        return VOCABULARY, weights

    def sample(self, rng: Optional[random.Random] = None, max_len: int = 32) -> str:
        """Fast autoregressive sampling."""
        if rng is None:
            rng = random.Random()

        current = START_TOKEN * (self.order - 1)
        output_chars = []

        for _ in range(max_len):
            chars, weights = self.get_sampling_dist(current)
            chosen = rng.choices(chars, weights=weights, k=1)[0]
            if chosen == END_TOKEN:
                break
            output_chars.append(chosen)
            current += chosen

        return "".join(output_chars)

    def perplexity(self, items: List[Tuple[str, int]]) -> float:
        """Compute model perplexity on a dataset."""
        total_logp = 0.0
        total_tokens = 0

        for pw, weight in items:
            lp = self.logprob(pw)
            total_logp += weight * lp
            total_tokens += weight * (len(pw) + 1)

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
            "continuation_counts": dict(self.continuation_counts),
            "total_continuation_types": self.total_continuation_types,
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
        m.continuation_counts = Counter(data.get("continuation_counts", {}))
        m.total_continuation_types = data.get("total_continuation_types", sum(m.continuation_counts.values()))

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
