"""
Probabilistic structure model for passwords.
Decomposes password probability into structure template distribution,
digit/symbol chunk distributions, case mask distributions, and a letter-level n-gram model.
Optimized with sampling cache.
"""

from collections import Counter, defaultdict
import math
import pickle
import random
import re
import string
from typing import Dict, List, Optional, Tuple

from pwlab.patterns.segmenter import segment, structure, case_mask, Segment
from pwlab.model.markov import MarkovModel


LETTERS = string.ascii_lowercase
DIGITS = string.digits
SYMBOLS = "".join(chr(i) for i in range(32, 127) if chr(i) not in string.ascii_letters and chr(i) not in string.digits)


class StructureModel:
    """
    Probabilistic Context-Free Grammar / Structure model.
    P(pw) = P(structure) * prod_i P(chunk_i | kind, length) * prod_j P(case_j | length)
    """

    def __init__(self, letter_ngram_order: int = 3, delta: float = 0.05):
        self.letter_ngram_order = letter_ngram_order
        self.delta = delta

        self.structure_counts: Counter = Counter()
        self.total_structures = 0

        self.chunk_counts: Dict[str, Dict[int, Counter]] = {
            "D": defaultdict(Counter),
            "S": defaultdict(Counter),
        }
        self.chunk_totals: Dict[str, Dict[int, int]] = {
            "D": defaultdict(int),
            "S": defaultdict(int),
        }

        self.case_counts: Dict[int, Counter] = defaultdict(Counter)
        self.case_totals: Dict[int, int] = defaultdict(int)

        self.letter_model = MarkovModel(order=letter_ngram_order, smoothing="backoff")
        self._cache_sample_structs: Optional[Tuple[List[str], List[int]]] = None

    def train(self, items: List[Tuple[str, int]]) -> None:
        """Train the structure model from (password, count) dataset."""
        self.structure_counts = Counter()
        self.total_structures = 0
        self.chunk_counts = {"D": defaultdict(Counter), "S": defaultdict(Counter)}
        self.chunk_totals = {"D": defaultdict(int), "S": defaultdict(int)}
        self.case_counts = defaultdict(Counter)
        self.case_totals = defaultdict(int)
        self._cache_sample_structs = None

        letter_items: List[Tuple[str, int]] = []

        for pw, weight in items:
            struct = structure(pw)
            self.structure_counts[struct] += weight
            self.total_structures += weight

            segs = segment(pw)
            for seg in segs:
                slen = seg.length
                if seg.kind == "D":
                    self.chunk_counts["D"][slen][seg.text] += weight
                    self.chunk_totals["D"][slen] += weight
                elif seg.kind == "S":
                    self.chunk_counts["S"][slen][seg.text] += weight
                    self.chunk_totals["S"][slen] += weight
                elif seg.kind == "L":
                    cmask = case_mask(seg.text)
                    self.case_counts[slen][cmask] += weight
                    self.case_totals[slen] += weight
                    letter_items.append((seg.text.lower(), weight))

        if letter_items:
            self.letter_model.train(letter_items)

    def logprob_structure(self, struct: str) -> float:
        """ln P(structure template) with additive smoothing."""
        cnt = self.structure_counts.get(struct, 0)
        vocab_structures = max(len(self.structure_counts), 100)
        p = (cnt + self.delta) / (self.total_structures + self.delta * vocab_structures)
        return math.log(max(p, 1e-15))

    def logprob_digit_chunk(self, text: str) -> float:
        """ln P(digit chunk | length)."""
        n = len(text)
        cnt = self.chunk_counts["D"][n].get(text, 0)
        total = self.chunk_totals["D"].get(n, 0)
        support = 10 ** min(n, 6)
        p = (cnt + self.delta) / (total + self.delta * support) if total > 0 else (1.0 / support)
        return math.log(max(p, 1e-15))

    def logprob_symbol_chunk(self, text: str) -> float:
        """ln P(symbol chunk | length)."""
        n = len(text)
        cnt = self.chunk_counts["S"][n].get(text, 0)
        total = self.chunk_totals["S"].get(n, 0)
        support = len(SYMBOLS) ** min(n, 4)
        p = (cnt + self.delta) / (total + self.delta * support) if total > 0 else (1.0 / support)
        return math.log(max(p, 1e-15))

    def logprob_case_mask(self, text: str) -> float:
        """ln P(case_mask | length)."""
        n = len(text)
        cmask = case_mask(text)
        cnt = self.case_counts[n].get(cmask, 0)
        total = self.case_totals.get(n, 0)
        p = (cnt + self.delta) / (total + self.delta * 4) if total > 0 else 0.25
        return math.log(max(p, 1e-15))

    def logprob(self, pw: str) -> float:
        """Compute total log-probability of password under the structure model."""
        if not pw:
            return -100.0

        struct = structure(pw)
        logp = self.logprob_structure(struct)

        segs = segment(pw)
        for seg in segs:
            if seg.kind == "D":
                logp += self.logprob_digit_chunk(seg.text)
            elif seg.kind == "S":
                logp += self.logprob_symbol_chunk(seg.text)
            elif seg.kind == "L":
                logp += self.letter_model.logprob(seg.text.lower())
                logp += self.logprob_case_mask(seg.text)

        return logp

    def sample(self, rng: Optional[random.Random] = None) -> str:
        """Sample a password according to the structure model."""
        if rng is None:
            rng = random.Random()

        if not self.structure_counts:
            return "password123"

        if self._cache_sample_structs is None:
            structs, weights = zip(*self.structure_counts.most_common(100))
            self._cache_sample_structs = (list(structs), list(weights))

        chosen_struct = rng.choices(self._cache_sample_structs[0], weights=self._cache_sample_structs[1], k=1)[0]
        tokens = re.findall(r"([LDS])(\d+)", chosen_struct)
        result_parts = []

        for kind, slen_str in tokens:
            slen = int(slen_str)
            if kind == "D":
                if self.chunk_counts["D"][slen]:
                    cands, cweights = zip(*self.chunk_counts["D"][slen].most_common(20))
                    chunk = rng.choices(cands, weights=cweights, k=1)[0]
                else:
                    chunk = "".join(rng.choices(DIGITS, k=slen))
                result_parts.append(chunk)

            elif kind == "S":
                if self.chunk_counts["S"][slen]:
                    cands, cweights = zip(*self.chunk_counts["S"][slen].most_common(10))
                    chunk = rng.choices(cands, weights=cweights, k=1)[0]
                else:
                    chunk = "".join(rng.choices(SYMBOLS, k=slen))
                result_parts.append(chunk)

            elif kind == "L":
                raw_letters = self.letter_model.sample(rng, max_len=slen)
                if len(raw_letters) < slen:
                    raw_letters = (raw_letters + "".join(rng.choices(LETTERS, k=slen)))[:slen]
                else:
                    raw_letters = raw_letters[:slen]

                if self.case_counts[slen]:
                    masks, mweights = zip(*self.case_counts[slen].items())
                    chosen_mask = rng.choices(masks, weights=mweights, k=1)[0]
                else:
                    chosen_mask = "lower"

                if chosen_mask == "UPPER":
                    raw_letters = raw_letters.upper()
                elif chosen_mask == "Capitalized":
                    raw_letters = raw_letters.capitalize()

                result_parts.append(raw_letters)

        return "".join(result_parts)

    def save(self, path: str) -> None:
        """Save model to disk."""
        data = {
            "letter_ngram_order": self.letter_ngram_order,
            "delta": self.delta,
            "structure_counts": dict(self.structure_counts),
            "total_structures": self.total_structures,
            "chunk_counts": {k: {slen: dict(c) for slen, c in self.chunk_counts[k].items()} for k in self.chunk_counts},
            "chunk_totals": {k: dict(self.chunk_totals[k]) for k in self.chunk_totals},
            "case_counts": {slen: dict(c) for slen, c in self.case_counts.items()},
            "case_totals": dict(self.case_totals),
        }
        letter_model_data = {
            "order": self.letter_model.order,
            "smoothing": self.letter_model.smoothing,
            "additive_delta": self.letter_model.additive_delta,
            "backoff_discount": self.letter_model.backoff_discount,
            "counts": {k: {ctx: dict(counter) for ctx, counter in self.letter_model.counts[k].items()} for k in self.letter_model.counts},
            "context_totals": {k: dict(totals) for k, totals in self.letter_model.context_totals.items()},
            "unigram_counts": dict(self.letter_model.unigram_counts),
            "total_tokens": self.letter_model.total_tokens,
        }
        data["letter_model"] = letter_model_data

        with open(path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str) -> "StructureModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        m = cls(letter_ngram_order=data["letter_ngram_order"], delta=data["delta"])
        m.structure_counts = Counter(data["structure_counts"])
        m.total_structures = data["total_structures"]

        for k in ("D", "S"):
            m.chunk_counts[k] = defaultdict(Counter)
            if k in data["chunk_counts"]:
                for slen, cdict in data["chunk_counts"][k].items():
                    m.chunk_counts[k][slen] = Counter(cdict)
            m.chunk_totals[k] = defaultdict(int)
            if k in data["chunk_totals"]:
                m.chunk_totals[k].update(data["chunk_totals"][k])

        m.case_counts = defaultdict(Counter)
        for slen, cdict in data["case_counts"].items():
            m.case_counts[slen] = Counter(cdict)
        m.case_totals = defaultdict(int, data["case_totals"])

        lm_data = data["letter_model"]
        m.letter_model = MarkovModel(
            order=lm_data["order"],
            smoothing=lm_data["smoothing"],
            additive_delta=lm_data["additive_delta"],
            backoff_discount=lm_data["backoff_discount"],
        )
        m.letter_model.total_tokens = lm_data["total_tokens"]
        m.letter_model.unigram_counts = Counter(lm_data["unigram_counts"])
        m.letter_model.counts = {}
        m.letter_model.context_totals = {}
        for k in range(1, m.letter_model.order + 1):
            m.letter_model.counts[k] = defaultdict(Counter)
            if k in lm_data["counts"]:
                for ctx, cdict in lm_data["counts"][k].items():
                    m.letter_model.counts[k][ctx] = Counter(cdict)
            m.letter_model.context_totals[k] = defaultdict(int)
            if k in lm_data["context_totals"]:
                m.letter_model.context_totals[k].update(lm_data["context_totals"][k])

        return m
