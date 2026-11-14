"""
Pattern-Aware Probabilistic Context-Free Grammar (PCFG) / Structure Model for passwords.
Decomposes password probability into:
1. Standard L/D/S structural template + lexicon-backed letter/digit/symbol/case distributions.
2. Semantic habit derivations (Leetspeak un-leeting, Keyboard Walks, Calendar Years, Dictionary words).
Combines derivations via log-sum-exp so human-habit passwords (e.g., 'P@ssw0rd2024!', '1qaz2wsx',
'Summer2026!') receive accurate probabilistic likelihoods directly from the grammar math.
"""

from collections import Counter, defaultdict
import math
import pickle
import random
import re
import string
from typing import Dict, List, Optional, Tuple

from pwlab.patterns.segmenter import segment, structure, case_mask
from pwlab.patterns.taggers import LEET_MAP, analyze
from pwlab.model.markov import MarkovModel


LETTERS = string.ascii_lowercase
DIGITS = string.digits
SYMBOLS = "".join(
    chr(i) for i in range(32, 127)
    if chr(i) not in string.ascii_letters and chr(i) not in string.digits
)


def _lse2(a: float, b: float) -> float:
    if a == -float("inf"):
        return b
    if b == -float("inf"):
        return a
    return a + math.log1p(math.exp(b - a)) if a >= b else b + math.log1p(math.exp(a - b))


class StructureModel:
    """
    Pattern-Aware Probabilistic Context-Free Grammar (PCFG) model.
    Blends L/D/S template derivations with lexicon-backed word probabilities,
    leetspeak un-leeting derivations, and keyboard walk productions.
    """

    def __init__(self, letter_ngram_order: int = 3, delta: float = 0.05):
        self.letter_ngram_order = letter_ngram_order
        self.delta = delta

        self.structure_counts: Counter = Counter()
        self.total_structures = 0

        self.chunk_counts: Dict[str, Dict[int, Counter]] = {
            "D": defaultdict(Counter),
            "S": defaultdict(Counter),
            "L": defaultdict(Counter),
        }
        self.chunk_totals: Dict[str, Dict[int, int]] = {
            "D": defaultdict(int),
            "S": defaultdict(int),
            "L": defaultdict(int),
        }

        self.case_counts: Dict[int, Counter] = defaultdict(Counter)
        self.case_totals: Dict[int, int] = defaultdict(int)

        self.letter_model = MarkovModel(order=letter_ngram_order, smoothing="backoff")
        self._cache_sample_structs: Optional[Tuple[List[str], List[int]]] = None

    def train(self, items: List[Tuple[str, int]]) -> None:
        """Train the PCFG structure model from (password, count) dataset."""
        self.structure_counts = Counter()
        self.total_structures = 0
        self.chunk_counts = {
            "D": defaultdict(Counter),
            "S": defaultdict(Counter),
            "L": defaultdict(Counter),
        }
        self.chunk_totals = {
            "D": defaultdict(int),
            "S": defaultdict(int),
            "L": defaultdict(int),
        }
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
                    lower_txt = seg.text.lower()
                    self.chunk_counts["L"][slen][lower_txt] += weight
                    self.chunk_totals["L"][slen] += weight

                    cmask = case_mask(seg.text)
                    self.case_counts[slen][cmask] += weight
                    self.case_totals[slen] += weight
                    letter_items.append((lower_txt, weight))

        if letter_items:
            self.letter_model.train(letter_items)

    def logprob_structure(self, struct: str) -> float:
        """ln P(structure template) with additive smoothing."""
        cnt = self.structure_counts.get(struct, 0)
        vocab_structures = max(len(self.structure_counts), 100)
        p = (cnt + self.delta) / (self.total_structures + self.delta * vocab_structures)
        return math.log(max(p, 1e-15))

    def logprob_digit_chunk(self, text: str) -> float:
        """ln P(digit chunk | length), incorporating year & sequence priors."""
        n = len(text)
        cnt = self.chunk_counts["D"][n].get(text, 0)
        total = self.chunk_totals["D"].get(n, 0)
        support = 10 ** min(n, 6)
        p_emp = (cnt + self.delta) / (total + self.delta * support) if total > 0 else (1.0 / support)

        # Semantic prior for 4-digit calendar years (1950-2035) and simple sequences/repeats
        p_prior = 0.0
        if n == 4 and text.isdigit():
            val = int(text)
            if 1950 <= val <= 2035:
                p_prior = 0.008  # Strong human preference for calendar years
        if len(set(text)) == 1 or text in "01234567890" or text in "9876543210":
            p_prior = max(p_prior, 0.015)

        p = 0.85 * p_emp + 0.15 * max(p_prior, p_emp)
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

    def logprob_letter_chunk(self, text: str) -> float:
        """
        ln P(letter chunk | length) blending empirical word lexicon frequency
        and character-level n-gram Markov transition probability.
        """
        lower_txt = text.lower()
        n = len(lower_txt)
        lp_ngram = self.letter_model.logprob(lower_txt)

        cnt = self.chunk_counts["L"][n].get(lower_txt, 0)
        total = self.chunk_totals["L"].get(n, 0)
        if total > 0 and cnt > 0:
            vocab_l = max(len(self.chunk_counts["L"][n]), 500)
            p_word = (cnt + self.delta) / (total + self.delta * vocab_l)
            lp_word = math.log(max(p_word, 1e-15))
            # Mixture of direct lexicon hit (0.65) and character n-gram model (0.35)
            return _lse2(math.log(0.65) + lp_word, math.log(0.35) + lp_ngram)

        return lp_ngram

    def _standard_derivation_logprob(self, pw: str) -> float:
        """Compute standard L/D/S structural derivation log-probability."""
        struct = structure(pw)
        logp = self.logprob_structure(struct)

        segs = segment(pw)
        for seg in segs:
            if seg.kind == "D":
                logp += self.logprob_digit_chunk(seg.text)
            elif seg.kind == "S":
                logp += self.logprob_symbol_chunk(seg.text)
            elif seg.kind == "L":
                logp += self.logprob_letter_chunk(seg.text)
                logp += self.logprob_case_mask(seg.text)

        return logp

    def _semantic_pattern_logprob(self, pw: str) -> float:
        """
        Evaluate semantic pattern grammar derivations (Leetspeak un-leeting & Keyboard Walks).
        Returns -inf if no semantic transformation applies.
        """
        best_alt = -float("inf")

        # 1. Check Leetspeak un-leeting derivation:
        # e.g., 'P@ssw0rd2024!' -> un-leet core letters 'Password2024!' + substitution cost
        # Only un-leet symbols/digits that are adjacent to letters so pure digit suffixes stay digits
        chars = list(pw)
        n = len(chars)
        num_subs = 0
        unleeted = list(chars)

        for i, ch in enumerate(chars):
            low = ch.lower()
            if low in LEET_MAP:
                # Check if surrounded by or adjacent to letters (or start of word followed by letters)
                prev_is_alpha = (i > 0 and unleeted[i - 1].isalpha())
                next_is_alpha = (i + 1 < n and (chars[i + 1].isalpha() or chars[i + 1].lower() in LEET_MAP))
                # Don't un-leet trailing '!' or trailing year digits
                if prev_is_alpha and next_is_alpha:
                    unleeted[i] = LEET_MAP[low]
                    num_subs += 1
                elif i == 0 and next_is_alpha and not ch.isdigit():
                    unleeted[i] = LEET_MAP[low]
                    num_subs += 1

        if num_subs > 0:
            unleeted_pw = "".join(unleeted)
            # Cost of applying leetspeak rule (~0.08 probability per substituted character)
            leet_penalty = math.log(0.15) + num_subs * math.log(0.18)
            lp_unleeted = self._standard_derivation_logprob(unleeted_pw) + leet_penalty
            best_alt = _lse2(best_alt, lp_unleeted)

        # 2. Check Keyboard Walk derivation:
        analysis = analyze(pw)
        walk_tags = [t for t in analysis.tags if t.tag_type == "keyboard_walk"]
        if walk_tags:
            covered = sum(t.span[1] - t.span[0] for t in walk_tags)
            if covered >= max(4, len(pw) // 2):
                # Keyboard walk grammar: ~200 common walks of length 4-10
                rem_len = max(0, len(pw) - covered)
                lp_walk = math.log(0.02) - math.log(250.0) * len(walk_tags) - rem_len * math.log(30.0)
                best_alt = _lse2(best_alt, lp_walk)

        return best_alt

    def logprob(self, pw: str) -> float:
        """Compute total log-probability of password under the Pattern-Aware PCFG model."""
        if not pw:
            return -100.0

        lp_std = self._standard_derivation_logprob(pw)
        lp_sem = self._semantic_pattern_logprob(pw)

        if lp_sem > -float("inf"):
            return _lse2(math.log(0.85) + lp_std, math.log(0.15) + lp_sem)
        return lp_std

    def sample(self, rng: Optional[random.Random] = None) -> str:
        """Sample a password from the learned PCFG distribution."""
        if rng is None:
            rng = random.Random()

        if not self.structure_counts:
            return "password123"

        if self._cache_sample_structs is None:
            structs, weights = zip(*self.structure_counts.most_common(500))
            self._cache_sample_structs = (list(structs), list(weights))

        chosen_struct = rng.choices(self._cache_sample_structs[0], weights=self._cache_sample_structs[1], k=1)[0]
        tokens = re.findall(r"([LDS])(\d+)", chosen_struct)
        result_parts = []

        for kind, slen_str in tokens:
            slen = int(slen_str)
            if kind == "D":
                if self.chunk_counts["D"][slen] and rng.random() < 0.85:
                    cands, cweights = zip(*self.chunk_counts["D"][slen].most_common(200))
                    chunk = rng.choices(cands, weights=cweights, k=1)[0]
                else:
                    chunk = "".join(rng.choices(DIGITS, k=slen))
                result_parts.append(chunk)

            elif kind == "S":
                if self.chunk_counts["S"][slen] and rng.random() < 0.85:
                    cands, cweights = zip(*self.chunk_counts["S"][slen].most_common(50))
                    chunk = rng.choices(cands, weights=cweights, k=1)[0]
                else:
                    chunk = "".join(rng.choices(SYMBOLS, k=slen))
                result_parts.append(chunk)

            elif kind == "L":
                if self.chunk_counts["L"][slen] and rng.random() < 0.65:
                    cands, cweights = zip(*self.chunk_counts["L"][slen].most_common(500))
                    raw_letters = rng.choices(cands, weights=cweights, k=1)[0]
                else:
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
            "continuation_counts": dict(self.letter_model.continuation_counts),
            "total_continuation_types": self.letter_model.total_continuation_types,
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

        for k in ("D", "S", "L"):
            m.chunk_counts[k] = defaultdict(Counter)
            if k in data.get("chunk_counts", {}):
                for slen, cdict in data["chunk_counts"][k].items():
                    m.chunk_counts[k][slen] = Counter(cdict)
            m.chunk_totals[k] = defaultdict(int)
            if k in data.get("chunk_totals", {}):
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
        m.letter_model.continuation_counts = Counter(lm_data.get("continuation_counts", {}))
        m.letter_model.total_continuation_types = lm_data.get(
            "total_continuation_types", sum(m.letter_model.continuation_counts.values())
        )
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
