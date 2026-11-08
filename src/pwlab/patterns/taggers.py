"""
Pattern taggers for identifying human password creation habits.
Detects:
- Dictionary words (common English words, names)
- Leetspeak substitutions (@->a, 0->o, 3->e, etc.)
- Years and dates (1950-2035, MMDDYYYY, DDMMYYYY, etc.)
- Sequences (12345, abcde) and repeats (aaaa, 1212)
- Keyboard walks on QWERTY layouts (qwerty, 1qaz2wsx, asdf)
- Common passwords (top breached passwords)
- Positional habits (trailing digits/symbols, leading capitalization)
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import string
from typing import Any, Dict, List, Optional, Set, Tuple

from pwlab.patterns.segmenter import Segment, segment, structure, case_mask


@dataclass(frozen=True)
class Tag:
    tag_type: str  # e.g. 'dictionary', 'leet', 'year', 'date', 'sequence', 'repeat', 'keyboard_walk', 'common_password', 'position_habit'
    span: Tuple[int, int]  # (start, end)
    matched_text: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Analysis:
    password: str
    segments: List[Segment]
    structure: str
    tags: List[Tag]

    def to_dict(self) -> dict:
        return {
            "password": self.password,
            "structure": self.structure,
            "segments": [
                {"text": s.text, "kind": s.kind, "start": s.start, "end": s.end, "case": case_mask(s.text) if s.kind == "L" else None}
                for s in self.segments
            ],
            "tags": [
                {"type": t.tag_type, "span": list(t.span), "matched": t.matched_text, "detail": t.detail}
                for t in self.tags
            ],
        }


# ============================================================================
# 1. DICTIONARY & LEETSPEAK TAGGER
# ============================================================================

LEET_MAP: Dict[str, str] = {
    "@": "a",
    "4": "a",
    "8": "b",
    "3": "e",
    "1": "i",
    "!": "i",
    "0": "o",
    "$": "s",
    "5": "s",
    "7": "t",
    "+": "t",
}


class DictionaryTagger:
    """Tagger for common English words and names."""

    def __init__(self, wordlist_paths: Optional[List[str | Path]] = None, min_word_len: int = 3):
        self.min_word_len = min_word_len
        self.words: Set[str] = set()

        if wordlist_paths is None:
            default_dir = Path(__file__).parent / "wordlists"
            wordlist_paths = [
                default_dir / "english_words.txt",
                default_dir / "names.txt",
            ]

        for path in wordlist_paths:
            p = Path(path)
            if p.exists():
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        w = line.strip().lower()
                        if w and not w.startswith("#") and len(w) >= self.min_word_len:
                            self.words.add(w)

        # Fallback built-in roots if files are missing or empty
        if len(self.words) < 50:
            fallback = {
                "password", "pass", "word", "secret", "admin", "welcome", "summer", "winter", "spring",
                "autumn", "love", "loveyou", "forever", "dragon", "monkey", "shadow", "super", "master",
                "football", "soccer", "baseball", "computer", "security", "orange", "apple", "yellow",
                "michael", "jessica", "daniel", "ashley", "nicole", "andrew", "qwerty", "starwars",
            }
            self.words.update(fallback)

    def is_word(self, candidate: str) -> bool:
        return candidate.lower() in self.words

    def find_words(self, text: str) -> List[Tag]:
        """Find dictionary words using longest-match substring scanning."""
        tags: List[Tag] = []
        lower = text.lower()
        n = len(text)

        # Scan substrings of length >= min_word_len, prioritizing longer matches
        i = 0
        while i < n:
            longest_match = None
            longest_end = -1

            # Check from longest possible down to min_word_len
            for j in range(n, i + self.min_word_len - 1, -1):
                sub = lower[i:j]
                if sub in self.words:
                    longest_match = sub
                    longest_end = j
                    break

            if longest_match:
                tags.append(Tag(
                    tag_type="dictionary",
                    span=(i, longest_end),
                    matched_text=text[i:longest_end],
                    detail={"root": longest_match, "length": len(longest_match)}
                ))
                i = longest_end
            else:
                i += 1

        return tags


class LeetTagger:
    """Tagger for leetspeak substitutions (e.g. P@ssw0rd -> password)."""

    def __init__(self, dict_tagger: DictionaryTagger):
        self.dict_tagger = dict_tagger

    def unleet(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Convert leet characters to standard alphabet and track substitutions."""
        unleeted_chars = []
        subs_used = {}
        for c in text:
            lower_c = c.lower()
            if lower_c in LEET_MAP:
                repl = LEET_MAP[lower_c]
                unleeted_chars.append(repl)
                subs_used[c] = repl
            else:
                unleeted_chars.append(lower_c)
        return "".join(unleeted_chars), subs_used

    def find_leet(self, text: str) -> List[Tag]:
        tags: List[Tag] = []
        n = len(text)

        # Fast check: does text contain any leet substitution characters?
        has_leet_char = any(c in LEET_MAP or c.lower() in LEET_MAP for c in text)
        if not has_leet_char:
            return tags

        for i in range(n):
            for j in range(n, i + self.dict_tagger.min_word_len - 1, -1):
                sub = text[i:j]
                # Only check if this specific substring actually contains leet characters
                if not any(c in LEET_MAP or c.lower() in LEET_MAP for c in sub):
                    continue

                unleeted, subs = self.unleet(sub)
                if self.dict_tagger.is_word(unleeted):
                    # It un-leets to a valid dictionary word!
                    tags.append(Tag(
                        tag_type="leet",
                        span=(i, j),
                        matched_text=sub,
                        detail={"root": unleeted, "substitutions": subs, "length": len(sub)}
                    ))
                    break  # Longest match for this start index

        return tags


# ============================================================================
# 2. DATE & YEAR TAGGER
# ============================================================================

class DateYearTagger:
    """Tagger for 4-digit years (1920-2035) and calendar dates."""

    YEAR_RE = re.compile(r"\b(19\d{2}|20[0-3]\d)\b")
    # Date formats: YYYYMMDD, DDMMYYYY, MMDDYYYY (8 digits), YYMMDD, DDMMYY (6 digits)
    DATE_8_RE = re.compile(r"\b(\d{8})\b")
    DATE_6_RE = re.compile(r"\b(\d{6})\b")

    def find_years_and_dates(self, text: str) -> List[Tag]:
        tags: List[Tag] = []

        # Find 4-digit years (embedded in digits or standalone)
        # Scan all 4-character digit substrings
        for i in range(len(text) - 3):
            chunk = text[i:i + 4]
            if chunk.isdigit():
                # Avoid false positives: repeated digits like '1111', '8888' are repeats, not years
                if len(set(chunk)) == 1:
                    continue
                # Sequential runs like '1234' are sequences, not years
                if chunk in "0123456789" or chunk in "9876543210":
                    continue

                val = int(chunk)
                if 1920 <= val <= 2035:
                    tags.append(Tag(
                        tag_type="year",
                        span=(i, i + 4),
                        matched_text=chunk,
                        detail={"year": val}
                    ))

        # Full 8-digit date check (e.g. 19980514 or 14051998)
        for i in range(len(text) - 7):
            chunk = text[i:i + 8]
            if chunk.isdigit() and len(set(chunk)) > 2:
                # Check YYYYMMDD
                y, m, d = int(chunk[:4]), int(chunk[4:6]), int(chunk[6:])
                if 1920 <= y <= 2035 and 1 <= m <= 12 and 1 <= d <= 31:
                    tags.append(Tag(
                        tag_type="date",
                        span=(i, i + 8),
                        matched_text=chunk,
                        detail={"format": "YYYYMMDD", "year": y, "month": m, "day": d}
                    ))
                    continue

                # Check DDMMYYYY or MMDDYYYY
                d_or_m, m_or_d, y2 = int(chunk[:2]), int(chunk[2:4]), int(chunk[4:])
                if 1920 <= y2 <= 2035 and 1 <= d_or_m <= 31 and 1 <= m_or_d <= 31:
                    tags.append(Tag(
                        tag_type="date",
                        span=(i, i + 8),
                        matched_text=chunk,
                        detail={"format": "DDMMYYYY/MMDDYYYY", "year": y2}
                    ))

        return tags


# ============================================================================
# 3. SEQUENCE & REPEAT TAGGER
# ============================================================================

class SequenceTagger:
    """Tagger for alphanumeric sequences (12345, abcde) and repeated patterns."""

    NUM_SEQ_FWD = "0123456789012345"
    NUM_SEQ_REV = "9876543210987654"
    ALPHA_SEQ_FWD = "abcdefghijklmnopqrstuvwxyz"
    ALPHA_SEQ_REV = "zyxwvutsrqponmlkjihgfedcba"

    def find_sequences_and_repeats(self, text: str) -> List[Tag]:
        tags: List[Tag] = []
        n = len(text)
        lower = text.lower()

        # 1. Single character repeats of length >= 3 (e.g. 'aaa', '1111', '!!!')
        i = 0
        while i < n:
            j = i + 1
            while j < n and text[j] == text[i]:
                j += 1
            if j - i >= 3:
                tags.append(Tag(
                    tag_type="repeat",
                    span=(i, j),
                    matched_text=text[i:j],
                    detail={"char": text[i], "count": j - i}
                ))
                i = j
            else:
                i += 1

        # 2. Sequential ascending/descending runs (digits or alphabet) of length >= 3
        i = 0
        while i < n:
            longest_seq = None
            longest_end = -1
            seq_type = None

            for j in range(i + 3, n + 1):
                sub = lower[i:j]
                if sub in self.NUM_SEQ_FWD or sub in self.NUM_SEQ_REV:
                    longest_seq = text[i:j]
                    longest_end = j
                    seq_type = "numeric_sequence"
                elif sub in self.ALPHA_SEQ_FWD or sub in self.ALPHA_SEQ_REV:
                    longest_seq = text[i:j]
                    longest_end = j
                    seq_type = "alpha_sequence"

            if longest_seq:
                tags.append(Tag(
                    tag_type="sequence",
                    span=(i, longest_end),
                    matched_text=longest_seq,
                    detail={"type": seq_type, "length": len(longest_seq)}
                ))
                i = longest_end
            else:
                i += 1

        # 3. Repeated pattern substrings (e.g. 'abab', '1212', 'abcabc')
        for pattern_len in (2, 3, 4):
            for i in range(len(text) - pattern_len * 2 + 1):
                pattern = text[i:i + pattern_len]
                # Check how many times it repeats
                k = 0
                while i + (k + 1) * pattern_len <= len(text) and text[i + k * pattern_len:i + (k + 1) * pattern_len] == pattern:
                    k += 1
                if k >= 2 and len(set(pattern)) > 1:  # Not single char repeat (handled above)
                    span_end = i + k * pattern_len
                    tags.append(Tag(
                        tag_type="repeat_pattern",
                        span=(i, span_end),
                        matched_text=text[i:span_end],
                        detail={"pattern": pattern, "repeats": k}
                    ))

        return tags


# ============================================================================
# 4. KEYBOARD WALK TAGGER (QWERTY)
# ============================================================================

class KeyboardWalkTagger:
    """Tagger for spatial keyboard walks on standard QWERTY layout."""

    # 4 rows of QWERTY layout
    ROWS = [
        "`1234567890-=",
        "qwertyuiop[]\\",
        "asdfghjkl;'",
        "zxcvbnm,./",
    ]

    # Precomputed horizontal runs of length >= 3
    HORIZ_FWD: List[str] = []
    HORIZ_REV: List[str] = []
    VERT_RUNS: List[str] = []

    def __init__(self, min_walk_len: int = 3):
        self.min_walk_len = min_walk_len
        self._build_walk_tables()

    def _build_walk_tables(self):
        for r in self.ROWS:
            self.HORIZ_FWD.append(r)
            self.HORIZ_REV.append(r[::-1])

        # Common vertical columns / slanted walks (e.g. 1qaz, 2wsx, 3edc, 4rfv, 5tgb, 6yhn, 7ujm, 8ik,, 9ol., 0p;/)
        cols = [
            "1qaz", "2wsx", "3edc", "4rfv", "5tgb", "6yhn", "7ujm", "8ik,", "9ol.", "0p;/", "-['", "=/\\"
        ]
        for c in cols:
            self.VERT_RUNS.append(c)
            self.VERT_RUNS.append(c[::-1])

    def find_walks(self, text: str) -> List[Tag]:
        tags: List[Tag] = []
        lower = text.lower()
        n = len(text)

        i = 0
        while i < n:
            longest_match = None
            longest_end = -1
            walk_direction = None

            for j in range(i + self.min_walk_len, n + 1):
                sub = lower[i:j]

                # Check horizontal
                if any(sub in row for row in self.HORIZ_FWD) or any(sub in row for row in self.HORIZ_REV):
                    longest_match = text[i:j]
                    longest_end = j
                    walk_direction = "horizontal_walk"
                # Check vertical / column
                elif any(sub in col for col in self.VERT_RUNS):
                    longest_match = text[i:j]
                    longest_end = j
                    walk_direction = "vertical_walk"

            if longest_match:
                tags.append(Tag(
                    tag_type="keyboard_walk",
                    span=(i, longest_end),
                    matched_text=longest_match,
                    detail={"direction": walk_direction, "length": len(longest_match)}
                ))
                i = longest_end
            else:
                i += 1

        return tags


# ============================================================================
# 5. COMMON PASSWORDS & POSITIONAL HABITS
# ============================================================================

COMMON_TOP_PASSWORDS = {
    "password", "123456", "123456789", "12345", "iloveyou", "1234567", "12345678",
    "rockyou", "qwerty", "abc123", "password1", "admin", "welcome", "monkey",
    "princess", "football", "dragon", "master", "shadow", "michael", "daniel",
    "baseball", "superman", "batman", "letmein", "login", "trustno1", "starwars",
}


class CommonPasswordTagger:
    """Tagger for well-known and top leaked passwords."""

    def __init__(self, top_passwords: Optional[Set[str]] = None):
        self.top_passwords = top_passwords or COMMON_TOP_PASSWORDS

    def check_common(self, text: str) -> List[Tag]:
        lower = text.lower()
        if lower in self.top_passwords:
            return [Tag(
                tag_type="common_password",
                span=(0, len(text)),
                matched_text=text,
                detail={"matched_root": lower}
            )]
        return []


class PositionTagger:
    """Tagger for common structural positioning habits."""

    def check_position_habits(self, text: str, segments: List[Segment]) -> List[Tag]:
        tags: List[Tag] = []
        if not segments:
            return tags

        # 1. Digits appended at the end (e.g. Summer2026, password123)
        if len(segments) >= 2 and segments[-1].kind == "D":
            tags.append(Tag(
                tag_type="position_habit",
                span=(segments[-1].start, segments[-1].end),
                matched_text=segments[-1].text,
                detail={"habit": "trailing_digits", "digits": segments[-1].text}
            ))

        # 2. Symbol appended at the end (e.g. Password1!)
        if segments[-1].kind == "S":
            tags.append(Tag(
                tag_type="position_habit",
                span=(segments[-1].start, segments[-1].end),
                matched_text=segments[-1].text,
                detail={"habit": "trailing_symbol", "symbol": segments[-1].text}
            ))

        # 3. Capital first letter only (e.g. Summer, Password, London)
        if segments[0].kind == "L" and case_mask(segments[0].text) == "Capitalized":
            tags.append(Tag(
                tag_type="position_habit",
                span=(segments[0].start, segments[0].start + 1),
                matched_text=segments[0].text[0],
                detail={"habit": "leading_capital", "segment": segments[0].text}
            ))

        # 4. Enclosing symbols (e.g. !password! or #secret#)
        if len(segments) >= 3 and segments[0].kind == "S" and segments[-1].kind == "S":
            tags.append(Tag(
                tag_type="position_habit",
                span=(0, len(text)),
                matched_text=text,
                detail={"habit": "enclosing_symbols"}
            ))

        return tags


# ============================================================================
# 6. UNIFIED PATTERN ANALYZER
# ============================================================================

class PatternAnalyzer:
    """Unified engine that coordinates all segmenters and taggers."""

    def __init__(self, wordlist_paths: Optional[List[str | Path]] = None):
        self.dict_tagger = DictionaryTagger(wordlist_paths=wordlist_paths)
        self.leet_tagger = LeetTagger(dict_tagger=self.dict_tagger)
        self.date_tagger = DateYearTagger()
        self.seq_tagger = SequenceTagger()
        self.walk_tagger = KeyboardWalkTagger()
        self.common_tagger = CommonPasswordTagger()
        self.pos_tagger = PositionTagger()

    def analyze(self, password: str) -> Analysis:
        """Analyze a password: decompose into segments and extract habit tags."""
        if not password:
            return Analysis(password="", segments=[], structure="", tags=[])

        segs = segment(password)
        struct = structure(password)

        raw_tags: List[Tag] = []

        # Run all taggers
        raw_tags.extend(self.common_tagger.check_common(password))
        raw_tags.extend(self.dict_tagger.find_words(password))
        raw_tags.extend(self.leet_tagger.find_leet(password))
        raw_tags.extend(self.date_tagger.find_years_and_dates(password))
        raw_tags.extend(self.seq_tagger.find_sequences_and_repeats(password))
        raw_tags.extend(self.walk_tagger.find_walks(password))
        raw_tags.extend(self.pos_tagger.check_position_habits(password, segs))

        # Filter duplicates and resolve overlaps
        resolved_tags = self._resolve_tags(raw_tags)

        return Analysis(
            password=password,
            segments=segs,
            structure=struct,
            tags=resolved_tags,
        )

    def _resolve_tags(self, tags: List[Tag]) -> List[Tag]:
        """Deduplicate and sort tags in order of appearance."""
        seen = set()
        unique: List[Tag] = []
        for t in tags:
            # Key: (tag_type, span, matched_text)
            k = (t.tag_type, t.span, t.matched_text)
            if k not in seen:
                seen.add(k)
                unique.append(t)

        # Sort tags primarily by start index, secondarily by longest span
        unique.sort(key=lambda t: (t.span[0], -(t.span[1] - t.span[0])))
        return unique


# Global convenience function
_DEFAULT_ANALYZER: Optional[PatternAnalyzer] = None


def analyze(password: str) -> Analysis:
    """Convenience function using the default shared PatternAnalyzer instance."""
    global _DEFAULT_ANALYZER
    if _DEFAULT_ANALYZER is None:
        _DEFAULT_ANALYZER = PatternAnalyzer()
    return _DEFAULT_ANALYZER.analyze(password)

