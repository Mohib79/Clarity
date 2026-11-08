# Plan 02: Structural Segmentation & Pattern Tagging

**Goal:** Split passwords into chunks and tag predictable human habits. These tags feed the model (Plan 03) and the explanations (Plan 05).

## Segmenter (`src/pwlab/patterns/segmenter.py`)
- [x] `segment(pw)` → list of `Segment(text, kind ∈ {L,D,S}, start, end)`.
  Example: `"Summer2026!"` → `["Summer"(L), "2026"(D), "!"(S)]`
- [x] `structure(pw)` → template string, e.g. `L6D4S1`.
- [x] `case_mask(segment)` → `lower | UPPER | Capitalized | mixed`.

## Taggers (`src/pwlab/patterns/taggers.py`)
Each tagger returns `Tag(type, span, detail)`. Overlaps are resolved by preferring the longest or most specific match.

| Tagger | Detects |
|---|---|
| Dictionary | English words / names (public frequency word lists in `wordlists/`) |
| Leet | Character substitutions (`@→a`, `0→o`, `3→e`, `$→s`, `4→a`, `1→i`...) that reveal a dictionary word |
| Date/Year | Years 1920–2035 and valid date formats (DDMMYYYY, MMDDYYYY, YYYYMMDD...) |
| Sequence/Repeat | `123456`, `abcd`, `9876`, `aaaa`, `abab` |
| Keyboard pattern | Adjacent-key runs on QWERTY (`qwerty`, `asdf`, `1qaz2wsx`) |
| Common password | Whole password is among the top-N most frequent training entries |
| Position habits | Digits or symbol at the end, only the first letter capitalized, enclosing symbols |

- [x] Public API: `analyze(pw) -> Analysis(segments, structure, tags)`.
- [x] `cli analyze-train` writes aggregate pattern statistics to `reports/results/pattern_stats.csv` (top structures, % per tag type).

## Acceptance criteria
- [x] `P@ssw0rd2024!` → leet word "password", year 2024, trailing symbol.
- [x] `1qaz2wsx` → keyboard walk. `123456789` → sequence.
- [x] Analyzes 20,000 entries in under 10 seconds.

## Tests (`tests/test_patterns.py`)
- [x] 38 hand-written unit test cases covering all taggers and false-positive checks passing via `pytest`.
