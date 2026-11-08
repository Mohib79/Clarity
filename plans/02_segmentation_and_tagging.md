# Plan 02: Structural Segmentation & Pattern Tagging

**Goal:** Split passwords into chunks and tag predictable human habits. These tags feed the model (Plan 03) and the explanations (Plan 05).

## Segmenter (`src/pwlab/patterns/segmenter.py`)
- [ ] `segment(pw)` → list of `Segment(text, kind ∈ {L,D,S}, start, end)`.
  Example: `"Summer2026!"` → `["Summer"(L), "2026"(D), "!"(S)]`
- [ ] `structure(pw)` → template string, e.g. `L6D4S1`.
- [ ] `case_mask(segment)` → `lower | UPPER | Capitalized | mixed`.

## Taggers (`src/pwlab/patterns/taggers.py`)
Each tagger returns `Tag(type, span, detail)`. Overlaps are resolved by preferring the longest or most specific match.

| Tagger | Detects |
|---|---|
| Dictionary | English words / names (public frequency word lists in `wordlists/`) |
| Leet | Character substitutions (`@→a`, `0→o`, `3→e`, `$→s`...) that reveal a dictionary word |
| Date/Year | Years 1950–2035 and valid date formats (DDMMYYYY, MMDDYY...) |
| Sequence/Repeat | `123456`, `abcd`, `9876`, `aaaa`, `abab` |
| Keyboard pattern | Adjacent-key runs on QWERTY (`qwerty`, `asdf`, `1qaz2wsx`) |
| Common password | Whole password is among the top-N most frequent training entries |
| Position habits | Digits or symbol at the end, only the first letter capitalized |

- [ ] Public API: `analyze(pw) -> Analysis(segments, structure, tags)`.
- [ ] `cli analyze-train` writes aggregate pattern statistics to `reports/results/pattern_stats.csv` (top structures, % per tag type).

## Acceptance criteria
- [ ] `P@ssw0rd2024!` → leet word "password", year 2024, trailing symbol, capitalized first letter.
- [ ] `1qaz2wsx` → keyboard pattern. `123456789` → sequence.
- [ ] Analyzes 1M entries in under 2 min.

## Tests (`tests/test_patterns.py`)
- About 40 hand-written cases, including false-positive checks (`8888` is a repeat, not a year).

