# Plan 05: Explainable Strength Meter (Defender's View)

**Goal:** For any typed password, estimate how guessable it is, label it Weak / Medium / Strong, and explain why in plain language.

## Guess-number estimator (`src/pwlab/meter/estimator.py`)
- [x] **Monte Carlo strength estimation** (Dell'Amico & Filippone, CCS 2015):
  1. Draw N (10,000+) samples from the model with their probabilities. Sort by probability.
  2. Precompute the cumulative sum of `1 / (N · p_i)`.
  3. For a password with probability `p`, its estimated guess number = the cumulative sum over samples with `p_i > p` (binary search, O(log N)).
- [x] Precompute once and save to `models/mc_table.npz`.
- [x] Return `log10_guesses` and estimated guesses.

## Scorer (`src/pwlab/meter/scorer.py`)
- [x] Thresholds (in `config.yaml`, justified in the report):
  - **Weak:** < 10⁶ estimated guesses (falls to online/throttled guessing)
  - **Medium:** 10⁶ – 10¹⁰
  - **Strong:** ≥ 10¹⁰
- [x] Pattern overrides: a whole-password common-list hit forces Weak. Short passwords (< 6) forced to Weak. Password made only of dictionary word plus year/walk capped at Medium.
- [x] Output: `{label, log10_guesses, estimated_guesses, reasons[], suggestions[], structure, segments, tags, baselines}`.

## Explanations (`src/pwlab/meter/explain.py`)
- [x] Turn the Plan 02 tags into templated sentences, ordered by impact:
  - "Combines a common root word ('{root}') with a 4-digit calendar year ('{year}')."
  - "Uses predictable character substitutions ('@' for 'a', '0' for 'o')."
  - "Follows an easily guessable keyboard pattern ('qwerty')."
  - "Only capitalizes the first letter, following a predictable checklist compliance habit."
  - "Appends a single symbol ('!') at the very end."
- [x] Actionable recommendations: passphrases of 4+ random words, avoid years and names, avoid predictable leetspeak substitutions.

## Comparative Baselines (`src/pwlab/meter/baselines.py`)
- [x] **LUDS Checklist Rule Checker:** Evaluates Length, Uppercase, Lowercase, Digits, Symbols. Shows why checklist meters give a false sense of security (e.g. `Password1!` is rated "Strong").
- [x] **zxcvbn:** Integrated for academic benchmark comparison.

## CLI
- [x] `python -m pwlab.cli score "Password1!"` → pretty-printed result.

## Acceptance criteria
- [x] `Password1!` → Weak (while LUDS checklist says Strong 5/5).
- [x] Random 16-character string → Strong.
- [x] Scoring executes in under 20 ms per password once the model is loaded.

## Tests (`tests/test_meter.py`)
- [x] 6 unit tests passing via `pytest` (monotonicity, boundaries, LUDS flaw demonstration, weak samples, strong samples, empty input). Total test suite: 62/62 passing.
