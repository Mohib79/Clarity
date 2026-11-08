# Plan 05: Explainable Strength Meter (Defender's View)

**Goal:** For any typed password, estimate how guessable it is, label it Weak / Medium / Strong, and explain why in plain language.

## Guess-number estimator (`src/pwlab/meter/estimator.py`)
- [ ] **Monte Carlo strength estimation** (Dell'Amico & Filippone, CCS 2015):
  1. Draw N (e.g. 10,000–100,000) samples from the model with their probabilities. Sort by probability.
  2. Precompute the cumulative sum of `1 / (N · p_i)`.
  3. For a password with probability `p`, its estimated guess number = the cumulative sum over samples with `p_i > p` (binary search, O(log N)).
- [ ] Precompute once and save to `models/mc_table.npz`.
- [ ] Return `log10_guesses`.

## Scorer (`src/pwlab/meter/scorer.py`)
- [ ] Thresholds (in `config.yaml`, justified in the report):
  - **Weak:** < 10⁶ estimated guesses (falls to online/throttled guessing)
  - **Medium:** 10⁶ – 10¹⁰
  - **Strong:** ≥ 10¹⁰
- [ ] Pattern overrides: a whole-password common-list hit forces Weak. A password made only of one dictionary word plus a year/sequence is capped at Medium.
- [ ] Output: `{label, log10_guesses, reasons[], suggestions[], segments}`.

## Explanations (`src/pwlab/meter/explain.py`)
- [ ] Turn the Plan 02 tags into templated sentences, ordered by impact:
  - "Based on a common word (`summer`) with a 4-digit year added at the end."
  - "Uses predictable letter substitutions (`@` for `a`, `0` for `o`)."
  - "Contains a keyboard pattern (`qwerty`)."
  - "Only the first letter is capitalized, which is a very common habit."
- [ ] Suggestions: "Add more unrelated words", "Avoid years and names", "Use a passphrase of 4+ random words".
- [ ] Never echo the full password in the output, only the highlighted segment spans.

## CLI
`python -m pwlab.cli score "Summer2026!"` → pretty-printed result.

## Acceptance criteria
- [ ] `Password1!` → Weak, with reasons. A random 16-character string → Strong. A 4-random-word passphrase → Medium/Strong.
- [ ] Scoring takes under 50 ms per password once the model is loaded.

## Tests (`tests/test_meter.py`)
- Monotonicity (lower probability means more guesses), threshold boundaries, explanation text for each tag type.

