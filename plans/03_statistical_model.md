# Plan 03: Statistical Model (n-gram + Structure)

**Goal:** Assign a probability to any password, learned from the training split.

## Character n-gram model (`src/pwlab/model/markov.py`)
- [x] Alphabet: 95 printable ASCII characters + an end token. Pad the start with `n-1` start tokens.
- [x] Count `context → next char` weighted by frequency.
- [x] Smoothing: additive (Laplace/δ) and interpolated backoff (Witten-Bell heuristic across orders). Compared on validation split.
- [x] API: `logprob(pw)`, `next_char_dist(context)`, `sample(rng)` (needed by the Plan 05 estimator).
- [x] Evaluated `n ∈ {2..5}` and smoothing settings by **perplexity on the validation slice** (saved to `reports/results/model_selection.csv`).
- [x] `save/load` (pickle persistence).

## Structure model (`src/pwlab/model/structure.py`)
- [x] P(structure template), P(digit chunk | length), P(symbol chunk | length), P(case mask), all smoothed.
- [x] Letter chunks are scored by a letters-only n-gram model, so unseen words still get a probability.
- [x] `logprob(pw)` = sum of the component log-probs. Supports `sample(rng)`.

## Facade (`src/pwlab/model/facade.py`)
- [x] `PasswordModel.train(counter, val_items)`, `.logprob(pw, kind="markov|structure|hybrid")`, `.sample(rng, kind)`, `.save/.load`.
- [x] Hybrid = log-mixture of the two models using numerically stable log-sum-exp, with λ tuned on validation.

## CLI
- [x] `python -m pwlab.cli train` → `models/model.pkl` and `reports/results/model_selection.csv`.

## Acceptance criteria
- [x] `logprob("password") > logprob("Summer2026!") > logprob("x7#Qp!vR2m")` (-9.59 vs -40.67 vs -74.61).
- [x] Next-character distributions sum to 1 (±1e-6).
- [x] Save/load round-trip gives identical scores.

## Tests (`tests/test_model.py`)
- [x] 10 unit tests passing via `pytest` (normalization, non-zero probability for unseen strings, toy corpus MLE preference, relative likelihood ordering, sampling, and save/load round-trips). Total test suite: 56/56 passing.
