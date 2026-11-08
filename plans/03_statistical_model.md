# Plan 03: Statistical Model (n-gram + Structure)

**Goal:** Assign a probability to any password, learned from the training split.

## Character n-gram model (`src/pwlab/model/markov.py`)
- [ ] Alphabet: 95 printable ASCII characters + an end token. Pad the start with `n-1` start tokens.
- [ ] Count `context → next char` weighted by frequency.
- [ ] Smoothing: additive (Laplace/δ) first, then interpolated backoff (Witten-Bell or absolute discounting). Compare the two.
- [ ] API: `logprob(pw)`, `next_char_dist(context)`, `sample(rng)` (needed by the Plan 05 estimator).
- [ ] Choose `n ∈ {2..5}` and the smoothing settings by **perplexity on the validation slice**.
- [ ] `save/load` (pickle + JSON metadata). Prune rare contexts if the model gets too large.

## Structure model (`src/pwlab/model/structure.py`)
- [ ] P(structure template), P(digit chunk | length), P(symbol chunk | length), P(case mask), all smoothed.
- [ ] Letter chunks are scored by a letters-only n-gram model, so unseen words still get a probability.
- [ ] `logprob(pw)` = sum of the component log-probs. Also supports `sample(rng)`.

## Facade (`src/pwlab/model/facade.py`)
- [ ] `PasswordModel.train(counter)`, `.logprob(pw, kind="markov|structure|hybrid")`, `.sample(rng, kind)`, `.save/.load`.
- [ ] Hybrid = log-mixture of the two models, with λ tuned on validation.

## CLI
`python -m pwlab.cli train` → `models/model.pkl` and `reports/results/model_selection.csv`.

## Acceptance criteria
- [ ] `logprob("password") > logprob("Summer2026!") > logprob("x7#Qp!vR2m")`.
- [ ] Next-character distributions sum to 1 (±1e-9).
- [ ] Save/load round-trip gives identical scores.

## Tests (`tests/test_model.py`)
- Normalisation, non-zero probability for unseen strings, a hand-computed tiny-corpus example, round-trip.

