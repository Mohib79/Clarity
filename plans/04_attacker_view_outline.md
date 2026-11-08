# Plan 04: Attacker's View: Offline Model Evaluation (Outline)

> [!NOTE]
> This plan is deliberately kept high-level. The team implements it from the published literature, strictly offline against the held-out test split of the research dataset.

## Objective
Show how well the trained model anticipates human password choices, compared with a simple frequency-wordlist baseline. This is the empirical evidence behind the strength meter.

## References to base the implementation on
- Weir et al., *Password Cracking Using Probabilistic Context-Free Grammars*, IEEE S&P 2009.
- Dürmuth et al., *OMEN: Faster Password Guessing Using an Ordered Markov Enumerator*, ESSoS 2015.
- Ma et al., *A Study of Probabilistic Password Models*, IEEE S&P 2014 (evaluation methodology).
- Ur et al., *Measuring Real-World Accuracies and Biases in Modeling Password Guessability*, USENIX Security 2015.

## Evaluation outline
1. Baseline: the frequency-ordered training list.
2. Model-based ranking from Plan 03, as described in the references.
3. Metric: % of test occurrences covered vs. guess budget (log scale, 10³–10⁷), plotted for each method.
4. Supplementary metric that needs no enumeration: **Monte Carlo guess-number estimates** for test passwords (re-use the Plan 05 estimator), giving the same kind of curve analytically.

## Deliverables
- `src/pwlab/eval/model_eval.py`, `reports/results/model_eval.csv`, `reports/figures/coverage_curve.png`.

## Constraints
- Plaintext comparison against our own test split only. No hashes, no external targets, no output files of generated lists in the repo.

