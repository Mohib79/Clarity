# Plan 07: Benchmarking & Security Analysis Report

**Goal:** Evaluate the meter and model empirically against baselines and write up the findings.

## Meter benchmark (`src/pwlab/eval/meter_eval.py`)
- [ ] Baselines (`src/pwlab/meter/baselines.py`):
  - **LUDS rule checker:** length ≥ 8 and character-class count → Weak/Medium/Strong.
  - **zxcvbn:** map score 0–1 to Weak, 2 to Medium, 3–4 to Strong. Also keep `guesses_log10`.
- [ ] Ground truth for each test password: its guessability under the model-based estimate and the frequency-baseline rank from Plan 04 (minimum across methods, following Ur et al. 2015).
- [ ] Metrics on a stratified test sample (e.g. 50k):
  - **Unsafe-Strong rate:** % rated Strong but guessable within 10⁶ (the key security metric).
  - **Over-Weak rate:** % rated Weak but not guessable within 10¹⁰ (hurts usability).
  - Spearman correlation between each meter's estimate and the ground truth.
  - Confusion matrices for each meter.
- [ ] Case-study table: about 15 illustrative common-pattern examples (`Password1!`, `P@ssw0rd`, `Summer2026!`, `qwerty123`, passphrases...) with the verdict from each meter.

## Figures (`src/pwlab/eval/plots.py` → `reports/figures/`)
- Coverage curve (from Plan 04), meter confusion matrices, scatter plot of estimated vs. ground-truth guesses, tag-frequency bar chart, structure-template top-20.

## Report (`reports/final_report.md`)
1. Introduction and motivation (why checklist meters fail)
2. Dataset and ethics
3. Method (segmentation, tagging, n-gram and structure models, MC estimation)
4. Attacker's View results
5. Defender's View results and baseline comparison
6. Security analysis: what human habits most reduce security, limitations (single-dataset bias, English-centric dictionaries, sampling error in MC), threats to validity
7. Recommendations for sign-up policies
8. Conclusion and references

## Acceptance criteria
- [ ] `python -m pwlab.cli evaluate --all` regenerates every table and figure from scratch.
- [ ] Report complete, with all figures referenced.

