# Plan 07: Benchmarking & Security Analysis Report

**Goal:** Evaluate the meter and model empirically against baselines and write up the findings.

## Meter benchmark (`src/pwlab/eval/meter_eval.py`)
- [x] Baselines (`src/pwlab/meter/baselines.py`):
  - **LUDS rule checker:** length ≥ 8 and character-class count → Weak/Medium/Strong.
  - **zxcvbn:** map score 0–1 to Weak, 2 to Medium, 3–4 to Strong.
- [x] Ground truth evaluation: Monte Carlo guess rank estimation across held-out test split.
- [x] Metrics on test sample:
  - **Unsafe-Strong rate:** % rated Strong but guessable within 10⁶.
  - **Over-Weak rate:** % rated Weak but requiring ≥ 10¹⁰ guesses.
  - Comparison table saved to `reports/results/meter_comparison.csv`.
- [x] Case-study table: 15 illustrative passwords (`Password1!`, `Summer2026!`, `P@ssw0rd2024!`, `qwerty123`, passphrases...) saved to `reports/results/case_studies.csv`.

## Figures (`src/pwlab/eval/plots.py` → `reports/figures/`)
- [x] `coverage_curve.png`: Model vs. frequency wordlist coverage curve.
- [x] `habit_frequencies.png`: Bar chart of top human predictable habits.
- [x] `top_structures.png`: Bar chart of top 10 structural grammar templates.
- [x] `meter_unsafe_strong_comparison.png`: Comparison of Unsafe-Strong failure rates.

## Report (`reports/final_report.md`)
- [x] Complete publication-grade academic term project report written, covering:
  1. Abstract & Introduction (why checklist meters fail)
  2. Dataset & Ethics
  3. Methodology (Segmentation, Pattern Tags, Markov & Structure Models, Monte Carlo Rank Estimation)
  4. Attacker's Perspective: Offline Guessability Evaluation
  5. Defender's Perspective: Explainable Strength Meter Evaluation
  6. Comparative Benchmarking vs. LUDS & zxcvbn
  7. 15-Password Case Studies
  8. Recommendations for Modern Authentication Policies
  9. Conclusion & Academic References (Weir et al., Dürmuth et al., Dell'Amico & Filippone, Ur et al.)

## Acceptance criteria
- [x] `python -m pwlab.cli evaluate --all` regenerates every table and figure from scratch.
- [x] Report complete with all figures and data referenced.
DONE!