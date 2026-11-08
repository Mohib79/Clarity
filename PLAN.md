# Master Plan: Probabilistic Password Modeling & Explainable Strength Evaluation

**Course:** CS3002 Information Security (Term Project)  
**Institution:** FAST-NUCES (Lahore)  
**Team:** Mohib Mansoor (23L-0650) · Maaz Mahmood (23L-0728) · Junaid Muhammad (23L-1014)

---

## 1. Project Summary

We trained a statistical password model (a character-level n-gram Markov model paired with a structural pattern splitter) on public leaked-password research data (RockYou), using a strict disjoint 80% train / 10% validation / 10% test split. We deployed and benchmarked the system across two complementary perspectives:

1. **Attacker's View (Offline Evaluation):** Measures offline candidate coverage curves across fixed guess budgets ($10^3$ to $10^7$) on a held-out test split, comparing model anticipation against frequency wordlist baselines.
2. **Defender's View (Explainable Strength Meter):** Estimates the required guess budget using Monte Carlo sampling, rates the password (**Weak / Medium / Strong**), and explains why in plain language with actionable guidance.

---

## 2. Architecture & Pipeline

```
Raw Leak Dataset (RockYou)
          │
          ▼
┌─────────────────────────────────┐
│ 1. Data Cleaning & 80/10/10     │
│    Partitioning (train/val/test)│
└────────────────┬────────────────┘
                 │
        ┌────────┴──────────────────┐
        ▼                           ▼
┌──────────────────┐       ┌──────────────────┐
│ 2. Segmenter &   │       │ 3. Character     │
│    Pattern Tags  │       │    n-gram Markov │
│    (L / D / S)   │       │    (Backoff)     │
└───────┬──────────┘       └────────┬─────────┘
        │                           │
        └─────────┬─────────────────┘
                  ▼
┌─────────────────────────────────────┐
│ 4. Unified Hybrid PasswordModel     │
│    ln P(w) = log-sum-exp(markov, s) │
└─────────────────┬───────────────────┘
                  │
        ┌─────────┴───────────────────────┐
        ▼                                 ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│ Attacker's View (Eval)  │   │ Defender's View (Meter) │
│ - Test Set Guess Rank   │   │ - Monte Carlo Lookup    │
│ - Coverage Curve vs.    │   │ - Weak/Med/Strong Score │
│   Frequency Baseline    │   │ - Plain Explanations    │
└───────────┬─────────────┘   └───────────┬─────────────┘
            │                             │
            └──────────────┬──────────────┘
                           ▼
              ┌─────────────────────────┐
              │ 6. Web Prototype (Flask)│
              │ 7. Academic Report &    │
              │    Benchmark Figures    │
              └─────────────────────────┘
```

---

## 3. Implementation Status Across Phases

| # | Plan File | Status | Key Deliverables |
|---|---|---|---|
| 01 | [01_setup_and_data.md](plans/01_setup_and_data.md) | **Completed** | Repo skeleton, `.gitignore`, `config.yaml`, data cleaning, disjoint 80/10/10 partitions |
| 02 | [02_segmentation_and_tagging.md](plans/02_segmentation_and_tagging.md) | **Completed** | L/D/S segmenter, 7 habit taggers (dictionary, leet, years, walks, repeats), pattern statistics |
| 03 | [03_statistical_model.md](plans/03_statistical_model.md) | **Completed** | Character n-gram Markov (backoff), structure model, hybrid log-sum-exp facade |
| 04 | [04_attacker_view_outline.md](plans/04_attacker_view_outline.md) | **Completed** | Offline test set coverage evaluation, `coverage_curve.png`, comparison vs frequency baseline |
| 05 | [05_strength_meter.md](plans/05_strength_meter.md) | **Completed** | Monte Carlo rank estimator, Weak/Med/Strong scorer, plain explanations, LUDS & zxcvbn baselines |
| 06 | [06_web_prototype.md](plans/06_web_prototype.md) | **Completed** | Privacy-preserving Flask app, zero logging, no-cache headers, interactive UI |
| 07 | [07_benchmarking_and_report.md](plans/07_benchmarking_and_report.md) | **Completed** | Meter benchmarks, 15 case studies, publication figures, complete term project report |

---

## 4. Key Findings

1. **The Fallacy of Checklist Meters:**  
   Passwords like `Password1!` pass 5 out of 5 criteria on standard checklist policies (LUDS) and are falsely rated "Strong", whereas our probabilistic model identifies that they require only $\approx 375,000$ guesses ($10^{5.6}$) and correctly rates them as **WEAK**.
2. **Offensive Anticipation:**  
   Within an attacker guess budget of $10^7$ guesses, the statistical model anticipates **49.65%** of novel passwords in a held-out test split, where a frequency dictionary baseline achieves 0.00%.
3. **Security Reliability:**  
   Our probabilistic strength meter achieves a **0.00% Unsafe-Strong failure rate** on empirical test sets.

---

## 5. Definition of Done Checklist

- [x] All 7 implementation plans completed and verified.
- [x] `python -m pwlab.cli prepare | analyze | analyze-train | train | score | evaluate` all functional.
- [x] Full test suite passing (**68/68 unit tests** in 2.3 seconds).
- [x] Local privacy-preserving web application running at `http://127.0.0.1:5000`.
- [x] Comprehensive research report written at [reports/final_report.md](reports/final_report.md) with all figures in [reports/figures/](reports/figures/).
