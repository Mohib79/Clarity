# Clarity - Password Modeling & Explainable Strength Evaluation

**Information Security (CS3002) Term Project**  
---

## 1. Project Overview

Conventional password strength meters rely on static checklist rules (Length $\ge 8$, Uppercase, Lowercase, Digits, Symbols — LUDS). Under these checklist rules, predictable passwords like `Password1!` appear "Strong", despite being trivial for real-world statistical models and dictionary attacks to crack.

This project implements a complete statistical password modeling engine trained on public leaked-password research data (RockYou), evaluating password security across two complementary perspectives:

1. **Attacker's View (Offline Evaluation):** Measures candidate coverage curves across fixed guess budgets ($10^3$ to $10^7$) on a strictly held-out test split, comparing model anticipation against frequency wordlist baselines.
2. **Defender's View (Explainable Strength Meter):** Estimates how guessable a typed password is using a Monte Carlo rank estimator, assigning an actionable rating (**Weak / Medium / Strong**) alongside plain-language explanations of predictable human habits.

---

## 2. Key Results & Findings

- **The Checklist Fallacy:** Passwords like `Password1!` pass 5/5 criteria on standard LUDS checklist meters and are rated "Strong", but our probabilistic model proves they are guessable within $\approx 375,000$ guesses ($10^{5.6}$) and correctly rates them as **WEAK**.
- **Antagonist Coverage:** Within an attacker guess budget of $10^7$ guesses, the hybrid statistical model anticipates **49.65%** of novel passwords on a held-out test set, whereas a frequency wordlist baseline covers 0.00%.
- **Passphrase Security:** Long passphrases like `correct-horse-battery-staple` achieve $> 10^{34}$ guess complexity, proving far more secure than short 8-character mixed strings.

---

## 3. Installation & Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/Mohib79/Clarity.git
cd Clarity

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements and package in editable mode
pip install -r requirements.txt
pip install -e .
```

---


## 4. Running the Privacy-Preserving Web Application

Start the local web prototype:

```bash
python web/app.py
```
Open your browser at **`http://127.0.0.1:5000`**.

**Privacy Guarantees:**
- Passwords typed in the browser are processed strictly in volatile RAM.
- Server access logs for password requests are suppressed.
- Evaluated via `POST /api/score` (never GET), with `Cache-Control: no-store` headers.

---

## 5. CLI Usage Guide

The project provides a unified CLI entry point via `python -m pwlab.cli`:

### Step 1: Clean and Partition the Dataset
```bash
# Rapid development run (100k sample)
python -m pwlab.cli prepare --sample 100000

# Full dataset run (all cleaned entries from data/raw/rockyou.txt)
python -m pwlab.cli prepare --full
```

### Step 2: Analyze Structural Patterns in Training Data
```bash
python -m pwlab.cli analyze-train --limit 20000
```

### Step 3: Train and Tune the Statistical Model
```bash
python -m pwlab.cli train
```
*Trains character n-gram Markov models with backoff smoothing across orders 2–5, tunes mixture weight $\lambda$ on the held-out validation slice, and saves artifacts to `models/model.pkl`.*

### Step 4: Evaluate a Password's Strength (With Plain Explanations)
```bash
python -m pwlab.cli score "Password1!"
python -m pwlab.cli score "Summer2026!"
python -m pwlab.cli score "correct-horse-battery-staple"
```

## 6. Running Unit Tests

Run the full automated test suite (68 tests):

```bash
python -m pytest tests/ -v
```

---

## 7. Reports & Documentation

- **Final Term Project Report:** [reports/final_report.md](reports/final_report.md)
- **Experimental Results:** [reports/results/](reports/results/)
  - `model_eval.csv`: Test set crack coverage vs. guess budgets.
  - `meter_comparison.csv`: Unsafe-Strong failure rates across meters.
  - `case_studies.csv`: 15-password case study comparison table.
  - `pattern_stats.csv`: Prevalence of human habits in leaked passwords.
- **Publication Figures:** [reports/figures/](reports/figures/)
  - `coverage_curve.png`: Offline guessability coverage curve.
  - `habit_frequencies.png`: Prevalence of predictable human habits.
  - `top_structures.png`: Top structural grammar templates.
  - `meter_unsafe_strong_comparison.png`: Unsafe-Strong failure rate comparison.
