# Probabilistic Password Modeling & Explainable Strength Evaluation

**Information Security (IS) Term Project**  
*FAST-NUCES (Lahore)*

**Team Members:**
- Mohib Mansoor (23L-0650)
- Maaz Mahmood (23L-0728)
- Junaid Muhammad (23L-1014)

---

## 1. Project Overview

Conventional password strength meters rely on simple checklist rules (e.g. at least 8 characters, 1 uppercase, 1 number, 1 symbol). Under these static heuristics, predictable passwords like `Password1!` appear "strong", despite being trivial for real-world statistical models and dictionary-based attacks to guess.

This project implements a statistical password modeling engine trained on public leaked-password research datasets. The system provides two complementary perspectives:

1. **Attacker's View (Offline Evaluation):** Generates candidate password guesses in order of statistical likelihood and measures crack coverage across fixed guess budgets on a strictly held-out test split.
2. **Defender's View (Explainable Strength Meter):** Estimates how guessable a typed password is, assigning an actionable rating (**Weak / Medium / Strong**) alongside plain-language explanations of predictable human habits (e.g., dictionary words, leetspeak, years, sequences, keyboard patterns).

---

## 2. Ethics & Security Guidelines

- **Offline Research Only:** All modeling, training, and testing are conducted offline against public research corpora. No live attacks, network scanning, or unauthorized password testing.
- **Privacy Preservation:** The interactive strength meter prototype processes queries locally and **never stores, logs, or transmits** user passwords.
- **Data Hygiene:** Leaked password datasets remain in `data/raw/` and are strictly excluded from version control via `.gitignore`. Only aggregated metrics and summary plots are published.

---

## 3. Installation

Clone the repository and install dependencies in your virtual environment:

```bash
git clone https://github.com/Mohib79/IS_Project.git
cd IS_Project

# Create and activate virtual environment (optional but recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

## 4. Dataset Setup & Preparation

Place the research dataset (e.g., `rockyou.txt`) into `data/raw/rockyou.txt`.

To run the data cleaning, filtering, and 80/10/10 (train/val/test) split pipeline:

```bash
# Rapid development run (sampled subset of 500k entries)
python -m pwlab.cli prepare --sample 500000

# Full dataset run (all cleaned entries)
python -m pwlab.cli prepare --full
```

The preparation pipeline performs:
- Text decoding with fallback and newline normalization.
- Filtering to printable ASCII characters (0x20–0x7E) and length constraints [4, 32].
- Partitioning into disjoint `train.tsv`, `val.tsv`, and `test.tsv`.
- Exporting dataset statistics to `reports/results/data_stats.json`.

---

## 5. Running Tests

Run unit tests via `pytest`:

```bash
python -m pytest tests/ -v
```

---

## 6. Implementation Roadmap

- [x] **Plan 01:** Setup, Configuration, Data Cleaning & Partitioning
- [ ] **Plan 02:** Structural Segmentation (L/D/S) & Pattern Tagging
- [ ] **Plan 03:** Statistical n-gram Markov & Structure Models
- [ ] **Plan 04:** Attacker's View: Offline Likelihood-Ordered Evaluation
- [ ] **Plan 05:** Defender's View: Explainable Strength Meter
- [ ] **Plan 06:** Privacy-Preserving Web Prototype
- [ ] **Plan 07:** Empirical Benchmarks (vs. LUDS & zxcvbn) and Security Report

