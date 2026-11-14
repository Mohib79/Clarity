# Clarity v2.0 — 4-Pillar Statistical & Deep Neural Password Modeling & Explainable Strength Evaluation

**Information Security (CS3002) Term Project**  
**Team:** Mohib Mansoor (23L-0650) · Maaz Mahmood (23L-0728) · Junaid Muhammad (23L-1014)  
---

## 1. Project Overview

Conventional password strength meters rely on static checklist rules (Length $\ge 8$, Uppercase, Lowercase, Digits, Symbols — LUDS). Under these checklist rules, predictable passwords like `Password1!` appear "Strong", despite being trivial for real-world statistical and neural models to crack.

**Clarity v2.0** implements a **4-Pillar Statistical & Deep Neural Super-Ensemble** trained on 250,000 unique leaked passwords (5.59 million occurrences from RockYou / SecLists):
1. **Kneser-Ney / Backoff Character N-Gram Markov Model (Order 5):** Captures local character transitions and continuation probabilities.
2. **Pattern-Aware Semantic PCFG (`StructureModel`):** Models structural templates (`L6D4S1`), lexicon root words, leetspeak un-leeting (`P@ssw0rd` $\rightarrow$ `password`), QWERTY keyboard walks (`1qaz2wsx`), and calendar years (`1950–2035`).
3. **Deep Autoregressive Character-GRU (`CharGRUModel`):** Gated Recurrent Unit neural network trained via Backpropagation Through Time (BPTT) and AdamW to capture long-range sequential dependencies.
4. **PassGPT Causal Decoder Transformer (`PassGPTModel`):** GPT-style Multi-Head Causal Self-Attention Transformer that models global sequence structure and computes **Per-Character Entropy / Surprise Heatmaps** ($-\log_2 P(c_i \mid c_{<i})$).

---

## 2. Headline Results & Findings

- **53% Reduction in Perplexity:** The 4-Pillar Super-Ensemble achieves a validation perplexity of **5.55** on held-out passwords (outperforming Markov-5 at `6.45`, Char-GRU at `8.97`, PassGPT at `9.63`, and basic Markov-4 at `11.90`).
- **The Checklist Fallacy:** `Password1!` passes 5/5 LUDS criteria ("Strong"), while Clarity proves it requires only $\approx 398,000$ guesses ($10^{5.6}$) and rates it **WEAK**. Similarly, leetspeak passwords like `P@ssw0rd2024!` are automatically un-leeted and rated **MEDIUM** ($10^{7.6}$).
- **Offensive Coverage:** Within a budget of $10^7$ guesses, the Clarity Super-Ensemble anticipates **74.72%** of unseen held-out test passwords (and **56.27%** within $10^6$ guesses), compared to **9.70%** for a Rule-Augmented Wordlist baseline.
- **0.00% Unsafe-Strong Failure Rate:** On a 5,000-password benchmark against LUDS and `zxcvbn`, Clarity achieves a **0.00%** Unsafe-Strong rate.

---

## 3. Running the Privacy-Preserving Web Application

Start the local web application:

```bash
.\.venv\Scripts\python.exe web/app.py
```
Open your browser at **`http://127.0.0.1:5000`**.

**Interactive v2.0 Features:**
- **Step 1 (Entry):** Live RAM evaluation with preset test cases.
- **Step 2 (Defender's View):** **Per-Character Entropy / Surprise Heatmap** ($-\log_2 P(c_i \mid c_{<i})$ bits per character), plain-language flaw explanations, and **Smart Hardened Alternatives** (click any generated alternative to test it live).
- **Step 3 (Attacker's View):** **4-Pillar Model Architecture Breakdown** (PassGPT vs. Char-GRU vs. KN-Markov vs. PCFG) and **Live Attacker Neighborhood Guess Stream**.
- **Step 4 (Baseline Verification):** Side-by-side comparison against LUDS Checklist and `zxcvbn`.

---

## 4. CLI Usage Guide

The project provides a unified CLI entry point via `python -m pwlab.cli`:

```bash
# 1. Auto-download RockYou corpus and create 180k/20k/50k train/val/test partitions
.\.venv\Scripts\python.exe -m pwlab.cli prepare --full

# 2. Analyze structural habits across training data
.\.venv\Scripts\python.exe -m pwlab.cli analyze-train --limit 50000

# 3. Train all 4 Pillars (Markov + PCFG + Char-GRU + PassGPT) & calibrate 50k Monte Carlo table
.\.venv\Scripts\python.exe -m pwlab.cli train

# 4. Generate ordered top candidate guesses using the Priority-Queue Guess Generator
.\.venv\Scripts\python.exe -m pwlab.cli generate --count 25

# 5. Score any password with 4-pillar breakdown & hardened alternatives
.\.venv\Scripts\python.exe -m pwlab.cli score "Password1!"
.\.venv\Scripts\python.exe -m pwlab.cli score "P@ssw0rd2024!"

# 6. Run full multi-model evaluation and regenerate all CSVs & publication figures
.\.venv\Scripts\python.exe -m pwlab.cli evaluate --all --test-limit 5000
```

---

## 5. Running Unit Tests

Run the full automated test suite (**70/70 tests passing**):

```bash
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

---

## 6. Reports & Publication Figures

- **Final Term Project Report:** [reports/final_report.md](reports/final_report.md)
- **Experimental Results:** [reports/results/](reports/results/)
  - `model_selection.csv`: Perplexity comparison across Markov orders, Char-GRU, PassGPT, and Super-Ensemble.
  - `model_eval.csv`: Multi-model test set crack coverage vs. guess budgets ($10^3$ to $10^7$).
  - `meter_comparison.csv`: Unsafe-Strong and Over-Weak failure rates across meters.
  - `case_studies.csv`: 15-password case study comparison table.
  - `pattern_stats.csv`: Prevalence of human habits in leaked passwords.
- **Publication Figures:** [reports/figures/](reports/figures/)
  - `coverage_curve.png`: Multi-model offline guessability coverage curve.
  - `model_perplexity_comparison.png`: Validation perplexity across statistical & neural architectures.
  - `habit_frequencies.png`: Prevalence of predictable human habits.
  - `top_structures.png`: Top structural grammar templates.
  - `meter_unsafe_strong_comparison.png`: Unsafe-Strong failure rate comparison.
