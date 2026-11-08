"""
Analytical Model Evaluation and Benchmark Coverage Pipeline.
Evaluates model guessability and cumulative test set coverage across fixed guess budgets
against the frequency-wordlist baseline, and generates publication-grade figures.
"""

from collections import Counter
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from pwlab.config import AppConfig, load_config
from pwlab.model.facade import PasswordModel
from pwlab.meter.estimator import MonteCarloEstimator
from pwlab.model.train import load_dataset_tsv


def evaluate_test_set_coverage(
    model: PasswordModel,
    estimator: MonteCarloEstimator,
    train_items: List[Tuple[str, int]],
    test_items: List[Tuple[str, int]],
    budgets: Optional[List[int]] = None,
    output_csv: str | Path = "reports/results/model_eval.csv",
    output_plot: str | Path = "reports/figures/coverage_curve.png",
) -> Dict[str, any]:
    """
    Evaluates what fraction of held-out test passwords fall within each guess budget
    under the statistical model vs. the frequency-wordlist baseline.
    """
    if budgets is None:
        budgets = [1000, 10000, 100000, 1000000, 10000000]

    out_csv = Path(output_csv)
    out_plot = Path(output_plot)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting offline evaluation on {len(test_items):,} held-out test passwords...")

    # Build frequency baseline rank lookup from train items (sorted descending by count)
    train_sorted = sorted(train_items, key=lambda x: x[1], reverse=True)
    freq_rank_map = {}
    for rank_idx, (pw, _) in enumerate(train_sorted, start=1):
        if pw not in freq_rank_map:
            freq_rank_map[pw] = rank_idx

    total_test_occurrences = sum(cnt for _, cnt in test_items)

    # Accumulate hits per budget
    model_cracked = {b: 0 for b in budgets}
    baseline_cracked = {b: 0 for b in budgets}

    for idx, (pw, weight) in enumerate(test_items):
        if idx > 0 and idx % 5000 == 0:
            print(f"    ... evaluated {idx:,} / {len(test_items):,} test items", flush=True)

        # 1. Model estimated rank
        lp = model.logprob(pw, kind="hybrid")
        est_guesses, _ = estimator.estimate_guesses(lp)

        for b in budgets:
            if est_guesses <= b:
                model_cracked[b] += weight

        # 2. Baseline frequency rank
        base_rank = freq_rank_map.get(pw, float("inf"))
        for b in budgets:
            if base_rank <= b:
                baseline_cracked[b] += weight

    # Calculate percentages
    results_rows = []
    model_pcts = []
    baseline_pcts = []

    for b in budgets:
        m_pct = (model_cracked[b] / total_test_occurrences) * 100.0 if total_test_occurrences > 0 else 0.0
        b_pct = (baseline_cracked[b] / total_test_occurrences) * 100.0 if total_test_occurrences > 0 else 0.0
        model_pcts.append(m_pct)
        baseline_pcts.append(b_pct)
        results_rows.append({
            "budget": b,
            "budget_log10": np.log10(b),
            "model_cracked": model_cracked[b],
            "model_percentage": round(m_pct, 2),
            "baseline_cracked": baseline_cracked[b],
            "baseline_percentage": round(b_pct, 2),
        })

    # Save results to CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Guess_Budget", "Budget_Log10", "Model_Cracked", "Model_Percent", "Baseline_Cracked", "Baseline_Percent"])
        for r in results_rows:
            writer.writerow([r["budget"], f"{r['budget_log10']:.1f}", r["model_cracked"], f"{r['model_percentage']}%", r["baseline_cracked"], f"{r['baseline_percentage']}%"])

    print(f"\n[+] Offline Evaluation Results saved to: {out_csv}")
    print("    ---------------------------------------------------------------")
    print("    Budget      Log10    Statistical Model (%)    Frequency Baseline (%)")
    print("    ---------------------------------------------------------------")
    for r in results_rows:
        print(f"    {r['budget']:<10,d}  10^{r['budget_log10']:.0f}      {r['model_percentage']:>8.2f}%              {r['baseline_percentage']:>8.2f}%")
    print("    ---------------------------------------------------------------")

    # Generate plot using Matplotlib
    try:
        plt.figure(figsize=(8, 5), dpi=300)
        plt.plot(budgets, model_pcts, marker="o", color="#2563eb", linewidth=2.5, label="Statistical Model (Hybrid)")
        plt.plot(budgets, baseline_pcts, marker="s", color="#dc2626", linewidth=2.0, linestyle="--", label="Frequency Wordlist Baseline")

        plt.xscale("log")
        plt.xlabel("Attacker Guess Budget (Log Scale)", fontsize=11, fontweight="bold")
        plt.ylabel("Test Set Crack Coverage (%)", fontsize=11, fontweight="bold")
        plt.title("Offline Password Guessability: Model vs. Frequency Baseline", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="lower right", fontsize=10)
        plt.tight_layout()

        plt.savefig(out_plot)
        plt.close()
        print(f"[+] Coverage curve chart saved to: {out_plot}")
    except Exception as e:
        print(f"[-] Could not render plot: {e}")

    return {
        "results": results_rows,
        "csv_path": str(out_csv),
        "plot_path": str(out_plot),
    }


def run_evaluation_cli(config: Optional[AppConfig] = None, test_limit: int = 10000):
    """CLI runner for offline evaluation."""
    if config is None:
        config = load_config()

    print("[*] Loading PasswordModel and MonteCarloEstimator...")
    model = PasswordModel.load(config.paths.model_file)
    estimator = MonteCarloEstimator.load(config.paths.mc_table)

    print("[*] Loading datasets...")
    train_items = load_dataset_tsv(config.paths.train_data)
    test_items = load_dataset_tsv(config.paths.test_data, max_rows=test_limit)

    evaluate_test_set_coverage(
        model=model,
        estimator=estimator,
        train_items=train_items,
        test_items=test_items,
        budgets=config.evaluation.guess_budgets,
        output_csv=config.paths.results_dir + "/model_eval.csv",
        output_plot=config.paths.figures_dir + "/coverage_curve.png",
    )

