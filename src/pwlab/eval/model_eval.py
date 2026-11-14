"""
Analytical Multi-Model Evaluation and Benchmark Coverage Pipeline.
Evaluates password guessability and cumulative test set coverage across fixed guess budgets
for all 4 Pillars (Kneser-Ney Markov, Pattern-Aware PCFG, Char-GRU, PassGPT Transformer)
plus the Clarity Super-Ensemble against a Rule-Augmented Wordlist Baseline.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from pwlab.config import AppConfig, load_config
from pwlab.model.facade import PasswordModel
from pwlab.meter.estimator import MonteCarloEstimator
from pwlab.model.train import load_dataset_tsv


def _build_rule_wordlist_ranks(train_items: List[Tuple[str, int]], max_roots: int = 15000) -> Dict[str, int]:
    """
    Build a realistic Attacker Wordlist + Basic Mangling Rules rank table from top training roots.
    Unlike a naive exact-match lookup on a disjoint split, a rule-based wordlist applies
    standard single-step mutations (Capitalize, append '1', '123', '!', '2024', '2026')
    to top roots in descending frequency order.
    """
    train_sorted = sorted(train_items, key=lambda x: x[1], reverse=True)
    rank_map: Dict[str, int] = {}
    rank_counter = 1

    # Pass 1: Direct wordlist entries
    for pw, _ in train_sorted:
        if pw not in rank_map:
            rank_map[pw] = rank_counter
            rank_counter += 1

    # Pass 2: Standard mangling rules (Best-12 style) on top roots
    suffixes = ["1", "123", "!", "1!", "2024", "2025", "2026", "99", "01", "12"]
    for pw, _ in train_sorted[:max_roots]:
        root = pw.lower()
        variants = [root.capitalize()]
        for s in suffixes:
            variants.append(root + s)
            variants.append(root.capitalize() + s)
        for v in variants:
            if v not in rank_map:
                rank_map[v] = rank_counter
                rank_counter += 1

    return rank_map


def evaluate_test_set_coverage(
    model: PasswordModel,
    estimator: MonteCarloEstimator,
    train_items: List[Tuple[str, int]],
    test_items: List[Tuple[str, int]],
    budgets: Optional[List[int]] = None,
    output_csv: str | Path = "reports/results/model_eval.csv",
    output_plot: str | Path = "reports/figures/coverage_curve.png",
) -> Dict[str, Any]:
    """
    Evaluates what fraction of held-out test passwords fall within each guess budget
    across all model pillars vs. the rule-augmented frequency wordlist baseline.
    """
    if budgets is None:
        budgets = [1000, 10000, 100000, 1000000, 10000000]

    out_csv = Path(output_csv)
    out_plot = Path(output_plot)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_plot.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Starting multi-model offline evaluation on {len(test_items):,} held-out test passwords...")

    freq_rank_map = _build_rule_wordlist_ranks(train_items)
    total_test_occurrences = sum(cnt for _, cnt in test_items)

    # Batch compute neural logprobs if available for fast evaluation
    test_pws = [pw for pw, _ in test_items]
    lp_markov_list = [model.markov_model.logprob(p) for p in test_pws]
    lp_struct_list = [model.structure_model.logprob(p) for p in test_pws]
    lp_gru_list = (
        model.gru_model.logprob_batch(test_pws, batch_size=512)
        if (model.gru_model and model.gru_model.is_trained)
        else lp_markov_list
    )
    lp_gpt_list = (
        model.gpt_model.logprob_batch(test_pws, batch_size=512)
        if (model.gpt_model and model.gpt_model.is_trained)
        else lp_markov_list
    )
    lp_hybrid_list = [model.logprob(p, kind="hybrid") for p in test_pws]

    cracked_by_kind: Dict[str, Dict[int, int]] = {
        "ensemble": {b: 0 for b in budgets},
        "gpt": {b: 0 for b in budgets},
        "gru": {b: 0 for b in budgets},
        "markov": {b: 0 for b in budgets},
        "pcfg": {b: 0 for b in budgets},
        "baseline": {b: 0 for b in budgets},
    }

    for idx, (pw, weight) in enumerate(test_items):
        est_ens, _ = estimator.estimate_guesses(lp_hybrid_list[idx])
        est_gpt, _ = estimator.estimate_guesses(lp_gpt_list[idx])
        est_gru, _ = estimator.estimate_guesses(lp_gru_list[idx])
        est_m, _ = estimator.estimate_guesses(lp_markov_list[idx])
        est_s, _ = estimator.estimate_guesses(lp_struct_list[idx])
        base_rank = freq_rank_map.get(pw, float("inf"))

        for b in budgets:
            if est_ens <= b:
                cracked_by_kind["ensemble"][b] += weight
            if est_gpt <= b:
                cracked_by_kind["gpt"][b] += weight
            if est_gru <= b:
                cracked_by_kind["gru"][b] += weight
            if est_m <= b:
                cracked_by_kind["markov"][b] += weight
            if est_s <= b:
                cracked_by_kind["pcfg"][b] += weight
            if base_rank <= b:
                cracked_by_kind["baseline"][b] += weight

    results_rows = []
    pct_curves: Dict[str, List[float]] = {k: [] for k in cracked_by_kind}

    for b in budgets:
        for k in cracked_by_kind:
            pct = (cracked_by_kind[k][b] / total_test_occurrences) * 100.0 if total_test_occurrences > 0 else 0.0
            pct_curves[k].append(round(pct, 2))

        results_rows.append({
            "budget": b,
            "budget_log10": np.log10(b),
            "model_cracked": cracked_by_kind["ensemble"][b],
            "model_percentage": pct_curves["ensemble"][-1],
            "gpt_percentage": pct_curves["gpt"][-1],
            "gru_percentage": pct_curves["gru"][-1],
            "markov_percentage": pct_curves["markov"][-1],
            "pcfg_percentage": pct_curves["pcfg"][-1],
            "baseline_cracked": cracked_by_kind["baseline"][b],
            "baseline_percentage": pct_curves["baseline"][-1],
        })

    # Save results to CSV
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Guess_Budget", "Budget_Log10",
            "Model_Cracked", "Model_Percent",
            "PassGPT_Percent", "CharGRU_Percent", "Markov_Percent", "PCFG_Percent",
            "Baseline_Cracked", "Baseline_Percent",
        ])
        for r in results_rows:
            writer.writerow([
                r["budget"], f"{r['budget_log10']:.1f}",
                r["model_cracked"], f"{r['model_percentage']}%",
                f"{r['gpt_percentage']}%", f"{r['gru_percentage']}%",
                f"{r['markov_percentage']}%", f"{r['pcfg_percentage']}%",
                r["baseline_cracked"], f"{r['baseline_percentage']}%",
            ])

    print(f"\n[+] Offline Evaluation Results saved to: {out_csv}")
    print("    -----------------------------------------------------------------------------------------")
    print("    Budget      Ensemble    PassGPT     Char-GRU    KN-Markov   PCFG        Rule-Wordlist")
    print("    -----------------------------------------------------------------------------------------")
    for r in results_rows:
        print(
            f"    10^{r['budget_log10']:.0f}       {r['model_percentage']:>6.2f}%     "
            f"{r['gpt_percentage']:>6.2f}%     {r['gru_percentage']:>6.2f}%     "
            f"{r['markov_percentage']:>6.2f}%     {r['pcfg_percentage']:>6.2f}%      {r['baseline_percentage']:>6.2f}%"
        )
    print("    -----------------------------------------------------------------------------------------")

    # Generate multi-model publication figure
    try:
        plt.figure(figsize=(9, 5.5), dpi=300)
        plt.plot(budgets, pct_curves["ensemble"], marker="o", color="#1d4ed8", linewidth=2.8, label="Clarity Super-Ensemble (All 4 Pillars)")
        plt.plot(budgets, pct_curves["gpt"], marker="^", color="#7c3aed", linewidth=2.0, linestyle="-.", label="PassGPT (Causal Transformer)")
        plt.plot(budgets, pct_curves["gru"], marker="d", color="#059669", linewidth=2.0, linestyle="--", label="Char-GRU (3-Layer Deep RNN)")
        plt.plot(budgets, pct_curves["markov"], marker="v", color="#d97706", linewidth=1.8, linestyle=":", label="Kneser-Ney 4-Gram Markov")
        plt.plot(budgets, pct_curves["pcfg"], marker="p", color="#0891b2", linewidth=1.8, linestyle=":", label="Pattern-Aware PCFG")
        plt.plot(budgets, pct_curves["baseline"], marker="s", color="#dc2626", linewidth=2.0, linestyle="--", label="Wordlist + Mangling Rules Baseline")

        plt.xscale("log")
        plt.xlabel("Attacker Guess Budget (Log Scale)", fontsize=11, fontweight="bold")
        plt.ylabel("Held-Out Test Set Crack Coverage (%)", fontsize=11, fontweight="bold")
        plt.title("Offline Password Guessability Across Architectures (RockYou Benchmark)", fontsize=12, fontweight="bold")
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(loc="upper left", fontsize=9.5, framealpha=0.92)
        plt.tight_layout()

        plt.savefig(out_plot)
        plt.close()
        print(f"[+] Multi-model coverage curve chart saved to: {out_plot}")
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
