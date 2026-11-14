"""
Plotting and figure generation for research report.
Generates publication-quality charts using Matplotlib (no external pandas dependency).
"""

import csv
from pathlib import Path
import matplotlib.pyplot as plt


def generate_all_figures(
    pattern_stats_csv: str = "reports/results/pattern_stats.csv",
    meter_comparison_csv: str = "reports/results/meter_comparison.csv",
    model_selection_csv: str = "reports/results/model_selection.csv",
    output_dir: str = "reports/figures",
):
    """Generate all analytical figures from experimental CSV results."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Habit Frequencies Plot
    p_stat = Path(pattern_stats_csv)
    if p_stat.exists():
        habits = []
        counts = []
        with open(p_stat, "r", encoding="utf-8") as f:
            lines = f.readlines()
            capture = False
            for line in lines:
                if "Human Habit Prevalence" in line:
                    capture = True
                    continue
                if capture and line.startswith("Habit,"):
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        habits.append(parts[1].replace("position:", "").replace("_", " ").title())
                        counts.append(int(parts[2]))

        if habits and counts:
            plt.figure(figsize=(9, 5), dpi=300)
            y_pos = range(len(habits[:8]))
            plt.barh(y_pos, counts[:8], color="#3b82f6", edgecolor="#1d4ed8")
            plt.yticks(y_pos, habits[:8], fontsize=10)
            plt.gca().invert_yaxis()
            plt.xlabel("Occurrences in Training Corpus", fontsize=11, fontweight="bold")
            plt.title("Prevalence of Human Predictable Habits in Leaked Passwords", fontsize=12, fontweight="bold")
            plt.grid(axis="x", linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(out_dir / "habit_frequencies.png")
            plt.close()
            print(f"[+] Saved: {out_dir / 'habit_frequencies.png'}")

    # 2. Top Structural Templates Plot
    if p_stat.exists():
        structures = []
        s_counts = []
        with open(p_stat, "r", encoding="utf-8") as f:
            lines = f.readlines()
            capture = False
            for line in lines:
                if "Top 20 Structural Templates" in line:
                    capture = True
                    continue
                if capture and line.startswith("---"):
                    break
                if capture and line.startswith("Structure,"):
                    parts = line.strip().split(",")
                    if len(parts) >= 3:
                        structures.append(parts[1])
                        s_counts.append(int(parts[2]))

        if structures and s_counts:
            plt.figure(figsize=(9, 5), dpi=300)
            x_pos = range(len(structures[:10]))
            plt.bar(x_pos, s_counts[:10], color="#6366f1", edgecolor="#4338ca")
            plt.xticks(x_pos, structures[:10], rotation=45, fontsize=10, fontweight="bold")
            plt.ylabel("Occurrences in Training Corpus", fontsize=11, fontweight="bold")
            plt.title("Top 10 Structural Grammar Templates in Leaked Passwords", fontsize=12, fontweight="bold")
            plt.grid(axis="y", linestyle=":", alpha=0.6)
            plt.tight_layout()
            plt.savefig(out_dir / "top_structures.png")
            plt.close()
            print(f"[+] Saved: {out_dir / 'top_structures.png'}")

    # 3. Meter Unsafe-Strong Failure Rate Comparison Plot
    p_comp = Path(meter_comparison_csv)
    if p_comp.exists():
        meters = []
        unsafe_rates = []
        with open(p_comp, "r", encoding="utf-8") as f:
            reader = list(f.readlines())
            for line in reader[1:]:
                parts = line.strip().split(",")
                if len(parts) >= 6:
                    meters.append(parts[0])
                    rate_str = parts[5].replace("%", "")
                    unsafe_rates.append(float(rate_str) if rate_str else 0.0)

        if meters:
            plt.figure(figsize=(7, 4.5), dpi=300)
            colors = ["#10b981", "#ef4444", "#3b82f6"]
            bars = plt.bar(meters, unsafe_rates, color=colors[:len(meters)], width=0.55)
            plt.ylabel("Unsafe-Strong Failure Rate (%)", fontsize=11, fontweight="bold")
            plt.title("Security Comparison: % of 'Strong' Ratings Guessable within 10^6", fontsize=11, fontweight="bold")
            plt.grid(axis="y", linestyle=":", alpha=0.6)
            for bar in bars:
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
            plt.ylim(0, max(max(unsafe_rates) * 1.25, 10))
            plt.tight_layout()
            plt.savefig(out_dir / "meter_unsafe_strong_comparison.png")
            plt.close()
            print(f"[+] Saved: {out_dir / 'meter_unsafe_strong_comparison.png'}")

    # 4. Multi-Architecture Perplexity Comparison Plot
    p_sel = Path(model_selection_csv)
    if p_sel.exists():
        model_names = []
        ppls = []
        with open(p_sel, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                m_name = row.get("Model") or f"Order-{row.get('Order')} ({row.get('Smoothing')})"
                val_p = float(row.get("Validation_Perplexity", 0.0))
                if val_p > 0:
                    model_names.append(m_name)
                    ppls.append(val_p)

        if model_names and ppls:
            plt.figure(figsize=(9.5, 5), dpi=300)
            bar_colors = [
                "#1d4ed8" if "Ensemble" in m else ("#7c3aed" if "PassGPT" in m or "GRU" in m else "#64748b")
                for m in model_names
            ]
            bars = plt.barh(range(len(model_names)), ppls, color=bar_colors, height=0.6)
            plt.yticks(range(len(model_names)), model_names, fontsize=9.5, fontweight="bold")
            plt.gca().invert_yaxis()
            plt.xlabel("Held-Out Validation Perplexity (Lower is Better)", fontsize=11, fontweight="bold")
            plt.title("Model Architecture Comparison: Statistical N-Grams vs. Deep Neural & Super-Ensemble", fontsize=11.5, fontweight="bold")
            plt.grid(axis="x", linestyle=":", alpha=0.6)
            for bar in bars:
                w = bar.get_width()
                plt.text(w + 0.15, bar.get_y() + bar.get_height() / 2.0, f"{w:.2f}", va="center", ha="left", fontsize=9, fontweight="bold")
            plt.xlim(0, max(ppls) * 1.15)
            plt.tight_layout()
            plt.savefig(out_dir / "model_perplexity_comparison.png")
            plt.close()
            print(f"[+] Saved: {out_dir / 'model_perplexity_comparison.png'}")
