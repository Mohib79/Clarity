"""
Meter evaluation, comparative benchmarking, and case study generation.
Evaluates our Probabilistic Meter vs. LUDS Checklist and zxcvbn.
Computes Unsafe-Strong rates, Over-Weak rates, and generates case study tables.
"""

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pwlab.config import AppConfig, load_config
from pwlab.meter.scorer import PasswordScorer
from pwlab.meter.baselines import evaluate_luds, evaluate_zxcvbn
from pwlab.model.train import load_dataset_tsv


CASE_STUDY_PASSWORDS = [
    "123456",
    "password",
    "Password1!",
    "P@ssw0rd2024!",
    "Summer2026!",
    "qwerty123",
    "1qaz2wsx",
    "iloveyou99",
    "dragon2024",
    "admin2026!",
    "letmein123",
    "correct-horse-battery-staple",
    "blue-sky-flying-clouds",
    "x7#Qp!vR2m$9",
    "k9#mQ!vR2x$7pL0w",
]


def run_case_studies(
    scorer: Optional[PasswordScorer] = None,
    output_csv: str | Path = "reports/results/case_studies.csv",
) -> List[dict]:
    """Generate comparative case study table across 15 illustrative passwords."""
    if scorer is None:
        scorer = PasswordScorer()

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for pw in CASE_STUDY_PASSWORDS:
        res = scorer.score(pw, include_baselines=True)
        luds = res.baselines["luds"]
        zxc = res.baselines["zxcvbn"]

        primary_reason = res.reasons[0] if res.reasons else "N/A"
        row = {
            "Password": pw,
            "Template": res.structure,
            "Probabilistic_Meter": res.label,
            "Est_Log10_Guesses": f"10^{res.log10_guesses:.1f}",
            "LUDS_Checklist": f"{luds['label']} ({luds['score']})",
            "zxcvbn": f"{zxc['label']} ({zxc['score']}/4)",
            "Primary_Explanation": primary_reason,
        }
        rows.append(row)

    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return rows


def run_meter_benchmark(
    scorer: Optional[PasswordScorer] = None,
    test_items: Optional[List[Tuple[str, int]]] = None,
    sample_size: int = 5000,
    output_csv: str | Path = "reports/results/meter_comparison.csv",
) -> Dict[str, any]:
    """
    Benchmark meters on held-out test passwords.
    Measures:
    - Unsafe-Strong Rate: % of passwords rated 'Strong' by the meter,
      but guessable within 10^6 guesses (the critical security metric).
    - Over-Weak Rate: % of passwords rated 'Weak' by the meter,
      yet requiring >= 10^10 guesses (hurts usability).
    """
    if scorer is None:
        scorer = PasswordScorer()

    if test_items is None:
        cfg = load_config()
        test_items = load_dataset_tsv(cfg.paths.test_data, max_rows=sample_size)

    total_evaluated = len(test_items)
    print(f"[*] Benchmarking meters on {total_evaluated:,} test passwords...")

    # Metrics trackers
    stats = {
        "Probabilistic": {"Weak": 0, "Medium": 0, "Strong": 0, "unsafe_strong": 0, "over_weak": 0},
        "LUDS": {"Weak": 0, "Medium": 0, "Strong": 0, "unsafe_strong": 0, "over_weak": 0},
        "zxcvbn": {"Weak": 0, "Medium": 0, "Strong": 0, "unsafe_strong": 0, "over_weak": 0},
    }

    for idx, (pw, _) in enumerate(test_items):
        res = scorer.score(pw, include_baselines=True)
        luds_res = res.baselines["luds"]
        zxc_res = res.baselines["zxcvbn"]

        # Ground-truth guessability proxy: model Monte Carlo estimate
        est_log10 = res.log10_guesses
        is_empirically_guessable = est_log10 < 6.0  # < 10^6 guesses
        is_empirically_hard = est_log10 >= 10.0     # >= 10^10 guesses

        # 1. Our meter
        m_label = res.label
        stats["Probabilistic"][m_label] += 1
        if m_label == "Strong" and is_empirically_guessable:
            stats["Probabilistic"]["unsafe_strong"] += 1
        if m_label == "Weak" and is_empirically_hard:
            stats["Probabilistic"]["over_weak"] += 1

        # 2. LUDS
        l_label = luds_res["label"]
        stats["LUDS"][l_label] += 1
        if l_label == "Strong" and is_empirically_guessable:
            stats["LUDS"]["unsafe_strong"] += 1
        if l_label == "Weak" and is_empirically_hard:
            stats["LUDS"]["over_weak"] += 1

        # 3. zxcvbn
        z_label = zxc_res["label"]
        stats["zxcvbn"][z_label] += 1
        if z_label == "Strong" and is_empirically_guessable:
            stats["zxcvbn"]["unsafe_strong"] += 1
        if z_label == "Weak" and is_empirically_hard:
            stats["zxcvbn"]["over_weak"] += 1

    # Format summary table
    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for meter_name, data in stats.items():
        unsafe_pct = (data["unsafe_strong"] / max(data["Strong"], 1)) * 100.0 if data["Strong"] > 0 else 0.0
        over_weak_pct = (data["over_weak"] / max(data["Weak"], 1)) * 100.0 if data["Weak"] > 0 else 0.0
        summary_rows.append({
            "Meter": meter_name,
            "Weak_Count": data["Weak"],
            "Medium_Count": data["Medium"],
            "Strong_Count": data["Strong"],
            "Unsafe_Strong_Count": data["unsafe_strong"],
            "Unsafe_Strong_Rate": f"{unsafe_pct:.2f}%",
            "Over_Weak_Count": data["over_weak"],
            "Over_Weak_Rate": f"{over_weak_pct:.2f}%",
        })

    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "summary": summary_rows,
        "output_path": str(out_p),
    }

