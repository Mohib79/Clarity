"""
Corpus-level pattern analysis and statistics generation.
Computes frequency of structural templates and human habits across the dataset.
"""

from collections import Counter, defaultdict
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

from pwlab.patterns.segmenter import structure
from pwlab.patterns.taggers import analyze, PatternAnalyzer


def compute_pattern_stats(
    train_path: str | Path,
    output_csv: str | Path = "reports/results/pattern_stats.csv",
    max_passwords: Optional[int] = 50000,
    analyzer: Optional[PatternAnalyzer] = None,
) -> Dict[str, any]:
    """
    Analyze patterns across a training split and export aggregate statistics.
    """
    p = Path(train_path)
    if not p.exists():
        raise FileNotFoundError(f"Training data not found at: {train_path}")

    if analyzer is None:
        analyzer = PatternAnalyzer()

    structure_counts: Counter = Counter()
    tag_counts: Counter = Counter()
    total_evaluated = 0
    total_weight = 0

    with open(p, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)

        for idx, row in enumerate(reader):
            if not row:
                continue
            pw = row[0]
            count = int(row[1]) if len(row) > 1 and row[1].isdigit() else 1

            struct = structure(pw)
            structure_counts[struct] += count

            res = analyzer.analyze(pw)
            seen_types_for_pw = set()
            for t in res.tags:
                sub_key = t.tag_type
                if t.tag_type == "position_habit" and "habit" in t.detail:
                    sub_key = f"position:{t.detail['habit']}"
                seen_types_for_pw.add(sub_key)

            for st in seen_types_for_pw:
                tag_counts[st] += count

            total_evaluated += 1
            total_weight += count

            if max_passwords is not None and total_evaluated >= max_passwords:
                break

    out_p = Path(output_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    with open(out_p, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Name", "Count", "Percentage"])

        writer.writerow(["Metadata", "Total_Evaluated", total_evaluated, "100.0%"])
        writer.writerow(["Metadata", "Total_Weight", total_weight, "100.0%"])

        writer.writerow([])
        writer.writerow(["--- Top 20 Structural Templates ---", "", "", ""])
        for struct, count in structure_counts.most_common(20):
            pct = (count / total_weight) * 100 if total_weight > 0 else 0
            writer.writerow(["Structure", struct, count, f"{pct:.2f}%"])

        writer.writerow([])
        writer.writerow(["--- Human Habit Prevalence ---", "", "", ""])
        for tag_type, count in tag_counts.most_common():
            pct = (count / total_weight) * 100 if total_weight > 0 else 0
            writer.writerow(["Habit", tag_type, count, f"{pct:.2f}%"])

    return {
        "total_evaluated": total_evaluated,
        "total_weight": total_weight,
        "top_structures": structure_counts.most_common(10),
        "tag_prevalence": dict(tag_counts.most_common()),
        "output_path": str(out_p),
    }

