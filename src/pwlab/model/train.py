"""
Model training, hyperparameter tuning, validation, and serialization pipeline.
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pwlab.config import AppConfig, load_config
from pwlab.model.facade import PasswordModel
from pwlab.model.markov import MarkovModel


def load_dataset_tsv(path: str | Path, max_rows: Optional[int] = None) -> List[Tuple[str, int]]:
    """Load (password, count) tuples from a TSV partition file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"TSV dataset not found: {path}")

    items: List[Tuple[str, int]] = []
    with open(p, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        for row in reader:
            if not row:
                continue
            pw = row[0]
            cnt = int(row[1]) if len(row) > 1 and row[1].isdigit() else 1
            items.append((pw, cnt))
            if max_rows is not None and len(items) >= max_rows:
                break
    return items


def run_model_training_pipeline(
    config: Optional[AppConfig] = None,
    train_path: Optional[str] = None,
    val_path: Optional[str] = None,
    output_model_path: Optional[str] = None,
    output_selection_csv: Optional[str] = None,
    quick_tune: bool = False,
    max_train_rows: Optional[int] = None,
) -> Tuple[PasswordModel, Dict[str, any]]:
    """
    Run complete training pipeline:
    1. Grid search across n-gram orders and smoothing options on validation data.
    2. Write model_selection.csv.
    3. Train final PasswordModel with optimal settings.
    4. Save model to models/model.pkl.
    """
    if config is None:
        config = load_config()

    tr_path = train_path or config.paths.train_data
    v_path = val_path or config.paths.val_data
    out_model = Path(output_model_path or config.paths.model_file)
    out_csv = Path(output_selection_csv or "reports/results/model_selection.csv")

    out_model.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading training data from: {tr_path}")
    train_items = load_dataset_tsv(tr_path, max_rows=max_train_rows)
    print(f"    Loaded {len(train_items):,} training passwords.")

    print(f"[*] Loading validation data from: {v_path}")
    val_items = load_dataset_tsv(v_path, max_rows=5000 if quick_tune else 10000)
    print(f"    Loaded {len(val_items):,} validation passwords.")

    # 1. Hyperparameter tuning & model selection
    print("\n[*] Evaluating n-gram orders and smoothing configurations on validation slice...")
    candidate_configs = [
        {"order": 2, "smoothing": "backoff", "delta": 0.01},
        {"order": 3, "smoothing": "backoff", "delta": 0.01},
        {"order": 4, "smoothing": "backoff", "delta": 0.01},
        {"order": 4, "smoothing": "additive", "delta": 0.01},
        {"order": 5, "smoothing": "backoff", "delta": 0.01},
    ] if not quick_tune else [
        {"order": 3, "smoothing": "backoff", "delta": 0.01},
        {"order": 4, "smoothing": "backoff", "delta": 0.01},
        {"order": 4, "smoothing": "additive", "delta": 0.01},
    ]

    results = []
    best_config = None
    best_ppl = float("inf")

    # Train and evaluate each candidate Markov order on train and validation
    for cfg in candidate_configs:
        order = cfg["order"]
        smoothing = cfg["smoothing"]
        delta = cfg["delta"]

        m = MarkovModel(order=order, smoothing=smoothing, additive_delta=delta)
        m.train(train_items)
        val_ppl = m.perplexity(val_items)

        print(f"    Order {order} ({smoothing:<8}) -> Validation Perplexity: {val_ppl:.2f}")
        results.append({
            "order": order,
            "smoothing": smoothing,
            "delta": delta,
            "val_perplexity": val_ppl,
        })

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_config = cfg

    # Export model selection results
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Order", "Smoothing", "Delta", "Validation_Perplexity"])
        for r in results:
            writer.writerow([r["order"], r["smoothing"], r["delta"], f"{r['val_perplexity']:.2f}"])

    print(f"\n[+] Best Markov configuration: Order {best_config['order']} ({best_config['smoothing']}) with Perplexity {best_ppl:.2f}")

    # 2. Train final unified PasswordModel (Markov + Structure)
    print("\n[*] Training unified PasswordModel (Markov + Structure Grammar)...")
    final_model = PasswordModel(
        ngram_order=best_config["order"],
        smoothing=best_config["smoothing"],
        additive_delta=best_config["delta"],
        mixture_lambda=config.model.mixture_lambda,
    )

    metrics = final_model.train(train_items, val_items=val_items, tune_lambda=True)
    tuned_lambda = metrics.get("tuned_lambda", config.model.mixture_lambda)
    hybrid_ppl = metrics.get("val_perplexity_hybrid", best_ppl)

    print(f"    Optimal mixture lambda (Markov vs Structure): {tuned_lambda:.2f}")
    print(f"    Hybrid Validation Perplexity:                 {hybrid_ppl:.2f}")

    # 3. Save model
    print(f"\n[*] Saving trained model to: {out_model}")
    final_model.save(out_model)

    # 4. Sanity check log-probabilities
    sanity_passwords = ["password", "Summer2026!", "x7#Qp!vR2m"]
    print("\n[+] Sanity Check: Log-probabilities on benchmark inputs:")
    for pw in sanity_passwords:
        lp_m = final_model.logprob(pw, kind="markov")
        lp_s = final_model.logprob(pw, kind="structure")
        lp_h = final_model.logprob(pw, kind="hybrid")
        print(f"    '{pw:<15}' -> Markov: {lp_m:8.2f} | Struct: {lp_s:8.2f} | Hybrid: {lp_h:8.2f}")

    summary = {
        "best_config": best_config,
        "best_markov_ppl": best_ppl,
        "hybrid_ppl": hybrid_ppl,
        "tuned_lambda": tuned_lambda,
        "selection_csv": str(out_csv),
        "model_file": str(out_model),
    }

    return final_model, summary

