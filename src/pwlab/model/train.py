"""
Model training, hyperparameter tuning, neural GPU training (Char-GRU & PassGPT),
Monte Carlo calibration, and serialization pipeline.
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pwlab.config import AppConfig, load_config
from pwlab.model.facade import PasswordModel
from pwlab.model.markov import MarkovModel
from pwlab.meter.estimator import MonteCarloEstimator


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
    train_neural: bool = True,
    neural_epochs: int = 4,
    calibrate_mc: bool = True,
) -> Tuple[PasswordModel, Dict[str, Any]]:
    """
    Run complete 4-Pillar training pipeline:
    1. Grid search across n-gram orders and smoothing options on validation data.
    2. Train Pattern-Aware PCFG, CUDA Char-GRU, and CUDA PassGPT Transformer.
    3. Tune 4-Pillar Super-Ensemble weights on held-out validation perplexity.
    4. Export complete model_selection.csv across all architectures.
    5. Save model artifacts and calibrate Monte Carlo rank estimator table (mc_table.npz).
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
    val_items = load_dataset_tsv(v_path, max_rows=3000 if quick_tune else 8000)
    print(f"    Loaded {len(val_items):,} validation passwords.")

    # 1. Hyperparameter tuning & model selection for Markov models
    print("\n[*] Evaluating Markov n-gram orders & Kneser-Ney / Backoff smoothing on validation slice...")
    candidate_configs = [
        {"model": "Markov-2", "order": 2, "smoothing": "backoff", "delta": 0.01},
        {"model": "Markov-3", "order": 3, "smoothing": "backoff", "delta": 0.01},
        {"model": "Markov-4 (Additive)", "order": 4, "smoothing": "additive", "delta": 0.01},
        {"model": "Markov-4 (Backoff)", "order": 4, "smoothing": "backoff", "delta": 0.01},
        {"model": "Markov-4 (Kneser-Ney)", "order": 4, "smoothing": "kneser_ney", "delta": 0.01},
        {"model": "Markov-5 (Backoff)", "order": 5, "smoothing": "backoff", "delta": 0.01},
    ] if not quick_tune else [
        {"model": "Markov-3", "order": 3, "smoothing": "backoff", "delta": 0.01},
        {"model": "Markov-4 (Backoff)", "order": 4, "smoothing": "backoff", "delta": 0.01},
        {"model": "Markov-4 (Kneser-Ney)", "order": 4, "smoothing": "kneser_ney", "delta": 0.01},
    ]

    results = []
    best_config = None
    best_ppl = float("inf")

    for cfg in candidate_configs:
        order = cfg["order"]
        smoothing = cfg["smoothing"]
        delta = cfg["delta"]

        m = MarkovModel(order=order, smoothing=smoothing, additive_delta=delta)
        m.train(train_items)
        val_ppl = m.perplexity(val_items)

        print(f"    {cfg['model']:<22} -> Validation Perplexity: {val_ppl:.2f}")
        results.append({
            "model": cfg["model"],
            "order": order,
            "smoothing": smoothing,
            "delta": delta,
            "val_perplexity": val_ppl,
        })

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_config = cfg

    print(f"\n[+] Best Markov configuration: {best_config['model']} with Perplexity {best_ppl:.2f}")

    # 2. Train unified 4-Pillar PasswordModel (Markov + Pattern-Aware PCFG + Char-GRU + PassGPT)
    print("\n[*] Training Unified 4-Pillar PasswordModel (Kneser-Ney Markov + PCFG + Char-GRU + PassGPT)...")
    final_model = PasswordModel(
        ngram_order=best_config["order"],
        smoothing=best_config["smoothing"],
        additive_delta=best_config["delta"],
        mixture_lambda=config.model.mixture_lambda,
    )

    hw_info = final_model.get_hardware_info()
    print(f"    Hardware Target: {hw_info['device_name']} (CUDA={hw_info['cuda_available']})")

    metrics = final_model.train(
        train_items,
        val_items=val_items,
        tune_lambda=True,
        train_neural=train_neural,
        neural_epochs=2 if quick_tune else neural_epochs,
    )

    tuned_lambda = metrics.get("tuned_lambda", config.model.mixture_lambda)
    hybrid_ppl = metrics.get("val_perplexity_hybrid", best_ppl)
    gru_ppl = metrics.get("val_perplexity_gru", 0.0)
    gpt_ppl = metrics.get("val_perplexity_gpt", 0.0)

    if gru_ppl > 0:
        results.append({"model": "Char-GRU (3-Layer Deep RNN)", "order": 34, "smoothing": "neural_gru", "delta": 0.0, "val_perplexity": gru_ppl})
    if gpt_ppl > 0:
        results.append({"model": "PassGPT (4-Layer Transformer)", "order": 34, "smoothing": "causal_attn", "delta": 0.0, "val_perplexity": gpt_ppl})
    results.append({"model": "Clarity Super-Ensemble", "order": best_config["order"], "smoothing": "ensemble", "delta": 0.0, "val_perplexity": hybrid_ppl})

    # Export model selection results
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Order", "Smoothing", "Delta", "Validation_Perplexity"])
        for r in results:
            writer.writerow([r["model"], r["order"], r["smoothing"], r["delta"], f"{r['val_perplexity']:.2f}"])

    print(f"\n    Ensemble Weights:             {final_model.ensemble_weights}")
    if gru_ppl > 0:
        print(f"    Char-GRU Validation PPL:      {gru_ppl:.2f}")
    if gpt_ppl > 0:
        print(f"    PassGPT Validation PPL:       {gpt_ppl:.2f}")
    print(f"    Super-Ensemble Val PPL:       {hybrid_ppl:.2f}")

    # 3. Save model
    print(f"\n[*] Saving trained 4-Pillar model to: {out_model}")
    final_model.save(out_model)

    # 4. Calibrate and save Monte Carlo Rank Estimator
    if calibrate_mc:
        mc_samples = 20000 if quick_tune else config.meter.mc_samples
        estimator = MonteCarloEstimator.calibrate(final_model, n_samples=mc_samples, seed=config.seed)
        estimator.save(config.paths.mc_table)
        print(f"[+] Saved calibrated Monte Carlo rank lookup table to: {config.paths.mc_table}")

    # 5. Sanity check log-probabilities across all 4 models + ensemble
    sanity_passwords = ["password", "Password1!", "Summer2026!", "P@ssw0rd2024!", "x7#Qp!vR2m$9"]
    print("\n[+] Sanity Check: Log-probabilities across all model pillars:")
    for pw in sanity_passwords:
        bd = final_model.model_breakdown(pw)
        print(
            f"    '{pw:<15}' -> Markov: {bd['markov']:7.2f} | PCFG: {bd['structure']:7.2f} "
            f"| GRU: {bd['gru']:7.2f} | PassGPT: {bd['gpt']:7.2f} | Ensemble: {bd['hybrid']:7.2f}"
        )

    summary = {
        "best_config": best_config,
        "best_markov_ppl": best_ppl,
        "gru_ppl": gru_ppl,
        "gpt_ppl": gpt_ppl,
        "hybrid_ppl": hybrid_ppl,
        "tuned_lambda": tuned_lambda,
        "ensemble_weights": final_model.ensemble_weights,
        "selection_csv": str(out_csv),
        "model_file": str(out_model),
    }

    return final_model, summary
