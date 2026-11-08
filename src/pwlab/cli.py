"""
pwlab command-line interface.

Usage:
    python -m pwlab.cli prepare [--raw PATH] [--sample N] [--max-raw-lines M] [--full] [--config PATH]
    python -m pwlab.cli analyze "Summer2026!"
    python -m pwlab.cli analyze-train [--limit N]
    python -m pwlab.cli train [--max-train N] [--quick] [--order K] [--smoothing TYPE]
    python -m pwlab.cli score "Password123!"
    python -m pwlab.cli evaluate
"""

import argparse
import sys
from pathlib import Path

from pwlab.config import load_config
from pwlab.data.prepare import prepare_dataset
from pwlab.patterns.taggers import analyze
from pwlab.patterns.stats import compute_pattern_stats
from pwlab.model.train import run_model_training_pipeline


def cmd_prepare(args):
    """Handle dataset preparation command."""
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.seed = args.seed

    print(f"[*] Starting dataset preparation pipeline...", flush=True)
    print(f"    Raw file: {args.raw or cfg.paths.raw_data}", flush=True)
    print(f"    Seed: {cfg.seed}", flush=True)

    if args.full:
        print("    Mode: FULL dataset (all cleaned entries)", flush=True)
        sample_size = None
    elif args.sample is not None:
        print(f"    Mode: Sampled subset ({args.sample:,} entries)", flush=True)
        sample_size = args.sample
    else:
        sample_size = cfg.sampling.default_dev_sample if cfg.sampling.enable_sample_by_default else None
        if sample_size:
            print(f"    Mode: Default dev sample ({sample_size:,} entries)", flush=True)
        else:
            print("    Mode: Full dataset", flush=True)

    if args.max_raw_lines:
        print(f"    Max raw lines limit: {args.max_raw_lines:,}", flush=True)

    stats, paths = prepare_dataset(
        config=cfg,
        raw_path=args.raw,
        sample_size=sample_size,
        max_raw_lines=args.max_raw_lines,
        full=args.full,
    )

    print("\n[+] Data Preparation Complete!", flush=True)
    print(f"    Total raw lines read:       {stats.total_raw_lines:,}", flush=True)
    print(f"    Kept valid occurrences:     {stats.kept_total_occurrences:,}", flush=True)
    print(f"    Unique kept passwords:      {stats.unique_kept_passwords:,}", flush=True)
    print(f"    Dropped (non-ASCII):        {stats.dropped_non_ascii:,}", flush=True)
    print(f"    Dropped (too short < {cfg.cleaning.min_length}):  {stats.dropped_too_short:,}", flush=True)
    print(f"    Dropped (too long > {cfg.cleaning.max_length}):  {stats.dropped_too_long:,}", flush=True)
    print(f"    Dropped (whitespace):       {stats.dropped_empty_or_whitespace:,}", flush=True)
    print(f"\n[+] Partitions Created:", flush=True)
    print(f"    Train:      {stats.train_count:,} unique passwords -> {paths['train']}", flush=True)
    print(f"    Validation: {stats.val_count:,} unique passwords -> {paths['val']}", flush=True)
    print(f"    Test:       {stats.test_count:,} unique passwords -> {paths['test']}", flush=True)
    print(f"    Stats JSON: {paths['stats']}", flush=True)


def cmd_analyze(args):
    """Analyze a single password and show structural segmentation + detected habits."""
    pw = args.password
    res = analyze(pw)
    print(f"\n[+] Password Pattern Analysis: '{pw}'")
    print(f"    Structural Template: {res.structure}")
    print(f"    Segments ({len(res.segments)}):")
    for s in res.segments:
        extra = f" (case: {s.text})" if s.kind == "L" else ""
        print(f"      - '{s.text}' [{s.kind}] (span {s.start}:{s.end}){extra}")

    print(f"\n    Detected Habit Tags ({len(res.tags)}):")
    if not res.tags:
        print("      (No common predictable habits detected)")
    for t in res.tags:
        detail_str = ", ".join(f"{k}={v}" for k, v in t.detail.items())
        print(f"      - [{t.tag_type.upper()}] '{t.matched_text}' (span {t.span[0]}:{t.span[1]}) -> {detail_str}")
    print()


def cmd_analyze_train(args):
    """Compute aggregate pattern and structure frequencies across training data."""
    cfg = load_config(args.config)
    train_file = args.train or cfg.paths.train_data

    print(f"[*] Analyzing patterns across training dataset: {train_file}")
    res = compute_pattern_stats(
        train_path=train_file,
        output_csv=args.output or "reports/results/pattern_stats.csv",
        max_passwords=args.limit,
    )

    print("\n[+] Pattern Analysis Complete!")
    print(f"    Evaluated passwords: {res['total_evaluated']:,}")
    print(f"    Output CSV:          {res['output_path']}")
    print("\n    Top 5 Structural Templates:")
    for struct, cnt in res["top_structures"][:5]:
        pct = (cnt / res["total_weight"]) * 100
        print(f"      - {struct:<10} {cnt:,} ({pct:.1f}%)")

    print("\n    Most Common Human Habits:")
    for habit, cnt in list(res["tag_prevalence"].items())[:6]:
        pct = (cnt / res["total_weight"]) * 100
        print(f"      - {habit:<25} {cnt:,} ({pct:.1f}%)")


def cmd_train(args):
    """Train the statistical n-gram and structure models."""
    cfg = load_config(args.config)
    run_model_training_pipeline(
        config=cfg,
        train_path=args.train,
        val_path=args.val,
        output_model_path=args.output,
        quick_tune=args.quick,
        max_train_rows=args.max_train,
    )


def cmd_score(args):
    """Stub for password scoring."""
    print(f"[*] Password scoring module (Plan 05). Password: {args.password}")


def cmd_evaluate(args):
    """Stub for offline evaluation."""
    print("[*] Offline benchmarking module (Plan 07).")


def main():
    parser = argparse.ArgumentParser(
        prog="pwlab",
        description="Probabilistic Password Modeling & Explainable Strength Evaluation",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to configuration YAML file")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # prepare
    p_prepare = subparsers.add_parser("prepare", help="Clean, filter, and split the password dataset")
    p_prepare.add_argument("--raw", default=None, help="Path to raw dataset (defaults to config.yaml)")
    p_prepare.add_argument("--sample", type=int, default=None, help="Number of unique passwords to sample")
    p_prepare.add_argument("--max-raw-lines", type=int, default=None, help="Limit maximum raw lines read from file")
    p_prepare.add_argument("--full", action="store_true", help="Process the full dataset without sampling")
    p_prepare.add_argument("--seed", type=int, default=None, help="Random seed for splitting")
    p_prepare.set_defaults(func=cmd_prepare)

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Analyze structural habits of a given password")
    p_analyze.add_argument("password", type=str, help="Password string to analyze")
    p_analyze.set_defaults(func=cmd_analyze)

    # analyze-train
    p_atrain = subparsers.add_parser("analyze-train", help="Compute corpus-wide pattern statistics")
    p_atrain.add_argument("--train", default=None, help="Path to train.tsv (defaults to config)")
    p_atrain.add_argument("--limit", type=int, default=50000, help="Max passwords to analyze")
    p_atrain.add_argument("--output", default="reports/results/pattern_stats.csv", help="Output CSV path")
    p_atrain.set_defaults(func=cmd_analyze_train)

    # train
    p_train = subparsers.add_parser("train", help="Train the statistical n-gram and structure models")
    p_train.add_argument("--train", default=None, help="Path to train.tsv (defaults to config)")
    p_train.add_argument("--val", default=None, help="Path to val.tsv (defaults to config)")
    p_train.add_argument("--output", default=None, help="Output model path (defaults to models/model.pkl)")
    p_train.add_argument("--max-train", type=int, default=None, help="Limit number of train items")
    p_train.add_argument("--quick", action="store_true", help="Run quick validation tuning")
    p_train.set_defaults(func=cmd_train)

    # score
    p_score = subparsers.add_parser("score", help="Evaluate password strength and generate explanation")
    p_score.add_argument("password", type=str, help="Password string to analyze")
    p_score.set_defaults(func=cmd_score)

    # evaluate
    p_eval = subparsers.add_parser("evaluate", help="Run offline benchmarking against baselines")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
