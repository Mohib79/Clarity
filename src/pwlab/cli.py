"""
pwlab command-line interface.

Usage:
    python -m pwlab.cli prepare [--raw PATH] [--sample N] [--max-raw-lines M] [--full] [--config PATH]
    python -m pwlab.cli analyze "Summer2026!"
    python -m pwlab.cli analyze-train [--limit N]
    python -m pwlab.cli train [--max-train N] [--quick]
    python -m pwlab.cli score "Password123!"
    python -m pwlab.cli evaluate [--all] [--test-limit N]
"""

import argparse
import sys
from pathlib import Path

from pwlab.config import load_config
from pwlab.data.prepare import prepare_dataset
from pwlab.patterns.taggers import analyze
from pwlab.patterns.stats import compute_pattern_stats
from pwlab.model.train import run_model_training_pipeline
from pwlab.meter.scorer import PasswordScorer
from pwlab.eval.model_eval import run_evaluation_cli
from pwlab.eval.meter_eval import run_case_studies, run_meter_benchmark
from pwlab.eval.plots import generate_all_figures


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
    """Evaluate password strength and generate plain-language explanations."""
    cfg = load_config(args.config)
    scorer = PasswordScorer(config=cfg)
    res = scorer.score(args.password, include_baselines=True)

    badge_symbols = {
        "Weak": "[!] WEAK",
        "Medium": "[*] MEDIUM",
        "Strong": "[+] STRONG",
    }
    badge = badge_symbols.get(res.label, res.label)

    print("\n" + "=" * 65)
    print(f"   PASSWORD STRENGTH EVALUATION: '{res.password}'")
    print("=" * 65)
    print(f"  Rating:             {badge}")
    print(f"  Estimated Guesses:  10^{res.log10_guesses:.2f} (~{res.estimated_guesses:,.0f} guesses)")
    print(f"  Model Log-Prob:     {res.logprob:.2f}")
    print(f"  Structural Pattern: {res.structure}")

    print("\n  Structural Segmentation:")
    for seg in res.segments:
        print(f"    - '{seg['text']}' [{seg['kind']}]")

    print("\n  Why is it rated this way? (Plain-Language Explanation):")
    for r in res.reasons:
        print(f"    * {r}")

    print("\n  Actionable Recommendations:")
    for s in res.suggestions:
        print(f"    > {s}")

    print("\n  Comparative Benchmarking vs. Industry Baselines:")
    luds = res.baselines.get("luds", {})
    zxc = res.baselines.get("zxcvbn", {})
    print(f"    - LUDS Checklist Rule Meter: {luds.get('label', 'N/A')} ({luds.get('score', 'N/A')} criteria passed)")
    print(f"    - zxcvbn Estimator:          {zxc.get('label', 'N/A')} (Score: {zxc.get('score', 'N/A')}/4, ~10^{zxc.get('log10_guesses', 0):.1f} guesses)")
    print("=" * 65 + "\n")


def cmd_evaluate(args):
    """Run offline benchmarking against held-out test set."""
    cfg = load_config(args.config)

    # 1. Test set coverage evaluation vs baseline
    print("[*] Running Test Set Coverage Evaluation (Plan 04)...")
    run_evaluation_cli(config=cfg, test_limit=args.test_limit)

    if args.all:
        print("\n[*] Running Meter Benchmarks vs. LUDS & zxcvbn (Plan 07)...")
        scorer = PasswordScorer(config=cfg)
        run_meter_benchmark(scorer=scorer, sample_size=args.test_limit)

        print("\n[*] Running 15-Password Case Studies...")
        run_case_studies(scorer=scorer)

        print("\n[*] Generating Publication Figures...")
        generate_all_figures()

        print("\n[+] Full Evaluation Complete! All CSVs and Figures generated in reports/")


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
    p_eval.add_argument("--test-limit", type=int, default=5000, help="Number of test passwords to evaluate")
    p_eval.add_argument("--all", action="store_true", help="Run full benchmark suite including case studies and plots")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
