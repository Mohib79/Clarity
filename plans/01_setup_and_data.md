# Plan 01: Project Setup & Data Preparation

**Goal:** Set up a reproducible repo and produce clean, disjoint 80% train / 20% test splits.

## Tasks
- [x] Create the folder layout from `PLAN.md §5`, with `__init__.py` files.
- [x] `requirements.txt`: numpy, matplotlib, pyyaml, flask, zxcvbn, pytest, tqdm.
- [x] `.gitignore`: `data/raw/`, `data/processed/`, `models/`, `.venv/`, `__pycache__/`.
- [x] `config.yaml`: paths, `seed: 42`, `min_len: 4`, `max_len: 32`, `ngram_order: 4`, meter thresholds, sample size.
- [x] `src/pwlab/config.py` loads the config into a dataclass. `src/pwlab/cli.py` has argparse subcommands.
- [x] README: documented architecture, ethics rules, setup instructions, and CLI usage.

## Cleaning (`src/pwlab/data/prepare.py`)
- [x] Decode as UTF-8 with a latin-1 fallback. Strip newlines.
- [x] Keep printable ASCII only (0x20 - 0x7E). Apply the length filter [4, 32].
- [x] Auto-detect plain format or `count<space>password` format.
- [x] Write stats to `reports/results/data_stats.json` (kept/dropped counts, length and character-class histograms).

## Split
- [x] Split into train (80%) and test (20%), plus a held-out validation slice (10% of train = 8% of total) -> `val.tsv`.
- [x] Save partitions as `data/processed/train.tsv`, `val.tsv`, `test.tsv` (`password\tcount`).
- [x] Add `--sample N` flag for fast development runs, `--full` for final runs, and `--max-raw-lines M` for fast prototyping.

## Acceptance criteria
- [x] Deterministic output for a fixed seed. Ratio matches 80/20 (and 10% validation). Totals add up.
- [x] High-performance streaming: cleans and filters 1,000,000 raw lines in a few seconds.

## Tests (`tests/test_prepare.py`)
- [x] 8/8 unit tests passing via `pytest` (filtering, character classification, bounds, non-ASCII, empty, split ratios, disjointness, end-to-end pipeline).
