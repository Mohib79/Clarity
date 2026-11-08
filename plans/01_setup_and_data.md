# Plan 01: Project Setup & Data Preparation

**Goal:** Set up a reproducible repo and produce clean, disjoint 80% train / 20% test splits.

## Tasks
- [ ] Create the folder layout from `PLAN.md §5`, with `__init__.py` files.
- [ ] `requirements.txt`: numpy, matplotlib, pyyaml, flask, zxcvbn, pytest, tqdm.
- [ ] `.gitignore`: `data/raw/`, `data/processed/`, `models/`, `.venv/`, `__pycache__/`.
- [ ] `config.yaml`: paths, `seed: 42`, `min_len: 4`, `max_len: 32`, `ngram_order: 4`, meter thresholds, sample size.
- [ ] `src/pwlab/config.py` loads the config into a dataclass. `src/pwlab/cli.py` has argparse subcommands (stubs at first).
- [ ] README: the team manually places the dataset at `data/raw/`. Research use only, never committed.

## Cleaning (`src/pwlab/data/prepare.py`)
- [ ] Decode as UTF-8 with a latin-1 fallback. Strip newlines.
- [ ] Keep printable ASCII only. Apply the length filter.
- [ ] Auto-detect plain format or `count<space>password` format.
- [ ] Write stats to `reports/results/data_stats.json` (kept/dropped counts, length and character-class histograms).

## Split
- [ ] Split **by occurrence** (expand counts, shuffle with the seed, cut 80/20). Save as `train.tsv` / `test.tsv` (`password\tcount`).
- [ ] If only unique entries are available, split unique strings and note this limitation.
- [ ] Hold out 10% of train as a **validation** slice for tuning. Never tune on test.
- [ ] `--sample N` flag for fast development runs, `--full` for final runs.

## Acceptance criteria
- [ ] Deterministic output for a fixed seed. Ratio within ±0.1%. Totals add up.

## Tests (`tests/test_prepare.py`)
- Filtering, both formats, determinism, ratio.

