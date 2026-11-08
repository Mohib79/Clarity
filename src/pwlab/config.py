"""
Configuration loader and data classes for pwlab.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List
import yaml


@dataclass
class PathsConfig:
    raw_data: str = "data/raw/rockyou.txt"
    processed_dir: str = "data/processed"
    train_data: str = "data/processed/train.tsv"
    val_data: str = "data/processed/val.tsv"
    test_data: str = "data/processed/test.tsv"
    data_stats: str = "reports/results/data_stats.json"
    models_dir: str = "models"
    model_file: str = "models/model.pkl"
    mc_table: str = "models/mc_table.npz"
    wordlists_dir: str = "src/pwlab/patterns/wordlists"
    reports_dir: str = "reports"
    figures_dir: str = "reports/figures"
    results_dir: str = "reports/results"


@dataclass
class CleaningConfig:
    min_length: int = 4
    max_length: int = 32
    ascii_only: bool = True
    drop_whitespace_only: bool = True


@dataclass
class SplitConfig:
    train_ratio: float = 0.80
    test_ratio: float = 0.20
    val_ratio_from_train: float = 0.10


@dataclass
class SamplingConfig:
    default_dev_sample: int = 500000
    enable_sample_by_default: bool = False


@dataclass
class ModelConfig:
    ngram_order: int = 4
    smoothing: str = "backoff"
    additive_delta: float = 0.01
    backoff_discount: float = 0.75
    mixture_lambda: float = 0.5


@dataclass
class MeterConfig:
    weak_threshold: float = 6.0
    strong_threshold: float = 10.0
    mc_samples: int = 50000


@dataclass
class EvaluationConfig:
    test_eval_sample: int = 50000
    guess_budgets: List[int] = field(default_factory=lambda: [1000, 10000, 100000, 1000000, 10000000])


@dataclass
class AppConfig:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    cleaning: CleaningConfig = field(default_factory=CleaningConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    meter: MeterConfig = field(default_factory=MeterConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """Load configuration from a YAML file, falling back to defaults if missing."""
    p = Path(config_path)
    if not p.exists():
        return AppConfig()

    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    paths = PathsConfig(**data.get("paths", {}))
    cleaning = CleaningConfig(**data.get("cleaning", {}))
    split = SplitConfig(**data.get("split", {}))
    sampling = SamplingConfig(**data.get("sampling", {}))
    model = ModelConfig(**data.get("model", {}))
    meter = MeterConfig(**data.get("meter", {}))
    evaluation = EvaluationConfig(**data.get("evaluation", {}))

    return AppConfig(
        seed=data.get("seed", 42),
        paths=paths,
        cleaning=cleaning,
        split=split,
        sampling=sampling,
        model=model,
        meter=meter,
        evaluation=evaluation,
    )

