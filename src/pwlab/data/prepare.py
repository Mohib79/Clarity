"""
Data preparation, cleaning, statistics collection, and train/val/test splitting.
High performance implementation for processing multi-million line datasets.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Tuple

from pwlab.config import AppConfig


COUNT_PREFIX_REGEX = re.compile(r"^\s*(\d+)\s+(.+)$")


@dataclass
class DataStats:
    total_raw_lines: int = 0
    kept_total_occurrences: int = 0
    unique_kept_passwords: int = 0
    dropped_non_ascii: int = 0
    dropped_too_short: int = 0
    dropped_too_long: int = 0
    dropped_empty_or_whitespace: int = 0
    length_histogram: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    character_classes: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["length_histogram"] = dict(sorted(self.length_histogram.items()))
        d["character_classes"] = dict(self.character_classes)
        return d


def classify_password(pw: str) -> str:
    """Classify character composition of a password (optimized single pass)."""
    has_alpha = False
    has_digit = False
    has_symbol = False

    for c in pw:
        if ("a" <= c <= "z") or ("A" <= c <= "Z"):
            has_alpha = True
        elif "0" <= c <= "9":
            has_digit = True
        else:
            has_symbol = True

        if has_alpha and has_digit and has_symbol:
            return "mixed_with_symbols"

    if has_alpha and not has_digit and not has_symbol:
        return "alpha_only"
    elif has_digit and not has_alpha and not has_symbol:
        return "digit_only"
    elif has_alpha and has_digit and not has_symbol:
        return "alphanumeric"
    elif has_symbol and not has_alpha and not has_digit:
        return "symbol_only"
    else:
        return "mixed_with_symbols"


def parse_line(
    raw_bytes: bytes,
    min_length: int = 4,
    max_length: int = 32,
    ascii_only: bool = True,
    drop_whitespace_only: bool = True,
) -> Tuple[Optional[str], int, Optional[str]]:
    """
    Parse and validate a raw line from a password dataset.
    Returns: (password, count, drop_reason)
    """
    try:
        line_str = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        try:
            line_str = raw_bytes.decode("latin-1")
        except Exception:
            return None, 0, "decode_error"

    line_str = line_str.rstrip("\r\n")

    count = 1
    # Check for count prefix only if string starts with space or digit (major optimization)
    if line_str and (line_str[0].isdigit() or line_str[0].isspace()):
        match = COUNT_PREFIX_REGEX.match(line_str)
        if match:
            try:
                count = int(match.group(1))
                pw = match.group(2)
            except ValueError:
                pw = line_str
        else:
            pw = line_str
    else:
        pw = line_str

    if drop_whitespace_only and not pw.strip():
        return None, 0, "empty_or_whitespace"

    if ascii_only:
        # Fast ASCII printable check (0x20 through 0x7E)
        if not pw.isascii() or any(c < " " or c > "~" for c in pw):
            return None, 0, "non_ascii"

    pw_len = len(pw)
    if pw_len < min_length:
        return None, 0, "too_short"
    if pw_len > max_length:
        return None, 0, "too_long"

    return pw, count, None


def load_and_clean_dataset(
    raw_path: str | Path,
    min_length: int = 4,
    max_length: int = 32,
    ascii_only: bool = True,
    drop_whitespace_only: bool = True,
    max_lines: Optional[int] = None,
    log_interval: int = 2000000,
) -> Tuple[Counter, DataStats]:
    """
    Stream and clean the raw dataset, collecting aggregate frequencies and statistics.
    """
    stats = DataStats()
    counts: Counter = Counter()

    p = Path(raw_path)
    if not p.exists():
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_path}")

    with open(p, "rb") as f:
        for idx, raw_line in enumerate(f):
            if max_lines is not None and idx >= max_lines:
                break

            if idx > 0 and idx % log_interval == 0:
                print(f"    ... processed {idx:,} raw lines ({len(counts):,} unique kept so far)", flush=True)

            stats.total_raw_lines += 1
            pw, count, drop_reason = parse_line(
                raw_line,
                min_length=min_length,
                max_length=max_length,
                ascii_only=ascii_only,
                drop_whitespace_only=drop_whitespace_only,
            )

            if drop_reason == "non_ascii" or drop_reason == "decode_error":
                stats.dropped_non_ascii += 1
            elif drop_reason == "too_short":
                stats.dropped_too_short += 1
            elif drop_reason == "too_long":
                stats.dropped_too_long += 1
            elif drop_reason == "empty_or_whitespace":
                stats.dropped_empty_or_whitespace += 1
            elif pw is not None:
                counts[pw] += count
                stats.kept_total_occurrences += count
                stats.length_histogram[len(pw)] += count
                cclass = classify_password(pw)
                stats.character_classes[cclass] += count

    stats.unique_kept_passwords = len(counts)
    return counts, stats


def split_dataset(
    counts: Counter,
    train_ratio: float = 0.80,
    test_ratio: float = 0.20,
    val_ratio_from_train: float = 0.10,
    seed: int = 42,
    sample_size: Optional[int] = None,
) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """
    Split the dataset into disjoint train, val, and test partitions.
    Partitions are deterministic given the seed.
    """
    items = list(counts.items())

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(items)

    # If sampling requested
    if sample_size is not None and sample_size < len(items):
        items = items[:sample_size]

    total = len(items)
    test_count = int(round(total * test_ratio))
    train_val_count = total - test_count

    val_count = int(round(train_val_count * val_ratio_from_train))
    train_count = train_val_count - val_count

    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]

    return train_items, val_items, test_items


def save_tsv(items: List[Tuple[str, int]], output_path: str | Path) -> None:
    """Save password items to a clean TSV file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write("password\tcount\n")
        for pw, count in items:
            f.write(f"{pw}\t{count}\n")


def prepare_dataset(
    config: Optional[AppConfig] = None,
    raw_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    sample_size: Optional[int] = None,
    max_raw_lines: Optional[int] = None,
    full: bool = False,
) -> Tuple[DataStats, Dict[str, str]]:
    """
    Main entry point for data preparation pipeline.
    """
    if config is None:
        config = AppConfig()

    input_file = raw_path or config.paths.raw_data
    out_dir = Path(output_dir or config.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts, stats = load_and_clean_dataset(
        raw_path=input_file,
        min_length=config.cleaning.min_length,
        max_length=config.cleaning.max_length,
        ascii_only=config.cleaning.ascii_only,
        drop_whitespace_only=config.cleaning.drop_whitespace_only,
        max_lines=max_raw_lines,
    )

    effective_sample = None
    if not full:
        if sample_size is not None:
            effective_sample = sample_size
        elif config.sampling.enable_sample_by_default:
            effective_sample = config.sampling.default_dev_sample

    train_items, val_items, test_items = split_dataset(
        counts=counts,
        train_ratio=config.split.train_ratio,
        test_ratio=config.split.test_ratio,
        val_ratio_from_train=config.split.val_ratio_from_train,
        seed=config.seed,
        sample_size=effective_sample,
    )

    stats.train_count = len(train_items)
    stats.val_count = len(val_items)
    stats.test_count = len(test_items)

    train_path = out_dir / "train.tsv"
    val_path = out_dir / "val.tsv"
    test_path = out_dir / "test.tsv"
    stats_path = Path(config.paths.data_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    save_tsv(train_items, train_path)
    save_tsv(val_items, val_path)
    save_tsv(test_items, test_path)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats.to_dict(), f, indent=2)

    file_paths = {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
        "stats": str(stats_path),
    }

    return stats, file_paths
