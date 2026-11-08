"""
Unit tests for data preparation, cleaning, and dataset splitting.
Compatible with both pytest and standard unittest.
"""

import unittest
from pathlib import Path
from collections import Counter

from pwlab.config import AppConfig
from pwlab.data.prepare import (
    parse_line,
    classify_password,
    split_dataset,
    prepare_dataset,
)


class TestDataPreparation(unittest.TestCase):
    def test_classify_password(self):
        self.assertEqual(classify_password("hello"), "alpha_only")
        self.assertEqual(classify_password("123456"), "digit_only")
        self.assertEqual(classify_password("hello123"), "alphanumeric")
        self.assertEqual(classify_password("!@#$%"), "symbol_only")
        self.assertEqual(classify_password("hello123!"), "mixed_with_symbols")

    def test_parse_line_clean(self):
        pw, count, reason = parse_line(b"summer2026\n", min_length=4, max_length=16)
        self.assertEqual(pw, "summer2026")
        self.assertEqual(count, 1)
        self.assertIsNone(reason)

    def test_parse_line_count_prefixed(self):
        pw, count, reason = parse_line(b"   42  secretPass!\r\n", min_length=4, max_length=16)
        self.assertEqual(pw, "secretPass!")
        self.assertEqual(count, 42)
        self.assertIsNone(reason)

    def test_parse_line_length_bounds(self):
        # Too short (< 4)
        pw, count, reason = parse_line(b"abc\n", min_length=4, max_length=16)
        self.assertIsNone(pw)
        self.assertEqual(reason, "too_short")

        # Too long (> 16)
        pw, count, reason = parse_line(b"this_password_is_way_too_long_for_limit\n", min_length=4, max_length=16)
        self.assertIsNone(pw)
        self.assertEqual(reason, "too_long")

    def test_parse_line_non_ascii(self):
        # Contains non-ASCII (e.g. café with 0xe9)
        pw, count, reason = parse_line(b"caf\xe9123\n", min_length=4, max_length=16, ascii_only=True)
        self.assertIsNone(pw)
        self.assertEqual(reason, "non_ascii")

    def test_parse_line_empty_or_whitespace(self):
        pw, count, reason = parse_line(b"   \r\n", min_length=4, max_length=16)
        self.assertIsNone(pw)
        self.assertEqual(reason, "empty_or_whitespace")

    def test_split_dataset_determinism_and_proportions(self):
        counts = Counter({f"password_{i}": 1 for i in range(1000)})

        train_1, val_1, test_1 = split_dataset(counts, train_ratio=0.8, test_ratio=0.2, val_ratio_from_train=0.1, seed=42)
        train_2, val_2, test_2 = split_dataset(counts, train_ratio=0.8, test_ratio=0.2, val_ratio_from_train=0.1, seed=42)

        # Determinism check
        self.assertEqual(train_1, train_2)
        self.assertEqual(val_1, val_2)
        self.assertEqual(test_1, test_2)

        # Ratio check
        # Total = 1000
        # test = 200 (20%)
        # train_val = 800 (80%)
        # val = 10% of 800 = 80
        # train = 800 - 80 = 720
        self.assertEqual(len(test_1), 200)
        self.assertEqual(len(val_1), 80)
        self.assertEqual(len(train_1), 720)

        # Disjointness check
        set_train = {p for p, _ in train_1}
        set_val = {p for p, _ in val_1}
        set_test = {p for p, _ in test_1}

        self.assertTrue(set_train.isdisjoint(set_val))
        self.assertTrue(set_train.isdisjoint(set_test))
        self.assertTrue(set_val.isdisjoint(set_test))

    def test_end_to_end_prepare_pipeline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            synthetic_raw = tmp_path / "raw.txt"
            synthetic_raw.write_bytes(
                b"123456\n"
                b"password\n"
                b"short\n"
                b"abc\n"  # too short (< 4)
                b"valid_pass_1\n"
                b"valid_pass_2\n"
                b"valid_pass_3\n"
                b"valid_pass_4\n"
                b"caf\xe9_bad\n"  # non-ascii
            )

            out_dir = tmp_path / "processed"
            stats_file = tmp_path / "reports" / "data_stats.json"

            cfg = AppConfig()
            cfg.paths.raw_data = str(synthetic_raw)
            cfg.paths.processed_dir = str(out_dir)
            cfg.paths.data_stats = str(stats_file)
            cfg.cleaning.min_length = 4
            cfg.cleaning.max_length = 20

            stats, paths = prepare_dataset(config=cfg, raw_path=str(synthetic_raw), output_dir=str(out_dir), full=True)

            self.assertEqual(stats.total_raw_lines, 9)
            self.assertEqual(stats.dropped_too_short, 1)
            self.assertEqual(stats.dropped_non_ascii, 1)
            self.assertEqual(stats.unique_kept_passwords, 7)

            self.assertTrue(Path(paths["train"]).exists())
            self.assertTrue(Path(paths["val"]).exists())
            self.assertTrue(Path(paths["test"]).exists())
            self.assertTrue(Path(paths["stats"]).exists())


if __name__ == "__main__":
    unittest.main()
