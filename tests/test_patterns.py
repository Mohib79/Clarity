"""
Unit tests for structural segmentation and pattern tagging.
Tests around 40 distinct cases including boundary conditions and false-positive checks.
"""

import unittest
from pwlab.patterns.segmenter import segment, structure, case_mask, Segment
from pwlab.patterns.taggers import (
    DictionaryTagger,
    LeetTagger,
    DateYearTagger,
    SequenceTagger,
    KeyboardWalkTagger,
    CommonPasswordTagger,
    PositionTagger,
    PatternAnalyzer,
    analyze,
)


class TestSegmentation(unittest.TestCase):
    def test_segment_mixed(self):
        segs = segment("Summer2026!")
        self.assertEqual(len(segs), 3)
        self.assertEqual(segs[0], Segment("Summer", "L", 0, 6))
        self.assertEqual(segs[1], Segment("2026", "D", 6, 10))
        self.assertEqual(segs[2], Segment("!", "S", 10, 11))

    def test_segment_all_alpha(self):
        segs = segment("password")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].kind, "L")
        self.assertEqual(segs[0].text, "password")

    def test_segment_all_digits(self):
        segs = segment("123456789")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].kind, "D")
        self.assertEqual(segs[0].text, "123456789")

    def test_segment_all_symbols(self):
        segs = segment("!@#$%^")
        self.assertEqual(len(segs), 1)
        self.assertEqual(segs[0].kind, "S")

    def test_segment_alternating(self):
        segs = segment("a1b2c3")
        self.assertEqual(len(segs), 6)
        self.assertEqual([s.kind for s in segs], ["L", "D", "L", "D", "L", "D"])

    def test_segment_empty(self):
        self.assertEqual(segment(""), [])

    def test_structure_template(self):
        self.assertEqual(structure("Summer2026!"), "L6D4S1")
        self.assertEqual(structure("admin123"), "L5D3")
        self.assertEqual(structure("!@#pass123"), "S3L4D3")
        self.assertEqual(structure(""), "")

    def test_case_masks(self):
        self.assertEqual(case_mask("summer"), "lower")
        self.assertEqual(case_mask("SUMMER"), "UPPER")
        self.assertEqual(case_mask("Summer"), "Capitalized")
        self.assertEqual(case_mask("sUmMeR"), "mixed")
        self.assertEqual(case_mask("pAssword"), "mixed")
        self.assertEqual(case_mask(""), "empty")


class TestTaggers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = PatternAnalyzer()
        cls.dict_tagger = cls.analyzer.dict_tagger
        cls.leet_tagger = cls.analyzer.leet_tagger
        cls.date_tagger = cls.analyzer.date_tagger
        cls.seq_tagger = cls.analyzer.seq_tagger
        cls.walk_tagger = cls.analyzer.walk_tagger
        cls.common_tagger = cls.analyzer.common_tagger
        cls.pos_tagger = cls.analyzer.pos_tagger

    # --- Dictionary tests ---
    def test_dict_exact_words(self):
        for word in ["password", "summer", "dragon", "monkey", "secret", "michael", "jessica"]:
            tags = self.dict_tagger.find_words(word)
            self.assertTrue(len(tags) >= 1)
            self.assertEqual(tags[0].detail["root"], word)

    def test_dict_case_insensitivity(self):
        tags = self.dict_tagger.find_words("SuMMeR")
        self.assertEqual(len(tags), 1)
        self.assertEqual(tags[0].detail["root"], "summer")

    def test_dict_substring(self):
        tags = self.dict_tagger.find_words("bluecat")
        roots = [t.detail["root"] for t in tags]
        self.assertIn("blue", roots)
        self.assertIn("cat", roots)

    # --- Leetspeak tests ---
    def test_leet_p_at_ssw0rd(self):
        tags = self.leet_tagger.find_leet("P@ssw0rd")
        self.assertTrue(len(tags) >= 1)
        self.assertEqual(tags[0].detail["root"], "password")
        self.assertIn("@", tags[0].detail["substitutions"])
        self.assertIn("0", tags[0].detail["substitutions"])

    def test_leet_dr_at_g0n(self):
        tags = self.leet_tagger.find_leet("dr@g0n")
        self.assertTrue(len(tags) >= 1)
        self.assertEqual(tags[0].detail["root"], "dragon")

    def test_leet_s3cr3t(self):
        tags = self.leet_tagger.find_leet("s3cr3t")
        self.assertTrue(len(tags) >= 1)
        self.assertEqual(tags[0].detail["root"], "secret")

    def test_leet_no_substitutions_ignored(self):
        # Plain English word should not be tagged as leet
        tags = self.leet_tagger.find_leet("password")
        self.assertEqual(len(tags), 0)

    # --- Date & Year tests ---
    def test_year_valid_modern(self):
        for yr in ["1984", "1999", "2000", "2018", "2024", "2026"]:
            tags = self.date_tagger.find_years_and_dates(f"test{yr}")
            self.assertTrue(any(t.tag_type == "year" and t.detail["year"] == int(yr) for t in tags))

    def test_year_false_positive_repeats(self):
        # 8888 or 1111 should NOT be tagged as years
        tags = self.date_tagger.find_years_and_dates("test8888")
        year_tags = [t for t in tags if t.tag_type == "year"]
        self.assertEqual(len(year_tags), 0)

    def test_year_false_positive_sequence(self):
        # 1234 is a sequence, not a year
        tags = self.date_tagger.find_years_and_dates("test1234")
        year_tags = [t for t in tags if t.tag_type == "year"]
        self.assertEqual(len(year_tags), 0)

    def test_date_full_format(self):
        tags = self.date_tagger.find_years_and_dates("19980514")
        self.assertTrue(any(t.tag_type == "date" for t in tags))

    # --- Sequence & Repeat tests ---
    def test_sequence_numeric_fwd(self):
        tags = self.seq_tagger.find_sequences_and_repeats("1234567")
        self.assertTrue(any(t.tag_type == "sequence" for t in tags))

    def test_sequence_numeric_rev(self):
        tags = self.seq_tagger.find_sequences_and_repeats("987654")
        self.assertTrue(any(t.tag_type == "sequence" for t in tags))

    def test_sequence_alpha_fwd(self):
        tags = self.seq_tagger.find_sequences_and_repeats("abcdef")
        self.assertTrue(any(t.tag_type == "sequence" for t in tags))

    def test_repeat_single_char(self):
        tags = self.seq_tagger.find_sequences_and_repeats("aaaaaa")
        self.assertTrue(any(t.tag_type == "repeat" and t.detail["char"] == "a" for t in tags))

    def test_repeat_pattern(self):
        tags = self.seq_tagger.find_sequences_and_repeats("ababab")
        self.assertTrue(any(t.tag_type == "repeat_pattern" for t in tags))

    # --- Keyboard walk tests ---
    def test_keyboard_walk_qwerty(self):
        tags = self.walk_tagger.find_walks("qwerty")
        self.assertTrue(any(t.tag_type == "keyboard_walk" for t in tags))

    def test_keyboard_walk_asdfgh(self):
        tags = self.walk_tagger.find_walks("asdfgh")
        self.assertTrue(any(t.tag_type == "keyboard_walk" for t in tags))

    def test_keyboard_walk_vertical_1qaz(self):
        tags = self.walk_tagger.find_walks("1qaz")
        self.assertTrue(any(t.tag_type == "keyboard_walk" for t in tags))

    def test_keyboard_walk_reverse(self):
        tags = self.walk_tagger.find_walks("poiuyt")
        self.assertTrue(any(t.tag_type == "keyboard_walk" for t in tags))

    # --- Common password tests ---
    def test_common_passwords(self):
        for cp in ["password", "123456", "iloveyou", "rockyou"]:
            tags = self.common_tagger.check_common(cp)
            self.assertEqual(len(tags), 1)
            self.assertEqual(tags[0].tag_type, "common_password")

    def test_non_common_password(self):
        tags = self.common_tagger.check_common("x7#Qp!vR2m_random")
        self.assertEqual(len(tags), 0)

    # --- Position habits tests ---
    def test_position_trailing_digits(self):
        segs = segment("password123")
        tags = self.pos_tagger.check_position_habits("password123", segs)
        self.assertTrue(any(t.detail.get("habit") == "trailing_digits" for t in tags))

    def test_position_trailing_symbol(self):
        segs = segment("Summer2026!")
        tags = self.pos_tagger.check_position_habits("Summer2026!", segs)
        self.assertTrue(any(t.detail.get("habit") == "trailing_symbol" for t in tags))

    def test_position_leading_capital(self):
        segs = segment("Summer2026")
        tags = self.pos_tagger.check_position_habits("Summer2026", segs)
        self.assertTrue(any(t.detail.get("habit") == "leading_capital" for t in tags))

    def test_position_enclosing_symbols(self):
        segs = segment("!secret!")
        tags = self.pos_tagger.check_position_habits("!secret!", segs)
        self.assertTrue(any(t.detail.get("habit") == "enclosing_symbols" for t in tags))

    # --- Unified analyze tests ---
    def test_unified_analyze_summer2026_exclamation(self):
        res = analyze("Summer2026!")
        self.assertEqual(res.structure, "L6D4S1")
        tag_types = {t.tag_type for t in res.tags}
        self.assertIn("dictionary", tag_types)
        self.assertIn("year", tag_types)
        self.assertIn("position_habit", tag_types)

    def test_unified_analyze_p_at_ssw0rd2024(self):
        res = analyze("P@ssw0rd2024!")
        tag_types = {t.tag_type for t in res.tags}
        self.assertIn("leet", tag_types)
        self.assertIn("year", tag_types)
        self.assertIn("position_habit", tag_types)

    def test_unified_analyze_walk(self):
        res = analyze("1qaz2wsx")
        tag_types = {t.tag_type for t in res.tags}
        self.assertIn("keyboard_walk", tag_types)

    def test_unified_empty(self):
        res = analyze("")
        self.assertEqual(len(res.segments), 0)
        self.assertEqual(len(res.tags), 0)


if __name__ == "__main__":
    unittest.main()

