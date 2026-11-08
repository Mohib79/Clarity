"""
Unit tests for the explainable strength meter (Defender's View).
Tests monotonicity, threshold categorizations, explanation generation,
pattern overrides, and comparative baselines.
"""

import unittest
from pwlab.meter.scorer import PasswordScorer, score_password
from pwlab.meter.baselines import evaluate_luds, evaluate_zxcvbn
from pwlab.meter.estimator import MonteCarloEstimator
from pwlab.model.facade import PasswordModel


class TestStrengthMeter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = PasswordScorer()

    def test_empty_password(self):
        res = self.scorer.score("")
        self.assertEqual(res.label, "Weak")
        self.assertEqual(res.estimated_guesses, 1.0)
        self.assertTrue(len(res.reasons) > 0)

    def test_weak_passwords(self):
        weak_samples = ["123456", "password", "qwerty", "admin", "iloveyou"]
        for pw in weak_samples:
            res = self.scorer.score(pw)
            self.assertEqual(res.label, "Weak", f"Expected '{pw}' to be rated Weak, got {res.label}")
            self.assertLess(res.log10_guesses, 6.0)

    def test_password1_exclamation_luds_flaw(self):
        """
        The classic proposal example:
        'Password1!' looks Strong to static checklists (LUDS 5/5),
        but our probabilistic model rates it Weak.
        """
        res = self.scorer.score("Password1!")
        self.assertEqual(res.label, "Weak")
        self.assertEqual(res.baselines["luds"]["label"], "Strong")
        self.assertEqual(res.baselines["luds"]["passed_rules"], 5)
        # Should explain why it is weak
        reasons_text = " ".join(res.reasons).lower()
        self.assertIn("password", reasons_text)

    def test_strong_random_password(self):
        """A long high-entropy string should be rated Strong."""
        res = self.scorer.score("k9#mQ!vR2x$7pL0w")
        self.assertEqual(res.label, "Strong")
        self.assertGreaterEqual(res.log10_guesses, 10.0)

    def test_monotonicity(self):
        """Higher model probability should correspond to fewer estimated guesses."""
        res_easy = self.scorer.score("password")
        res_med = self.scorer.score("Summer2026!")
        res_hard = self.scorer.score("vR2#x9mQ!7pL0w")

        self.assertLess(res_easy.estimated_guesses, res_med.estimated_guesses)
        self.assertLess(res_med.estimated_guesses, res_hard.estimated_guesses)

    def test_baselines_eval(self):
        luds = evaluate_luds("Password1!")
        self.assertEqual(luds.label, "Strong")
        self.assertEqual(luds.score_raw, "5/5")

        zxc = evaluate_zxcvbn("123456")
        self.assertEqual(zxc.label, "Weak")
        self.assertEqual(zxc.score_raw, 0)


if __name__ == "__main__":
    unittest.main()

