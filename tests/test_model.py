"""
Unit tests for MarkovModel, StructureModel, and PasswordModel facade.
Tests probability normalization, smoothing, toy corpus verification,
relative likelihood ordering, sampling, and save/load round-trips.
"""

import math
import tempfile
from pathlib import Path
import unittest

from pwlab.model.markov import MarkovModel, VOCABULARY, VOCAB_SIZE
from pwlab.model.structure import StructureModel
from pwlab.model.facade import PasswordModel, log_sum_exp_pair


class TestMarkovModel(unittest.TestCase):
    def setUp(self):
        self.toy_data = [
            ("password", 50),
            ("pass123", 30),
            ("summer2026", 20),
            ("winter2024", 15),
            ("admin", 25),
        ]
        self.model = MarkovModel(order=3, smoothing="backoff")
        self.model.train(self.toy_data)

    def test_probability_normalization_backoff(self):
        """Verify sum of probabilities over all vocabulary tokens equals 1.0."""
        contexts = ["pa", "ss", "20", "zz", ""]
        for ctx in contexts:
            total_p = sum(self.model.char_prob(ctx, c) for c in VOCABULARY)
            self.assertAlmostEqual(total_p, 1.0, places=6, msg=f"Probabilities did not sum to 1.0 for context '{ctx}'")

    def test_probability_normalization_additive(self):
        """Verify additive smoothing sums to 1.0."""
        add_model = MarkovModel(order=3, smoothing="additive", additive_delta=0.01)
        add_model.train(self.toy_data)
        for ctx in ["pa", "12", "xy"]:
            total_p = sum(add_model.char_prob(ctx, c) for c in VOCABULARY)
            self.assertAlmostEqual(total_p, 1.0, places=6)

    def test_smoothing_handles_unseen(self):
        """Unseen transitions and characters must receive positive non-zero probabilities."""
        p_unseen = self.model.char_prob("zzz", "$")
        self.assertGreater(p_unseen, 0.0)
        self.assertLess(p_unseen, 1.0)

        # Unseen password must not have logprob of -inf
        lp = self.model.logprob("unseen_password!#$99")
        self.assertTrue(math.isfinite(lp))
        self.assertLess(lp, 0.0)

    def test_toy_corpus_mle_preference(self):
        """Transitions frequent in training should have higher probability than rare ones."""
        # 'p' -> 'a' was trained ('password', 'pass123')
        p_a_given_p = self.model.char_prob("p", "a")
        # 'p' -> 'z' was never seen
        p_z_given_p = self.model.char_prob("p", "z")
        self.assertGreater(p_a_given_p, p_z_given_p)

    def test_sampling(self):
        """Sampling should produce non-empty printable ASCII strings."""
        for _ in range(10):
            pw = self.model.sample(max_len=16)
            self.assertIsInstance(pw, str)
            self.assertTrue(all(32 <= ord(c) <= 126 for c in pw))


class TestStructureModel(unittest.TestCase):
    def setUp(self):
        self.data = [
            ("password123", 40),
            ("Summer2026!", 30),
            ("Winter2024!", 25),
            ("admin1", 20),
            ("123456", 50),
        ]
        self.model = StructureModel(letter_ngram_order=3)
        self.model.train(self.data)

    def test_structure_logprob_finite(self):
        lp1 = self.model.logprob("Summer2026!")
        lp2 = self.model.logprob("x9#Qp!vR2m")
        self.assertTrue(math.isfinite(lp1))
        self.assertTrue(math.isfinite(lp2))
        # Common structure should score higher than completely random gibberish
        self.assertGreater(lp1, lp2)

    def test_structure_sampling(self):
        pw = self.model.sample()
        self.assertIsInstance(pw, str)
        self.assertGreater(len(pw), 0)


class TestPasswordModelFacade(unittest.TestCase):
    def setUp(self):
        self.train_data = [
            ("password", 100),
            ("123456", 80),
            ("password123", 60),
            ("Summer2026!", 40),
            ("Winter2024!", 30),
            ("admin", 25),
            ("iloveyou", 50),
        ]
        self.model = PasswordModel(ngram_order=3, smoothing="backoff", mixture_lambda=0.5)
        self.model.train(self.train_data)

    def test_relative_likelihood_ordering(self):
        """
        Fundamental project goal:
        logprob('password') > logprob('Summer2026!') > logprob('x7#Qp!vR2m')
        """
        lp_common = self.model.logprob("password")
        lp_structured = self.model.logprob("Summer2026!")
        lp_random = self.model.logprob("x7#Qp!vR2m")

        self.assertGreater(lp_common, lp_structured, "Common password should be more probable than structured password")
        self.assertGreater(lp_structured, lp_random, "Structured password should be more probable than random password")

    def test_hybrid_log_sum_exp(self):
        """Test numerically stable log-sum-exp helper."""
        a = math.log(0.3)
        b = math.log(0.7)
        res = log_sum_exp_pair(a, b)
        self.assertAlmostEqual(res, 0.0, places=6)  # log(0.3 + 0.7) = log(1.0) = 0.0

    def test_save_load_roundtrip(self):
        """Verify reloaded model produces identical scores to original model."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "test_model.pkl"
            self.model.save(save_path)

            loaded = PasswordModel.load(save_path)

            test_candidates = ["password", "Summer2026!", "123456", "admin", "qwert987!"]
            for cand in test_candidates:
                orig_m = self.model.logprob(cand, kind="markov")
                load_m = loaded.logprob(cand, kind="markov")
                self.assertAlmostEqual(orig_m, load_m, places=6)

                orig_s = self.model.logprob(cand, kind="structure")
                load_s = loaded.logprob(cand, kind="structure")
                self.assertAlmostEqual(orig_s, load_s, places=6)

                orig_h = self.model.logprob(cand, kind="hybrid")
                load_h = loaded.logprob(cand, kind="hybrid")
                self.assertAlmostEqual(orig_h, load_h, places=6)


if __name__ == "__main__":
    unittest.main()

