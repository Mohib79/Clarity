"""
Unit tests for the privacy-preserving Flask web application.
Verifies endpoints, privacy headers, and input bounds.
"""

import unittest
from web.app import app


class TestWebApp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def test_index_route(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Probabilistic Password Strength Meter", resp.data)

    def test_privacy_headers(self):
        resp = self.client.get("/")
        self.assertIn("no-store", resp.headers.get("Cache-Control", ""))
        self.assertIn("no-cache", resp.headers.get("Cache-Control", ""))
        self.assertEqual(resp.headers.get("Referrer-Policy"), "no-referrer")
        self.assertEqual(resp.headers.get("X-Content-Type-Options"), "nosniff")

    def test_api_score_post_valid(self):
        resp = self.client.post("/api/score", json={"password": "Password1!"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["label"], "Weak")
        self.assertEqual(data["structure"], "L8D1S1")
        self.assertIn("reasons", data)
        self.assertIn("baselines", data)
        self.assertEqual(data["baselines"]["luds"]["label"], "Strong")

    def test_api_score_get_rejected(self):
        # GET should be rejected with 405 to prevent passwords leaking into URLs
        resp = self.client.get("/api/score?password=secret")
        self.assertEqual(resp.status_code, 405)

    def test_api_score_non_json_rejected(self):
        resp = self.client.post("/api/score", data="password=test")
        self.assertEqual(resp.status_code, 400)

    def test_api_score_oversized_rejected(self):
        long_pw = "a" * 200
        resp = self.client.post("/api/score", json={"password": long_pw})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()

