"""
Privacy-Preserving Flask Web Application for Explainable Password Strength Evaluation.
"""

import logging
import os
import sys
from pathlib import Path

# Ensure src directory is in sys.path regardless of how/where app.py is launched
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from flask import Flask, jsonify, request, send_from_directory

from pwlab.config import load_config
from pwlab.meter.scorer import PasswordScorer

# Suppress request logging to ensure zero password leakage in logs
log = logging.getLogger("werkzeug")
log.setLevel(logging.ERROR)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 1024  # Max 1 KB payload

# Initialize shared scorer once at startup
config = load_config(str(ROOT_DIR / "config.yaml"))
scorer = PasswordScorer(config=config)


@app.after_request
def apply_privacy_headers(response):
    """Enforce strict no-cache, no-store, and privacy headers."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0, private"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self' 'unsafe-inline'; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "img-src 'self' data:;"
    )
    return response


@app.route("/")
def index():
    """Serve the interactive web interface."""
    return send_from_directory(str(Path(__file__).parent / "templates"), "index.html")


@app.route("/api/score", methods=["POST"])
def api_score():
    """
    Evaluate a password and return strength rating and structural explanation.
    Never accepts GET requests to prevent passwords appearing in query strings or server access logs.
    """
    if not request.is_json:
        return jsonify({"error": "Request must be JSON with Content-Type: application/json"}), 400

    data = request.get_json(silent=True) or {}
    password = data.get("password", "")

    if len(password) > 128:
        return jsonify({"error": "Password exceeds maximum allowable length of 128 characters"}), 400

    res = scorer.score(password, include_baselines=True)
    return jsonify(res.to_dict())


if __name__ == "__main__":
    print("[*] Starting Privacy-Preserving Password Strength Web App...")
    print("    URL: http://127.0.0.1:5000")
    print("    Privacy Notice: Passwords are evaluated in memory and NEVER logged or stored.")
    app.run(host="127.0.0.1", port=5000, debug=False)
