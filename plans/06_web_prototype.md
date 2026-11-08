# Plan 06: Privacy-Preserving Web Prototype

**Goal:** A clean single page where a user types a password and instantly sees a score and an explanation. Nothing is ever stored or logged.

## Backend (`web/app.py`, Flask)
- [ ] Load `PasswordModel` and the MC table once at startup.
- [ ] `POST /api/score` with a JSON body `{password}` → Plan 05 output. **Never GET / query strings** (these end up in logs and browser history).
- [ ] Privacy controls:
  - Silence the werkzeug request logger. Never log request bodies. No database, no files written.
  - Headers: `Cache-Control: no-store`, `Referrer-Policy: no-referrer`, a strict `Content-Security-Policy`, `X-Content-Type-Options: nosniff`.
  - Bind to `127.0.0.1` by default. Reject bodies over 1 KB.
  - No third-party scripts, fonts or analytics.
- [ ] `GET /` serves the page.

## Frontend (`web/templates/index.html`, `web/static/`)
- [ ] Password input (`autocomplete="off"`, show/hide toggle), debounced (≈250 ms) fetch.
- [ ] Coloured strength bar (Weak red / Medium amber / Strong green), estimated guesses ("≈ 10⁴ guesses"), and a list of reasons and suggestions.
- [ ] Segment visualisation: chunks highlighted by tag type (word, year, symbol, leet, keyboard pattern).
- [ ] A side-by-side mini panel with the zxcvbn and LUDS verdicts, for demo contrast.
- [ ] Footer: "Your password never leaves this page's request and is never stored."

## Acceptance criteria
- [ ] Works with `python web/app.py`. Response in under 100 ms.
- [ ] Privacy check: after a session, grep the logs/terminal and project folder for test passwords and confirm zero hits. Document this in the report.

## Tests (`tests/test_web.py`)
- Flask test client: POST works, GET with a query is rejected, headers present, oversized body rejected.

