# Plan 06: Privacy-Preserving Web Prototype

**Goal:** A clean single page where a user types a password and instantly sees a score and an explanation. Nothing is ever stored or logged.

## Backend (`web/app.py`, Flask)
- [x] Load `PasswordModel` and the MC table once at startup.
- [x] `POST /api/score` with a JSON body `{password}` → Plan 05 output. **Never GET / query strings** (preventing password leakage in URLs and logs).
- [x] Privacy controls:
  - Silenced the werkzeug request logger to prevent terminal and access log leakage.
  - Zero disk storage, zero database, in-memory volatile evaluation only.
  - Strict security headers: `Cache-Control: no-store, no-cache, must-revalidate`, `Referrer-Policy: no-referrer`, strict `Content-Security-Policy`, `X-Content-Type-Options: nosniff`.
  - Bound to `127.0.0.1` by default. Reject oversized bodies (> 1 KB).
  - Standalone frontend with zero third-party external CDN dependencies.
- [x] `GET /` serves the user interface.

## Frontend (`web/templates/index.html`)
- [x] Password input (`autocomplete="off"`, `spellcheck="false"`, show/hide toggle), debounced (200 ms) fetch.
- [x] Coloured strength badge (Weak red / Medium amber / Strong green), animated strength bar, and estimated guess budget.
- [x] Structural template and segment pills highlighting character categories (Letters, Digits, Symbols).
- [x] Plain-language explanations ("Why is it rated this way?") and actionable security guidance.
- [x] Side-by-side comparison panel contrasting our Probabilistic Meter vs. the flawed LUDS checklist rule meter and zxcvbn.
- [x] Prominent Privacy Guarantee badge ("Evaluated in volatile RAM only. Zero logging or data storage.").

## Acceptance criteria
- [x] Works with `python web/app.py` or through the CLI. Instant response (< 30 ms).
- [x] Privacy verified: GET query parameters rejected, logging silenced, no-store headers confirmed.

## Tests (`tests/test_web.py`)
- [x] 6 unit tests passing via `pytest` (endpoint status, valid score payload, GET rejection, non-JSON rejection, size bounds, and privacy headers). Total test suite: 68/68 passing.
