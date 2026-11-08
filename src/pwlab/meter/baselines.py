"""
Baseline password strength evaluators for benchmarking:
1. LUDS checklist rule-checker (Length, Uppercase, Digits, Symbols).
2. zxcvbn (standard dictionary and entropy-based estimator).
"""

from dataclasses import dataclass
import string
from typing import Optional
import zxcvbn


@dataclass
class BaselineResult:
    meter_name: str
    label: str  # 'Weak', 'Medium', 'Strong'
    score_raw: any
    log10_guesses: Optional[float]
    details: dict


def evaluate_luds(password: str) -> BaselineResult:
    """
    Conventional checklist meter: Length (>=8), Uppercase, Lowercase, Digits, Symbols.
    Demonstrates the false sense of security provided by static checklist policies.
    """
    has_upper = any("A" <= c <= "Z" for c in password)
    has_lower = any("a" <= c <= "z" for c in password)
    has_digit = any("0" <= c <= "9" for c in password)
    has_symbol = any(c in string.punctuation or (32 <= ord(c) <= 126 and not c.isalnum()) for c in password)
    has_len8 = len(password) >= 8

    passed_rules = sum([has_upper, has_lower, has_digit, has_symbol, has_len8])

    # Rule-based categorization:
    # 5/5 -> Strong
    # 3-4 -> Medium
    # 0-2 -> Weak
    if passed_rules == 5:
        label = "Strong"
    elif passed_rules >= 3:
        label = "Medium"
    else:
        label = "Weak"

    # Static keyspace estimate: N^L
    keyspace = 0
    if has_lower:
        keyspace += 26
    if has_upper:
        keyspace += 26
    if has_digit:
        keyspace += 10
    if has_symbol:
        keyspace += 32

    import math
    est_log10 = len(password) * math.log10(max(keyspace, 10)) if keyspace > 0 else 0.0

    return BaselineResult(
        meter_name="LUDS Checklist",
        label=label,
        score_raw=f"{passed_rules}/5",
        log10_guesses=round(est_log10, 2),
        details={
            "length_ge_8": has_len8,
            "has_uppercase": has_upper,
            "has_lowercase": has_lower,
            "has_digits": has_digit,
            "has_symbols": has_symbol,
            "passed_count": passed_rules,
        },
    )


def evaluate_zxcvbn(password: str) -> BaselineResult:
    """
    Evaluates password using zxcvbn.
    Score: 0, 1 -> Weak; 2 -> Medium; 3, 4 -> Strong.
    """
    res = zxcvbn.zxcvbn(password)
    score = res["score"]  # 0 to 4
    guesses_log10 = res["guesses_log10"]

    if score <= 1:
        label = "Weak"
    elif score == 2:
        label = "Medium"
    else:
        label = "Strong"

    return BaselineResult(
        meter_name="zxcvbn",
        label=label,
        score_raw=score,
        log10_guesses=round(guesses_log10, 2),
        details={
            "calc_time": res.get("calc_time", 0),
            "crack_time_online_throttled": res.get("crack_times_display", {}).get("online_throttling_100_per_hour", ""),
            "crack_time_offline_fast": res.get("crack_times_display", {}).get("offline_fast_hashing_1e10_per_second", ""),
            "feedback": res.get("feedback", {}),
        },
    )

