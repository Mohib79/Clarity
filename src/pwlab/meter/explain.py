"""
Plain-language explanation generator for the password strength meter.
Translates detected structural tags, patterns, and predictability metrics
into clear, actionable, user-friendly feedback.
"""

from typing import List, Tuple
from pwlab.patterns.taggers import Analysis, Tag


def generate_explanations(analysis: Analysis, log10_guesses: float, label: str) -> Tuple[List[str], List[str]]:
    """
    Generate prioritized plain-language reasons for password weakness/strength,
    along with actionable security suggestions.
    Returns: (reasons, suggestions)
    """
    reasons: List[str] = []
    suggestions: List[str] = []

    tag_types = {t.tag_type for t in analysis.tags}
    dict_tags = [t for t in analysis.tags if t.tag_type == "dictionary"]
    leet_tags = [t for t in analysis.tags if t.tag_type == "leet"]
    year_tags = [t for t in analysis.tags if t.tag_type == "year"]
    date_tags = [t for t in analysis.tags if t.tag_type == "date"]
    seq_tags = [t for t in analysis.tags if t.tag_type == "sequence"]
    repeat_tags = [t for t in analysis.tags if t.tag_type == "repeat"]
    rep_pat_tags = [t for t in analysis.tags if t.tag_type == "repeat_pattern"]
    walk_tags = [t for t in analysis.tags if t.tag_type == "keyboard_walk"]
    common_tags = [t for t in analysis.tags if t.tag_type == "common_password"]
    pos_tags = [t for t in analysis.tags if t.tag_type == "position_habit"]

    # 1. Exact common breach hit
    if common_tags:
        reasons.append("This is one of the most frequently breached passwords in history and is tested in the first few guesses of any attack.")
        suggestions.append("Change this immediately: known leaked passwords can be cracked in less than a second.")

    # 2. Leetspeak substitutions
    if leet_tags:
        for lt in leet_tags[:2]:
            root = lt.detail.get("root", "word")
            subs = lt.detail.get("substitutions", {})
            sub_strs = [f"'{k}' for '{v}'" for k, v in subs.items()]
            sub_desc = ", ".join(sub_strs) if sub_strs else "character substitutions"
            reasons.append(f"Uses predictable character substitutions ({sub_desc}) based on the dictionary word '{root}'. Modern password tools un-leet these automatically.")
        suggestions.append("Avoid substituting numbers or symbols for letters (e.g. '@' for 'a' or '0' for 'o'), as attackers test these rules first.")

    # 3. Dictionary word + Year combination
    if dict_tags and year_tags:
        word_root = dict_tags[0].detail.get("root", "word")
        yr_val = year_tags[0].detail.get("year", "year")
        reasons.append(f"Combines a common root word ('{word_root}') with a 4-digit calendar year ('{yr_val}'), a very common human password habit.")
        suggestions.append("Avoid combining dictionary words with recent or birth years.")
    elif dict_tags and not leet_tags:
        for dt in dict_tags[:2]:
            root = dt.detail.get("root", "word")
            reasons.append(f"Contains the common dictionary word or name '{root}', drastically reducing effective randomness.")
        suggestions.append("Avoid using single dictionary words or names.")

    # 4. Standalone Year or Date
    if year_tags and not dict_tags:
        yr = year_tags[0].detail.get("year", "year")
        reasons.append(f"Contains a 4-digit calendar year ('{yr}').")
        suggestions.append("Avoid using years, which narrow the search space to a tiny fraction of possibilities.")
    if date_tags:
        reasons.append("Contains an 8-digit calendar date format (such as a birthday or anniversary).")
        suggestions.append("Do not use calendar dates or birthdays in passwords.")

    # 5. Keyboard walks
    if walk_tags:
        for wt in walk_tags[:2]:
            direction = wt.detail.get("direction", "keyboard pattern").replace("_", " ")
            reasons.append(f"Follows an easily guessable {direction} ('{wt.matched_text}') on a QWERTY keyboard.")
        suggestions.append("Avoid sequential keyboard paths like 'qwerty' or '1qaz2wsx'.")

    # 6. Sequences and Repeats
    if seq_tags:
        for st in seq_tags[:2]:
            reasons.append(f"Contains a predictable sequence ('{st.matched_text}') like consecutive numbers or letters.")
        suggestions.append("Avoid ascending or descending sequences (e.g. '123456' or 'abcdef').")

    if repeat_tags:
        for rt in repeat_tags[:2]:
            char = rt.detail.get("char", "character")
            count = rt.detail.get("count", 3)
            reasons.append(f"Contains '{char}' repeated {count} times consecutively.")
        suggestions.append("Avoid repeating the same character.")

    if rep_pat_tags:
        pat = rep_pat_tags[0].detail.get("pattern", "")
        reasons.append(f"Repeats a simple sub-pattern ('{pat}').")

    # 7. Positional habits (trailing numbers, trailing symbols, leading capital)
    for pt in pos_tags:
        habit = pt.detail.get("habit")
        if habit == "trailing_digits":
            reasons.append(f"Tacks digits ('{pt.detail.get('digits')}') onto the end of the password.")
        elif habit == "trailing_symbol":
            reasons.append(f"Appends a single symbol ('{pt.detail.get('symbol')}') at the very end.")
        elif habit == "leading_capital":
            reasons.append("Only capitalizes the first letter, following a predictable checklist compliance habit.")
        elif habit == "enclosing_symbols":
            reasons.append("Encloses words between matching symbols.")

    # If the password is strong and has no tags
    if label == "Strong" and not reasons:
        reasons.append("High statistical entropy: character transitions do not follow predictable dictionary or lexical patterns.")
        reasons.append("Contains sufficient length and structural variation to resist offline candidate guessing.")

    # General recommendations
    if label in ("Weak", "Medium"):
        if len(analysis.password) < 12:
            suggestions.append(f"Length is currently {len(analysis.password)} characters. Increasing length to 14+ characters exponentially increases guess difficulty.")
        suggestions.append("Best Practice: Create a passphrase of 4 or more random, unrelated words (e.g. 'correct-horse-battery-staple').")

    # Deduplicate while preserving order
    def dedupe(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for it in items:
            if it not in seen:
                seen.add(it)
                out.append(it)
        return out

    return dedupe(reasons), dedupe(suggestions)

