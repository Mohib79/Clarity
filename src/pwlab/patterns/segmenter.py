"""
Structural segmentation for passwords.
Decomposes passwords into contiguous chunks of Letters (L), Digits (D), and Symbols (S).
"""

from dataclasses import dataclass
import string
from typing import List


@dataclass(frozen=True)
class Segment:
    text: str
    kind: str  # 'L' (Letters), 'D' (Digits), 'S' (Symbols/Special)
    start: int
    end: int  # exclusive

    @property
    def length(self) -> int:
        return self.end - self.start


def get_char_kind(c: str) -> str:
    """Return character category: L (Letter), D (Digit), or S (Symbol/Special)."""
    if ("a" <= c <= "z") or ("A" <= c <= "Z"):
        return "L"
    elif "0" <= c <= "9":
        return "D"
    else:
        return "S"


def case_mask(text: str) -> str:
    """
    Determine the capitalization pattern of a letter segment.
    Returns: 'lower', 'UPPER', 'Capitalized', or 'mixed'.
    """
    if not text:
        return "empty"
    if text.islower():
        return "lower"
    if text.isupper():
        return "UPPER"
    if text[0].isupper() and (len(text) == 1 or text[1:].islower()):
        return "Capitalized"
    return "mixed"


def segment(password: str) -> List[Segment]:
    """
    Segment a password into maximal contiguous chunks of the same character kind (L, D, S).
    Example: 'Summer2026!' -> [Segment('Summer', 'L', 0, 6), Segment('2026', 'D', 6, 10), Segment('!', 'S', 10, 11)]
    """
    if not password:
        return []

    segments: List[Segment] = []
    current_start = 0
    current_kind = get_char_kind(password[0])

    for i in range(1, len(password)):
        k = get_char_kind(password[i])
        if k != current_kind:
            segments.append(Segment(text=password[current_start:i], kind=current_kind, start=current_start, end=i))
            current_start = i
            current_kind = k

    segments.append(Segment(text=password[current_start:len(password)], kind=current_kind, start=current_start, end=len(password)))
    return segments


def structure(password: str) -> str:
    """
    Compute the structural template of a password.
    Example: 'Summer2026!' -> 'L6D4S1'
    """
    segments = segment(password)
    return "".join(f"{s.kind}{s.length}" for s in segments)

