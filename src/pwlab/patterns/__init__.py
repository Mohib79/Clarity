"""
pwlab.patterns: Structural segmentation and human pattern tagging.
"""

from pwlab.patterns.segmenter import Segment, segment, structure, case_mask
from pwlab.patterns.taggers import (
    Tag,
    Analysis,
    PatternAnalyzer,
    analyze,
    DictionaryTagger,
    LeetTagger,
    DateYearTagger,
    SequenceTagger,
    KeyboardWalkTagger,
    CommonPasswordTagger,
    PositionTagger,
)

__all__ = [
    "Segment",
    "segment",
    "structure",
    "case_mask",
    "Tag",
    "Analysis",
    "PatternAnalyzer",
    "analyze",
    "DictionaryTagger",
    "LeetTagger",
    "DateYearTagger",
    "SequenceTagger",
    "KeyboardWalkTagger",
    "CommonPasswordTagger",
    "PositionTagger",
]

