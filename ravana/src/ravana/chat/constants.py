"""
Shared constants for RAVANA cognitive architecture.
Auto-extracted from scripts/ravana_chat.py via split_engine.py.
"""
import json, os

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))), "data")

with open(os.path.join(_DATA_DIR, "constants.json"), encoding="utf-8") as _f:
    _CONSTANTS = json.load(_f)

TEEN_CONCEPTS = [(label, " ".join(kws)) for label, kws in _CONSTANTS["teen_concepts"]]
WEB_GARBAGE = set(_CONSTANTS["web_garbage"])
STOP_WORDS = set(_CONSTANTS["stop_words"])

# Re-export for backwards compatibility
TEEN_CONCEPT_LABELS = {label.lower() for label, _ in TEEN_CONCEPTS}
TEEN_CONCEPT_KEYWORDS = {label: kw.lower().split() for label, kw in TEEN_CONCEPTS}


from collections import Counter
import re


def _is_word_salad(text: str) -> bool:
    """Detect if generated text is a word salad (meaningless word lists or repetitions)."""
    if not text:
        return True
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return True
    for i in range(len(words) - 1):
        if words[i] == words[i+1]:
            if words[i] not in ("that", "had", "bye", "hello", "no", "yeah", "well", "good"):
                return True
    content_words = [w for w in words if len(w) >= 3 and w not in (
        "the", "and", "for", "with", "from", "that", "this", "they", "them", "have"
    )]
    if content_words:
        counts = Counter(content_words)
        max_rep = max(counts.values())
        if max_rep >= 3:
            return True
        rep_count = sum(1 for c in counts.values() if c >= 2)
        if rep_count >= 2 and len(content_words) < 20:
            return True
    unique_words = set(words)
    ttr = len(unique_words) / len(words)
    if len(words) >= 10 and ttr < 0.5:
        return True
    return False
