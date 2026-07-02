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
KNOWN_VERBS = set(_CONSTANTS.get("known_verbs", []))
KNOWN_ADJS = set(_CONSTANTS.get("known_adjs", []))
FUNCTION_WORDS = set(_CONSTANTS.get("function_words", []))
FUNCTION_POS = dict(_CONSTANTS.get("function_pos", {}))

def classify_word_pos(word: str) -> str:
    """Dynamically classify the POS tag of a word using constants.json rules."""
    ll = word.lower()
    
    is_verb = ll in KNOWN_VERBS
    if not is_verb:
        if ll.endswith('s'):
            if ll[:-1] in KNOWN_VERBS:
                is_verb = True
            elif ll.endswith('es') and ll[:-2] in KNOWN_VERBS:
                is_verb = True

    if is_verb:
        return 'verb'
    if ll in KNOWN_ADJS:
        return 'adj'
        
    verb_suffixes = ['ing', 'ed', 'ize', 'ify', 'ate', 'en', 'ish']
    adj_suffixes = ['able', 'ible', 'ful', 'less', 'ous', 'al', 'ic', 'ive']
    
    if any(len(ll) > len(s) + 1 and ll.endswith(s) for s in verb_suffixes):
        return 'verb'
    if any(len(ll) > len(s) + 1 and ll.endswith(s) for s in adj_suffixes):
        return 'adj'
    if ll in FUNCTION_WORDS:
        return FUNCTION_POS.get(ll, 'adv')
        
    return 'noun'

class ConceptPosDict(dict):
    def __missing__(self, key):
        val = classify_word_pos(key)
        self[key] = val
        return val

    def get(self, key, default=None):
        if key not in self:
            val = classify_word_pos(key)
            self[key] = val
        return super().get(key, default)

# Re-export for backwards compatibility
TEEN_CONCEPT_LABELS = {label.lower() for label, _ in TEEN_CONCEPTS}
TEEN_CONCEPT_KEYWORDS = {label: kw.lower().split() for label, kw in TEEN_CONCEPTS}


from collections import Counter
import re


def _is_word_salad(text: str, allow_content_only: bool = False) -> bool:
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
    # Structural word safety check for sentences of 5 or more words
    # Skip for neural decoder output which may be content-word-only
    if not allow_content_only and len(words) >= 5:
        structural_words = {
            "the", "a", "an", "of", "to", "in", "for", "on", "by", "at", "with", "from",
            "and", "or", "but", "is", "are", "was", "were", "has", "have", "had", "does",
            "do", "did", "can", "could", "will", "would", "should", "it", "they", "he",
            "she", "i", "we", "you", "my", "your", "his", "her", "their", "our", "its"
        }
        if not any(w in structural_words for w in words):
            return True

    # Repeated words with zero structure check for short sentences
    # Skip for neural decoder
    if not allow_content_only and len(words) >= 4:
        structural_words = {
            "the", "a", "an", "of", "to", "in", "for", "on", "by", "at", "with", "from",
            "and", "or", "but", "is", "are", "was", "were", "has", "have", "had", "does",
            "do", "did", "can", "could", "will", "would", "should", "it", "they", "he",
            "she", "i", "we", "you", "my", "your", "his", "her", "their", "our", "its"
        }
        if not any(w in structural_words for w in words) and len(set(words)) < len(words):
            return True

    return False
