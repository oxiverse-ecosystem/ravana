"""
Shared constants for RAVANA cognitive architecture.
Auto-extracted from scripts/ravana_chat.py via split_engine.py.
"""
import json, os
from typing import Optional

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


def _is_word_salad(text: str, allow_content_only: bool = False, subject: Optional[str] = None) -> bool:
    """Detect if generated text is a word salad (meaningless word lists or repetitions)."""
    if not text:
        return True
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return True

    # Consecutive identical words check (only within the same sentence)
    raw_tokens = text.lower().split()
    for i in range(len(raw_tokens) - 1):
        t1, t2 = raw_tokens[i], raw_tokens[i+1]
        w1 = re.sub(r'\W+', '', t1)
        w2 = re.sub(r'\W+', '', t2)
        if w1 == w2 and w1:
            if not any(p in t1 for p in ('.', '?', '!')):
                if w1 not in ("that", "had", "bye", "hello", "no", "yeah", "well", "good"):
                    return True

    content_words = [w for w in words if len(w) >= 3 and w not in (
        "the", "and", "for", "with", "from", "that", "this", "they", "them", "have"
    )]
    if content_words:
        # Exclude words that are part of the subject from repetition counts
        # (since repeating the subject in a multi-sentence paragraph is grammatically valid)
        subject_words = set(re.findall(r"\b\w+\b", subject.lower())) if subject else set()
        filtered_content_words = [w for w in content_words if w not in subject_words]
        
        # If all content words were part of the subject, fall back to content_words
        counts_words = filtered_content_words if filtered_content_words else content_words
        counts = Counter(counts_words)
        if counts:
            max_rep = max(counts.values())
            if max_rep >= 3:
                return True
            rep_count = sum(1 for c in counts.values() if c >= 2)
            if rep_count >= 2 and len(counts_words) < 20:
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

    # Check for long runs of consecutive content words without any structural words or punctuation stoppers.
    if not allow_content_only:
        stoppers = {
            "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
            "can", "could", "will", "would", "should", "it", "they", "he", "she",
            "i", "we", "you", "who", "which", "that", "this", "these", "those",
            "refers", "means", "describes", "occurs", "about",
            "the", "a", "an", "of", "to", "in", "for", "on", "by", "at", "with", "from",
            "and", "or", "but", "as", "its"
        }
        raw_tokens = text.lower().split()
        run = 0
        for token in raw_tokens:
            clean_token = re.sub(r'\W+', '', token)
            if (any(p in token for p in ('.', '?', '!', ',', ':', ';', '-', '—')) or 
                clean_token in stoppers or 
                not clean_token):
                run = 0
            else:
                run += 1
                if run >= 5:
                    return True

    # Grammatical anchor density check to filter out plain noun/adjective list pile-ups from the decoder.
    if not allow_content_only and len(words) >= 5:
        grammatical_anchors = {
            "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
            "can", "could", "will", "would", "should", "it", "they", "he", "she",
            "i", "we", "you", "who", "which", "that", "this", "these", "those",
            "refers", "means", "describes", "occurs", "about"
        }
        anchor_count = sum(1 for w in words if w in grammatical_anchors)
        if anchor_count == 0:
            return True
        if len(words) >= 10 and anchor_count < 2:
            return True

    return False
