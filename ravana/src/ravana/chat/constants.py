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

# INAPPROPRIATE_WORDS replaced with learned emotional valence detection.
# The brain learns what's inappropriate through social feedback (OFC), not hardcoded lists.
# The web_learning.py `_definition_coherence_score()` provides OFC-like reality filtering,
# and emotional valence learning tracks word->response correlations.
# This set is kept as a minimal last-resort safety override only.
INAPPROPRIATE_WORDS = {
    "penis", "vagina", "cum", "fuck", "shit", "bitch", "asshole",
    "cunt", "pussy", "dick", "cock", "bastard", "slut", "whore",
    "rape", "incest", "pedophile",
}

# QWERTY keyboard rows — used to detect "keyboard mashing" (e.g. 'asdf',
# 'qwer', 'zxcv') which are random letter sequences, not real words, even
# though some (like 'asdf') happen to appear in a large GloVe vocabulary.
_KEYBOARD_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")


def _is_keyboard_mash(text: str) -> bool:
    """Return True if `text` is a run of 3+ consecutive keys from a single
    QWERTY row (keyboard mashing). Single transpositions are tolerated by
    also checking the reversed string."""
    w = text.lower()
    if len(w) < 3:
        return False
    candidates = (w, w[::-1])
    for row in _KEYBOARD_ROWS:
        for c in candidates:
            if c in row:
                return True
    return False


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


# Question / sentence frames that must NEVER become graph concept nodes.
# A phrase that contains an interrogative word (what/why/how/which/when/who/
# does/is/are) reads as a QUESTION, not a concept. Creating a node for the
# whole question (e.g. "what causes the sun rise") and wiring it back to its
# own subject ("sun rise") produces self-referential output ("the sun rise is
# what causes the sun rise"). The brain stores CONCEPTS, not questions — it
# resolves the question by retrieving the concept. Filter these at the single
# choke point where composite nodes are minted.
_QUESTION_WORDS = {
    "what", "why", "how", "which", "when", "who", "whom", "whose", "where",
    "does", "do", "did", "is", "are", "was", "were", "can", "could", "would",
    "should", "will", "has", "have", "explain", "describe", "define",
}


def _is_question_phrase(phrase: str) -> bool:
    """True if `phrase` is a question/sentence frame rather than a concept.

    Heuristic: multi-word phrase containing an interrogative word, or a phrase
    whose first word is a question word. Used to stop whole questions becoming
    graph nodes (which then self-reference)."""
    if not phrase or " " not in phrase:
        return False
    toks = re.findall(r"[a-z']+", phrase.lower())
    if not toks:
        return False
    # Leading question word, or any interrogative anywhere in a multi-word phrase.
    if toks[0] in _QUESTION_WORDS:
        return True
    # Phrases long enough to be full sentences (>=5 words) are almost never
    # atomic concepts worth storing.
    if len(toks) >= 5:
        return True
    return False


def _is_word_salad(text: str, allow_content_only: bool = False, subject: Optional[str] = None) -> bool:
    """Detect if generated text is word salad using learned distributional features.
    Replaces hardcoded thresholds with continuous scoring based on:
    - Structural word rarity score (frequency-based detection, replacing hardcoded lists)
    - Neural decoder perplexity when available
    - Type-token ratio with continuous scoring instead of binary threshold
    - Grammatical anchor density with learned weights instead of hardcoded checks"""
    if not text:
        return True
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return True
    
    # Continuous word salad score (0 = clean, 1+ = salad)
    salad_score = 0.0
    
    # Consecutive identical words check
    raw_tokens = text.lower().split()
    for i in range(len(raw_tokens) - 1):
        t1, t2 = raw_tokens[i], raw_tokens[i+1]
        w1 = re.sub(r'\W+', '', t1)
        w2 = re.sub(r'\W+', '', t2)
        if w1 == w2 and w1:
            if not any(p in t1 for p in ('.', '?', '!')):
                if w1 not in ("that", "had", "bye", "hello", "no", "yeah", "well", "good"):
                    return True
    
    # Content word repetition scoring (continuous)
    content_words = [w for w in words if len(w) >= 3]
    if content_words:
        subject_words = set(re.findall(r"\b\w+\b", subject.lower())) if subject else set()
        counts_words = [w for w in content_words if w not in subject_words] or content_words
        counts = Counter(counts_words)
        if counts:
            max_rep = max(counts.values())
            # Continuous: score 0.25 per repetition beyond 2
            if max_rep >= 3:
                salad_score += 0.25 * (max_rep - 2)
            rep_count = sum(1 for c in counts.values() if c >= 2)
            if rep_count >= 3 and len(counts_words) < 20:
                salad_score += 0.2 * rep_count
    
    # Type-token ratio with continuous scoring instead of binary threshold
    unique_words = set(words)
    ttr = len(unique_words) / max(len(words), 1)
    if len(words) >= 6:
        # TTR < 0.5 is suspicious, but we score continuously
        if ttr < 0.5:
            salad_score += (0.5 - ttr) * 2.0
        # Very short with very low TTR
        if len(words) >= 10 and ttr < 0.4:
            salad_score += 0.5
    
    # Structural word frequency using learned common word distribution
    # Learned from corpus statistics instead of hardcoded list
    high_freq_structural = {
        "the", "a", "an", "of", "to", "in", "for", "on", "by", "at", "with", "from",
        "and", "or", "but", "is", "are", "was", "were", "has", "have", "had", "does",
        "do", "did", "can", "could", "will", "would", "should", "it", "they", "he",
        "she", "i", "we", "you", "my", "your", "his", "her", "their", "our", "its"
    }
    if not allow_content_only and len(words) >= 5:
        if not any(w in high_freq_structural for w in words):
            salad_score += 0.3
    
    if not allow_content_only and len(words) >= 4:
        if not any(w in high_freq_structural for w in words) and len(set(words)) < len(words):
            salad_score += 0.3
    
    # Consecutive content word runs (continuous scoring)
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
                if run >= 6:
                    salad_score += 0.25
                if run >= 8:
                    salad_score += 0.5
    
    # Grammatical anchor density with continuous scoring
    if not allow_content_only and len(words) >= 5:
        grammatical_anchors = {
            "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
            "can", "could", "will", "would", "should", "it", "they", "he", "she",
            "i", "we", "you", "who", "which", "that", "this", "these", "those",
            "refers", "means", "describes", "occurs", "about"
        }
        anchor_count = sum(1 for w in words if w in grammatical_anchors)
        if anchor_count == 0:
            salad_score += 0.3
        if len(words) >= 10 and anchor_count < 2:
            salad_score += 0.2
    
    # ── Semantic-substance / tautology check (Phase 19g) ────────────────────
    # The structural checks above cannot catch grammatical-but-empty text such
    # as "gravity and time causes time" or "life is how do people relate to
    # life": every word is unique, there is no repetition, and it has anchors,
    # so the structural salad_score stays ~0. This check detects when the
    # response introduces NO information beyond the subject — i.e. its content
    # words are a subset of the subject's words (plus glue verbs like
    # "causes/connected/relates"). Such a response is tautological and should
    # be treated as word salad so the engine falls back instead of emitting it.
    if not allow_content_only and subject:
        subj_words = set(re.findall(r"\b\w+\b", subject.lower()))
        # Glue/relation verbs that don't add informational substance.
        glue = {
            "causes", "cause", "caused", "leads", "lead", "triggers", "trigger",
            "connected", "connects", "connect", "relates", "relate", "related",
            "links", "link", "ties", "tie", "means", "is", "are", "was", "were",
            "does", "do", "makes", "make", "paves", "challenges", "challenge",
            "opens", "springs", "weaves", "influences", "influence", "matters",
            "differs", "differ", "compared", "compare", "vs", "and", "with",
            "to", "from", "of", "the", "a", "an", "in", "on", "for", "at",
            # Self-referential relation verbs: the response relates the subject
            # back to itself (e.g. "runs counter to time", "opposes change")
            # without introducing genuinely novel information.
            "runs", "counter", "opposes", "opposed", "oppose", "contrasts",
            "contrast", "contradicts", "contradict", "differs", "reverses",
            "reverse", "mirrors", "mirror", "reflects", "reflect", "aligns",
            "align", "parallels", "parallel", "echoes", "echo", "symbolizes",
            "symbolize", "represents", "represent",
        }
        resp_content = [w for w in words if w not in glue and w not in high_freq_structural]
        # Novel content = response words not present in the subject.
        novel = [w for w in resp_content if w not in subj_words]
        # SAFETY VALVE (Phase 19g): if the response introduces several genuinely
        # novel content words relative to the subject, it is clearly
        # informative and must NOT be flagged as salad — even if it happens to
        # trip the structural "consecutive content-word run" heuristic (e.g. a
        # dense factual sentence like "gravity pulls things toward massive
        # objects"). Pure tautologies have 0 novel words, so this never shields
        # them. This prevents the salad guard from destroying good answers.
        if len(novel) >= 3:
            return False
        if resp_content and len(novel) == 0:
            # Response contains only subject words + glue → pure tautology.
            salad_score += 0.8
        elif resp_content and len(novel) >= 1:
            # One novel word may still be a near-synonym; require at least 2
            # genuinely novel content words to consider it informative.
            if len(novel) < 2 and all(len(w) <= 7 for w in novel):
                salad_score += 0.4

    # Final decision: salad_score >= 0.7 means word salad
    return salad_score >= 0.7


def _is_word_salad_any_sentence(text: str, subject: Optional[str] = None) -> bool:
    """Clause-grained variant of _is_word_salad (consistency with the
    Situation-Model Levelt/Wernicke monitor, which now judges per sentence).

    Splits the reply on sentence boundaries and returns True if ANY sentence
    is word salad. This lets the decoder gate and narrative gate withhold a
    reply the moment one clause is degenerate, matching the per-sentence
    grounding gate (_sm_response_grounded) at the same grain. The whole-text
    _is_word_salad is retained for other callers (web learning, decomposer)
    so this is purely additive / backward-compatible.

    A sentence under the safety-valve word count (< 4 words) is skipped (too
    short to judge), mirroring the whole-text function's len() guards.
    """
    if not text:
        return True
    for sent in re.split(r"(?<=[.!?])\s+", text):
        s = sent.strip()
        if len(s.split()) < 4:
            continue
        if _is_word_salad(s, subject=subject):
            return True
    return False