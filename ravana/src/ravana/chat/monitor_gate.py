"""M10 observability — fluent-tautology signature detector (permanent CI gate).

The fluent-tautological failure class (Wernicke's receptive-aphasia analog):
grammatically fluent, subject-anchored, semantically EMPTY text such as
"Life semantic people, which semantic cannot" or "black holes bend is black
holes bend". This standalone, PURE function reproduces the M4 signature
detector so it can run as a regression-proof CI gate over a fixed corpus (the
Q3/Q5/Q8 set from experiments/measure_sm_grounding.py). If the signature ever
reappears on any production path, the build fails.

Signature definition (faithful to the M4 catch): the clause is subject-anchored
+ glue/copula + vague metawords, with NO genuinely novel content predicate verb
(a real verb like "pulls", "grow", "escape") — i.e. it asserts no relation
between distinct content concepts — OR it is a truncated-subject repetition
("X bend is X bend"). A bare noun phrase with no metaword (e.g. "gravitational
attraction between masses") is NOT fluent-tautological; it is simply an
incomplete definition (handled by M7's copula gate, not here).
"""

import re
from typing import Optional

# Vague metawords that carry no independent content (Wernicke residual class).
_METAWORDS = {
    "semantic", "contrastive", "causal", "great", "even", "cannot", "which",
    "related", "connected", "linked", "tied", "means", "does", "makes",
    "opens", "springs", "weaves", "influences", "matters", "differs",
    "compared", "vs", "runs", "counter", "opposes", "contrasts", "contradicts",
    "reverses", "mirrors", "reflects", "aligns", "parallels", "echoes",
    "symbolizes", "represents",
}

# Copulas / relation verbs / determiners / coordinators — glue that does not
# by itself assert a real predicate argument.
_GLUE = {
    "is", "are", "was", "were", "be", "been", "being", "does", "do", "did",
    "makes", "make", "causes", "cause", "connected", "connect", "relates",
    "relate", "linked", "link", "tied", "tie", "means", "refers", "describes",
    "opens", "springs", "weaves", "influences", "influence", "matters",
    "differs", "compare", "vs", "with", "from", "of", "the", "a", "an",
    "in", "on", "for", "at", "to", "and", "but", "or",
}

# Seed set of content (predicate) verbs. Presence of ANY of these in a clause
# means it asserts a real relation, so it is NOT fluent-tautological. This is a
# regression gate over a FIXED corpus, not a general verb lexicon — the negative
# corpus verbs are enumerated here; the signature strings contain none.
_CONTENT_VERBS = {
    "pull", "pulls", "pulled", "attract", "attracts", "attraction",
    "bend", "bends", "bended", "escape", "escapes", "grow", "grows", "grew",
    "grow", "reproduce", "reproduces", "adapt", "adapts", "rely", "relies",
    "relying", "care", "cares", "cared", "believe", "believes", "learn",
    "learns", "think", "thinks", "know", "knows", "happen", "happens",
    "exist", "exists", "live", "lives", "die", "dies", "change", "changes",
    "move", "moves", "fall", "falls", "rise", "rises", "create", "creates",
}


def _real_words(clause: str, subj_words) -> list:
    return [w for w in re.findall(r"\b\w+\b", clause.lower())
            if w not in _GLUE and w not in _METAWORDS and w not in subj_words]


def _has_truncated_repetition(clause: str) -> bool:
    """True if the clause (glue removed) contains a repeated >=2-token run."""
    toks = [w for w in re.findall(r"\b\w+\b", clause.lower())
            if w not in _GLUE]
    if len(toks) < 4:
        return False
    for i in range(len(toks) - 1):
        bigram = (toks[i], toks[i + 1])
        # look for the same bigram later in the clause
        if bigram in list(zip(toks[i + 2:], toks[i + 3:])):
            return True
    return False


def detects_fluent_tautology(text: str, subject: Optional[str] = None) -> bool:
    """Return True if ``text`` carries the fluent-tautological signature.

    Fires only when the WHOLE text is tautological — a single incidental filler
    clause inside an otherwise real answer does not trip it.
    """
    if not text or not text.strip():
        return False
    subj_words = set(re.findall(r"\b\w+\b", (subject or "").lower())) if subject else set()
    clauses = [c.strip() for c in re.split(r"(?<=[.!?])\s+", text) if c.strip()]
    if not clauses:
        return False
    judged = [c for c in clauses if len(re.findall(r"\b\w+\b", c.lower())) >= 3]
    if not judged:
        return False

    def _is_taut(c: str) -> bool:
        words = re.findall(r"\b\w+\b", c.lower())
        has_metaword = any(w in _METAWORDS for w in words)
        has_content_verb = any(w in _CONTENT_VERBS for w in words)
        real = _real_words(c, subj_words)
        # (a) pure glue/metaword/subject — no real word at all.
        if not real:
            return True
        # (b) subject-anchored + metawords, but no content predicate verb
        #     asserting a relation (only empty nouns/glue remain).
        if has_metaword and not has_content_verb:
            return True
        # (c) truncated-subject repetition ("X bend is X bend").
        if _has_truncated_repetition(c):
            return True
        return False

    return all(_is_taut(c) for c in judged)
