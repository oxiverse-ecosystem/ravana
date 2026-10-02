"""Canonical naming for DERIVATION relations — where one thing is named after another.

Why this module exists
----------------------
A derivation disclosure ("the sourdough starter is named after my uncle
bartholomew", "my espresso machine is called after my grandfather salvatore")
asserts a relation between TWO things: the subject the user owns (the starter,
the machine) and the referent the derivation points at (the uncle, the
grandfather).

The miner had no vocabulary for this relation class at all. Its kin-miner
regex (``\\bmy\\s+([a-z][a-z-]+)``) matched the *possessive inside the
prepositional phrase* — "my uncle" — so the whole disclosure was re-read as a
statement about the USER, producing:

    ('i', 'uncle bartholomew', 'bartholomew')

which is wrong three ways at once: the subject is the user rather than the
starter, the derivation relation ("named after") is discarded entirely, and the
value is a substring of the attribute so recall rendered the broken
"your uncle bartholomew is bartholomew."

Nothing about that is specific to "uncle" or to "named after" — it is the
absence of a shared vocabulary for derivation relations, so the miner fell back
to mis-parsing the wrong clause. This module supplies that vocabulary, following
the same pattern as :mod:`relation_attrs` (kin) and :mod:`pet_slots` (species):
ONE shared notion so the miner and every recall site agree by construction
instead of via N copies of a hand-kept list.

The vocabulary below is SEED structure, not an answer table. It maps a surface
predicate -> a canonical relation, and it is extended at runtime by
:func:`learn_derivation`. It is never a source of reply text and never a
per-person table.
"""
from typing import Dict, Optional
import re

# Seed derivation vocabulary: surface predicate -> canonical relation.
# A derivation predicate is a NAMING/ATTRIBUTION verb that takes a referent
# ("named after", "called after", "inspired by"). Extended at runtime via
# learn_derivation(); never a source of reply text. Single source of truth
# shared by the miner and the recaller.
_DERIVATION_SEED: Dict[str, str] = {
    "named after": "named after",
    "called after": "named after",
    "named for": "named after",
    "called for": "named after",
    "christened after": "named after",
    "in honor of": "named after",
    "in honour of": "named after",
    "named in honor of": "named after",
    "in memory of": "named after",
    "named in memory of": "named after",
    "a tribute to": "named after",
    "named as a tribute to": "named after",
    "inspired by": "inspired by",
    "inspired": "inspired by",
    "modeled on": "modeled on",
    "modelled on": "modeled on",
    "modeled after": "modeled on",
    "based on": "based on",
    "derived from": "based on",
    "adapted from": "based on",
    "taken from": "based on",
    "copied from": "based on",
}

# Runtime-grown extension of the seed table.
_DERIVATION_LEARNED: Dict[str, str] = {}

# The naming verbs that can head a derivation predicate. This is a closed-class
# seed (a small set of high-frequency English verbs), extended at runtime. It is
# vocabulary, NOT reply text: it decides whether a word may be treated as the
# naming head of an attribution, exactly as relation_attrs decides whether a
# word is a relationship.
_NAMING_VERB_SEED = {
    "name", "call", "christen", "title", "nickname", "dub", "label",
    "rename", "dedicate",
}

_NAMING_VERB_LEARNED: set = set()

# Attribution prepositions. A derivation predicate is a naming verb followed by
# one of these, so the class is recognized MORPHOLOGICALLY rather than as a
# hand-kept list of whole phrases: an unseen combination ("dubbed after",
# "tagged after") reduces correctly without a table entry.
_ATTRIBUTION_PREPS = {
    "after", "for", "by", "from", "to", "in honor of", "in honour of",
    "in memory of", "on",
}

# Tokens that can never head a naming verb (closed class / pronouns), so
# learn_naming_verb cannot be talked into learning a function word.
_NAMING_STOP = {
    "i", "me", "my", "you", "your", "he", "she", "it", "we", "they", "this",
    "that", "these", "those", "there", "here", "what", "which", "who", "is",
    "was", "are", "were", "be", "been", "being", "and", "or", "but", "if",
    "the", "a", "an", "of", "in", "on", "at", "to", "with", "from", "for",
    "not", "no", "so", "as", "then", "than", "when", "while", "after",
    "before", "because", "about",
}

# (stem, prep) -> canonical relation, derived from the seed table itself so the
# canonical form of an inflected seed verb ("named after" -> "name" + "after")
# folds back onto the seed KEY a later surface question can resolve.
_SEED_BY_STEM_PREP: Dict[tuple, str] = {}


def _build_seed_index() -> None:
    for phrase, canon in _DERIVATION_SEED.items():
        for size in (3, 2, 1):
            toks = phrase.split()
            if len(toks) <= size:
                continue
            prep = " ".join(toks[-size:])
            if prep not in _ATTRIBUTION_PREPS:
                continue
            stem = naming_verb_of(toks[-size - 1])
            if stem:
                _SEED_BY_STEM_PREP.setdefault((stem, prep), canon)
            break


def learn_naming_verb(word: str) -> Optional[str]:
    """Register a verb seen heading a derivation predicate; return its stem.

    Growth path for the naming-verb seed. A naming verb RAVANA has not heard of
    ("my track is DUBBED after my cousin ...") becomes addressable for later
    recall without a code change. Regular verb morphology is folded onto the
    stem so "tags"/"tagged"/"tagging" resolve to one relation. Returns None when
    the word cannot be a naming verb (empty, a pronoun, a closed-class token, or
    a bare noun carrying no verb inflection), so every call site can refuse to
    build a derivation from it.
    """
    w = (word or "").strip().lower()
    if not w:
        return None
    if w in _NAMING_STOP:
        return None
    known = naming_verb_of(w)
    if known:
        return known
    # An unseen token may only be learned when it is MORPHOLOGICALLY a verb
    # form — it carries a verb inflection. A bare noun carries none, so being
    # scanned by the miner is not enough to register it ("starter" stays out).
    # Decidable from morphology alone, so an unseen verb is still learnable.
    stem = _inflected_stem(w)
    if stem is None or len(stem) < 3:
        return None
    _NAMING_VERB_LEARNED.add(stem)
    return stem


def _known_naming_verbs() -> set:
    """Seed + runtime-learned naming verbs (the resolvable vocabulary)."""
    return _NAMING_VERB_SEED | _NAMING_VERB_LEARNED


def _inflected_stem(word: str) -> Optional[str]:
    """Strip a verb inflection off a token, or None if it carries none.

    Handles the orthographic final-consonant doubling of a short CVC stem:
    "dub" -> "dubbed" / "tag" -> "tagged" double the last letter before the
    suffix, so the -ed/-ing rules have to try the shorter stem too.
    """
    w = (word or "").strip().lower()
    if not w:
        return None
    for suf, plain in (("ed", 2), ("ing", 3)):
        if not w.endswith(suf) or len(w) <= plain:
            continue
        stem = w[:-plain]
        # Doubled final consonant ("dubbed" -> "dub", not "dubb").
        if len(stem) > 1 and stem[-1] == stem[-2]:
            stem = stem[:-1]
        return stem
    if w.endswith("d") and len(w) > 2:
        return w[:-1]
    if w.endswith("es") and len(w) > 3:
        return w[:-2]
    if w.endswith("s") and len(w) > 2:
        return w[:-1]
    return None


def naming_verb_of(word: str) -> Optional[str]:
    """Resolve a surface verb form to its naming-verb stem, or None."""
    w = (word or "").strip().lower()
    if not w:
        return None
    known = _known_naming_verbs()
    if w in known:
        return w
    stem = _inflected_stem(w)
    if stem is None:
        return None
    if stem in known:
        return stem
    # A dropped silent -e ("naming" -> "name").
    if (stem + "e") in known:
        return stem + "e"
    return None


def learn_derivation(phrase: str) -> str:
    """Register a derivation predicate seen in a live disclosure.

    Growth path for the seed vocabulary: an attribution RAVANA has never seen
    becomes addressable for later recall without a code change. Returns the
    canonical relation (the phrase's own words, normalized) so a caller can
    store it as an attribute key immediately.
    """
    p = _norm_phrase(phrase)
    if not p:
        return ""
    known = derivation_of(p)
    if known:
        return known
    _DERIVATION_LEARNED[p] = p
    return p


def _norm_phrase(phrase: str) -> str:
    return re.sub(r"\s+", " ", (phrase or "").strip().lower())


def derivation_of(phrase: str) -> Optional[str]:
    """Resolve a derivation predicate to its canonical relation, or None.

    Resolution is morphological, not a phrase lookup: a naming verb (seed or
    RAVANA-learned) followed by an attribution preposition is a derivation, so
    an unseen combination resolves without a table entry.
    """
    p = _norm_phrase(phrase)
    if not p:
        return None
    if p in _DERIVATION_SEED:
        return _DERIVATION_SEED[p]
    if p in _DERIVATION_LEARNED:
        return _DERIVATION_LEARNED[p]
    toks = p.split()
    if len(toks) < 2:
        return None
    # Try the longest attribution preposition first so "in honor of" is not
    # shadowed by a bare "of"/"in" split.
    for size in (3, 2, 1):
        if len(toks) <= size:
            continue
        prep = " ".join(toks[-size:])
        if prep not in _ATTRIBUTION_PREPS:
            continue
        stem = naming_verb_of(toks[-size - 1])
        if stem is None:
            return None
        # Canonicalize the whole predicate under its own surface so an unseen
        # combination still yields ONE stable attribute key. When the stem is a
        # SEED naming verb the canonical form is the SEED KEY ("named after"),
        # never the stem ("name after") — a stem-folded attribute is one no later
        # surface question can resolve, so the fact would be written and never
        # recalled. Only a verb RAVANA has not seen keeps its de-inflected stem.
        canon = (_DERIVATION_SEED.get(p)
                 or _SEED_BY_STEM_PREP.get((stem, prep))
                 or f"{stem} {prep}")
        _DERIVATION_LEARNED.setdefault(p, canon)
        return canon
    return None


def is_derivation_attr(attr: str) -> bool:
    """True when a stored attribute names a derivation relation."""
    a = _norm_phrase(attr)
    if not a:
        return False
    return derivation_of(a) is not None


def render_derivation(subject: str, relation: str, value: str) -> str:
    """Render a stored derivation fact as a clause for a reply.

    "sourdough starter" + "named after" + "uncle bartholomew" ->
    "your sourdough starter is named after uncle bartholomew".

    Every content word comes from the store; the connective ("your", "is") is
    fixed grammar. This is a renderer over stored state, not an authored reply.
    """
    s = _norm_phrase(subject)
    r = _norm_phrase(relation)
    v = _norm_phrase(value)
    if not s or not v:
        return ""
    return f"your {s} is {r} {v}"


_build_seed_index()
