"""Canonical relationship-attribute naming for user-disclosed kin / relatives.

Why this module exists
----------------------
A relationship disclosure ("my grandmother Indira weaves baskets", "my brother
Arjun climbs mountains", "my uncle Vivek repairs radios") is mined into the
PersonalFactStore under a (subject, attribute, value) triple. The miner keys the
fact by a COMBINED attribute: ``"<kin> <name>"`` (e.g. ``"grandmother indira"``),
and cued recall resolves it by the relationship head.

When RAVANA had no SHARED relationship lexicon, the kin word list lived as a
local variable *inside* the miner (user_model.py). That is the same trap the
pet-slots lesson calls out: the recall/enumeration sites must agree on which
words count as a relationship, or some relatives get enumerated and others
don't depending on which copy of the list each path sees. This module is the
single source of truth, so the miner, the cued-recall renderers, AND the new
category-enumeration recaller all resolve "is this a relationship fact?" through
one function.

The vocabulary below is SEED structure, not an answer table: it maps surface
relationship words -> a canonical relation, and is extended at runtime by
:func:`learn_relation` whenever a disclosure names a kin word the seed has not
seen, so RAVANA grows its own relationship vocabulary from conversation. It is
never a source of reply text and never a per-person table.
"""
from typing import Dict, Optional

import re

# Seed relationship vocabulary: surface form -> canonical relation.
# Covers blood kin AND non-kin ROLE words (mentor, teacher, coach, friend,
# neighbour, boss, colleague, ...). Extended at runtime via learn_relation();
# never a source of reply text. Single source of truth shared by the miner,
# the pet miner's relation_of() guard, and the recaller so all three agree on
# what counts as a relationship by construction.
_RELATION_SEED: Dict[str, str] = {
    "grandmother": "grandmother", "grandma": "grandmother", "granny": "grandmother",
    "nan": "grandmother", "nana": "grandmother", "nani": "grandmother",
    "grandfather": "grandfather", "grandpa": "grandfather", "granddad": "grandfather",
    "gran": "grandfather", "poppop": "grandfather", "nana ji": "grandfather",
    "mother": "mother", "mom": "mother", "mum": "mother", "mama": "mother",
    "maa": "mother", "amma": "mother",
    "father": "father", "dad": "father", "papa": "father", "appa": "father",
    "daddy": "father", "abba": "father",
    "sister": "sister", "sis": "sister",
    "brother": "brother", "bro": "brother", "bhai": "brother", "bhaiya": "brother",
    "aunt": "aunt", "auntie": "aunt", "aunty": "aunt",
    "uncle": "uncle", "mama ji": "uncle", "chacha": "uncle",
    "cousin": "cousin", "cousin brother": "cousin", "cousin sister": "cousin",
    "niece": "niece",
    "nephew": "nephew",
    "daughter": "daughter",
    "son": "son",
    "wife": "wife", "spouse": "wife",
    "husband": "husband", "spouse": "husband",
    "partner": "partner", "girlfriend": "partner", "boyfriend": "partner",
    "fiance": "partner", "fiancee": "partner",
    "stepmother": "stepmother", "stepmom": "stepmother",
    "stepfather": "stepfather", "stepdad": "stepfather",
    "stepsister": "stepsister", "stepbrother": "stepbrother",
    "halfsister": "halfsister", "halfbrother": "halfbrother",
    "grandson": "grandson", "granddaughter": "granddaughter",
    "motherinlaw": "motherinlaw", "mother in law": "motherinlaw",
    "fatherinlaw": "fatherinlaw", "father in law": "fatherinlaw",
    # Non-kin ROLE words (mentors, teachers, coaches, friends, neighbours,
    # bosses, colleagues, ...). These are RELATIONSHIPS the user discloses
    # about themselves, exactly like kin — "my mentor Dr. Okonkwo taught
    # me astronomy" must mine + recall the SAME way as "my brother Arjun".
    # They live in the SHARED relationship seed (not a second local list in
    # the miner) so the pet miner's relation_of() guard, the role miner, and
    # the recaller ALL agree on what counts as a relationship by
    # construction. This closes the slot-collision bug where a non-kin role
    # word was unknown to relation_of at pet-mining time and got mis-stored
    # as a pet species ("my mentor Dr..." -> ('i','mentor','dr')), which
    # then polluted recall with a truncated "your mentor is dr." and a
    # double-output. Seed structure: RAVANA also grows its own via
    # learn_relation at runtime; removing any entry degrades gracefully.
    "mentor": "mentor", "mentors": "mentor", "mentor's": "mentor",
    "teacher": "teacher", "teachers": "teacher",
    "coach": "coach", "coaches": "coach",
    "tutor": "tutor",
    "friend": "friend", "friends": "friend",
    "bestfriend": "friend", "bestie": "friend",
    "neighbor": "neighbor", "neighbour": "neighbor",
    "neighbors": "neighbor", "neighbours": "neighbor",
    "boss": "boss", "bosses": "boss",
    "manager": "manager", "managers": "manager",
    "supervisor": "supervisor",
    "colleague": "colleague", "colleagues": "colleague",
    "coworker": "coworker", "co-worker": "coworker",
    "roommate": "roommate", "roommates": "roommate",
    "housemate": "housemate",
    "landlord": "landlord", "landlords": "landlord",
    "landlady": "landlady",
    "rival": "rival", "rivals": "rival",
    "enemy": "enemy", "enemies": "enemy",
    "godparent": "godparent", "godparents": "godparent",
    "guardian": "guardian", "guardians": "guardian",
    "carer": "carer", "carers": "carer",
    "caregiver": "caregiver", "caregivers": "caregiver",
}

# Runtime-grown extension of the seed table.
_RELATION_LEARNED: Dict[str, str] = {}


def learn_relation(word: str) -> str:
    """Register a relationship word seen in a live disclosure and return its canon.

    Growth path for the seed vocabulary: a kin word RAVANA has never heard of
    ("my bhabhi neha paints") becomes addressable for later recall/enumeration
    without any code change. A trailing plural "s" is folded onto the singular so
    the plural form of the same word resolves to one canonical relation.
    """
    w = (word or "").strip().lower()
    if not w:
        return ""
    known = relation_of(w)
    if known:
        return known
    canon = w[:-1] if len(w) > 3 and w.endswith("s") else w
    _RELATION_LEARNED[w] = canon
    _RELATION_LEARNED[canon] = canon
    if not canon.endswith("s"):
        _RELATION_LEARNED[canon + "s"] = canon
    return canon


def relation_of(word: str) -> Optional[str]:
    """Canonical relationship for a surface kin word, or None if not a relation.

    Two-token forms ("mother in law") are tried both joined and with a hyphen so
    that surface variations resolve to one canonical relation. Multi-word kin
    MODIFIERS are normalized to a single head ("great-aunt" / "great aunt" /
    "grand-aunt" -> "aunt", "step-brother" -> "brother") so a disclosure like
    "my great-aunt Hortense ..." mines + recalls through the SAME path as "my
    aunt" — no second branch. Round 2026-08-21T2156Z defect D2: the bare word
    "great" is not a relationship, so "great-aunt" was previously unmapped and
    the disclosure was dropped. The mapping is seed structure (RAVANA-extendable
    via learn_relation); it names NO person, only the modifier->head collapse.
    """
    w = (word or "").strip().lower()
    if not w:
        return None
    # Normalize multi-word kin modifiers to a single canonical head.
    _norm = w.replace("-", " ").strip()
    _norm = re.sub(r"\s+", " ", _norm)
    if _norm != w:
        _parts = _norm.split()
        # Guard against all-symbol/empty tokens ("--", "//") that split to
        # nothing — they are not relationships. Without this, _norm.split()[-1]
        # raises IndexError and aborts the whole miner pass.
        if not _parts:
            return None
        _head = _parts[-1]  # "great aunt" -> "aunt", "step brother" -> "brother"
        _canon = _RELATION_SEED.get(_head) or _RELATION_LEARNED.get(_head)
        if _canon is not None:
            return _canon
    joined = w.replace(" ", "")
    return (_RELATION_SEED.get(w) or _RELATION_SEED.get(joined)
            or _RELATION_LEARNED.get(w) or _RELATION_LEARNED.get(joined))


def is_relation_attribute(attr: str) -> bool:
    """True when a stored PersonalFactStore attribute is a relationship slot.

    A relationship attribute is a combined "<kin> <name>" key (e.g. "grandmother
    indira") whose head word is a known relation. A bare kin word with no name
    (e.g. "sister" from "my sister climbs rocks") also counts, so name-less
    disclosures are enumerable too.
    """
    a = str(attr or "").strip().lower()
    if not a:
        return False
    if relation_of(a) is not None:
        return True
    _head = a.split()[0] if a.split() else ""
    return relation_of(_head) is not None


def base_relation(attr: str) -> Optional[str]:
    """The canonical relationship HEAD of a combined-attr fact.

    "grandmother indira" -> "grandmother"; "sister" -> "sister"; "mochi" -> None.
    """
    a = str(attr or "").strip().lower()
    if not a:
        return None
    _head = a.split()[0] if a.split() else a
    return relation_of(_head)


def render_relation(attr: str, value: str) -> str:
    """Render a stored relationship fact as a natural clause for a reply.

    "grandmother indira" + "weaves baskets" -> "your grandmother indira weaves
    baskets" (verb-phrase, no copula). Content comes from the store; only the
    connective is fixed. A name-less relation ("sister" + "climbs rocks") renders
    as "your sister climbs rocks".
    """
    return f"your {attr.strip().lower()} {value.strip().lower()}"
