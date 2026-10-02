"""Canonical slot naming for user-disclosed animal companions.

Why this module exists
----------------------
A possession disclosure ("my cat is pixel", "i have two cats named biscuit and
gravy") is mined into the PersonalFactStore under a (subject, attribute, value)
triple. An earlier fix collapsed EVERY species onto one flat ``pet_name``
attribute so that multi-name disclosures could be indexed. That lost the
species, with three consequences:

  * a user with both a cat and a dog had the second overwrite the first, since
    both landed on the same slot;
  * a cued recall ("what is my cat's name?") could not distinguish which animal
    was being asked about;
  * a correction ("no, my cat is milo") could not find the prior value to
    supersede, because the correction path looks the slot up by the species
    word the user actually said.

The fix is to keep the SPECIES in the attribute and put the multiplicity in a
numeric suffix: ``cat``, ``cat_2``, ``dog``. The species word is normalised to a
singular canonical form so "cats"/"kitten"/"cat" all address the same slot, and
recall maps the user's spoken animal word through the same normaliser. That
makes the miner and every recall site agree on one key by construction rather
than by three copies of a hand-kept synonym table.

The synonym table below is SEED structure, not an answer table: it maps surface
word -> canonical species and is extended at runtime by
:func:`learn_species` whenever a disclosure names an animal the table has not
seen, so RAVANA grows its own species vocabulary from conversation.
"""
from typing import Dict, Optional
import re

# Seed species vocabulary: surface form -> canonical singular species.
# Extended at runtime via learn_species(); never a source of reply text.
_SPECIES_SEED: Dict[str, str] = {
    "cat": "cat", "cats": "cat", "kitten": "cat", "kittens": "cat",
    "kitty": "cat",
    "dog": "dog", "dogs": "dog", "puppy": "dog", "puppies": "dog", "pup": "dog",
    "bird": "bird", "birds": "bird", "parrot": "bird", "parrots": "bird",
    "fish": "fish",
    "rabbit": "rabbit", "rabbits": "rabbit", "bunny": "rabbit",
    "hamster": "hamster", "hamsters": "hamster",
    "horse": "horse", "horses": "horse", "pony": "horse",
    "owl": "owl", "owls": "owl",
    "pet": "pet", "pets": "pet",
}

# Runtime-grown extension of the seed table.
_SPECIES_LEARNED: Dict[str, str] = {}

# Pronoun / function-word stop-set. A possession disclosure ("i named my dog
# Rex") can leave the word immediately before "named"/"called" as a first-person
# pronoun (e.g. "i keep a sourdough starter i named doris" -> group(1) == "i").
# Such words are never animals, so learn_species must reject them outright —
# otherwise a bogus species slot ("i" -> "doris") is created and can leak on an
# UNKNOWN-entity cued recall, violating RAVANA's confabulation bar.
_PRONOUN_STOP = frozenset({
    "i", "me", "my", "mine", "you", "your", "yours", "we", "our", "us",
    "it", "its", "they", "them", "their", "he", "she", "his", "her",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "a", "an", "the", "some", "any", "one",
})


# ── CLOSED-CLASS FUNCTION WORDS (FIX-RV-18) ───────────────────────────────────
# A FUNCTION WORD carries no content of its own: it is an article, a pronoun,
# a preposition, a conjunction, or a SUPPORT/AUXILIARY verb ("is/was/was being",
# "do/does/did", "have/has/had", "will/can/would/should"). None of them can be
# the PREDICATE of a disclosure.
#
# Why this class exists (the defect): an open-class miner that takes "the token
# after the subject pronoun" as the relation head will happily seat a support
# verb in that slot. "when i WAS A teenager i lived in mumbai" put "a" — an
# ARTICLE — in the head slot, stored the relation-less fact ("i", "does:a",
# "a teenager"), and in doing so CONSUMED the rest of the clause, so the real
# predicate ("lived") and the real disclosed content ("mumbai") were dropped
# from the store entirely. The user's location — the most recallable fact in
# the sentence — was silently lost.
#
# This is STRUCTURAL VOCABULARY (a closed class of function words), which is
# legitimate: it names grammatical word classes, not answers. It lives HERE,
# in the one shared slot-naming module, so the miner and every recall site
# agree on what may occupy a head slot BY CONSTRUCTION rather than through N
# hand-kept copies of a synonym table. It is a SEED: :func:`learn_function_word`
# grows it at runtime, so a function word RAVANA meets in the wild joins the
# class without a code change.
_FUNCTION_SEED: frozenset = frozenset({
    # articles / determiners / quantifiers
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "each", "every", "all", "both", "few", "many", "much", "more", "most",
    "other", "another", "such", "no", "nor", "one", "ones", "several",
    # pronouns
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
    "you", "your", "yours", "yourself", "he", "him", "his", "she", "her",
    "hers", "it", "its", "they", "them", "their", "theirs", "who", "whom",
    "whose", "what", "which", "there", "here",
    # prepositions / particles
    "in", "on", "at", "by", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "to",
    "from", "up", "down", "of", "off", "over", "under", "near", "behind",
    "beyond", "among", "onto", "upon", "across", "throughout", "via",
    # conjunctions / discourse markers
    "and", "or", "but", "nor", "so", "because", "although", "though",
    "while", "whereas", "if", "unless", "since", "when", "whenever", "where",
    "whether", "than", "then", "also", "plus", "however", "though",
    # SUPPORT / AUXILIARY VERBS + COPULAS (the RV-18 defect class)
    "be", "am", "is", "are", "was", "were", "been", "being",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "ought", "need", "dare",
    # interrogative / relativiser / negation function words
    "not", "n't", "never", "if", "than", "as",
})

# Runtime-grown extension of the function-word seed.
_FUNCTION_LEARNED: set = set()


def learn_function_word(word: str) -> bool:
    """Register a closed-class function word seen in a live disclosure.

    Growth path for the seed class, exactly as :func:`learn_species` is the
    growth path for the species seed: a function word RAVANA has never had
    classified joins the class at runtime, so a miner that consults
    :func:`is_function_word` stops seating it in a relation-head slot without
    any code change. Returns True when the word is (now) a known function word.

    This is a WORD-CLASS learner, not an answer table: it records that a token
    belongs to a grammatical class, never what to say about it.
    """
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    if w in _FUNCTION_SEED or w in _FUNCTION_LEARNED:
        return True
    _FUNCTION_LEARNED.add(w)
    return True


def is_function_word(word: str) -> bool:
    """True when `word` is a closed-class function word.

    The single gate every relation/activity/attribute head slot consults before
    storing a fact. A TRUE answer means the word must never become a
    relation head: the miner has to look further for the real predicate.
    Seed + :func:`learn_function_word` growth; no content, no replies.
    """
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return True   # an empty slot can never carry a disclosure
    return w in _FUNCTION_SEED or w in _FUNCTION_LEARNED


# ── RESIDENCE VERBS + LOCATIVE PREPOSITIONS (FIX-RV-18) ───────────────────────
# A residence verb paired with a locative preposition is a PLACE disclosure
# ("i live in berlin", "when i was a teenager i lived in mumbai", "i moved to
# porto"). Those belong to the location miner, which stores the place as a
# first-class ("location", <place>) fact; an activity miner must not ALSO store
# the same disclosure as a verb-phrase half-fact.
#
# Both classes are closed-class structural vocabulary and both grow at runtime
# (:func:`learn_residence_verb`, :func:`learn_locative_preposition`), mirroring
# the species/function-word seeds above.
_RESIDENCE_SEED: frozenset = frozenset({
    "live", "lives", "lived", "living", "stay", "stays", "stayed", "staying",
    "move", "moves", "moved", "moving", "remain", "remains", "remained",
    "reside", "resides", "resided", "residing", "settle", "settles",
    "settled", "relocate", "relocates", "relocated", "relocating",
    "grow", "grew", "grown", "born", "based", "located", "stationed",
    "situated", "stay",
})
_RESIDENCE_LEARNED: set = set()

_LOCATIVE_PREP_SEED: frozenset = frozenset({
    "in", "at", "near", "from", "to", "onto", "into", "outside", "inside",
    "around", "by", "throughout", "across", "abroad", "overseas",
})
_LOCATIVE_PREP_LEARNED: set = set()


def learn_residence_verb(word: str) -> bool:
    """Register a residence/place verb seen in a live disclosure."""
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    _RESIDENCE_LEARNED.add(w)
    return True


def learn_locative_preposition(word: str) -> bool:
    """Register a locative preposition seen in a live disclosure."""
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    _LOCATIVE_PREP_LEARNED.add(w)
    return True


def is_residence_verb(word: str) -> bool:
    """True when `word` places the subject somewhere (a place disclosure)."""
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    if w in _RESIDENCE_SEED or w in _RESIDENCE_LEARNED:
        return True
    # Inflected forms of a seed stem ("residing" from "reside").
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and w[: -len(suf)] in _RESIDENCE_SEED:
            return True
    return False


def is_locative_preposition(word: str) -> bool:
    """True when `word` introduces a place ("lived IN mumbai")."""
    w = (word or "").strip().lower().strip(".,!?;:'\"")
    if not w:
        return False
    return w in _LOCATIVE_PREP_SEED or w in _LOCATIVE_PREP_LEARNED


def learn_species(word: str) -> Optional[str]:
    """Register an animal word seen in a live disclosure and return its canon.

    Growth path for the seed vocabulary: a species RAVANA has never heard of
    ("i have an axolotl named nyx") becomes addressable for later recall
    without any code change. A trailing plural "s" is folded onto the singular
    so the plural form of the same word resolves to one slot. Returns None when
    the word cannot be a species (empty or a pronoun / function word), so every
    call site's ``if _species is not None`` guard refuses to store a bogus slot.
    """
    w = (word or "").strip().lower()
    if not w:
        return None
    # Defense-in-depth: never register a pronoun / function word as a species.
    # The miner branches guard this too, but learn_species is the single
    # chokepoint every pet-disclosure path routes through, so rejecting here
    # guarantees no bogus slot (e.g. "i" -> "doris") can ever be learned.
    if w in _PRONOUN_STOP:
        return None
    known = species_of(w)
    if known:
        return known
    canon = w[:-1] if len(w) > 3 and w.endswith("s") else w
    _SPECIES_LEARNED[w] = canon
    _SPECIES_LEARNED[canon] = canon
    if not canon.endswith("s"):
        _SPECIES_LEARNED[canon + "s"] = canon
    return canon


def species_of(word: str) -> Optional[str]:
    """Canonical species for a surface animal word, or None if not an animal."""
    w = (word or "").strip().lower()
    if w.endswith("'s"):
        w = w[:-2]
    if not w:
        return None
    return _SPECIES_SEED.get(w) or _SPECIES_LEARNED.get(w)


def is_pet_attribute(attr: str) -> bool:
    """True when a stored PersonalFactStore attribute is a pet-name slot."""
    return species_of(base_species(attr)) is not None


def base_species(attr: str) -> str:
    """Strip the multiplicity suffix from a pet slot: ``cat_2`` -> ``cat``."""
    return re.sub(r"_\d+$", "", str(attr or "").strip().lower())


def slot_for(species: str, index: int = 1) -> str:
    """Slot attribute for the Nth pet of a species.

    The first pet of a species uses the bare species name so that the common
    single-pet case reads as a plain attribute (``cat``), and the correction
    path — which looks a slot up by the species word the user said — finds it
    without knowing how many pets there are.
    """
    canon = species_of(species) or learn_species(species)
    return canon if index <= 1 else f"{canon}_{index}"


def render(attr: str, value: str) -> str:
    """Render a stored pet slot as a natural clause for a recall reply."""
    return f"your {base_species(attr)} is {value}"


def render_pair(ent: str, attr: str, value: str) -> Optional[str]:
    """Render a pet clause when EITHER side names the species, else None.

    Pet facts reach the recall renderers in two shapes depending on which
    store they came from: the entity index keys them as (species, index) and
    the fact store as ("i", species_slot). One helper resolves both so the
    three call sites stay a single ``elif`` instead of repeating the
    entity-or-attribute dance.
    """
    sp = species_of(str(ent)) or (base_species(attr) if is_pet_attribute(attr) else None)
    if sp is None:
        return None
    # FIX-RV-13 (round auto/round-20260925T0823-fix-7): this is the LAST copula
    # site and it was the unconditional one. It renders the species-keyed pet
    # slot, which for an animal the user never NAMED holds the predicate they
    # disclosed about it -- the possession+predicate miner on this branch puts
    # 'had surgery last month' in exactly that slot. So a past-finite predicate
    # came out as "your dog is had surgery last month".
    #
    # The copula belongs to the same shared grammar rule every other site uses.
    # Importing it here (rather than passing it in) keeps pet_slots -- the
    # module that owns the species vocabulary -- free of any dependency on the
    # chat engine, and guarantees a value rendered as a pet clause and a value
    # rendered as a generic fact get the SAME grammar.
    try:
        from .user_model import drops_copula as _dc
    except Exception:  # import cycle / standalone use of this module
        _dc = None
    if _dc is not None and _dc(str(value).strip()):
        return f"your {sp} {value}"
    return f"your {sp} is {value}"
