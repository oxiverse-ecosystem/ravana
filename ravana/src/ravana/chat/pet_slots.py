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
    "pet": "pet", "pets": "pet",
}

# Runtime-grown extension of the seed table.
_SPECIES_LEARNED: Dict[str, str] = {}


def learn_species(word: str) -> str:
    """Register an animal word seen in a live disclosure and return its canon.

    Growth path for the seed vocabulary: a species RAVANA has never heard of
    ("i have an axolotl named nyx") becomes addressable for later recall
    without any code change. A trailing plural "s" is folded onto the singular
    so the plural form of the same word resolves to one slot.
    """
    w = (word or "").strip().lower()
    if not w:
        return ""
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
    return f"your {sp} is {value}" if sp else None
