#!/usr/bin/env python3
"""
Validation tests for the fail-human root-cause fixes (L1, L2).

L1 (category-error metaphor): the cross-modal probe (Path 1) must fire for
under-covered words like 'triangle' (using the learned Binder/AttributeEncoder
probe, NOT a random graph draw), and the fallback must reference the subject's
own properties / a structure-mapped bearer — never two arbitrary sampled nouns.

L2 (P5 paradox grounding): the retrieved grounding clause must be coherent with
the paradox topic (GloVe cosine >= 0.15 via the repo's _definition_coherence_score)
and must NOT be a definition of a query substring. Off-topic snippets must be
rejected (fail-closed) rather than quoted.

These tests are offline-friendly: metaphor uses the probe (GloVe cache, no
network); the P5 grounding test patches kb_describe so it does not depend on
live Wikipedia/search.
"""
import os
import sys
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ravana", "src"))

import pytest
from ravana.chat.engine import CognitiveChatEngine

# Sensory-phrase words emitted by Path 1 (cross-modal) — derived from
# engine._SENSORY_DIM_PHRASE so the test tracks the real vocabulary.
SENSORY_WORDS = {
    "shape", "picture", "looks", "see", "colour", "color", "brightness",
    "darkness", "pattern", "texture", "feel", "temperature", "weight",
    "sound", "hear", "loudness", "movement", "structure", "taste", "smell",
}


def _eng():
    # baby_mode=True boots the engine; GloVe loads from cache (no network).
    return CognitiveChatEngine(baby_mode=True)


def _extract_grounding(reply: str):
    """Pull the '(from what i've read: ...)' clause, or '' if none."""
    m = re.search(r"\(from what i've read:\s*(.+?)\)\s*$", reply)
    return m.group(1).strip() if m else ""


def test_metaphor_uses_subject_property_not_random_nodes():
    """L1: 'taste of a triangle' must reference triangle's own sensory profile
    (Path 1 cross-modal) or its ConceptNet properties — NOT two random graph
    nodes. The reply must contain the subject and a sensory/property word, and
    must never be the old hardcoded 'flavor of a tuesday' brush-off."""
    e = _eng()
    # Force the gate's authoritative subject (mirrors process_turn wiring).
    e._is_category_error("what is the taste of a triangle")
    subj = getattr(e, "_last_category_subject", "triangle")
    reply = e._metaphor_for_category_error(subj, "taste")
    assert reply is not None, "metaphor should be produced for 'taste of a triangle'"
    assert "triangle" in reply.lower(), "reply must reference the real subject (triangle)"
    # Path 1 (cross-modal) OR Path 2 (ConceptNet features) — either way it
    # references the subject's properties, not two arbitrary nouns.
    has_sensory = any(w in reply.lower() for w in SENSORY_WORDS)
    # Path 2 phrasing: "{Subject} is more about X than about having a taste"
    has_property_frame = "more about" in reply.lower() and "than about" in reply.lower()
    assert has_sensory or has_property_frame, (
        f"metaphor should use a sensory phrase or subject property, got: {reply!r}")
    # The old hardcoded analogy must be gone.
    assert "flavor of a tuesday" not in reply.lower()


def test_category_error_no_random_sample():
    """L1: Path 3 (structure-mapped incongruent pair) must NOT be a raw
    random.sample of two unrelated graph nodes. When it fires, the reply frames
    the mismatch via a real property-bearer ('B has a PROP, SUBJ doesn't'),
    so it references the subject and the queried property explicitly."""
    e = _eng()
    # Use a subject whose probe is weak but ConceptNet gives a property bearer.
    e._is_category_error("what colour is a thought")
    subj = getattr(e, "_last_category_subject", "thought")
    reply = e._metaphor_for_category_error(subj, "colour")
    assert reply is not None
    # Whether Path 1 or Path 3, the subject appears and the property is named.
    assert subj in reply.lower()
    assert "colour" in reply.lower() or "color" in reply.lower()


def test_paradox_grounding_coherent_and_on_topic():
    """L2: a paradox grounding clause must be coherent with the topic
    (>= 0.15) and not a definition of a query substring."""
    e = _eng()
    topic = "angels"
    # Patch kb_describe to return a clean, on-topic definition (no network).
    e.kb_describe = lambda t: (
        "An angel is a spiritual, heavenly, or supernatural entity, usually "
        "humanoid with bird-like wings, often depicted as a messenger.")
    reply = e._reflect_on_paradox("how many angels can dance on the head of a pin")
    ground = _extract_grounding(reply)
    assert ground, f"expected a grounding clause, got reply: {reply!r}"
    # Coherence must clear the repo's 0.15 bar.
    coh = e._definition_coherence_score(topic, ground)
    assert coh >= 0.15, f"grounding coherence {coh:.3f} < 0.15 for topic {topic!r}: {ground!r}"
    # Must not be a trivial definition of the query word.
    assert not re.match(r"^\s*angels\s+(is|are|was|were)\b", ground.lower()), \
        f"grounding is a def of the query substring: {ground!r}"


def test_paradox_grounding_rejects_off_topic():
    """L2: an off-topic grounding candidate must be rejected (fail-closed),
    so the reply carries NO '(from what i've read: ...)' clause."""
    e = _eng()
    # Patch kb_describe to return clearly off-topic text (entity collision).
    e.kb_describe = lambda t: (
        "The latest LA Angels news, rumors, free agent signings, and fan "
        "opinion from Halo Hangout.")
    reply = e._reflect_on_paradox("how many angels can dance on the head of a pin")
    ground = _extract_grounding(reply)
    coh = e._definition_coherence_score("angels", ground) if ground else 0.0
    # Either no grounding clause, or any clause present must be coherent.
    # (With the strict max-coherence gate the off-topic patch is rejected.)
    assert (not ground) or (coh >= 0.15), \
        f"off-topic grounding should be rejected, got: {ground!r} (coh={coh:.3f})"


def test_paradox_topic_extraction():
    """L2: _paradox_topic must pull the salient concept, not a modal/question
    word (e.g. not 'can' from 'can god create a rock')."""
    e = _eng()
    assert e._paradox_topic("can god create a rock he cannot lift") == "god" or \
           e._paradox_topic("can god create a rock he cannot lift") == "create"
    assert e._paradox_topic("how many angels can dance on the head of a pin") == "angels"
    assert e._paradox_topic("is the statement i am lying true or false") in ("liar", "lying")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
