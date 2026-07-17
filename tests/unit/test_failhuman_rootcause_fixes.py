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


def test_metaphor_path1_fires_for_broad_subjects():
    """L1 / Fix A.1 outcome guard: the learned cross-modal probe (Path 1) must
    fire for a BROAD set of category-error subjects -- geometric, abstract,
    sensory, physical -- NOT fall through to Path 3 random/structure-mapped
    sampling. Guards against a future probe change silently regressing L1 back
    to arbitrary pairs. The reply must carry the Path-1 signature
    ('more in terms of its <sensory phrase>'), proving the probe -- not a
    random draw -- generated it."""
    e = _eng()
    subjects = ["triangle", "square", "circle", "cube", "line", "number",
                "justice", "love", "time", "equation", "silence", "memory",
                "dream", "thought", "idea", "freedom", "peace", "anger",
                "color", "red", "music", "gravity", "atom", "cell", "tree",
                "stone", "wind", "shadow", "language", "soul", "infinity"]
    fired = 0
    for subj in subjects:
        reply = e._metaphor_for_category_error(subj, "taste")
        if reply is None:
            continue
        # Path-1 signature: cross-modal phrasing referencing the subject's own
        # sensorimotor profile. (Path 3 would say 'can have the taste of <Y>'.)
        # Path-1 signature after the B3 semantic-control rewrite: the reply
        # ANCHORS on the asked property first ("doesn't really have a <prop>"),
        # then bridges to the subject's own cross-modal dimension via
        # "more by its <phrase>" / "think of it by its <phrase>" /
        # "relate it to its <phrase>". The old literal "more in terms of its"
        # substring was replaced by the better (property-anchored) phrasing.
        if subj in reply.lower() and (
                "more by its" in reply.lower()
                or "think of it by its" in reply.lower()
                or "relate it to its" in reply.lower()):
            fired += 1
        else:
            # Allow Path 2 (ConceptNet feature frame) as an acceptable derived
            # path, but NOT a bare structure-mapped/random pair without the
            # subject's profile. The key guard: the subject word appears and a
            # derived (non-random) framing is used.
            assert subj in reply.lower(), f"reply omits subject {subj!r}: {reply!r}"
    # The learned probe must cover the large majority of subjects (the literal
    # A.1 goal: widen empirical base so Path 1 fires). Require >= 90% Path-1.
    rate = fired / len(subjects)
    assert rate >= 0.9, f"Path-1 coverage only {rate:.2f} ({fired}/{len(subjects)})"


def test_wide_probe_coverage_via_lancaster():
    """Fix A.1 guard: the production cross-modal probe must be the
    WIDE-COVERAGE Lancaster probe (39,707 human-rated words), with the 535-word
    Binder probe retained only as a fine-grained fallback. The combined encoder
    must load Lancaster as primary, and it must generalize a sensory profile to
    words far outside the Binder training set -- proving the cross-modal read-out
    is no longer limited to 535 lemmas (the original A.1 coverage gap)."""
    import os
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    binder = os.path.join(repo_root, "data", "attribute_encoder.npz")
    lanc = os.path.join(repo_root, "data", "lancaster_encoder.npz")
    if not (os.path.exists(binder) and os.path.exists(lanc)):
        pytest.skip("encoder artifacts not present")
    from ravana.ontology.attribute_encoder import (
        load_combined_encoder, build_glove64_lookup)
    lut, _ = build_glove64_lookup(
        os.path.join(repo_root, "data", "ravana_glove_cache.npz"))
    enc = load_combined_encoder(binder, lanc)
    # Wider probe is actually wired (primary), Binder kept as fallback.
    assert enc.lancaster is not None, "Lancaster wide-coverage probe must be wired"
    assert enc.binder is not None, "Binder fine-grained probe must remain as fallback"
    # Rare/abstract words well outside the 535-word Binder base must still
    # receive a sensible sensory profile from the wider probe.
    rare = ["quotient", "umbra", "cathedral", "serendipity", "algorithm",
            "neuron", "equinox", "lattice", "cipher", "monastery", "kiln",
            "cobalt", "tundra", "quartz"]
    present = [w for w in rare if w in lut]
    covered = 0
    for w in present:
        av = enc.attribute_vector(lut[w])
        sensory = max(av[enc.dim_index[d]] for d in
                      ("Vision", "Color", "Shape", "Sound", "Taste", "Smell",
                       "Touch", "Texture", "Weight", "Pattern"))
        if sensory > 0.5:
            covered += 1
    # The wider probe must activate sensory dims for the majority of these
    # out-of-Binder-base words (coverage, not just 535 lemmas).
    assert covered >= int(0.6 * len(present)), (
        f"wide probe only covered {covered}/{len(present)} rare words")


def test_engine_wires_combined_encoder_with_lancaster():
    """Regression guard for the actual Fix A.1 wiring bug: ConceptNetOntology
    .load() is a @classmethod that builds a FRESH object, discarding the
    attribute_encoder passed to the constructor. If the engine does not
    re-attach it after load(), ont.attribute_encoder ends up None and the
    wide-coverage Lancaster probe never drives Path 1 (the gate silently
    falls back to the Binder-only lazy-load). Assert the live engine exposes
    the COMBINED encoder (Lancaster primary + Binder fallback)."""
    e = _eng()
    enc = e._cn_ontology.attribute_encoder
    from ravana.ontology.attribute_encoder import CombinedAttributeEncoder
    assert isinstance(enc, CombinedAttributeEncoder), (
        f"engine must load the combined encoder, got {type(enc).__name__}")
    assert enc.lancaster is not None, "Lancaster wide probe must be wired into the engine"
    assert enc.binder is not None, "Binder fallback must remain wired"
    # And it must drive Path 1 for an out-of-Binder-base word.
    e._is_category_error("what is the taste of a triangle")
    subj = getattr(e, "_last_category_subject", "triangle")
    reply = e._metaphor_for_category_error(subj, "taste")
    # B3 rewrite: Path 1 now anchors on the asked property ("doesn't really
    # have a taste") then bridges to the subject's cross-modal read via
    # "more by its <phrase>" — the old literal "more in terms of its" is gone
    # by design. Assert the Lancaster-driven cross-modal profile fires and
    # stays subject-anchored (no random draw, no 'flavor of a tuesday').
    assert reply and subj in reply.lower() and (
        "more by its" in reply.lower() or "think of it by its" in reply.lower()
        or "relate it to its" in reply.lower()), (
        f"Lancaster-driven Path 1 should fire: {reply!r}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
