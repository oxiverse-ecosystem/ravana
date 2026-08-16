"""Regression tests for round 2026-08-15T0830Z fixes.

Three defects found by a fresh-persona (ASTRID) rotating-probe chat round:

1. Double-gerund (Bug 1): `_gerund_of("restoring")` must NOT produce
   "restoringing". The date-recall realizer ("you started restoring radios")
   was collapsing to "restoringing".
2. Degenerate lead (Bug 2): the assertion-lead lexicon must never plant a
   raw/garbled clause into a `got it — {topic}` / `right, {topic}` template.
   Verified by asserting the lexicon pools contain no `{topic}`-slotting phrase
   (the garble source) — the runtime override file is the source of truth.
3. Empathy detection gap (Bug 3 root cause): a first-person feeling-copula
   ("i felt electrified", "i feel giddy") whose affect word is absent from the
   closed VAD lexicon must still be detected as an affective disclosure and met
   with empathy (naming the user's own word), NOT fall through to "noted." /
   a leaked internal valence number. Factual disclosures ("i keep bees",
   "i moved to X") must remain None (fail-closed).

Run: RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest \
        tests/unit/test_round_2026_08_15T0830Z_empathy_gerund.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))  # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "ravana", "src"))

import pytest

from ravana.chat.engine import CognitiveChatEngine, _gerund_of, _extract_user_affect_word
from ravana.chat.realizer_lexicon import _SEED_POOLS

try:
    import json
    _LEX_PATH = os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "data", "realizer_lexicon.json")
    with open(_LEX_PATH, encoding="utf-8") as _f:
        _LEX = json.load(_f)
    _RUNTIME_POOLS = _LEX.get("pools", {})
except Exception:
    _RUNTIME_POOLS = {}


# ── Bug 1: double-gerund ──────────────────────────────────────────────────────
def test_gerund_does_not_double_an_ing_stem():
    for stem in ("restoring", "building", "studying", "running", "ringing"):
        assert _gerund_of(stem) == stem, f"_gerund_of({stem!r}) doubled"


def test_gerund_of_regular_verb():
    assert _gerund_of("keep") == "keeping"
    assert _gerund_of("restore") == "restoring"


def test_gerund_phrase_restoring_radios():
    from ravana.chat.engine import _verb_phrase_to_gerund
    assert _verb_phrase_to_gerund("restoring radios") == "restoring radios"


# ── Bug 2: no topic-slotting garble (the reflect-the-topic lead gate) ─────────
# The lexicon intentionally KEEPS the "{topic}" templates (so the de-hardcoding
# externalized-lexicon test still finds "yeah, {topic}." in other_leads). The
# garble protection is NOT "no {topic} in the pool" — it is the consumer gate
# realizer_lexicon.has_clean_topic(), which only permits a reflect-the-topic lead
# when the topic is a clean noun phrase. This test verifies that gate directly:
# clauses / verbs / copulas / opinions / comparatives must NEVER be reflected,
# while a clean noun IS. The runtime and this test share the same function, so
# they cannot drift.
from ravana.chat.realizer_lexicon import has_clean_topic

# Each entry: (topic_subject, original_utterance, expected_clean)
_GARBLE_CASES = [
    # The original bug's degenerate clauses — must be rejected.
    ("saying loud", "i keep saying loud about it", False),
    ("believe nuclear energy", "i believe nuclear energy is clean", False),
    ("jazz seriously overrated", "jazz seriously overrated imo", False),
    # Opinions are not self-descriptions -> never "so you're <opinion>".
    ("cassettes", "i think i was wrong about cassettes", False),
    ("tape", "i prefer tape", False),
    # Comparatives resolve to a bare noun head that lacks markers -> still reject.
    ("ceramics meditative painting", "ceramics is more meditative than painting", False),
    ("painting", "painting is better than ceramics", False),
    # Copula / verb subjects must not be planted.
    ("am a teacher", "i am a teacher", False),
    ("feel tired", "i feel tired", False),
    # Multi-word clause subjects (>2 tokens) are never a clean noun phrase.
    ("the thing he said about", "the thing he said about it", False),
]

_CLEAN_CASES = [
    ("gravity", "i'm studying gravity", True),
    ("a teacher", "i'm a teacher", True),
    ("bees", "i keep bees", True),
    ("ravens", "ravens are clever", True),
]


def test_clean_topic_gate_rejects_garble_clauses():
    for topic, text, expected in _GARBLE_CASES:
        assert has_clean_topic(topic, text) is expected, (
            f"has_clean_topic({topic!r}, {text!r}) should be {expected}")


def test_clean_topic_gate_allows_clean_nouns():
    for topic, text, expected in _CLEAN_CASES:
        assert has_clean_topic(topic, text) is expected, (
            f"has_clean_topic({topic!r}, {text!r}) should be {expected}")


def test_lead_pools_keep_externalized_topic_templates():
    # The de-hardcoding plan moved the templates into the pool, NOT deleted them.
    # The reflect-the-topic exemplar must remain reachable (feature preserved).
    pools = _RUNTIME_POOLS or _SEED_POOLS
    assert any("{topic}" in c for c in pools.get("other_leads", [])), \
        "externalized 'yeah, {topic}.' exemplar missing from other_leads"


# ── Bug 3: empathy detection for lexicon-absent feeling words ─────────────────
@pytest.fixture(scope="module")
def engine():
    os.environ["RAVANA_OFFLINE"] = "1"
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="test_0830z_empathy")


def test_feeling_copula_detected_even_when_word_not_in_vad(engine):
    # "electrified" / "giddy" are NOT in the closed VAD lexicon; the copula
    # fallback must still return a disclosure so empathy fires.
    for u in ("i felt electrified the first time a swarm settled",
              "i feel giddy about it",
              "i feel hollow today"):
        d = engine._detect_emotional_disclosure(ctx=None, text=u)
        assert d is not None, f"missed affective disclosure: {u!r}"
        assert d[1], f"disclosure has no felt word: {u!r} -> {d!r}"


def test_factual_disclosure_still_fails_closed(engine):
    # These are NOT affective; the detector must return None so facts reach
    # autobiographical storage (no false empathy, no leaked number).
    for u in ("i keep bees and tend a cabin",
              "i moved to sundby in 2014",
              "i started restoring radios in 2016"):
        assert engine._detect_emotional_disclosure(ctx=None, text=u) is None, \
            f"false-positive empathy on factual disclosure: {u!r}"


def test_extract_user_affect_word_returns_lexicon_words():
    # _extract_user_affect_word only returns words IN the broad affect lexicon
    # (electrified is NOT). The detector's copula fallback is what catches
    # out-of-lexicon words; this test just pins the in-lexicon behavior.
    assert _extract_user_affect_word("i felt terrified") == "terrified"
    assert _extract_user_affect_word("i felt electrified") == ""


def test_i_am_name_not_treated_as_affect(engine):
    """Regression: 'I am Noor' / 'I'm Corvin' must NOT be treated as affective
    disclosures; the copula fallback is restricted to is_affect_term-approved
    words for 'I am' / 'I'm' forms."""
    for u in ("I am Noor", "I'm Corvin", "i am Ahmed"):
        d = engine._detect_emotional_disclosure(ctx=None, text=u)
        assert d is None, f"false-positive affect on name disclosure: {u!r} -> {d!r}"


def test_i_feel_explicit_still_works(engine):
    """Regression: explicit 'I feel' / 'I felt' forms remain affect disclosures
    even for non-lexicon words, preserving the existing copula fallback."""
    for u in ("I feel electrified", "I felt giddy", "I feel hollow"):
        d = engine._detect_emotional_disclosure(ctx=None, text=u)
        assert d is not None, f"missed affective disclosure: {u!r}"
        assert d[1], f"disclosure has no felt word: {u!r} -> {d!r}"
