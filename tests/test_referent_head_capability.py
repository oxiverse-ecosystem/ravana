"""Referent-head extraction (round 2026-09-30T1031Z, card t_159df91e).

Two defects from the round's chat probe motivated this capability, and both
reproduce without it:

1. A NEGATION particle became the extracted topic.
   "tell me something you don't know much about" grounded to the subject
   ``don't``, and the uncertainty frame rendered it as
   "i don't really have a solid grasp on don't so far".
2. A whole PREDICATE CLAUSE became the stance target and was interpolated
   verbatim: "i'm still forming a view on naming things matters".

Both are the same root cause — no shared notion of which tokens can *denote*
a referent. Every extractor kept its own hand-kept closed-class list and got
it wrong differently.

The capability is ``ravana.chat.topic_head``: a contraction expander, a
closed-class ``FunctionClass`` with two evidence-driven growth paths, and a
clause-segmenting head reducer. The unit tests below pin the module contract;
the engine tests pin the two live defects.

Gate: reverting the capability (the two engine call sites) turns the engine
tests red. Proven by `git stash` of the src diff — see the round report.
"""

import os
import sys

import pytest

_REPO = r"C:\Users\Likhith\Documents\Projects\ravana"
for _p in (_REPO, f"{_REPO}\\ravana_ml\\src", f"{_REPO}\\ravana\\src",
           f"{_REPO}\\ravana-v2\\src"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

from ravana.chat.topic_head import (  # noqa: E402
    FunctionClass,
    expand_contractions,
    referent_head,
)


# ── contraction morphology ────────────────────────────────────────────────
def test_expand_contractions_handles_unlisted_forms_by_rule():
    # The rule covers the whole n't class, not an enumerated list.
    assert expand_contractions("i don't know") == "i do not know"
    assert expand_contractions("she doesn't agree") == "she does not agree"
    assert expand_contractions("they hadn't come") == "they had not come"
    assert expand_contractions("we wouldn't go") == "we would not go"
    assert expand_contractions("it shouldn't matter") == "it should not matter"
    # Auxiliaries whose stem the seed has not seen still reduce to a stem.
    assert expand_contractions("it mightn't matter").endswith("not matter")
    # Irregular stems are the only ones needing a table entry.
    assert expand_contractions("he won't come") == "he will not come"
    assert expand_contractions("it ain't real") == "it is not real"
    assert expand_contractions("i can't go") == "i can not go"
    # A negative that is not a contraction is untouched.
    assert expand_contractions("i am not tired") == "i am not tired"


# ── closed class: the negation defect's root cause ────────────────────────
def test_negation_particles_are_grammatical_not_referents():
    fc = FunctionClass()
    for w in ("not", "never", "no", "nothing", "nobody", "neither"):
        assert fc.is_function(w), w
    # Contraction surface forms are rejected via their expansion, so the seed
    # does not need every surface form enumerated.
    for w in ("don't", "doesn't", "didn't", "can't", "won't", "isn't"):
        assert fc.is_function(w), w
    # Real content words are not grammar.
    for w in ("mangroves", "sourdough", "privacy", "naming"):
        assert not fc.is_function(w), w


def test_referent_head_drops_negation_entirely():
    # The exact round defect: nothing here denotes a referent, so the honest
    # answer is None and the caller must fail open to its own fallback.
    assert referent_head("you don't know much about") is None
    assert referent_head("don't") is None
    assert referent_head("i do not know anything at all") is None


# ── clause segmentation: the predicate-clause defect ──────────────────────
@pytest.mark.parametrize("phrase,expected", [
    ("naming things matters", "naming things"),        # the round defect
    ("silence is underrated", "silence"),              # copular clause
    ("matters", "matters"),                            # bare predicate is the head
    ("mangroves", "mangroves"),                        # already a head
    ("remote work", "remote work"),                    # NOT truncated
    ("public transit", "public transit"),
    ("sourdough starter", "sourdough starter"),
    ("people who talk too much", "people"),            # relative modifier dropped
    # "handle" is noun-ambiguous and POS tags it 'noun', so a learned
    # predicate alone must NOT cut here — that is the anti-regression case.
    ("cities handle heat", "cities handle heat"),
])
def test_referent_head_segments_clauses(phrase, expected):
    assert referent_head(phrase) == expected


def test_referent_head_never_invents_a_referent():
    # Fail-open contract: no content token -> None, never a fabricated topic.
    for phrase in ("", "   ", "?", "the a of and", "do you"):
        assert referent_head(phrase) is None, phrase


# ── online growth: the class is a seed, not a frozen table ─────────────────
def test_function_class_grows_from_a_parser_signal():
    fc = FunctionClass()
    # A word nobody seeded or tabulated is unknown...
    assert not fc.is_function("frobnicate")
    # ...until a parser reports it in function position.
    assert fc.learn("frobnicate") == {"frobnicate"}
    assert fc.is_function("frobnicate")
    # Removing the seed still leaves the learned member: graceful degradation.
    fc2 = FunctionClass(seed=set())
    fc2.learn("frobnicate")
    assert fc2.is_function("frobnicate")
    # A seed member removed from the seed alone stops being grammatical —
    # proving the seed is data, not a hardcoded table baked into the logic.
    fc3 = FunctionClass(seed={"mangroves"})
    assert fc3.is_function("mangroves")


def test_function_class_demotes_ungrounded_topic_tokens_online():
    """A token proposed as a topic that RAVANA can never ground is not a
    referent — the class learns that from extraction evidence alone."""
    fc = FunctionClass()
    # First observation: still allowed (one miss is not a pattern).
    assert fc.observe_topic_token("florp", grounded=False) is True
    # Second consecutive miss crosses the threshold.
    assert fc.observe_topic_token("florp", grounded=False) is False
    assert fc.is_function("florp")
    # A real concept is forgiven immediately — never permanently banned.
    assert fc.observe_topic_token("florp", grounded=True) is True
    assert not fc.is_function("florp")
    # An already-grounded token is never counted at all.
    assert fc.null_counts().get("mangroves") is None


def test_predicate_class_grows_and_requires_pos_agreement_to_cut():
    fc = FunctionClass()
    # A learned predicate alone does NOT cut: "work" is in the engine's verb
    # vocabularies but is a noun in "remote work".
    fc.learn_predicate("gorp")
    assert referent_head("remote work", func=FunctionClass(
        predicate_seed={"work"})) == "remote work"
    # With the caller's POS state agreeing it is a verb, it becomes a clause
    # edge — the growth path pays off.
    got = referent_head("naming things gorp",
                        pos_lookup=lambda w: "verb" if w == "gorp" else "noun",
                        func=fc)
    assert got == "naming things"


# ── live engine: the two round defects ────────────────────────────────────
@pytest.fixture(scope="module")
def engine():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="test_t159df91e")
    yield eng
    try:
        eng.stop_background_learning()
    except Exception:
        pass


def test_engine_negation_is_never_the_extracted_subject(engine):
    # The round defect, pinned on the live engine: the negation particle must
    # not survive as a referent. Previously the subject WAS "don't" and the
    # reply read "i don't really have a solid grasp on don't so far".
    fc = engine._function_class()
    assert fc is not None, "referent capability must be present"
    assert fc.is_function("don't")
    assert engine._referent_head("you don't know much about") is None
    # And the subject the engine actually extracts is not the particle.
    for q in ("tell me something you don't know much about",
              "what do you not know about?"):
        subj, _ = engine._extract_topic(q, [])
        assert (subj or "").strip().lower() not in ("don't", "not", "do"), \
            f"negation leaked as subject for {q!r}: {subj!r}"


def test_engine_predicate_clause_is_reduced_to_its_head(engine):
    # The second round defect: the raw clause was interpolated into the reply.
    assert engine._referent_head("naming things matters") == "naming things"


def test_engine_capability_survives_a_real_turn(engine):
    """The capability must work in the live pipeline, not just in isolation."""
    r = engine.process_turn("what do you think about naming things matters?")
    reply = r.get("response", "") if isinstance(r, dict) else str(r)
    assert "naming things matters" not in reply, \
        f"raw predicate clause leaked into the reply: {reply!r}"


def test_engine_learns_a_predicate_online(engine):
    """Growth path, live: a word with positive verb evidence is registered
    permanently in the engine's referent class, and the class is what the
    clause reducer consults.

    Registration is unconditional; whether a learned predicate actually cuts a
    clause additionally requires the POS state to agree (see the module test
    for why). Both halves are asserted so a future change cannot quietly make
    the growth path either inert or over-eager.
    """
    fc = engine._function_class()
    assert "glorp" not in fc.predicates
    engine._learn_predicates_from(["glorp"])
    assert "glorp" in fc.predicates
    # A POS-tagged verb the engine learned DOES become a clause edge.
    pos = dict(engine._concept_pos)
    pos["glorp"] = "verb"
    got = referent_head("naming things glorp",
                        pos_lookup=lambda w: pos.get(w), func=fc)
    assert got == "naming things", got
    # And the registration is permanent — a second call reports no new words.
    assert engine._learn_predicates_from(["glorp"]) == set()


def test_referent_class_growth_survives_save_and_load():
    """The growth path must be DURABLE.

    A class that forgets what it learned re-learns it every boot, which makes
    the online path decorative. This pins the save -> load round trip in both
    directions (function membership, predicate membership, ungrounded counts)
    and pins the fail-open behaviour on a malformed payload.
    """
    import importlib
    from ravana.chat.engine import CognitiveChatEngine

    suffix = "test_t159df91e_persist"
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    try:
        fc = eng._function_class()
        fc.learn("frobnicate")
        fc.learn_predicate("glorp")
        for _ in range(3):
            fc.observe_topic_token("zibble", grounded=False)
        eng.save()
    finally:
        try:
            eng.stop_background_learning()
        except Exception:
            pass

    eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
    try:
        fc2 = eng2._function_class()
        assert fc2.is_function("frobnicate"), "learned function word was lost"
        assert "glorp" in fc2.predicates, "learned predicate was lost"
        assert fc2.is_function("zibble"), "ungrounded-token demotion was lost"
        # Fail-open: a malformed payload must not corrupt the class.
        before = fc2.is_function("mangroves")
        fc2.load_state({"learned": "not-a-list", "null_counts": "not-a-dict"})
        assert fc2.is_function("mangroves") == before
        fc2.load_state(None)
    finally:
        try:
            eng2.stop_background_learning()
        except Exception:
            pass


def test_no_hardcoded_reply_strings_in_the_capability():
    """Doctrine gate: the capability is a predicate over state, never a source
    of prose. Any sentence-shaped literal in EXECUTABLE code is a violation.

    Docstrings are excluded (they are documentation, not replies) by parsing
    the module with ``ast`` and walking only the string literals that are not
    docstrings.
    """
    import ast
    import inspect
    import ravana.chat.topic_head as th

    tree = ast.parse(inspect.getsource(th))
    # Collect the docstring node of the module and of every function/class.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))

    prose = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) in docstrings:
                continue
            s = node.value
            # Prose = a long literal with spaces that reads as a sentence.
            if len(s) >= 25 and " " in s and s.strip().endswith("."):
                prose.append(s)
    assert not prose, f"prose literals found in capability module: {prose}"
