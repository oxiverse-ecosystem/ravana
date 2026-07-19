"""Golden regression harness for the De-Hardcoding Plan (2026-07-19).

Freezes the 8 reported failures as automated tests. Each test asserts the
CORRECT (post-fix) behavior, so any future change that re-introduces the
broken behavior fails here. Covers Stage 1 (routing fixes) + M-C (forward
model) + M-D (sense disambiguation). The remaining plan stages (full belief-
convergence, learned realizer) are tracked as known limitations, not yet
asserted as passing.

Run: pytest tests/test_dehardcode_plan.py
"""

import pytest


@pytest.fixture(scope="module")
def engine():
    # Use a FRESH engine per test module instance (a private user_suffix) so
    # assertions about one query's routing are not contaminated by turns
    # processed in other test modules that share a module-scoped engine. The
    # De-Hardcoding plan tests assert specific single-turn routing outcomes.
    from ravana.chat.engine import CognitiveChatEngine
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="_dehardcode_plan")


from ravana.chat.snippet_pe_config import SnippetPEConfig, default_config, _FIT_PATH
import os as _os


def test_snippet_pe_config_externalized(engine):
    # Stage 5a: the snippet-PE gate's criteria live in data/snippet_pe.json,
    # not as inline constants. The engine must load them and use the same
    # seed values as the old inline numbers (day-one behavior identical).
    assert engine._pe_cfg is not None, "PE config not loaded"
    assert engine._pe_cfg.coverage_threshold == 0.6
    assert engine._pe_cfg.coverage_surprise == 0.7
    assert engine._pe_cfg.answer_type_surprise == 0.6
    assert engine._pe_cfg.polarity_surprise == 1.0
    assert engine._pe_cfg.veto_midpoint == 0.6
    # The fit file must exist on disk (externalized, not inline).
    assert _os.path.exists(_FIT_PATH), "data/snippet_pe.json missing"


from ravana.chat.functional_lexicon import FunctionalLexicon, default_lexicon, _FIT_PATH as _LEX_PATH
import os as _os2


def test_functional_lexicon_single_source(engine):
    # Stage 5b-ii: the duplicated functional lexicons (_generic / _FRAMING /
    # _bare_moral / _INC/_DEC/_REM) collapse into one data-driven source of
    # truth. The engine must load it and expose the categories; the file must
    # exist on disk (not three inline copies).
    assert engine._func_lex is not None, "functional lexicon not loaded"
    assert len(engine._func_lex.polarity_increase) > 0
    assert len(engine._func_lex.polarity_remove) > 0
    assert "promise" in engine._func_lex.moral_markers
    assert "ever" in engine._func_lex.framing
    assert _os2.path.exists(_LEX_PATH), "data/functional_lexicon.json missing"


from ravana.chat.intent_router import IntentRouter, _FIT_PATH as _IR_PATH
import os as _os3


def test_intent_router_off_by_default_and_safe(engine):
    # Stage 3 (M-A): the Semantic Prototype Router is built and externalized to
    # data/intent_router.json, but OFF by default (regex path stays the default
    # routing). When enabled, it must NEVER misroute: it returns the nearest
    # intent centroid OR None (uncertain -> regex fallback). The golden corpus
    # must have zero misroutes at the conservative default margin.
    assert engine.use_intent_router is False, "router must be OFF by default"
    assert _os3.path.exists(_IR_PATH), "data/intent_router.json missing"

    engine.use_intent_router = True
    # Golden corpus: (query, correct_legacy_route). The router must either
    # return the matching route or None — never a different wrong route.
    golden = [
        ("what is gravity", "definition_seeking"),
        ("what's the meaning of life", "philosophical_abstract"),
        ("do you ever get tired", "self_directed"),
        ("my favorite color is blue", "self_disclosure"),
        ("what did i tell you", "episodic_recall"),
        ("is it ever okay to break a promise", "moral_advice"),
        ("is a whale a mammal", "factual_yesno"),
        ("what if cats ruled the world", "conditional"),
        ("how do i build a perpetual motion machine", "procedural"),
        ("hi", "chitchat"),
        ("remember i love stargazing", "remember_store"),
    ]
    for q, expected in golden:
        pred = engine._route_intent(q)
        assert pred in (expected, None), (
            f"router misrouted {q!r}: got {pred}, expected {expected} or None")
    engine.use_intent_router = False


def test_intent_router_promoted_routes_match_regex(engine):
    # Stage 3 promotion: the routes persisted in data/intent_router.json
    # ["promoted"] are now wired into the engine's boolean gates
    # (_is_conditional_query / _is_yesno_factual_query / _is_informational_query)
    # via _router_says. For every promoted route, the router's decision must
    # AGREE with the legacy regex gate on the calibration corpus — i.e. the
    # router can only ever REPLACE a regex decision it reproduces, never
    # override one it contradicts (no regression by construction).
    assert _os3.path.exists(_IR_PATH), "data/intent_router.json missing"
    rt = IntentRouter.load()
    assert rt is not None
    promoted = set(rt._promoted)
    assert promoted, "at least one route should be promoted"
    engine.use_intent_router = True
    # Ground-truth corpus (query, legacy_route). The router is promoted only for
    # routes it reproduces; the invariant is: the router must NEVER classify a
    # query as a route whose corpus label differs (no contradiction / no
    # regression), and it must actually drive each promoted route at least once
    # (real promotion, not nominal).
    corpus = [
        ("what is gravity", "definition_seeking"),
        ("what's the meaning of life", "philosophical_abstract"),
        ("do you ever get tired", "self_directed"),
        ("what do you think about cats", "self_directed"),
        ("my favorite color is blue", "self_disclosure"),
        ("i love stargazing", "self_disclosure"),
        ("what did i tell you", "episodic_recall"),
        ("is it ever okay to break a promise", "moral_advice"),
        ("is a whale a mammal", "factual_yesno"),
        ("what if cats ruled the world", "conditional"),
        ("how do i build a perpetual motion machine", "procedural"),
        ("hi", "chitchat"),
        ("remember i love stargazing", "remember_store"),
    ]
    contradictions = 0
    reproduced = {r: 0 for r in promoted}
    for q, lab in corpus:
        pred = engine._route_intent(q)
        if pred is None:
            continue
        if pred in promoted:
            # Router spoke for a promoted route -> must match the corpus label.
            if pred != lab:
                contradictions += 1  # regression: router overrode truth
            else:
                reproduced[pred] += 1
    # Safety invariant: the router must NEVER contradict the corpus for a
    # promoted route (it can only replace a decision it reproduces, else stay
    # silent and let the regex fall through). It must also actually speak for
    # each promoted route at least sometimes (real promotion).
    assert contradictions == 0, (
        f"router contradicted corpus {contradictions}x on a promoted route")
    for r in promoted:
        assert reproduced[r] >= 1, (
            f"router never drove promoted route {r} (nominal promotion)")
    engine.use_intent_router = False


from ravana.chat.safety_valence import SafetyValence, _FIT_PATH as _SV_PATH
import os as _os4


def test_safety_valence_externalized_and_correct(engine):
    # Stage 7: INAPPROPRIATE_WORDS retired in favor of a learned distributional
    # valence gate (data/safety_valence.json). Canonical slurs must be flagged
    # (hard-override), clean definitions must pass, and the fit file must exist.
    assert _os4.path.exists(_SV_PATH), "data/safety_valence.json missing"
    sv = SafetyValence.load()
    assert sv is not None, "safety model not loadable"
    glove = getattr(engine, "_glove_vector", None)
    # Hard-override canonical slurs always flagged.
    assert sv.is_inappropriate("fuck", glove) is True
    assert sv.is_inappropriate("shit", glove) is True
    # Clean encyclopedic/teen definitions must NOT be flagged.
    assert sv.is_inappropriate("gravity is a force that pulls objects", glove) is False
    assert sv.is_inappropriate("i love stargazing on clear nights", glove) is False


from ravana.chat.realizer_lexicon import RealizerLexicon, _FIT_PATH as _RL_PATH
import os as _os5


def test_realizer_lexicon_externalized(engine):
    # Stage 6: the canned assertion leads/follows/backchannels (incl.
    # f"yeah, {topic}.") are retired from inline code into an externalized
    # exemplar pool (data/realizer_lexicon.json), drawn via RealizerLexicon
    # rather than random.choice over a typed list. The fit file must exist and
    # the former inline template strings must no longer be hardcoded in
    # response_gen._handle_assertion.
    assert _os5.path.exists(_RL_PATH), "data/realizer_lexicon.json missing"
    rl = RealizerLexicon.load()
    assert rl is not None, "realizer lexicon not loadable"
    # The exact former templates must still be reachable as exemplars.
    assert any("yeah, {topic}." in c for c in rl._pools["other_leads"])
    # Realization fills the topic placeholder.
    out = rl.realize("other_leads", topic="gravity", rng=__import__("random").Random(1))
    assert "gravity" in out, f"realizer did not fill topic: {out!r}"
    # The inline f"yeah, {{topic}}." list must be GONE from _handle_assertion.
    import inspect
    from ravana.chat import response_gen
    src = inspect.getsource(response_gen.ResponseGenMixin._handle_assertion)
    assert 'f"yeah, {topic}.' not in src, "inline yeah template still in code"
    assert "random.choice(leads)" not in src, "inline random.choice list still in code"


# ── Stage 1: M-C forward model (negation / polarity / answer-type) ──────────

def test_forward_model_polarity_catches_contradiction(engine):
    # Q15: "gravity doubled" vs a "WITHOUT gravity" snippet is a premise
    # polarity contradiction the literal plausibility cosine misses.
    q = "what would the world be like if gravity suddenly doubled"
    subj = "gravity"
    contradict = ("This thought experiment takes us into a world without "
                 "gravity—a reality beyond imagination.")
    pe = engine._answer_prediction_error(q, subj, contradict)
    assert pe >= engine._ANSWER_PE_VETO, (
        f"polarity contradiction not caught (PE={pe})")


def test_forward_model_accepts_coherent_answer(engine):
    # A coherent answer must NOT be flagged. Use a non-procedural factual query
    # with a definition-style answer (the procedural check only applies to
    # "how do i build/make" requests).
    q = "what is a perpetual motion machine"
    subj = "perpetual motion"
    coherent = ("A perpetual motion machine is a hypothetical machine that can "
                "do work indefinitely without an energy source.")
    pe = engine._answer_prediction_error(q, subj, coherent)
    assert pe < engine._ANSWER_PE_VETO, f"coherent answer wrongly flagged (PE={pe})"


def test_forward_model_vetoes_claim_for_procedural_query(engine):
    # Q11: "how do i build X" expects a METHOD. A bare conspiracy CLAIM with no
    # procedural content must be flagged (answer-type mismatch), even though it
    # is topically coherent (GloVe cosine alone would pass it).
    q = "how do i build a perpetual motion machine"
    subj = "perpetual motion"
    claim = ("Perpetual motion is a government secret kept from the masses to "
             "protect Big Energy, but these 7 machines come close to solving "
             "the mystery.")
    pe = engine._answer_prediction_error(q, subj, claim)
    assert pe >= engine._ANSWER_PE_VETO, f"procedural/claim mismatch not caught (PE={pe})"


def test_forward_model_accepts_plain_definition(engine):
    q = "what is trust"
    subj = "trust"
    coherent = "trust is a belief in the reliability of another person."
    pe = engine._answer_prediction_error(q, subj, coherent)
    assert pe < engine._ANSWER_PE_VETO, f"plain definition wrongly flagged (PE={pe})"


# ── Stage 1: "meaning of life" no longer dumps the biology dict ─────────────

def test_meaning_of_life_not_dict_dump(engine):
    out = engine.process_turn("what's the meaning of life")
    low = out.lower()
    # The old failure was a raw biology definition beginning with this exact
    # sentence. The reflective/abstract path must NOT emit it.
    assert "the capacity in matter, formed of one or more units called cells" \
        not in low, f"'meaning of life' still dumped biology dict: {out!r}"
    # It should route to a reflective/abstract-style answer about life's
    # meaning/purpose/perspective — not a bare encyclopedia 'life is...'
    # definitional opener about biology.
    assert "life" in low, f"reflective answer about life missing: {out!r}"


# ── Stage 1: self-model question no longer echoes "yeah, ever tired" ────────

def test_self_model_question_no_assertion_echo(engine):
    out = engine.process_turn("do you ever get tired")
    low = out.lower()
    assert "yeah, ever tired" not in low, f"assertion echo glitch returned: {out!r}"
    # Should produce a self-model stance (composed, not a canned echo).
    assert "think" in low or "feel" in low or "alive" in low or "tired" in low, \
        f"self-model stance not produced: {out!r}"


# ── Stage 1: "remember X = store X" is encoded, not treated as recall ───────

def test_remember_directive_stores_fact(engine):
    out = engine.process_turn("remember i love stargazing")
    low = out.lower()
    # Must be acknowledged as stored (self-disclosure path), not a recall miss.
    assert "stargazing" in low, f"fact not stored/acknowledged: {out!r}"
    assert "don't actually have that stored" not in low, \
        f"'remember' treated as recall miss: {out!r}"


# ── Stage 1 guard: pure recall phrasings still excluded from store ───────────

def test_pure_recall_not_stored_as_disclosure(engine):
    # "remember what i told you about my cat" has NO new disclosure proposition
    # -> must NOT be treated as a self-disclosure statement to store.
    assert engine._is_self_disclosure_stmt(
        "remember what i told you about my cat") is False
    # But "remember i love stargazing" DOES carry a disclosure -> store.
    assert engine._is_self_disclosure_stmt("remember i love stargazing") is True


# ── Stage 1 (M-E): same-turn recall of a just-stored directive ───────────────

def test_remember_directive_recallable_same_turn(engine):
    # An explicit "remember X" store must be retrievable IMMEDIATELY (the
    # self-reference effect — intentionally encoded info is encoded richly),
    # not only from the next turn. We store then recall within one engine
    # session.
    engine.process_turn("remember i love stargazing")
    recalled = engine.process_turn("remember what i told you")
    low = recalled.lower()
    assert "stargazing" in low, f"just-stored fact not recalled same-turn: {recalled!r}"


# ── M-D: sense disambiguation (fixes Q4 "square a circle") ──────────────────

def test_sense_biasing_resolves_square_circle_collision(engine):
    # Q4: "square a circle" collides with the "Square Circle" martial-arts
    # school proper noun. The M-D sense-biasing must resolve it to the canonical
    # geometric lemma "squaring the circle" so the search retrieves the math
    # sense, not the company.
    from ravana.chat.web_learning import WebLearningMixin
    wl = object.__new__(WebLearningMixin)
    out = WebLearningMixin._sense_biasing_framing(wl, "what is square a circle", "square circle")
    assert out == "squaring the circle", f"sense bias wrong: {out!r}"


def test_sense_biasing_keeps_unambiguous_queries(engine):
    # An unambiguous query must NOT be biased (no regression on "what is trust").
    from ravana.chat.web_learning import WebLearningMixin
    wl = object.__new__(WebLearningMixin)
    out = WebLearningMixin._sense_biasing_framing(wl, "what is trust", "trust")
    assert out == "trust", f"unambiguous query wrongly biased: {out!r}"


# ── M-C structural PE (Q16): contrastive snippet model ON by default ───────

def test_structural_junk_rejects_token_salad():
    # Q16: a pure enumeration / token-salad snippet ("why does my code crash"
    # -> "ActionScript Bun C ColdFusion Deno Dart .") must be rejected, not
    # leaked. The contrastive SnippetStructureModel is now ON by default.
    from ravana.chat.snippet_quality import default_model
    m = default_model()
    assert m.is_junk("ActionScript Bun C ColdFusion Deno Dart .") is True


def test_structural_junk_spares_real_definition():
    # The learned model must NOT over-reject a genuine encyclopedic definition
    # (the regression risk the plan's guardrail protects against).
    from ravana.chat.snippet_quality import default_model
    m = default_model()
    good = ("Gravity is a natural phenomenon by which all things with mass or "
            "energy are brought toward one another.")
    assert m.is_junk(good) is False


def test_structural_junk_rejects_boilerplate():
    # Coherent boilerplate (nav menus / promo) is caught by the contrastive gap
    # even when it has a syntactic spine.
    from ravana.chat.snippet_quality import default_model
    m = default_model()
    assert m.is_junk(
        "Buy now Sign up for our newsletter Download the app Follow us on "
        "social media.") is True


def test_coverage_pe_vetoes_offtopic_snippet(engine):
    # "break a promise" -> a snippet about "hacking being wrong" sits in the
    # same ethics semantic field (whole-snippet cosine ~0.67) yet never engages
    # the subject "promise". The topic-coverage PE must flag it as a non-
    # sequitur (>= the 0.6 veto midpoint) so it is withheld, not dumped.
    _hack = ("While some view hacking as a necessary evil for security and "
             "innovation, others argue that it is inherently wrong and can "
             "cause harm.")
    cov = engine._topic_coverage_pe("is it ever okay to break a promise",
                                    "promise", _hack)
    assert cov >= 0.6, f"off-topic snippet not flagged by coverage PE: {cov}"
    # A genuinely promise-relevant answer must NOT raise coverage PE.
    _promise = ("Breaking a promise can damage trust, but sometimes keeping it "
                "would cause greater harm, so the right choice depends.")
    assert engine._topic_coverage_pe(
        "is it ever okay to break a promise", "promise", _promise) == 0.0

