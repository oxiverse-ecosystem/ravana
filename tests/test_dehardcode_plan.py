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
    #
    # Use an isolated data_dir (temp) so parallel pytest-xdist workers don't
    # contend on the same SQLite DB file (database-is-locked errors). The repo
    # GloVe cache is injected manually below regardless of data_dir.
    import tempfile
    _data_dir = tempfile.mkdtemp(prefix="ravana_dhp_")
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               data_dir=_data_dir,
                               user_suffix="_dehardcode_plan")
    _proj_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    glove_cache = _os.path.join(_proj_root, "data", "ravana_glove_cache.npz")
    if _os.path.exists(glove_cache):
        import numpy as np
        d = np.load(glove_cache, allow_pickle=True)
        eng._glove_vecs = {str(w).lower(): v for w, v in zip(d["words"].tolist(), d["vecs"])}
        eng._glove_proj = d["proj"].astype(np.float32)
        eng._glove_dim = int(d["proj"].shape[1])
    return eng


from ravana.chat.snippet_pe_config import SnippetPEConfig, default_config, _FIT_PATH
import os as _os


def test_snippet_pe_config_externalized(engine):
    # Stage 5a: the snippet-PE gate's criteria live in data/snippet_pe.json,
    # not as inline constants. The engine must load them, and every criterion
    # must be a finite number in its valid range.
    #
    # NOTE: the values are FITTED by experiments/measure_snippet_pe.py (EER on a
    # labeled corpus), so they are deliberately NOT pinned to the legacy seed
    # numbers — pinning them here would defeat the point of externalizing them.
    # What must hold is that the config is loaded, externalized, and sane.
    import math
    assert engine._pe_cfg is not None, "PE config not loaded"
    seeds = SnippetPEConfig()
    for field in ("coverage_threshold", "coverage_surprise",
                  "answer_type_surprise", "polarity_surprise",
                  "veto_midpoint"):
        val = getattr(engine._pe_cfg, field)
        assert isinstance(val, float), f"{field} not a float: {val!r}"
        assert math.isfinite(val), f"{field} is non-finite ({val}) — a gate " \
            f"that can never fire; the calibration harness must not emit inf"
        assert 0.0 <= val <= 2.0, f"{field} out of range: {val}"
    # Criteria must not drift wildly from the audited seeds.
    for field in ("coverage_threshold", "coverage_surprise",
                  "answer_type_surprise", "veto_midpoint"):
        assert abs(getattr(engine._pe_cfg, field)
                   - getattr(seeds, field)) <= 0.25, \
            f"{field} drifted too far from seed"
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


def test_intent_router_on_by_default_and_safe(engine):
    # Stage3 (M-A): the Semantic Prototype Router is built and externalized to
    # data/intent_router.json, and is ON by default (replacing the hardcoded
    # routing regex). When enabled, it must NEVER misroute: it returns the nearest
    # intent centroid OR None (uncertain -> regex fallback). The golden corpus
    # must have zero misroutes at the conservative default margin.
    assert engine.use_intent_router is True, "router must be ON by default"
    assert _os3.path.exists(_IR_PATH), "data/intent_router.json missing"

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
    # regex path must remain available as the fallback for uncertain/None routes
    engine.use_intent_router = False
    try:
        for q, expected in golden:
            pred = engine._route_intent(q)
            assert pred in (expected, None), (
                f"regex fallback misrouted {q!r}: got {pred}, expected {expected} or None")
    finally:
        engine.use_intent_router = True


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


def test_self_directed_promoted_pre_admit_no_empty_regression(engine):
    # Stage 3 residual-cluster completion: self_directed is promoted as a
    # PRE-ADMIT gate to _route_self_query (never a replacement). The exact
    # regression from the earlier attempt was "do you ever get tired" ->
    # EMPTY response (router admitted but the block's compositional answering
    # didn't fire). The pre-admit must call _route_self_query and return its
    # answer; if the block returns None it falls through to the legacy path.
    assert "self_directed" in set(IntentRouter.load()._promoted), \
        "self_directed must be in promoted for this gate to be active"
    engine.use_intent_router = True
    # Router classifies the agent-mind query without contradicting the corpus.
    assert engine._route_intent("do you ever get tired") == "self_directed"
    # End-to-end: must produce a non-empty self-model answer, NOT a web def.
    resp = engine.process_turn("do you ever get tired")
    assert resp and resp.strip(), "self_directed query regressed to empty response"
    assert "web source" not in resp.lower(), \
        "self_directed query wrongly routed to web definition"
    _self_keys = ("think", "feel", "weigh", "learn", "person", "ravana",
                  "ai", "conscious")
    assert any(k in resp.lower() for k in _self_keys), \
        f"self_directed query not answered from self-model: {resp!r}"
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


# ─────────────────────────────────────────────────────────────────────────────
# RAVANA round 2026-08-11T1328Z regression guards
# D1: reaction/affiliation gate must not swallow a genuine question.
# D2: a question must not be answered by echoing an unrelated prior turn.
# D3: bare "i'm X" copula must not store a transient phrase as the user's name.
# D4: activity/event miner must not store meta-discourse / inner-state objects.
# ─────────────────────────────────────────────────────────────────────────────

def test_D1_reaction_gate_spares_question(engine):
    # "so what's your real read on X" / "so, whose dog is it now" open with the
    # reaction lead-in "so" but are QUESTIONS. The affiliation gate must NOT
    # swallow them into a hollow "glad you felt that" ack; they must fall
    # through to the real pipeline.
    eng = engine
    r = eng.process_turn(
        "so what's your real read on the cave versus the radio, now you've heard all that?")
    assert "glad you felt that" not in r, f"D1 regression: question swallowed by reaction gate -> {r!r}"
    # A genuine reaction (no '?', no interrogative opener) must still route to
    # the affiliation frame without crashing.
    r2 = eng.process_turn("so, that really landed")
    assert isinstance(r2, str) and r2.strip()


def test_D3_bare_copula_does_not_poison_name(engine):
    # Transient copula phrases must NOT be stored as the user's NAME.
    eng = engine
    eng.process_turn("i'm not here, not really.")
    eng.process_turn("i'm most myself when the water's still.")
    eng.process_turn("i'm gone by the time you read this.")
    name = eng.user_model.user_name
    assert name not in ("Not Here", "Most Myself", "Gone"), \
        f"D3 regression: transient phrase stored as name -> {name!r}"
    # A real self-naming still works.
    eng.process_turn("i'm mara.")
    assert eng.user_model.user_name.lower() == "mara", \
        f"D3 regression: real name not captured -> {eng.user_model.user_name!r}"


def test_D4_activity_miner_skips_meta_discourse(engine):
    # "i keep saying it" / "i felt a kind of weight lift" / "i told a friend"
    # must NOT become ('i','does',...) / ('i','event',...) possession facts.
    eng = engine
    eng.process_turn("i keep saying it — the cave keeps its secrets.")
    eng.process_turn("i felt a kind of weight lift when i surfaced.")
    eng.process_turn("i told a friend about the sump last week.")
    facts = eng.user_model.personal_facts.facts
    junk = [k for k in facts if isinstance(k, tuple) and len(k) >= 3
            and k[1] in ("does", "event")
            and str(facts[k].value).lower() in
            ("keep saying", "felt kind", "told friend", "told friend drowned",
             "lose track", "felt weight")]
    assert not junk, f"D4 regression: meta-discourse stored as activity/event -> {junk}"


def test_D2_question_not_answered_by_unrelated_echo(engine):
    # A question whose best hippocampal match shares only a stray token must
    # not be answered by echoing an unrelated prior turn.
    import re as _re
    eng = engine
    eng.process_turn("i forage chanterelles up past the quarry, the western slope after rain.")
    q = "come on, the sump over the shack any day, right?"
    mem = eng._try_hippocampal_retrieval(
        type("_Ctx", (), {"subject": "sump"})(), q)
    if mem is not None:
        qt = {t for t in _re.findall(r"[a-zA-Z']+", q.lower()) if len(t) >= 3}
        ft = {t for t in _re.findall(r"[a-zA-Z']+", mem.lower()) if len(t) >= 3}
        assert len(qt & ft) >= 2, \
            f"D2 regression: unrelated fact echoed for a question: {mem!r}"


# ─────────────────────────────────────────────────────────────────────────────
# Feature (round 2026-08-11T1328Z): Agent Self-Stance Formation & Recall.
# Residual limitation from the round: a self-opinion question ("what's your
# read on X") fell through to "i'm still figuring that out" even when the USER
# had stated strong views on X across the conversation. RAVANA had no structured
# self-model to render a real lean from. This test asserts the NEW capability:
# the agent DERIVES, RECORDS, and RECALLS an informed stance grounded in the
# user's actual learned stance — and stays honestly silent only with no evidence.
# ─────────────────────────────────────────────────────────────────────────────

def test_F1_agent_self_stance_forms_from_user_stance(engine):
    # The user states a strong, repeated view on a topic. The agent's own
    # self-opinion question about that topic must now render a REAL grounded
    # lean ("i lean toward X" / "i am drawn to X") instead of the hollow
    # "still figuring that out" frame.
    eng = engine
    # Mine a confident user stance: "i really love chanterelles" (positive),
    # expressed twice so confidence clears the 0.35 floor.
    eng.process_turn("i really love chanterelles — they're the best thing i find.")
    eng.process_turn("i really love chanterelles more than any other mushroom.")
    # Sanity: the user stance was actually learned.
    _us = eng.user_model.opinions.resolve_topic("chanterelles")
    assert _us is not None, "precondition: user stance on 'chanterelles' not mined"
    _st = eng.user_model.opinions.query_stance(_us)
    assert _st is not None and _st.confidence >= 0.35, \
        f"precondition: user stance confidence too low: {_st}"
    # Now ask the agent for ITS read on the topic.
    r = eng.process_turn("what do you think about chanterelles?")
    assert "still figuring that out" not in r, \
        f"F1 regression: agent has no self-stance on a discussed topic -> {r!r}"
    # The agent must have DERIVED + RECORDED its own stance in the durable store.
    _key = eng._agent_stance_key("chanterelles")
    assert _key, "precondition: agent stance key derivation failed"
    _own = getattr(eng, "_agent_stances", {})
    assert _key in _own and _own[_key].confidence >= 0.35, \
        f"F1 regression: agent did not form/record a stance on 'chanterelles': {_own!r}"
    # The recorded stance is GROUNDED in the user's (positive) polarity, not a
    # confabulation: agent polarity should be positive (drawn toward), since the
    # user was strongly for it.
    assert _own[_key].polarity > 0.0, \
        f"F1 regression: agent stance polarity not grounded in user's positive view: {_own[_key].polarity}"


def test_F2_agent_self_stance_recalled_stably(engine):
    # The derived stance must persist in the engine's store and be recalled on
    # a LATER ask (personality continuity), not recomputed as a hollow answer
    # each time. This exercises the recall branch of the resolver.
    eng = engine
    # Mine a confident user stance the miner recognizes ("i really love X").
    eng.process_turn("i really love open hardware — it should be the standard.")
    eng.process_turn("i really love open hardware more than closed gear.")
    r1 = eng.process_turn("what do you think about open hardware?")
    assert "still figuring that out" not in r1, \
        f"F2 regression: first ask hollow -> {r1!r}"
    _key = eng._agent_stance_key("open hardware")
    _own = getattr(eng, "_agent_stances", {})
    assert _key in _own, f"F2 precondition: stance not recorded -> {_own!r}"
    # Ask again; the resolver must hit the RECALL branch (confidence preserved,
    # no reformation needed) and still render a grounded stance.
    r2 = eng.process_turn("what's your take on open hardware?")
    assert "still figuring that out" not in r2, \
        f"F2 regression: recall branch hollow on second ask -> {r2!r}"
    assert getattr(_own[_key], "confidence", 0.0) >= 0.35, \
        f"F2 regression: recorded stance confidence dropped -> {_own[_key]}"


def test_F3_agent_honest_when_no_evidence(engine):
    # When the user has expressed NO view on a topic and there is no
    # constitutive value, the agent must remain HONEST (no fabricated stance).
    # "blefuscu" is a topic neither seeded nor discussed — no evidence exists.
    eng = engine
    r = eng.process_turn("what do you think about blefuscu?")
    assert "still figuring that out" in r, \
        f"F3 regression: agent fabricated a stance on an unseen topic -> {r!r}"
    _key = eng._agent_stance_key("blefuscu")
    _own = getattr(eng, "_agent_stances", {})
    assert _key not in _own or _own.get(_key, None) is None, \
        f"F3 regression: a stance was stored for an evidence-less topic -> {_own!r}"


def test_F4_agent_stance_key_rejects_junk_topics(engine):
    # Documented LIMIT: the canonical key derivation (_agent_stance_key) must
    # reject non-topics ("right"/"source"/"it"/"that"/...) so a hollow or
    # deictic cue can never become a stored agent stance (the old
    # confabulation class). A real topic returns its stripped lowercase key.
    eng = engine
    # Junk / deictic tokens -> empty key (callers treat "" as "no stance").
    # These exactly match the _JUNK set in engine_self_query.py:_agent_stance_key.
    for _junk in ("right", "it", "that", "things", "really", "all", "matter",
                  "topic", "question", "ok", "okay", "", "   "):
        assert eng._agent_stance_key(_junk) == "", \
            f"F4 regression: junk topic '{_junk}' should yield empty key, " \
            f"got {eng._agent_stance_key(_junk)!r}"
    # A real topic yields a canonical lowercased key (multiword kept verbatim).
    assert eng._agent_stance_key("Chanterelles") == "chanterelles", \
        "F4 regression: real topic key not canonicalized"
    assert eng._agent_stance_key("open Hardware") == "open hardware", \
        "F4 regression: multiword topic key not lowercased"


