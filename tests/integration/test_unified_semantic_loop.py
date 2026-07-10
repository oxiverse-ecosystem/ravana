"""
End-to-end soak test: the FULL unified semantic loop (A1->N1->N4->N2->C-lite->N3->E).

Every stage is proven in isolation; this test proves they HOLD TOGETHER when
they interact across a real multi-turn chat, using the production objects wired
exactly as the live agent uses them.

It runs the SAME loop against BOTH runnable agents:
  * ChatInterface          — the fully-wired agent (pfc_workspace N2/N4 + web_learner C-lite/E)
  * CognitiveChatEngine    — the engine path that scripts/ravana_chat.py ACTUALLY runs
                             (pfc_workspace N2/N4 + WebLearningMixin C-lite/E, wired in this work).

Proving it on ChatInterface alone would leave the "proven in isolation, unproven
where it runs" gap one layer down — so the engine path is asserted directly.

The loop per turn:
  turn            -> agent.process_turn  (A1/N1 classify, N4 gate, N2 spawn, sleep)
  novel intent    -> PrefrontalWorkspace.learn_from_turn routes ABSTAIN -> EmergentCategoryLearner
  curiosity (E)   -> curiosity_e_step selects argmax-EFE topic (sparse neighbourhood)
  web-read (C)    -> C-lite writes typed triples into the REAL graph
  binding (N3)    -> DualCodeSpace encodes the new fact in the 2048-D HRR space
  sleep           -> PrefrontalWorkspace.sleep consolidates rehearsed candidates

Offline/deterministic: the "web read" is injected via the C-lite writer (exactly
what the learning loop calls) — no network. This isolates the cross-stage
interaction, which is the thing unit tests don't cover.

process_turn is ~18s, so the soak is a few turns, not hundreds.
"""

import os
import sys

import numpy as np
import pytest

_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in [
    os.path.join(_proj_root, "ravana_ml", "src"),
    os.path.join(_proj_root, "ravana", "src"),
    os.path.join(_proj_root, "ravana-v2", "src"),
    _proj_root,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

pytestmark = [pytest.mark.integration]

GLOVE_CACHE = os.path.join(_proj_root, "data", "ravana_glove_cache.npz")


def _load_glove(agent, glove_host):
    """Load the real GloVe cache into the agent's glove host so OOV detection
    is meaningful (mirrors a deployed run: genuine novel words -> None)."""
    if not os.path.exists(GLOVE_CACHE):
        return
    d = np.load(GLOVE_CACHE, allow_pickle=True)
    glove_host._glove_vecs = {str(w).lower(): v for w, v in
                              zip(d["words"].tolist(), d["vecs"])}
    glove_host._glove_proj = d["proj"].astype(np.float32)
    glove_host._glove_dim = int(d["proj"].shape[1])


# ── Agent builders ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ci_agent():
    from ravana.chat.interface import ChatInterface, ChatConfig
    # dim=64 to match the GloVe projection so loading the real cache is consistent
    config = ChatConfig(dim=64, seed=42, baby_mode=True, trace_enabled=False)
    ci = ChatInterface(config)
    ci._curiosity_drive_enabled = False  # we drive E explicitly, not the bg loop
    _load_glove(ci, ci.graph_engine)
    return ci


@pytest.fixture(scope="module")
def engine_agent():
    from ravana.chat.engine import CognitiveChatEngine
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    eng._curiosity_drive_enabled = False  # we drive E explicitly, not the bg loop
    _load_glove(eng, eng)  # engine itself is the glove host
    return eng


# ── Per-agent accessors (the two agents differ in attribute shape) ─────────

def _accessors(agent):
    """Return the loop's touch-points, abstracting over ChatInterface vs engine."""
    if hasattr(agent, "web_learner"):  # ChatInterface
        ge = agent.graph_engine
        edge_graph = agent.graph_engine.graph  # ConceptGraph (has get_edge)
        label_index = lambda: agent.graph_engine._all_labels
        web_learn = lambda text, src: agent.web_learner._web_to_graph.learn_text(text, source_url=src)
        curiosity = lambda cands: agent.web_learner.curiosity_e_step(candidate_topics=cands)
        gap = lambda t: agent.web_learner.knowledge_gap(t)
        last_fact = lambda: agent.web_learner.last_fact_count()
    else:  # CognitiveChatEngine (engine path)
        ge = agent.graph
        edge_graph = agent.graph  # ConceptGraph (has get_edge)
        # label -> first nid (mirrors the shim / GraphEngine._all_labels)
        label_index = lambda: {lab: nids[0] for lab, nids in agent._concept_keywords.items() if nids}
        web_learn = lambda text, src: agent._get_web_to_graph().learn_text(text, source_url=src)
        curiosity = lambda cands: agent.curiosity_e_step(candidate_topics=cands)
        gap = lambda t: agent.knowledge_gap(t)
        last_fact = lambda: agent._get_web_to_graph().fact_count()
    return ge, edge_graph, label_index, web_learn, curiosity, gap, last_fact


def _typed_edge_count_ge(edge_graph):
    """Count typed C-lite/web edges in the ConceptGraph by iterating .edges
    directly (robust for both ChatInterface.graph and engine.graph, avoids the
    label-index round-trip that the engine shim doesn't persist)."""
    n = 0
    for (src, tgt), e in getattr(edge_graph, "edges", {}).items():
        if e is not None and e.relation_type in (
                "is_a", "has_property", "causes", "located_in", "part_of"):
            n += 1
    return n


# ── The shared loop, run identically on both agents ───────────────────────

def _run_unified_loop(agent, name):
    ge, edge_graph, label_index, web_learn, curiosity, gap, last_fact = _accessors(agent)

    # 1) N2 spawn on a real novel turn
    before = 0
    resp = agent.process_turn("fnord blarg whipple quadrature")
    assert isinstance(resp, str) and len(resp) > 0, f"[{name}] conversation still works"
    n2 = agent.pfc_workspace._n2
    assert n2 is not None, f"[{name}] N2 learner must be live in the turn loop"
    assert len(n2.candidate_ids()) > before, f"[{name}] novel utterance must spawn (N2)"

    # known intent must NOT spawn
    cands_before = len(n2.candidate_ids())
    resp = agent.process_turn("what is trust?")
    assert isinstance(resp, str) and len(resp) > 0
    assert len(n2.candidate_ids()) == cands_before, f"[{name}] known intent must not spawn"

    # 2) E selects the sparse gap, C-lite writes, N3 binds, EFE closes
    # Make "water" KNOWN so its EFE gap is low.
    web_learn("Water is a chemical compound. Water is essential for life.", "web:water")
    novel = "quantum entanglement"
    target = curiosity(["water", novel, "relativity"])
    assert target == novel, f"[{name}] E must pick the sparse/novel topic (highest EFE), got {target}"
    efe_before = gap(target).efe

    edges_before = _typed_edge_count_ge(edge_graph)
    facts = web_learn(
        f"{target} is a physics concept. {target} relates to gravity. "
        f"{target} was studied by Einstein.", f"web:{target}")
    assert facts > 0, f"[{name}] C-lite must write facts into the graph"
    edges_after = _typed_edge_count_ge(edge_graph)
    assert edges_after > edges_before, f"[{name}] C-lite must add typed edges to the graph"

    # N3: encode the new fact in the 2048-D HRR space (dual-code binding)
    from ravana.core.dual_code_space import DualCodeSpace
    dcs = DualCodeSpace(GLOVE_CACHE)
    struct = dcs.encode_fact(target, "relates_to", "gravity")
    assert struct.shape == (dcs.hrr_dim,)
    recovered = dcs.recover_role_filler(struct, "object",
                                         [target, "gravity", "einstein", "water"])
    assert recovered == "gravity", f"[{name}] N3 binding must recover the fact's object"

    efe_after = gap(target).efe
    assert efe_after < efe_before, f"[{name}] E loop must close: EFE drops after learning"

    # 3) sleep consolidates rehearsed N2
    for cid in list(n2.candidate_ids()):
        n2.reinforce(cid)
        n2.reinforce(cid)
    res = agent.pfc_workspace.sleep()
    assert res["promoted"] >= 1, f"[{name}] rehearsed candidates must consolidate"
    assert len(n2.stable_ids()) >= 1

    # 4) still converses, and C-lite facts landed
    resp = agent.process_turn("tell me something about what you learned")
    assert isinstance(resp, str) and len(resp) > 0
    assert last_fact() > 0, f"[{name}] C-lite facts were written"


class TestUnifiedSemanticLoopChatInterface:
    def test_full_loop(self, ci_agent):
        _run_unified_loop(ci_agent, "ChatInterface")


class TestUnifiedSemanticLoopEnginePath:
    """The runnable agent: scripts/ravana_chat.py -> CognitiveChatEngine.
    This is the path that was previously dark for C-lite/E."""
    def test_full_loop(self, engine_agent):
        _run_unified_loop(engine_agent, "CognitiveChatEngine")


def test_wiring_finding_engine_path_now_has_c_lite(engine_agent):
    """The seam is closed: the engine path now exposes C-lite/E helpers.
    (Previously this was the documented gap; now it must hold.)"""
    from ravana.chat.engine import CognitiveChatEngine
    assert hasattr(CognitiveChatEngine, "curiosity_e_step"), \
        "engine path must expose curiosity_e_step (E)"
    assert hasattr(CognitiveChatEngine, "knowledge_gap"), \
        "engine path must expose knowledge_gap (C-lite/E bridge)"
    assert engine_agent._get_web_to_graph() is not None, \
        "engine path must build a WebToGraph over its own graph"
    # and the existing mixin behaviour is intact
    assert hasattr(engine_agent, "_auto_select_curiosity_topics")
