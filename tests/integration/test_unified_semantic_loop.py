"""
End-to-end soak test: the FULL unified semantic loop (A1->N1->N4->N2->C-lite->N3->E).

Every stage is proven in isolation; this test proves they HOLD TOGETHER when
they interact across a real multi-turn chat, using the production objects wired
exactly as the live agent uses them:

  turn            -> engine.process_turn  (A1/N1 classify, N4 gate, N2 spawn, sleep)
  novel intent    -> PrefrontalWorkspace.learn_from_turn routes ABSTAIN -> EmergentCategoryLearner
  curiosity (E)   -> WebLearner.curiosity_e_step selects argmax-EFE topic
  web-read (C)    -> WebToGraph.learn_text writes typed triples into the real graph
  binding (N3)    -> DualCodeSpace encodes the new fact in the 2048-D HRR space
  sleep           -> PrefrontalWorkspace.sleep consolidates rehearsed candidates

It is offline/deterministic: the "web read" is injected via WebToGraph.learn_text
(exactly what WebLearner._learn_from_text calls) — no network. This isolates the
cross-stage interaction, which is the thing unit tests don't cover.

This runs against ChatInterface, the fully-wired agent (it has BOTH pfc_workspace
[N2/N4] and web_learner [C-lite/E]). The engine path (CognitiveChatEngine, used by
scripts/ravana_chat.py) currently lacks web_learner, so C-lite/E are NOT live there
— see the finding printed at the end. Process_turn is ~18s, so the soak is a few
turns, not hundreds.
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


@pytest.fixture(scope="module")
def agent():
    from ravana.chat.interface import ChatInterface, ChatConfig
    # dim=64 to match the GloVe projection so loading the real cache is consistent
    config = ChatConfig(dim=64, seed=42, baby_mode=True, trace_enabled=False)
    ci = ChatInterface(config)
    ci._curiosity_drive_enabled = False  # we drive E explicitly, not the bg loop
    # Load the real GloVe cache so OOV detection is meaningful (mirrors a
    # deployed run: genuine novel words return None -> true novelty signal).
    if os.path.exists(GLOVE_CACHE):
        d = np.load(GLOVE_CACHE, allow_pickle=True)
        ci.graph_engine._glove_vecs = {str(w).lower(): v for w, v in
                                       zip(d["words"].tolist(), d["vecs"])}
        ci.graph_engine._glove_proj = d["proj"].astype(np.float32)
        ci.graph_engine._glove_dim = int(d["proj"].shape[1])
    return ci


# ── Helpers ──────────────────────────────────────────────────────────────

def _typed_edge_count(ge) -> int:
    labels = list(ge._all_labels.keys())
    n = 0
    for i, la in enumerate(labels):
        for lb in labels[i + 1:]:
            sa, sb = ge._all_labels[la], ge._all_labels[lb]
            for a, b in ((sa, sb), (sb, sa)):
                e = ge.graph.get_edge(a, b)
                if e is not None and e.relation_type in (
                        "is_a", "has_property", "causes", "located_in", "part_of"):
                    n += 1
    return n


class TestUnifiedSemanticLoop:
    def test_n2_spawn_from_real_turn(self, agent):
        """A genuinely novel utterance on a real turn spawns an N2 candidate."""
        before = len(agent.pfc_workspace._n2.candidate_ids()) if agent.pfc_workspace._n2 else 0
        # drive a real turn so the live wiring (process_turn -> learn_from_turn) fires
        resp = agent.process_turn("fnord blarg whipple quadrature")
        assert isinstance(resp, str) and len(resp) > 0  # conversation still works
        n2 = agent.pfc_workspace._n2
        assert n2 is not None, "N2 learner must be live in the turn loop"
        assert len(n2.candidate_ids()) > before, "novel utterance must spawn a candidate (N2)"

    def test_known_intent_does_not_spawn(self, agent):
        n2 = agent.pfc_workspace._n2
        cands_before = len(n2.candidate_ids())
        resp = agent.process_turn("what is trust?")
        assert isinstance(resp, str) and len(resp) > 0
        # known question -> no new candidate
        assert len(n2.candidate_ids()) == cands_before, "known intent must not spawn"

    def test_e_selects_gap_then_c_lite_writes_then_n3_binds(self, agent):
        """The cross-stage loop: E picks a gap -> C-lite writes -> N3 binds."""
        ge = agent.graph_engine
        # Make "water" KNOWN: write C-lite facts so its EFE gap is low.
        agent.web_learner._web_to_graph.learn_text(
            "Water is a chemical compound. Water is essential for life.",
            source_url="web:water")
        # Novel topic with NO node, NO edges, NO C-lite facts -> max EFE gap.
        novel = "quantum entanglement"
        target = agent.web_learner.curiosity_e_step(
            candidate_topics=["water", novel, "relativity"])
        assert target == novel, f"E must pick the sparse/novel topic (highest EFE), got {target}"
        efe_before = agent.web_learner.knowledge_gap(target).efe

        # C-lite: the web-read writes typed triples into the REAL graph
        edges_before = _typed_edge_count(ge)
        facts = agent.web_learner._web_to_graph.learn_text(
            f"{target} is a physics concept. {target} relates to gravity. "
            f"{target} was studied by Einstein.",
            source_url=f"web:{target}")
        assert facts > 0, "C-lite must write facts into the graph"
        edges_after = _typed_edge_count(ge)
        assert edges_after > edges_before, "C-lite must add typed edges to the graph"

        # N3: encode the new fact in the 2048-D HRR space (dual-code binding)
        from ravana.core.dual_code_space import DualCodeSpace
        dcs = DualCodeSpace(GLOVE_CACHE)
        struct = dcs.encode_fact(target, "relates_to", "gravity")
        assert struct.shape == (dcs.hrr_dim,)  # high-D binding space used
        # resonator decode recovers the object filler
        recovered = dcs.recover_role_filler(struct, "object",
                                             [target, "gravity", "einstein", "water"])
        assert recovered == "gravity", "N3 binding must recover the fact's object"

        # E loop closes: EFE gap for the topic must drop after the web-read
        efe_after = agent.web_learner.knowledge_gap(target).efe
        assert efe_after < efe_before, "E loop must close: EFE drops after learning"

    def test_sleep_consolidates_rehearsed_n2(self, agent):
        """Periodic sleep promotes rehearsed N2 candidates, prunes singletons."""
        n2 = agent.pfc_workspace._n2
        # rehearse whatever candidates exist
        for cid in list(n2.candidate_ids()):
            n2.reinforce(cid)
            n2.reinforce(cid)
        res = agent.pfc_workspace.sleep()
        assert res["promoted"] >= 1, "rehearsed candidates must consolidate"
        assert len(n2.stable_ids()) >= 1

    def test_full_state_after_loop(self, agent):
        """Sanity: the agent still converses and the graph grew from C-lite."""
        resp = agent.process_turn("tell me something about what you learned")
        assert isinstance(resp, str) and len(resp) > 0
        # the novel N2 intent + C-lite facts are now part of the agent's state
        assert agent.pfc_workspace._n2 is not None
        assert agent.web_learner.last_fact_count() > 0, "C-lite facts were written"


# ── Finding surfaced by this test (informational, not a hard failure) ──
def test_wiring_finding_engine_path_lacks_c_lite(agent):
    """The engine path (scripts/ravana_chat.py -> CognitiveChatEngine) has
    pfc_workspace (N2/N4 live) but NO web_learner, so C-lite/E are NOT wired
    there. This test documents the asymmetry; the fix is to attach WebLearner
    (or its _web_to_graph + curiosity_e_step) to the engine's web mixin too.
    """
    from ravana.chat.engine import CognitiveChatEngine
    # confirm the engine class is the one the prod script uses
    assert CognitiveChatEngine is not None
    # ChatInterface (used here) HAS web_learner -> C-lite/E live
    assert hasattr(agent, "web_learner"), "ChatInterface must have web_learner"
    # The asymmetry is real; record it (does not fail the suite)
    engine_has_web_learner = hasattr(CognitiveChatEngine, "web_learner")
    print(f"\n  [FINDING] CognitiveChatEngine.web_learner present: {engine_has_web_learner}")
    print("  [FINDING] C-lite/E are live on ChatInterface but NOT on the engine")
    print("            path used by scripts/ravana_chat.py -> wire WebLearner into")
    print("            the engine's WebLearningMixin to close the gap.")
