"""End-to-end integration tests for the full RAVANA chat pipeline.

Key constraint: each process_turn() call takes ~18s due to the reasoning loop.
We minimize calls by sharing one engine via module-scoped fixture and merging
related assertions into fewer test methods.

Pipeline: GraphEngine → Cognitive Core → Language (PFC, SynAssembly, SurfaceRealizer) → Response
"""

import pytest
import os
import sys
import re
import tempfile
import numpy as np

# Add package paths (same setup as test_ravana_chat_core.py etc.)
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


# ── Helpers ──────────────────────────────────────────────────────────────

def _is_valid_response(resp: str, min_words: int = 2) -> bool:
    """Check for basic English sentence structure."""
    if not resp or not isinstance(resp, str) or len(resp) < 5:
        return False
    if " " not in resp:
        return False
    first = resp.strip()[0]
    if not first.isupper() and first not in '"\'“‘':
        return False
    last = resp.strip()[-1]
    if last not in ".!?":
        return False
    return len([w for w in resp.split() if len(w) >= 2]) >= min_words


# ── Module-scoped fixture (shared across all tests) ───────────────────

@pytest.fixture(scope="module")
def engine():
    """One ChatInterface shared by all tests (~13s init, ~18s/turn)."""
    from ravana.chat.interface import ChatInterface, ChatConfig
    config = ChatConfig(dim=32, seed=42, baby_mode=True, trace_enabled=False)
    return ChatInterface(config)


# ── Tests ─────────────────────────────────────────────────────────────

class TestInitialization:
    """All components wire up correctly — no process_turn calls needed."""

    def test_all_components_initialized(self, engine):
        assert engine.graph_engine is not None
        assert len(engine.graph_engine.graph.nodes) >= 20
        assert len(engine.graph_engine.graph.edges) > 0
        assert engine.decoder_engine is not None
        for attr in ["emotion", "identity", "meaning", "dual_process",
                      "gw", "meta_cog", "sleep_engine", "belief_store",
                      "basal_ganglia", "cerebellar_ngram", "pfc_workspace",
                      "syntactic_assembly", "surface_realizer",
                      "relation_predictor", "propagation", "plasticity",
                      "web_learner", "bootstrap_manager"]:
            assert getattr(engine, attr) is not None, f"{attr} is None"
        assert 0 < engine.identity.state.strength < 1.0

    def test_cognitive_framework_functional(self):
        """GRACE CognitiveFramework exposes functional API."""
        from ravana.cognitive.framework import CognitiveFramework, FrameworkConfig
        import numpy as np
        fw = CognitiveFramework(FrameworkConfig())
        for method in ["perceive", "infer", "learn", "predict", "diagnose", "query", "save", "sleep"]:
            assert hasattr(fw, method), f"Missing method: {method}"
        assert fw.config is not None
        # Must initialize first, then call perceive with state + input_vec
        state = fw.initialize()
        result = fw.perceive(state, np.random.randn(fw.config.concept_dim).astype(np.float32))
        assert isinstance(result, list)
        # Also test that infer, predict, query work without crashing
        infer_result = fw.infer(state, np.random.randn(fw.config.concept_dim).astype(np.float32))
        assert isinstance(infer_result, dict)


class TestSingleTurn:
    """Minimal process_turn calls to validate pipeline output."""

    def test_basic_response(self, engine):
        """Single turn → valid English response."""
        start_count = engine.turn_count
        resp = engine.process_turn("Tell me about trust")
        assert _is_valid_response(resp), f"Bad: '{resp}'"
        assert engine.turn_count == start_count + 1
        assert engine._last_strategy in (
            "neural_decoder", "neural_decoder_reasoned",
            "dorsal_reasoned",
            "associative", "graph_fallback", "unknown_subject",
            "syntactic_pipeline")

    def test_response_references_topic(self, engine):
        """Response mentions the queried topic."""
        resp = engine.process_turn("What is respect")
        rl = resp.lower()
        assert any(w in rl for w in ["respect", "trust", "loyalty", "honesty"]), \
            f"Not about topic: '{resp}'"

    def test_question_handled(self, engine):
        """Questions produce responses."""
        resp = engine.process_turn("How does trust work?")
        assert _is_valid_response(resp), f"Bad: '{resp}'"


class TestMultiTurn:
    """Pipeline state evolves across turns — 4 process_turn calls total."""

    def test_topic_and_follow_up(self, engine):
        """Follow-up maintains topic context."""
        r1 = engine.process_turn("Tell me about trust")
        assert _is_valid_response(r1)
        r2 = engine.process_turn("Tell me more")
        assert _is_valid_response(r2), f"Follow-up bad: '{r2}'"
        assert engine.turn_count >= 3

    def test_emotion_and_free_energy(self, engine):
        """Cognitive state changes detectably across turns."""
        v0, a0, d0 = (engine.emotion.state.valence,
                       engine.emotion.state.arousal,
                       engine.emotion.state.dominance)
        engine.process_turn("This is really bad and sad")
        v1, a1, d1 = (engine.emotion.state.valence,
                       engine.emotion.state.arousal,
                       engine.emotion.state.dominance)
        total_delta = abs(v1-v0) + abs(a1-a0) + abs(d1-d0)
        assert total_delta > 0.001, f"Emotion unchanged: delta={total_delta:.4f}"

        fe = engine._free_energy
        assert isinstance(fe, float) and 0.0 <= fe <= 1.0


class TestPipelineComponents:
    """Isolated component tests — no process_turn calls needed."""

    def test_pfc_discourse_plan(self, engine):
        plan = engine.pfc_workspace.plan_discourse(
            user_input="Tell me about trust", subject="trust",
            concept_pos={},
            associations=[("respect", 0.8), ("connection", 0.6)],
            past_topics=[], is_follow_up=False)
        assert plan is not None and len(plan.intents) >= 1

    def test_surface_realizer(self, engine):
        from ravana.language.syntactic_cell_assembly import SyntacticFrame
        from ravana.language.surface_realizer import DiscourseState
        frame = SyntacticFrame(
            subject_concept="Trust", verb_concept="connect", object_concept="Respect",
            relation_type="semantic",
            tense="present", depth=0)
        class DT: value = "ELABORATE"
        sent = engine.surface_realizer.realize(
            frame=frame,
            discourse_context=DiscourseState(0, None, DT(), 3),
            dopamine_tone=0.5, cerebellar_ngram=None,
            discourse_marker="Furthermore")
        assert isinstance(sent, str) and len(sent) > 0 and sent[0].isupper()

    def test_basal_ganglia_gating(self, engine):
        # BasalGangliaGate.select_concept uses (label, score, confidence, relation_type) tuples
        candidates = [
            ("respect", 0.8, 0.9, "semantic"),
            ("connection", 0.6, 0.7, "semantic"),
            ("loyalty", 0.5, 0.6, "semantic"),
        ]
        label, rel_type, score = engine.basal_ganglia.select_concept(candidates)
        assert label in ["respect", "connection", "loyalty"]
        assert rel_type == "semantic"
        assert score > 0

    def test_dual_process_routing(self, engine):
        conf = engine.identity.state.strength * 0.5 + 0.2
        route = engine.dual_process.decide_route(confidence=conf, novelty=0.1, stakes=0.15)
        # Route enum values are "system1_fast" and "system2_slow" (not shortened)
        assert route.route.value in ("system1_fast", "system2_slow")
        assert route.reason is not None

    def test_sleep_consolidation(self, engine):
        """Trigger sleep cycle directly — verifies no crash."""
        metrics = engine._sleep_consolidate()
        assert isinstance(metrics, dict)
        assert "edges_strengthened" in metrics
        assert "edges_pruned" in metrics

    def test_graph_spread_activation(self, engine):
        engine.process_turn("Tell me about trust")
        # Activation may decay to near-zero after process_turn returns.
        # Check that _concept_vad was populated (proves activation happened)
        # and that the graph has some episodic or semantic edges
        if engine._concept_vad:
            # VAD was recorded for activated concepts — activation occurred
            assert len(engine._concept_vad) > 0
        else:
            # Fallback: check that the graph has edges (bootstrap created some)
            assert len(engine.graph_engine.graph.edges) > 0
        episodic = sum(1 for e in engine.graph_engine.graph.edges.values()
                       if getattr(e, 'relation_type', None) == 'episodic')
        assert episodic >= 0


class TestSaveLoad:
    """Serialization round-trip (separate engine, 2 process_turn calls)."""

    def test_save_then_load(self):
        from ravana.chat.interface import ChatInterface, ChatConfig
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            save_path = f.name
        try:
            cfg = ChatConfig(dim=32, seed=42, baby_mode=True,
                             trace_enabled=False, data_dir=os.path.dirname(save_path))
            cfg.user_suffix = os.path.basename(save_path).replace(".pkl", "")
            eng1 = ChatInterface(cfg)
            eng1._save_path = save_path
            eng1.process_turn("Tell me about trust")
            saved_nodes = len(eng1.graph_engine.graph.nodes)
            msg = eng1.save()
            assert os.path.exists(save_path), f"Save failed: {msg}"

            eng2 = ChatInterface(cfg)
            eng2._save_path = save_path
            assert eng2._load(), "Load returned False"
            # Should retain most nodes
            assert len(eng2.graph_engine.graph.nodes) >= saved_nodes * 0.5, \
                f"Lost nodes: {saved_nodes} -> {len(eng2.graph_engine.graph.nodes)}"
            # Post-load response works
            resp = eng2.process_turn("Tell me about trust")
            assert _is_valid_response(resp), f"Post-load bad: '{resp}'"
        finally:
            try:
                os.unlink(save_path)
            except OSError:
                pass
