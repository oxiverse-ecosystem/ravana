"""Phase 1 regression tests: edge provenance + grounding-gate A/B flag.

1. Provenance bug fix (web_to_graph.py): ConceptEdge has `source_metadata`,
   NOT `metadata`. The old code's `hasattr(edge, "metadata")` guard was always
   False, so every web-fact edge arrived UNTAGGED — which blocked source-based
   pruning (couldn't tell a verified fact from noisy co-occurrence). The fix
   writes `edge.source_metadata` with source/relation/edge_kind="web_fact".
2. Co-occurrence edges (web/learner.py _learn_from_text) are now tagged
   edge_kind="co_occurrence" so pruning can distinguish them.
3. _disable_grounding_gate flag: when True, the SM dispatch bypasses the gate
   (arc_off arm of the pre/post benchmark); when False/absent, the gate still
   runs and withholds ungrounded fluent text.

Run from repo root:
    python -m pytest tests/unit/test_phase1_provenance.py -v
"""
import os
import sys
from unittest import mock

import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))
sys.path.insert(0, os.path.join(_PROJ, "ravana_ml", "src"))

from ravana.graph.engine import GraphEngine
from ravana.web.web_to_graph import WebToGraph
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext


def _openie_fact(subj, obj, rel="is_a"):
    class F:
        def __init__(s, a, b, r):
            s.subject, s.obj, s.relation, s.confidence = a, b, r, 0.9
    return F(subj, obj, rel)


def test_web_to_graph_tags_source_metadata():
    ge = GraphEngine(dim=64, seed=1)
    wt = WebToGraph(ge, source="https://example.com/fact")
    wt.openie = type("O", (), {"extract": lambda self, text: [
        _openie_fact("whale", "mammal", "is_a")]})()
    n = wt.learn_text("whales are mammals", source_url="https://example.com/fact")
    assert n >= 1, "expected at least one fact written"
    tagged = False
    for nid, node in ge.graph.nodes.items():
        if node.label and node.label.lower() == "whale":
            for tid, edge in ge.graph.get_outgoing(nid):
                if (hasattr(edge, "source_metadata")
                        and edge.source_metadata.get("edge_kind") == "web_fact"):
                    assert edge.source_metadata.get("source") == "https://example.com/fact"
                    assert edge.source_metadata.get("relation") == "is_a"
                    # Defaults preserved (merge, not overwrite).
                    assert edge.source_metadata.get("source_agent") == "system"
                    tagged = True
    assert tagged, "web-fact edge was NOT tagged with source_metadata"


def test_disable_grounding_gate_bypasses_dispatch():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              data_dir="/tmp/ravana_phase1_gate")
    # "trust" is a seeded teen concept (has a graph node) so cond_embs is
    # populated and the decoder block is reached. The garbage is what the
    # (broken) decoder would emit for an unknown aspect of a known subject.
    garbage = ("trust is the light and the space where the matter and the time "
               "bend in the quiet of the evening")
    ctx = CognitiveResponseContext(
        subject="trust", raw_input="what is trust?",
        associated_concepts=[("space", 0.5), ("gravity", 0.5), ("time", 0.5)])
    # The decoder path requires a non-zero situation_vector to populate
    # conditioning embeddings (otherwise it returns None before the decoder).
    ctx.situation_vector = np.random.RandomState(0).randn(eng.dim).astype(np.float32)

    # The decoder returns token *indices*; map them to our garbage words so the
    # decoder block treats the output as real generated text.
    words = garbage.split()
    idxs = list(range(100, 100 + len(words)))
    eng._decoder_idx_to_word = {i: w for i, w in zip(idxs, words)}

    # Isolate the flag at the REAL call site (response_gen.py:1633). Force the
    # grounding gate to block (returns False) and the neural decoder to emit
    # garbage; neutralize the salad pre-check so we test ONLY the flag.
    _GEN_STRATEGIES = ("situation_model_decoder", "situation_model_narrative")
    with mock.patch.object(eng, "_sm_response_grounded", return_value=False), \
         mock.patch("ravana.chat.response_gen._is_word_salad", return_value=False), \
         mock.patch("ravana.chat.response_gen._is_word_salad_any_sentence", return_value=False), \
         mock.patch.object(eng.neural_decoder, "generate", return_value=idxs):
        # Gate ON (default): garbage withheld -> no *generated* strategy returned
        # (the grounding gate blocks ungrounded fluent text).
        out = eng._generate_with_situation_model(ctx)
        assert out is None or out[1] not in _GEN_STRATEGIES, out
        # Gate OFF (arc_off arm): bypassed -> a situation-model *generated*
        # strategy is emitted (decoder if its output is fluent enough, else the
        # narrative paragraph path — both are gate-bypassed generation, which is
        # what this flag controls). The grounding gate no longer withholds it.
        eng._disable_grounding_gate = True
        out2 = eng._generate_with_situation_model(ctx)
        assert out2 is not None and out2[1] in _GEN_STRATEGIES, out2
        # The flag's job is to stop the grounding gate from *withholding*
        # ungrounded fluent text. When bypassed, a generated strategy is
        # emitted. That is either the decoder's literal garbage (decoder path)
        # OR the narrative paragraph (narrative path) — both prove the gate no
        # longer blocks generation. Accept either: substantial generated
        # content reached the surface.
        _generated = out2[0]
        assert (garbage.lower() in _generated.lower()) or len(_generated) > 15, out2


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
