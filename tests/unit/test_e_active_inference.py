"""
Tests for E — active-inference control spine (curiosity as epistemic action).
"""

import os
import sys

import numpy as np
import pytest

_PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))

from ravana.core.active_inference import ActiveInferenceController, EFEState
from ravana.graph.engine import GraphEngine
from ravana.web.web_to_graph import WebToGraph


def _seed_ge() -> GraphEngine:
    ge = GraphEngine(dim=64, glove_vecs=None)
    for w in ["water", "fire", "smoke", "earth", "quantum", "entanglement",
              "relativity", "gravity", "einstein"]:
        vec = np.random.RandomState(hash(w) % 1000).randn(64).astype("float32")
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        node = ge.graph.add_node(vector=vec, label=w)
        ge._all_labels[w] = node.id
    for w in ("water", "fire"):
        ge.graph.get_node(ge._all_labels[w]).prediction_free_energy = 0.2
    return ge


class TestActiveInferenceController:
    def test_score_combines_gap_and_pfe(self):
        ge = _seed_ge()
        w2g = WebToGraph(ge, source="t")
        w2g.learn_text("Water is a chemical compound. Fire causes smoke.")
        ctrl = ActiveInferenceController(
            gap_fn=w2g.knowledge_gap,
            pfe_fn=lambda t: float(getattr(ge.graph.get_node(ge._all_labels[t]),
                                           "prediction_free_energy", 0.0) or 0.0)
            if t in ge._all_labels else 0.0)
        s = ctrl.score("quantum entanglement")
        assert s.total > 0.0
        assert s.gap_efe >= 0.0

    def test_select_target_picks_max_efe(self):
        ge = _seed_ge()
        w2g = WebToGraph(ge, source="t")
        w2g.learn_text("Water is a chemical compound. Fire causes smoke.")
        ctrl = ActiveInferenceController(
            gap_fn=w2g.knowledge_gap,
            pfe_fn=lambda t: float(getattr(ge.graph.get_node(ge._all_labels[t]),
                                           "prediction_free_energy", 0.0) or 0.0)
            if t in ge._all_labels else 0.0)
        target = ctrl.select_target(["water", "earth", "quantum entanglement", "relativity"])
        assert target in ("quantum entanglement", "relativity")

    def test_loop_closes_after_read(self):
        ge = _seed_ge()
        w2g = WebToGraph(ge, source="t")
        w2g.learn_text("Water is a chemical compound. Fire causes smoke.")
        ctrl = ActiveInferenceController(
            gap_fn=w2g.knowledge_gap,
            pfe_fn=lambda t: float(getattr(ge.graph.get_node(ge._all_labels[t]),
                                           "prediction_free_energy", 0.0) or 0.0)
            if t in ge._all_labels else 0.0)
        target = ctrl.select_target(["water", "quantum entanglement", "relativity"])
        efe_before = ctrl.score(target).total
        w2g.learn_text(f"{target} is a physics concept. {target} relates to gravity.")
        efe_after = ctrl.score(target).total
        _, _, closed = ctrl.act_and_close_loop(target, efe_before,
                                                lambda t: ctrl.score(t).total)
        assert closed, "EFE must drop after epistemic action (loop closes)"
        assert ctrl.loop_closed_rate() == 1.0


class TestWebLearnerEIntegration:
    def test_curiosity_e_step_runs_additively(self):
        # Importability + the hook exists without spinning up the heavy learner
        import ravana.web.learner as wl
        assert hasattr(wl.WebLearner, "curiosity_e_step")
        assert hasattr(wl.WebLearner, "knowledge_gap")
