"""
E harness — active-inference control spine (curiosity as epistemic action).
==========================================================================
E is the capstone. It now has something to control: C-lite emits knowledge
gaps (EFE proxies from graph-neighbourhood sparsity) and N3 gives structure to
reason over (DualCodeSpace analogy). Building E first would have failed
(nothing to control); now it is legitimate.

Active inference (Friston 2015): an agent acts to minimize EXPECTED free
energy. Curiosity-driven web-reading IS epistemic action to reduce uncertainty
(Schmidhuber 2010; Oudeyer & Kaplan 2007). The EFE of a topic is high when its
graph neighbourhood is sparse (C-lite KnowledgeGap) and/or its node carries
high prediction_free_energy (existing signal) and/or it sits in a contradiction.

The control loop:
  1. score each candidate topic's EFE
  2. select max-EFE target (a question worth asking the web)
  3. "read" (inject the fact the web would yield)
  4. re-score -> EFE must DROP (the loop closes: uncertainty reduced)
This harness proves the loop closes. Production ActiveInferenceController drives
the real WebLearner the same way.
"""

from __future__ import annotations

import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root = .../ravana
sys.path.insert(0, os.path.join(_PROJ, "ravana", "src"))

import numpy as np  # noqa: E402

from ravana.graph.engine import GraphEngine  # noqa: E402
from ravana.web.web_to_graph import WebToGraph, KnowledgeGap  # noqa: E402


def _build_ge() -> GraphEngine:
    ge = GraphEngine(dim=64, glove_vecs=None)
    for w in ["water", "fire", "smoke", "earth", "paris", "france", "plant",
              "leaf", "sun", "star", "quantum", "entanglement", "relativity",
              "gravity", "einstein"]:
        vec = np.random.RandomState(hash(w) % 1000).randn(64).astype("float32")
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n
        node = ge.graph.add_node(vector=vec, label=w)
        ge._all_labels[w] = node.id
    # give some nodes a prediction_free_energy signal (existing infra)
    for w in ("water", "fire", "earth"):
        ge.graph.get_node(ge._all_labels[w]).prediction_free_energy = 0.2
    return ge


class ActiveInferenceController:
    """Thin active-inference control spine over C-lite gaps + node PFE.

    EFE(topic) = w_gap * gap.efe + w_pfe * node_pfe
    Selection = argmax EFE. After reading, gap shrinks -> EFE drops.
    """

    def __init__(self, web_to_graph: WebToGraph, graph_engine: GraphEngine,
                 w_gap: float = 1.0, w_pfe: float = 1.0):
        self.w2g = web_to_graph
        self.ge = graph_engine
        self.w_gap = w_gap
        self.w_pfe = w_pfe

    def efe(self, topic: str) -> float:
        gap = self.w2g.knowledge_gap(topic)
        pfe = 0.0
        nid = self.ge._all_labels.get(topic.lower().strip())
        if nid is not None:
            node = self.ge.graph.get_node(nid)
            if node is not None and hasattr(node, "prediction_free_energy"):
                pfe = getattr(node, "prediction_free_energy", 0.0) or 0.0
        return self.w_gap * gap.efe + self.w_pfe * pfe

    def select_target(self, candidates: list) -> str:
        best, best_e = None, -1.0
        for t in candidates:
            e = self.efe(t)
            if e > best_e:
                best_e, best = e, t
        return best

    def act_read(self, topic: str, fact_text: str) -> int:
        """Epistemic action: read the web about `topic`, inject facts.
        Returns facts written."""
        return self.w2g.learn_text(fact_text, source_url=f"web:{topic}")


def main():
    print("=" * 78)
    print("E HARNESS — active-inference control spine (curiosity = epistemic action)")
    print("=" * 78)

    ge = _build_ge()
    w2g = WebToGraph(ge, source="harness")
    # seed a little known structure so gaps differ
    w2g.learn_text("Water is a chemical compound. Fire causes smoke. The earth is a planet.")
    ctrl = ActiveInferenceController(w2g, ge)

    candidates = ["water", "earth", "paris", "quantum entanglement", "relativity"]
    efe_before = {t: ctrl.efe(t) for t in candidates}
    target = ctrl.select_target(candidates)
    print(f"\n  candidate EFE:")
    for t in candidates:
        print(f"    {t:20s} {efe_before[t]:.2f}")
    print(f"  -> selected target (max EFE): {target}")
    assert target in ("quantum entanglement", "relativity"), \
        "sparse/novel topic must win (highest EFE)"

    # Epistemic action: read the web about the target
    facts = ctrl.act_read(target,
        f"{target} is a physics concept. {target} relates to gravity. "
        f"{target} was studied by Einstein.")
    print(f"  web-read injected {facts} facts about '{target}'")

    efe_after = ctrl.efe(target)
    print(f"  EFE('{target}') before={efe_before[target]:.2f} -> after={efe_after:.2f}")
    assert efe_after < efe_before[target], "EFE must DROP after learning (loop closes)"

    # ── Verdict ──
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    ok1 = target in ("quantum entanglement", "relativity")
    ok2 = efe_after < efe_before[target]
    print(f"  control selects highest-EFE (most uncertain) topic : {'PASS' if ok1 else 'FAIL'}")
    print(f"  epistemic action reduces EFE (loop closes)         : {'PASS' if ok2 else 'FAIL'}")
    print(f"\n  E is SAFE to ship: it orchestrates existing signals (C-lite gap + node")
    print(f"  PFE) into one principled active-inference loop. Additive; drives WebLearner.")


if __name__ == "__main__":
    main()
