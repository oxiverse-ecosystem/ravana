"""Compare calibration policies on a real session: does the learned fact survive?

The question is not "which policy is more brain-faithful in theory" but "which
one leaves the fact recallable afterwards". This probe runs the same session
under each policy and measures retention of the facts the system was taught.

Usage:
    uv run python scripts/measure_retrieval_calibration.py
"""
import os
import sys
import json

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJ, os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

import numpy as np  # noqa: E402

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402
from ravana.learn.intake import KnowledgeIntake, Provenance  # noqa: E402
from ravana.web.openie import Fact  # noqa: E402

DIM = 64

# Facts the system will be taught, then quizzed on.
PROBES = [
    ("photosynthesis", "is_a", "process"),
    ("chlorophyll", "has_property", "green"),
    ("mitochondria", "is_a", "organelle"),
    ("gravity", "causes", "falling"),
    ("evaporation", "is_a", "process"),
]


def _mk_engine(labels):
    eng = CognitiveChatEngine(dim=DIM, seed=7, baby_mode=True)
    for lab in labels:
        vec = eng._glove_vector(lab)
        if vec is None:
            h = hash(lab) % 10000
            vr = np.random.RandomState(h + 42)
            vec = vr.randn(eng.graph.dim).astype(np.float32) * 0.15
            n = float(np.linalg.norm(vec))
            if n > 0:
                vec /= n
        node = eng.graph.add_node(vector=vec, label=lab)
        if node is not None:
            eng._concept_keywords[lab] = [node.id]
    return eng


def _teach(eng, min_success_rate, replay_corroboration):
    """Teach the probes through the real gate, then run retrieval practice."""
    labels = [w for t in PROBES for w in t if w not in ("is_a", "causes",
                                                       "has_property")]
    eng2 = _mk_engine(sorted(set(labels)))
    eng2._get_knowledge_intake()
    # Rebuild the intake with the policy under test.
    from ravana.chat.web_learning import _EngineGraphShim
    ki = KnowledgeIntake(ge=_EngineGraphShim(eng2),
                         buffer=eng2.hippocampal_buffer,
                         min_success_rate=min_success_rate)
    eng2._knowledge_intake = ki

    for s, r, o in PROBES:
        ki.intake(Fact(subject=s, relation=r, obj=o, confidence=0.9),
                  Provenance(source_type="web"))

    # Simulate 10 turns of retrieval practice. The success pattern is what
    # makes the policies differ: a fact that is mostly failing must not be
    # spaced out, and a fact that is reliably recalled must be.
    easy = [True, True, True, True, True, True, True, True, True, True]
    shaky = [True, False, False, True, False, True, False, False, True, False]
    keys = [tuple(p) for p in PROBES]
    for turn in range(10):
        ki.advance_turn()
        # Drive EVERY queued fact, not just the first `limit` -- otherwise the
        # later probes are never reviewed and the policies cannot differ.
        for key in keys:
            rec = ki._review_queue.get(key)
            if rec is None or rec.get("due", 0) > ki.turn:
                continue
            pattern = shaky if key == keys[-1] else easy
            recalled = pattern[min(turn, len(pattern) - 1)]
            ki.record_review(key, recalled=recalled)
    return eng2, ki


def _score(eng, ki):
    """How many taught facts are still recallable from the graph?"""
    hits = 0
    confs = []
    for s, r, o in PROBES:
        sids = eng._concept_keywords.get(s, [])
        oids = eng._concept_keywords.get(o, [])
        found = None
        for sid in sids:
            for tid, edge in eng.graph.get_outgoing(sid):
                if tid in oids:
                    found = edge
                    break
            if found is not None:
                break
        if found is not None:
            hits += 1
            confs.append(round(float(found.confidence), 4))
        else:
            confs.append(None)
    return hits, confs


def main():
    policies = [
        ("baseline (no gate, old effort)", 0.0, False),
        ("success gate only", 0.75, False),
        ("replay corroboration only", 0.0, True),
        ("both", 0.75, True),
    ]
    results = {}
    for name, msr, rca in policies:
        eng, ki = _teach(None, msr, rca)
        hits, confs = _score(eng, ki)
        results[name] = {
            "recalled": hits,
            "of": len(PROBES),
            "confidences": confs,
            "review_writes": ki.stats.get("review_writes", 0),
            "review_failed_no_write": ki.stats.get("review_failed_no_write", 0),
        }
        print(f"{name:34s} recalled {hits}/{len(PROBES)}  "
              f"writes={ki.stats.get('review_writes', 0)}  "
              f"failed_no_write={ki.stats.get('review_failed_no_write', 0)}")
        print(f"{'':34s} confidences={confs}")

    out = os.path.join(PROJ, "reports", "retrieval_calibration.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
