#!/usr/bin/env python3
"""Self-contained RSI role-binding EXPERIMENT — no production edits, no engine boot.

Loads the CURRENT, UNMODIFIED main code from this worktree (schema_induction,
response_gen, ontology GloVe loader), replays the recorded focused-live RSI
rows and the recorded GloVe cache, and compares:

  * vector-only baseline  = production `schema_association_scores_from_graph`
                            + production `rank_candidates_for_unseen_subject`;
  * role-aligned          = scratch-only complementary-role-centre scorer,
                            fed into the SAME production ranker through an
                            in-process monkeypatch that adds one optional
                            `role_scores` argument.

No file under ravana/src or scripts/ is edited. No CognitiveChatEngine is
constructed and no process_turn is called. Gold answers are used ONLY after
scoring, to evaluate picks; they never enter a trace, centre or score.

Usage (from the worktree root, with the MAIN repo's venv python):
  PYTHONPATH=<worktree>/ravana/src;<worktree>/ravana_ml/src \
    <main>/.venv/Scripts/python.exe scratch/role_binding_experiment.py
"""
from __future__ import annotations

import ast
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
MAIN = WORKTREE.parent / "ravana"
for _p in (str(WORKTREE / "ravana" / "src"),
           str(WORKTREE / "ravana_ml" / "src"), str(WORKTREE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.chat import response_gen as rg  # noqa: E402
from ravana.chat import schema_induction as si  # noqa: E402
from ravana.ontology.attribute_encoder import build_glove64_lookup  # noqa: E402

ROWS_PATH = MAIN / "scratch" / "rsi_focus_live_rows.json"
CENSUS_PATH = MAIN / "scratch" / "rsi_teach_form_probe2.log"
GLOVE_PATH = MAIN / "data" / "ravana_glove_cache.npz"
OUT_PATH = HERE / "role_binding_results.json"

REL_TYPES = {"CAUSE": "causal", "PART_OF": "part_of",
             "PRECEDES": "temporal", "RESEMBLES": "analogical"}

# Surface phrase -> role orientation. Copied verbatim from the CURRENT main
# harness scripts/measure_learn5_rsi.py:101-126 (inverse phrases swap roles).
PHRASES = {
    "CAUSE": ({"causes", "leads to", "produces", "brings about", "gives rise to"},
              {"is caused by", "results from", "is produced by", "arises from"}),
    "PART_OF": ({"is part of", "belongs to", "is a component of"},
                {"contains", "includes", "is composed of"}),
    "PRECEDES": ({"comes before", "precedes", "happens before"},
                 {"comes after", "follows", "happens after"}),
    "RESEMBLES": ({"resembles", "is similar to", "is akin to"}, set()),
}
FAMILIES = ("CAUSE", "PART_OF", "PRECEDES", "RESEMBLES")
CONDITIONS = ("near", "far", "foil", "negative")


# ── fakes (no ConceptGraph, no engine) ──────────────────────────────────────

@dataclass
class FakeNode:
    label: str
    vector: Any


class FakeGraph:
    def __init__(self, nodes: Dict[int, FakeNode]):
        self.nodes = nodes

    def get_node(self, nid):
        return self.nodes.get(nid)


def make_graph(labels: Sequence[str], lut: Dict[str, np.ndarray]):
    """One node per label with the engine's real projected GloVe-64 vector."""
    nodes: Dict[int, FakeNode] = {}
    missing: List[str] = []
    for nid, label in enumerate(sorted(set(labels)), start=1):
        v = lut.get(label.strip().lower())
        if v is None:
            missing.append(label)
        nodes[nid] = FakeNode(label, v)
    kw: Dict[str, List[int]] = {}
    for nid, node in nodes.items():
        kw.setdefault(node.label.lower(), []).append(nid)
    return nodes, kw, missing


def canonical_pair(triple: Sequence[str], family: str) -> Optional[Tuple[str, str]]:
    s, phrase, o = (str(x).strip().lower() for x in triple)
    if "not " in phrase or "never" in phrase or "n't" in phrase:
        return None                      # counterexample: no trusted assertion
    fwd, inv = PHRASES[family]
    if phrase in inv:
        return (o, s)
    if phrase in fwd:
        return (s, o)
    return None


# ── scratch-only role scorer (design §3.3/§3.4), reusing main's primitives ──
# Main's vector helpers are nested inside schema_association_scores_from_graph
# (response_gen.py:235-268) and are not importable. These are byte-for-byte
# copies of those three helpers, kept in the scratch script so production code
# stays untouched. The production scorer itself is still called unchanged.

def _graph_node(label, nodes, keyword_index):
    key = str(label).strip().lower()
    ids = list((keyword_index or {}).get(key, ()) or ())
    if not ids:
        for nid, nd in nodes.items():
            if str(getattr(nd, "label", "") or "").strip().lower() == key:
                ids.append(nid)
    for nid in ids:
        nd = nodes.get(nid)
        if nd is not None:
            return nd
    return None


def _graph_vec(nd, glove_vector=None):
    if nd is None:
        return None
    v = getattr(nd, "vector", None)
    if v is None and callable(glove_vector):
        try:
            v = glove_vector(getattr(nd, "label", ""))
        except Exception:
            v = None
    return v


def _graph_cos(a, b):
    if a is None or b is None:
        return 0.0
    try:
        den = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        if den <= 0:
            return 0.0
        return float(np.dot(a, b)) / den
    except Exception:
        return 0.0


def unit(v: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if v is None:
        return None
    try:
        n = float(np.linalg.norm(v))
        if n <= 0:
            return None
        return np.asarray(v, dtype=float) / n
    except Exception:
        return None


def role_aligned_scores(nodes, index, subject, candidates, pairs,
                        glove_fn=None) -> Dict[str, float]:
    """Complementary-role centre scoring; no subject<->candidate cosine.

    r1 centre = mean of demonstrated SOURCE vectors, r2 centre = mean of
    demonstrated TARGET vectors. The subject must fit r1 (slot-typing gate at
    the weakest observed within-role similarity, floored at 0); each option is
    scored ONLY by its cosine to the r2 centre. Missing vectors score 0.
    """
    svec = _graph_vec(_graph_node(subject, nodes, index), glove_fn)
    r1, r2 = [], []
    for a, b in (pairs or ()):
        va = unit(_graph_vec(_graph_node(a, nodes, index), glove_fn))
        vb = unit(_graph_vec(_graph_node(b, nodes, index), glove_fn))
        if va is not None:
            r1.append(va)
        if vb is not None:
            r2.append(vb)
    if svec is None or not r1 or not r2:
        return {}
    c1, c2 = unit(np.mean(np.stack(r1), axis=0)), unit(np.mean(np.stack(r2), axis=0))
    if c1 is None or c2 is None:
        return {}
    sims = [_graph_cos(r1[i], r1[j])
            for i in range(len(r1)) for j in range(i + 1, len(r1))]
    lower = max(0.0, min(sims)) if sims else 0.0
    if _graph_cos(svec, c1) <= lower:
        return {}                       # slot-typing gate fails closed
    out: Dict[str, float] = {}
    for c in (candidates or ()):
        key = str(c).strip().lower()
        if not key or key in out:
            continue
        cv = _graph_vec(_graph_node(c, nodes, index), glove_fn)
        out[key] = _graph_cos(cv, c2)
    return out


def role_gate_diagnostics(nodes, index, subject, pairs, glove_fn=None):
    """Report the subject slot-fit and the observed within-role lower bound."""
    svec = _graph_vec(_graph_node(subject, nodes, index), glove_fn)
    r1 = []
    for a, _b in (pairs or ()):
        va = unit(_graph_vec(_graph_node(a, nodes, index), glove_fn))
        if va is not None:
            r1.append(va)
    if svec is None or not r1:
        return {"subject_fit": None, "lower_bound": None, "passes": False}
    c1 = unit(np.mean(np.stack(r1), axis=0))
    sims = [_graph_cos(r1[i], r1[j])
            for i in range(len(r1)) for j in range(i + 1, len(r1))]
    lower = max(0.0, min(sims)) if sims else 0.0
    fit = _graph_cos(svec, c1)
    return {"subject_fit": round(fit, 4), "lower_bound": round(lower, 4),
            "passes": c1 is not None and fit > lower}


# ── in-process monkeypatch: one DEFAULTED optional ranker parameter ─────────
# This is the only production behaviour change, and it exists in memory only.
_ORIGINAL_RANK = si.rank_candidates_for_unseen_subject


def rank_with_roles(subject, candidates, hypotheses, observed_traces,
                    candidate_scores, instance_scores=None, min_margin=0.0, *,
                    role_scores=None):
    scores = candidate_scores if role_scores is None else role_scores
    return _ORIGINAL_RANK(subject, candidates, hypotheses, observed_traces,
                          scores, instance_scores, min_margin)


si.rank_candidates_for_unseen_subject = rank_with_roles


# ── replay ──────────────────────────────────────────────────────────────────

def recorded_cause_edges() -> Tuple[List[Tuple[str, str]], int]:
    """Parse the parent's recorded engine census (natural CAUSE teaching)."""
    text = CENSUS_PATH.read_text(encoding="utf-8", errors="replace")
    import re
    line = [ln for ln in text.splitlines() if ln.strip().startswith("natural trusted")]
    if not line:
        return [], 0
    m = re.search(r"natural trusted\s+(\d+)\s+(\[.*\])", line[-1])
    if not m:
        return [], 0
    count = int(m.group(1))
    pairs = ast.literal_eval(m.group(2))
    return [(str(a), str(b)) for a, b in pairs], count


def replay_case(row: dict, lut, cause_edges, cause_count):
    family = row["relation"]
    condition = row["condition"]
    item_id = row["id"]
    inp = row["input"]
    subject = inp["query_subject"]
    cands = list(inp["candidates"])
    gold = row["gold"]["target"]

    pairs = [canonical_pair(t, family) for t in inp["demonstrations"]]
    pairs = [p for p in pairs if p is not None]
    canonical_demo_pairs = sorted(set(pairs))

    # Edge provenance model. CAUSE uses the ACTUAL recorded trusted census;
    # other families are the post-write-fix trusted model (fake tests in
    # tests/unit/test_rsi_all_relations.py prove the write stamps them).
    negated = any("not " in str(t[1]).lower() or "never" in str(t[1]).lower()
                  for t in inp["demonstrations"])
    if family == "CAUSE" and cause_edges and not negated:
        edge_pairs = cause_edges
        edge_source = "recorded-engine-census"
        census_count = cause_count
    elif negated:
        edge_pairs = []                  # counterexamples write no trusted edge
        edge_source = "negated-demos-no-write"
        census_count = None
    else:
        edge_pairs = canonical_demo_pairs
        edge_source = "synthetic-trusted-post-write-fix"
        census_count = None
    edges = [si.EdgeView(s, REL_TYPES[family], o, "curated", "conversation", 0.9)
             for s, o in edge_pairs]

    labels = [subject] + cands + [x for p in pairs for x in p] + \
             [x for p in edge_pairs for x in p]
    nodes, index, missing = make_graph(labels, lut)
    glove_fn = lambda label: lut.get(str(label).strip().lower())  # noqa: E731

    traces = si.traces_from_edges(edges)
    hyps = si.induce_schema_hypotheses(traces)
    pairs_for_score = [p for h in hyps for p in h.instances]

    cscores, iscores = rg.schema_association_scores_from_graph(
        FakeGraph(nodes), index, subject, cands, pairs_for_score, glove_fn)
    baseline = si.rank_candidates_for_unseen_subject(
        subject, cands, hyps, traces, cscores, iscores)

    # Type check: score inside each single relation family, then take the best
    # family's role fit (no cross-family pooling).
    family_scores: Dict[str, float] = {}
    gate: Dict[str, Any] = {}
    for h in hyps:
        fam = role_aligned_scores(nodes, index, subject, cands,
                                  list(h.instances), glove_fn)
        gate[h.relation] = role_gate_diagnostics(nodes, index, subject,
                                                 list(h.instances), glove_fn)
        for k, v in fam.items():
            family_scores[k] = max(family_scores.get(k, 0.0), v)
    role = si.rank_candidates_for_unseen_subject(
        subject, cands, hyps, traces, cscores, iscores, role_scores=family_scores)

    def margins(scores):
        vals = sorted((float(v) for k, v in scores.items()
                       if k in {c.lower() for c in cands}), reverse=True)
        if not vals:
            return None
        return round(vals[0] - (vals[1] if len(vals) > 1 else 0.0), 4)

    floor = round(min(iscores.values()), 4) if iscores else None
    expect_reject = condition in ("foil", "negative")
    return {
        "item_id": item_id, "family": family, "condition": condition,
        "relation_type": REL_TYPES[family], "subject": subject,
        "candidates": cands, "gold_target": gold,
        "demo_count": len(canonical_demo_pairs),
        "trusted_edges_used": len(edges), "edge_source": edge_source,
        "recorded_census_edge_count": census_count,
        "traces": len(traces), "hypotheses": len(hyps),
        "support": [h.support for h in hyps],
        "congruence_floor": floor,
        "role_gate": gate,
        "baseline_pick": baseline, "role_pick": role,
        "baseline_correct": (baseline is None) if expect_reject else (baseline == gold),
        "role_correct": (role is None) if expect_reject else (role == gold),
        "baseline_margin": margins(cscores), "role_margin": margins(family_scores),
        "baseline_scores": {k: round(float(v), 4) for k, v in sorted(cscores.items())},
        "role_scores": {k: round(float(v), 4) for k, v in sorted(family_scores.items())},
        "missing_glove_labels": missing,
        "gold_used_for": "evaluation only",
    }


def sanity_control():
    """Validate the instrument: role scoring CAN move a synthetic ranking.

    If this control fails, the scorer/ranker wiring is broken and every null
    result below is uninformative (F7). Vectors here are invented, and are
    labelled as such; they are not evidence about RSI or GloVe.
    """
    vecs = {
        "wheel": np.array([0, 1, 0, 0.0]), "car": np.array([0, .5, .5, .7071]),
        "petal": np.array([0, 0, 1, 0.0]), "flower": np.array([0, .5, .5, .7071]),
        "thaw": np.array([.8, .5, .1, 0.0]),
        "warmth": np.array([.98, .1, .1, 0.0]), "engine": np.array([0, 0, 0, 1.0]),
    }
    nodes, index, _ = make_graph(list(vecs), {k.lower(): v for k, v in vecs.items()})
    g = FakeGraph(nodes)
    traces = [si.TrustedTrace(si.Declaration(a, "part_of", b),
                              {"edge_kind": "curated", "source": "conversation"},
                              "part_of:%s->%s" % (a, b)) for a, b in
              [("wheel", "car"), ("petal", "flower")]]
    hyps = si.induce_schema_hypotheses(traces)
    pairs = list(hyps[0].instances)
    c, i = rg.schema_association_scores_from_graph(g, index, "thaw",
                                                    ["warmth", "engine"], pairs)
    r = role_aligned_scores(nodes, index, "thaw", ["warmth", "engine"], pairs)
    b = si.rank_candidates_for_unseen_subject("thaw", ["warmth", "engine"],
                                              hyps, traces, c, i)
    a = si.rank_candidates_for_unseen_subject("thaw", ["warmth", "engine"],
                                              hyps, traces, c, i, role_scores=r)
    return {"baseline_pick": b, "role_pick": a, "passes": (b == "warmth" and a == "engine"),
            "role_scores": {k: round(float(v), 4) for k, v in r.items()}}


def main() -> int:
    missing_inputs = [str(p) for p in (ROWS_PATH, CENSUS_PATH, GLOVE_PATH) if not p.exists()]
    if missing_inputs:
        print("MISSING INPUTS: " + ", ".join(missing_inputs), file=sys.stderr)
        return 2
    print("[experiment] ravana.chat.schema_induction from", si.__file__)
    print("[experiment] ravana.chat.response_gen  from", rg.__file__)
    assert str(WORKTREE).replace("\\", "/") in si.__file__.replace("\\", "/")
    assert str(WORKTREE).replace("\\", "/") in rg.__file__.replace("\\", "/")
    assert "rank_with_roles" not in str(_ORIGINAL_RANK)  # unmodified production ranker

    lut, dim = build_glove64_lookup(str(GLOVE_PATH))
    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    cause_edges, cause_count = recorded_cause_edges()
    selected = [r for r in rows if r.get("backend") == "live"
                and r.get("arm") == "structural" and r.get("k") == 8
                and r.get("condition") in CONDITIONS
                and r.get("relation") in FAMILIES]
    # One row per (family, condition); prefer primary pool, no permutation dupes.
    dedup = {}
    for r in selected:
        key = (r["relation"], r["condition"])
        if key not in dedup or r.get("pool") == "primary":
            dedup[key] = r
    results = [replay_case(dedup[k], lut, cause_edges, cause_count)
               for k in sorted(dedup)]

    summary = {}
    for family in FAMILIES:
        fam_rows = [r for r in results if r["family"] == family]
        summary[family] = {}
        for cond in CONDITIONS:
            rs = [r for r in fam_rows if r["condition"] == cond]
            summary[family][cond] = {
                "n": len(rs),
                "baseline_correct": sum(r["baseline_correct"] for r in rs),
                "role_correct": sum(r["role_correct"] for r in rs),
                "baseline_picks": [r["baseline_pick"] for r in rs],
                "role_picks": [r["role_pick"] for r in rs],
            }

    totals = {
        cond: {
            "n": sum(summary[f][cond]["n"] for f in FAMILIES),
            "baseline_correct": sum(summary[f][cond]["baseline_correct"] for f in FAMILIES),
            "role_correct": sum(summary[f][cond]["role_correct"] for f in FAMILIES),
        } for cond in CONDITIONS
    }

    sanity = sanity_control()
    payload = {
        "metadata": {
            "worktree": str(WORKTREE), "main_repo": str(MAIN),
            "commit": "dbb3e78",
            "production_files_edited": [],
            "code_loaded_from": [si.__file__, rg.__file__],
            "rows_source": str(ROWS_PATH), "rows_total": len(rows),
            "rows_selected": len(results),
            "census_source": str(CENSUS_PATH),
            "recorded_cause_trusted_pairs": cause_edges,
            "recorded_cause_edge_count": cause_count,
            "glove_cache": str(GLOVE_PATH), "glove_dim": int(dim),
            "ranker_patch": "in-process optional role_scores arg only; gates unchanged",
            "gold_use": "evaluation only, never passed to trace/scorer/ranker",
        },
        "sanity_control": sanity,
        "totals": totals,
        "summary": summary,
        "rows": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")
    print(json.dumps({"sanity": sanity, "summary": summary}, indent=2))
    print("[experiment] wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
