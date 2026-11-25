#!/usr/bin/env python3
"""Scratch-only experiment 2: NON-VECTOR symbolic role-signature matcher for RSI.

NO production edits. NO engine boot. The script imports the CURRENT, UNMODIFIED
main code from this worktree (schema_induction, response_gen, GloVe loader),
replays the SAME 16 recorded focused-live cases used by experiment 1, and adds
one in-process monkeypatch that gives the production ranker an optional
``role_scores`` argument (gates, congruence floor and strict margin stay in the
unmodified production code).

It compares, for every family x condition:

  * prod_baseline      - recomputed, vector-only (production scorer + ranker)
  * exp1_vector_role   - EXPERIMENT 1's recorded GloVe role-centre picks (loaded)
  * V1..V4             - symbolic role signatures, no GloVe except an explicit
                          tie-break in V4, under two floor treatments.

Symbolic representation: each trusted demonstration is
    (relation_type, r1_label_tokens, r2_label_tokens).
A query candidate is scored by (a) exact/partial relation-type match, (b) token
overlap of the candidate with the r2 (demonstrated-target) role tokens, and
(c) for V4 only, a small semantic (GloVe) fallback used ONLY to break exact
ties. V3 adds an identity-free (character 3-gram Jaccard) overlap and is
GloVe-free. All scores are fed to the SAME production ranker with the
congruence floor and strict margin preserved.

Usage (worktree root, main repo's venv python):
  PYTHONPATH=<worktree>/ravana/src;<worktree>/ravana_ml/src;<worktree> \
    <main>/.venv/Scripts/python.exe scratch/role_binding_symbolic_experiment.py
"""
from __future__ import annotations

import ast
import json
import re
import sys
from collections import Counter
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
EXP1_PATH = MAIN / "scratch" / "role_binding_results.json"
OUT_PATH = HERE / "role_binding_symbolic_results.json"

REL_TYPES = {"CAUSE": "causal", "PART_OF": "part_of",
             "PRECEDES": "temporal", "RESEMBLES": "analogical"}
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
    nodes: Dict[int, FakeNode] = {}
    missing: List[str] = []
    for nid, label in enumerate(sorted(set(labels)), start=1):
        v = lut.get(str(label).strip().lower())
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


# ── SYMBOLIC role-signature scorer (scratch only; NO vectors) ───────────────
def toks(label: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", str(label).lower()) if t]


def char_ngrams(word: str, n: int = 3) -> set:
    w = " " + str(word).lower() + " "
    return {w[i:i + n] for i in range(max(0, len(w) - n + 1))}


def ngram_jaccard(a: str, b: str) -> float:
    A, B = char_ngrams(a), char_ngrams(b)
    return (len(A & B) / len(A | B)) if (A | B) else 0.0


def overlap_fraction(candidate: str, role_counter: Counter) -> float:
    """Fraction of candidate tokens present in the r2 role-token multiset."""
    ct = Counter(toks(candidate))
    if not ct:
        return 0.0
    return sum(min(n, role_counter[t]) for t, n in ct.items()) / sum(ct.values())


def symbolic_scores(hyp: si.SchemaHypothesis, candidates: Sequence[str],
                    variant: str) -> Dict[str, float]:
    """Per-candidate symbolic score in [0, 1]; GloVe-free (V4 tie-break added later).

    rel   = exact relation-type match. The query's relation is the demonstrated
            family itself, and the hypothesis admits only that relation, so this
            is 1.0 for every candidate in a case. With only this component every
            candidate ties and the production strict-margin rule emits None.
    r2ov  = overlap of the candidate's tokens with the demonstrated r2 role.
    idfree= identity-free overlap: max char-3gram Jaccard of the candidate against
            any demonstrated r2 token (still no word identity, no vectors).
    """
    inst = list(hyp.instances)
    r2_counter: Counter = Counter()
    r2_words: List[str] = []
    for _a, b in inst:
        for t in toks(b):
            r2_counter[t] += 1
        r2_words.append(b)
    scores: Dict[str, float] = {}
    for c in candidates:
        key = str(c).strip().lower()
        if not key or key in scores:
            continue
        rel = 1.0                                    # (a) exact relation-type match
        r2ov = overlap_fraction(c, r2_counter)       # (b) r2-token overlap
        idfree = max((ngram_jaccard(c, w) for w in r2_words), default=0.0)
        if variant == "V1":
            s = rel                                  # exact relation only
        elif variant == "V2":
            s = r2ov                                 # relation + r2-token overlap
        elif variant == "V3":
            s = r2ov + idfree                        # + identity-free (no GloVe)
        elif variant == "V4":
            s = r2ov + idfree                        # tie-break appended outside
        else:
            raise ValueError("unknown variant %s" % variant)
        scores[key] = float(min(1.0, s))
    return scores


def symbolic_lower_bound(hyp: si.SchemaHypothesis, variant: str) -> float:
    """Rescaled congruence floor: weakest observed within-r2-role similarity.

    Mirrors exp-1's within-r1 lower bound (max(0, min pairwise similarity)).
    """
    r2 = [b for _a, b in hyp.instances]
    if variant in ("V2",):
        vals = []
        for i in range(len(r2)):
            for j in range(i + 1, len(r2)):
                vals.append(overlap_fraction(r2[i], Counter(toks(r2[j]))))
        return max(0.0, min(vals)) if vals else 0.0
    if variant in ("V3", "V4"):
        vals = [ngram_jaccard(r2[i], r2[j])
                for i in range(len(r2)) for j in range(i + 1, len(r2))]
        return max(0.0, min(vals)) if vals else 0.0
    return 1.0                                       # V1: constant relation match


def apply_glove_tiebreak(scores: Dict[str, float], subject_vec,
                         candidate_vecs: Dict[str, Any]) -> Dict[str, float]:
    """(c) Small semantic fallback ONLY as a strict top-tie-break.

    Only candidates already tied at the maximum symbolic score are re-ordered,
    and only by GloVe cos(subject, candidate). A 1e-6 nudge makes a strict
    winner; candidates below the top are untouched. This can never override a
    strict symbolic winner.
    """
    if not scores:
        return scores
    top = max(scores.values())
    tied = [k for k, v in scores.items() if abs(v - top) <= 1e-12]
    if len(tied) < 2 or subject_vec is None:
        return scores
    def cos(a, b):
        if a is None or b is None:
            return 0.0
        den = float(np.linalg.norm(a)) * float(np.linalg.norm(b))
        return float(np.dot(a, b)) / den if den > 0 else 0.0
    best = max(tied, key=lambda k: (cos(subject_vec, candidate_vecs.get(k)), k))
    out = dict(scores)
    out[best] = top + 1e-6
    return out


# ── in-process monkeypatch: one DEFAULTED optional ranker parameter ─────────
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
    text = CENSUS_PATH.read_text(encoding="utf-8", errors="replace")
    line = [ln for ln in text.splitlines() if ln.strip().startswith("natural trusted")]
    if not line:
        return [], 0
    m = re.search(r"natural trusted\s+(\d+)\s+(\[.*\])", line[-1])
    if not m:
        return [], 0
    return [(str(a), str(b)) for a, b in ast.literal_eval(m.group(2))], int(m.group(1))


def pick_for(hypotheses, traces, subject, cands, scores, instance_scores):
    return si.rank_candidates_for_unseen_subject(
        subject, cands, hypotheses, traces, scores, instance_scores,
        role_scores=scores)


def replay_case(row: dict, lut, cause_edges, cause_count, exp1_map) -> dict:
    family, condition, item_id = row["relation"], row["condition"], row["id"]
    inp = row["input"]
    subject = inp["query_subject"]
    cands = list(inp["candidates"])
    gold = row["gold"]["target"]

    pairs = [p for p in (canonical_pair(t, family) for t in inp["demonstrations"])
             if p is not None]
    canonical_demo_pairs = sorted(set(pairs))
    negated = any("not " in str(t[1]).lower() or "never" in str(t[1]).lower()
                  for t in inp["demonstrations"])
    if family == "CAUSE" and cause_edges and not negated:
        edge_pairs, edge_source, census_count = cause_edges, "recorded-engine-census", cause_count
    elif negated:
        edge_pairs, edge_source, census_count = [], "negated-demos-no-write", None
    else:
        edge_pairs, edge_source, census_count = canonical_demo_pairs, "synthetic-trusted-post-write-fix", None
    edges = [si.EdgeView(s, REL_TYPES[family], o, "curated", "conversation", 0.9)
             for s, o in edge_pairs]

    labels = [subject] + cands + [x for p in pairs for x in p] + \
             [x for p in edge_pairs for x in p]
    nodes, index, missing = make_graph(labels, lut)
    glove_fn = lambda label: lut.get(str(label).strip().lower())

    traces = si.traces_from_edges(edges)
    hyps = si.induce_schema_hypotheses(traces)
    pairs_for_score = [p for h in hyps for p in h.instances]

    cscores, iscores = rg.schema_association_scores_from_graph(
        FakeGraph(nodes), index, subject, cands, pairs_for_score, glove_fn)

    # production congruence floor (recomputed, unchanged)
    prod_floor = round(min(iscores.values()), 4) if iscores else None
    baseline = si.rank_candidates_for_unseen_subject(
        subject, cands, hyps, traces, cscores, iscores)
    prod_match_exp1 = (baseline == exp1_map[(family, condition)]["baseline_pick"])

    subject_vec = glove_fn(subject)
    cand_vecs = {c: glove_fn(c) for c in cands}

    arms: Dict[str, Any] = {}
    expect_reject = condition in ("foil", "negative")

    def record(name, pick, scores, floor):
        arms[name] = {
            "pick": pick,
            "correct": (pick is None) if expect_reject else (pick == gold),
            "scores": {k: round(float(v), 6) for k, v in sorted(scores.items())},
            "floor": round(float(floor), 6) if floor is not None else None,
        }

    record("prod_baseline", baseline, cscores, prod_floor)

    e1 = exp1_map[(family, condition)]
    arms["exp1_vector_role"] = {
        "pick": e1["role_pick"],
        "correct": (e1["role_pick"] is None) if expect_reject else (e1["role_pick"] == gold),
        "recorded": True,
        "scores": e1["role_scores"],
        "floor": e1["congruence_floor"],
    }

    for variant in ("V1", "V2", "V3", "V4"):
        s = {}
        for h in hyps:
            vs = symbolic_scores(h, cands, variant)
            for k, v in vs.items():
                s[k] = max(s.get(k, 0.0), v)       # best family fit per candidate
        if variant == "V4":
            s = apply_glove_tiebreak(s, subject_vec, cand_vecs)
        # (i) preserve the production congruence floor exactly as exp-1 did
        pick_prod = pick_for(hyps, traces, subject, cands, s, iscores)
        record("%s_prodfloor" % variant, pick_prod, s, prod_floor)
        # (ii) rescaled symbolic floor (weakest within-role similarity)
        lower = max((symbolic_lower_bound(h, variant) for h in hyps), default=0.0)
        sym_inst = {"r2_lower": lower}
        pick_sym = pick_for(hyps, traces, subject, cands, s, sym_inst)
        record("%s_symfloor" % variant, pick_sym, s, lower)

    return {
        "item_id": item_id, "family": family, "condition": condition,
        "subject": subject, "candidates": cands, "gold_target": gold,
        "demonstrations": inp["demonstrations"],
        "demo_count": len(canonical_demo_pairs),
        "trusted_edges_used": len(edges), "edge_source": edge_source,
        "traces": len(traces), "hypotheses": len(hyps),
        "prod_congruence_floor": prod_floor,
        "prod_baseline_matches_exp1": prod_match_exp1,
        "arms": arms,
        "missing_glove_labels": missing,
    }


ARM_NAMES = ["prod_baseline", "exp1_vector_role",
             "V1_prodfloor", "V2_prodfloor", "V3_prodfloor", "V4_prodfloor",
             "V1_symfloor", "V2_symfloor", "V3_symfloor", "V4_symfloor"]


def sanity_control():
    """Instrument validation: the symbolic scorer CAN move a production pick,
    and fails closed when r2 tokens are disjoint from every candidate."""
    vecs = {"car": np.array([0, 1, 0, 0.0]), "petal": np.array([0, 0, 1, 0.0]),
            "wheel": np.array([1, 0, 0, 0.0]), "axle": np.array([1, 0, 0, 0.0]),
            "thaw": np.array([.5, .5, 0, 0.0])}
    lut = {k.lower(): v for k, v in vecs.items()}
    nodes, index, _ = make_graph(list(vecs), lut)
    g = FakeGraph(nodes)
    traces = [si.TrustedTrace(si.Declaration(a, "part_of", b),
                              {"edge_kind": "curated", "source": "conversation"},
                              "part_of:%s->%s" % (a, b))
              for a, b in [("wheel", "car"), ("axle", "car")]]
    hyps = si.induce_schema_hypotheses(traces)
    s = symbolic_scores(hyps[0], ["car", "petal"], "V2")
    inst = {"r2_lower": symbolic_lower_bound(hyps[0], "V2")}
    pick = si.rank_candidates_for_unseen_subject(
        "thaw", ["car", "petal"], hyps, traces, {}, inst, role_scores=s)
    # disjoint control: candidate shares no r2 token
    s2 = symbolic_scores(hyps[0], ["petal"], "V2")
    pick2 = si.rank_candidates_for_unseen_subject(
        "thaw", ["petal"], hyps, traces, {}, inst, role_scores=s2)
    return {"overlap_scores": {k: round(v, 4) for k, v in s.items()},
            "overlap_pick": pick, "disjoint_pick": pick2,
            "passes": (pick == "car" and pick2 is None)}


def main() -> int:
    missing = [str(p) for p in (ROWS_PATH, CENSUS_PATH, GLOVE_PATH, EXP1_PATH)
               if not p.exists()]
    if missing:
        print("MISSING INPUTS: " + ", ".join(missing), file=sys.stderr)
        return 2
    print("[symbolic] schema_induction from", si.__file__)
    print("[symbolic] response_gen  from", rg.__file__)
    assert str(WORKTREE).replace("\\", "/") in si.__file__.replace("\\", "/")
    assert str(WORKTREE).replace("\\", "/") in rg.__file__.replace("\\", "/")
    assert "rank_with_roles" not in str(_ORIGINAL_RANK)

    lut, dim = build_glove64_lookup(str(GLOVE_PATH))
    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    exp1 = json.loads(EXP1_PATH.read_text(encoding="utf-8"))
    exp1_map = {(r["family"], r["condition"]): r for r in exp1["rows"]}
    cause_edges, cause_count = recorded_cause_edges()

    selected = [r for r in rows if r.get("backend") == "live"
                and r.get("arm") == "structural" and r.get("k") == 8
                and r.get("condition") in CONDITIONS and r.get("relation") in FAMILIES]
    dedup = {}
    for r in selected:
        key = (r["relation"], r["condition"])
        if key not in dedup or r.get("pool") == "primary":
            dedup[key] = r
    results = [replay_case(dedup[k], lut, cause_edges, cause_count, exp1_map)
               for k in sorted(dedup)]

    # note PRECEDES far/near/foil/negative are 'sealed' pool; exp1 still has them
    summary = {}
    for family in FAMILIES:
        summary[family] = {}
        for cond in CONDITIONS:
            rs = [r for r in results if r["family"] == family and r["condition"] == cond]
            summary[family][cond] = {
                "n": len(rs),
                "gold": rs[0]["gold_target"] if rs else None,
                "arms": {arm: {
                    "correct": sum(r["arms"][arm]["correct"] for r in rs),
                    "picks": [r["arms"][arm]["pick"] for r in rs],
                } for arm in ARM_NAMES},
            }

    totals = {arm: {cond: {
        "n": sum(summary[f][cond]["n"] for f in FAMILIES),
        "correct": sum(summary[f][cond]["arms"][arm]["correct"] for f in FAMILIES),
    } for cond in CONDITIONS} for arm in ARM_NAMES}

    sanity = sanity_control()
    reproduced = sum(1 for r in results if r["prod_baseline_matches_exp1"])

    # answer the parent's specific question over the key families
    key = {}
    for arm in ARM_NAMES:
        pf = [r for r in results if r["family"] in ("PART_OF", "PRECEDES")
              and r["condition"] in ("near", "far")]
        key[arm] = {
            "near_far_off_zero": sum(r["arms"][arm]["correct"] for r in pf
                                     if r["condition"] in ("near", "far")),
            "near_far_n": len(pf),
        }

    payload = {
        "metadata": {
            "worktree": str(WORKTREE), "main_repo": str(MAIN),
            "base_commit": "ca26488", "production_files_edited": [],
            "code_loaded_from": [si.__file__, rg.__file__],
            "rows_source": str(ROWS_PATH), "rows_total": len(rows),
            "rows_selected": len(results), "census_source": str(CENSUS_PATH),
            "glove_cache": str(GLOVE_PATH), "glove_dim": int(dim),
            "exp1_comparator": str(EXP1_PATH),
            "congruence_floor": "production floor preserved in *_prodfloor arms; "
                                "rescaled within-role lower bound in *_symfloor arms",
            "strict_margin": "unmodified production ranker (min_margin=0)",
            "gold_use": "evaluation only, never passed to trace/scorer/ranker",
        },
        "instrument_validation": {
            "sanity_control": sanity,
            "prod_baseline_reproduced_of_16": reproduced,
            "prod_baseline_matches_exp1_all": reproduced == 16,
        },
        "arm_names": ARM_NAMES,
        "totals": totals,
        "key_near_far_partof_precedes": key,
        "summary": summary,
        "rows": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    print("[symbolic] sanity:", json.dumps(sanity))
    print("[symbolic] baseline reproduced of 16:", reproduced)
    print("[symbolic] per-condition totals (correct/n):")
    for arm in ARM_NAMES:
        t = totals[arm]
        print("  %-18s near %d/%d far %d/%d foil %d/%d neg %d/%d" % (
            arm, t["near"]["correct"], t["near"]["n"],
            t["far"]["correct"], t["far"]["n"],
            t["foil"]["correct"], t["foil"]["n"],
            t["negative"]["correct"], t["negative"]["n"]))
    print("[symbolic] PART_OF/PRECEDES near+far off zero:",
          json.dumps(key))
    print("[symbolic] wrote", OUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
