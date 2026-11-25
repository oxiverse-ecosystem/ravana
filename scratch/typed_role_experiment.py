#!/usr/bin/env python3
"""Scratch-only experiment 3: TYPED ROLE SLOTS on the 16 recorded RSI cases.

NO production edits. NO engine boot. NO `process_turn`. NO store writes. NO
network. This script imports the CURRENT, UNMODIFIED main code from this
worktree and replays the SAME 16 recorded focused-live cases used by
experiments 1 and 2. It adds one in-process monkeypatch that gives the
production ranker a defaulted ``role_scores=None`` argument; the congruence
floor and strict-margin gate remain the unmodified production code.

ARMS (see docs/investigations/RSI_TYPED_ROLE_DESIGN.md):
  A0              recomputed production vector baseline (control).
  A1 ORACLE-DOMAIN     ORACLE / NOT DEPLOYABLE. Uses item-bank domain tags
                       re-joined by (id, condition); the engine never sees them.
  A2 ORACLE-ANSWER-KEY CONTAMINATION / CEILING ONLY. Uses world_model, the gold
                       answer key, keyed by exactly (relation, subject).
  A3 GRAPH-DERIVED     the ONLY deployable candidate: is_a edges + copula heads
                       recovered from the read-only store, no oracle labels.

Gold is used ONLY to score a pick after ranking, never supplied to a scorer.

Usage (worktree root, main repo's venv python):
  PYTHONHASHSEED=0 PYTHONPATH=<worktree>/ravana/src;<worktree>/ravana_ml/src;<worktree> \
    <main>/.venv/Scripts/python.exe scratch/typed_role_experiment.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import pickle
import re
import sqlite3
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

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
from ravana.learn.definition_proposition import proposition_from_definition  # noqa: E402
from ravana.ontology.attribute_encoder import build_glove64_lookup  # noqa: E402

ROWS_PATH = MAIN / "scratch" / "rsi_focus_live_rows.json"
SYM_PATH = MAIN / "scratch" / "role_binding_symbolic_results.json"
CENSUS_PATH = MAIN / "scratch" / "rsi_teach_form_probe2.log"
GLOVE_PATH = MAIN / "data" / "ravana_glove_cache.npz"
DB_PATH = MAIN / "weights" / "ravana_weights.db"
ITEM_BANK_PATH = WORKTREE / "scripts" / "measure_learn5_rsi.py"
OUT_PATH = HERE / "typed_role_results.json"

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

ARM_NAMES = [
    "A0_prodbaseline",
    "A1_oracle_domain_prodfloor", "A1_oracle_domain_symfloor",
    "A2_oracle_answerkey_prodfloor", "A2_oracle_answerkey_symfloor",
    "A3_graph_type_prodfloor", "A3_graph_type_symfloor",
]
ARM_STATUS = {
    "A0_prodbaseline": "CONTROL — recomputed unmodified production vector baseline",
    "A1_oracle_domain_prodfloor": "ORACLE / NOT DEPLOYABLE — item-bank domain tag",
    "A1_oracle_domain_symfloor": "ORACLE / NOT DEPLOYABLE — item-bank domain tag",
    "A2_oracle_answerkey_prodfloor": "CONTAMINATION / CEILING ONLY — world_model answer key",
    "A2_oracle_answerkey_symfloor": "CONTAMINATION / CEILING ONLY — world_model answer key",
    "A3_graph_type_prodfloor": "DEPLOYABLE CANDIDATE — graph-derived types only",
    "A3_graph_type_symfloor": "DEPLOYABLE CANDIDATE — graph-derived types only",
}


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


# ── in-process monkeypatch: one DEFAULTED optional ranker parameter ─────────
_ORIGINAL_RANK = si.rank_candidates_for_unseen_subject


def rank_with_roles(subject, candidates, hypotheses, observed_traces,
                    candidate_scores, instance_scores=None, min_margin=0.0, *,
                    role_scores=None):
    scores = candidate_scores if role_scores is None else role_scores
    return _ORIGINAL_RANK(subject, candidates, hypotheses, observed_traces,
                          scores, instance_scores, min_margin)


si.rank_candidates_for_unseen_subject = rank_with_roles


def pick_for(hypotheses, traces, subject, cands, scores, instance_scores):
    return si.rank_candidates_for_unseen_subject(
        subject, cands, hypotheses, traces, scores, instance_scores,
        role_scores=scores)


# ── typed scorers (scratch only) ────────────────────────────────────────────
def jaccard(a: Set[str], b: Set[str]) -> float:
    return (len(a & b) / len(a | b)) if (a | b) else 0.0


def typed_jaccard_scores(candidates: Sequence[str], demo_types: Set[str],
                         type_map: Dict[str, Set[str]]) -> Dict[str, float]:
    """A3: Jaccard overlap of a candidate's stored type set with the union of
    demonstration-r2 type sets. No oracle labels; only graph/store information."""
    out: Dict[str, float] = {}
    for c in candidates:
        key = str(c).strip().lower()
        if not key or key in out:
            continue
        out[key] = jaccard(type_map.get(key, set()), demo_types)
    return out


def _pool_of(word: str, bank) -> Optional[str]:
    w = str(word).strip().lower()
    for dom in bank.DOMAINS:
        if w in bank.OBJECTS[dom]:
            return dom
    return None


def oracle_domain_scores(item: dict, condition: str, candidates: Sequence[str],
                         bank) -> Dict[str, float]:
    """A1 ORACLE: domain tag from regenerated item bank, never seen by engine.

    near: candidate must be in the demo domain (item['domain']);
    far:  candidate must be in item['far_domain'].
    foil/negative: candidates are all in the demo domain -> constant 1.0.
    """
    demo_dom = item["domain"]
    far_dom = item["far_domain"]
    target_dom = far_dom if condition == "far" else demo_dom
    out: Dict[str, float] = {}
    for c in candidates:
        key = str(c).strip().lower()
        if not key or key in out:
            continue
        out[key] = 1.0 if _pool_of(key, bank) == target_dom else 0.0
    return out


def oracle_answerkey_scores(item: dict, subject: str,
                            candidates: Sequence[str]) -> Dict[str, float]:
    """A2 ORACLE / CONTAMINATION: 1 iff (rel, subject, candidate) in world_model."""
    rel = item["relation"]
    wm = {(k[0], k[1]): set(v) for k, v in item["world_model"].items()}
    gold_set = wm.get((rel, str(subject).strip().lower()), set())
    out: Dict[str, float] = {}
    for c in candidates:
        key = str(c).strip().lower()
        if not key or key in out:
            continue
        out[key] = 1.0 if key in gold_set else 0.0
    return out


# ── A3 substrate: read-only store -> type_map ───────────────────────────────
_SINGLE_WORD = re.compile(r"^[a-z][a-z\-]*$")


def build_type_map(db_path: Path, bank) -> Tuple[Dict[str, Set[str]], dict]:
    """Build word -> set(type labels) from the persisted store, read-only.

    (a) real is_a edges: both endpoints must exist in `concepts`; targets must
        be single words (excludes E15's multi-word junk phrase and dangling ids).
    (b) is_a heads recovered from metadata['definitions'] via the existing
        copula logic in learn/definition_proposition.py.
    """
    cx = sqlite3.connect("file:%s?mode=ro" % str(db_path).replace("\\", "/"),
                         uri=True)
    census: Dict[str, Any] = {}
    type_map: Dict[str, Set[str]] = {}

    # node census
    census["nodes_total"] = cx.execute("SELECT COUNT(*) FROM concepts").fetchone()[0]
    census["node_types"] = dict(cx.execute(
        "SELECT node_type, COUNT(*) FROM concepts GROUP BY node_type").fetchall())
    census["relation_types"] = dict(cx.execute(
        "SELECT relation_type, COUNT(*) FROM edges GROUP BY relation_type").fetchall())

    labels = {str(r[0]).strip().lower(): str(r[1] or "")
              for r in cx.execute("SELECT id, label FROM concepts")}
    id_to_label = {str(r[0]): str(r[1] or "") for r in cx.execute(
        "SELECT id, label FROM concepts")}
    label_set = {str(v).strip().lower() for v in labels.values() if v}

    used = 0
    rejected_multiword = 0
    rejected_dangling = 0
    for sid, tid, rtype in cx.execute(
            "SELECT source_id, target_id, relation_type FROM edges "
            "WHERE relation_type = 'is_a'"):
        s = str(id_to_label.get(str(sid), "")).strip().lower()
        t = str(id_to_label.get(str(tid), "")).strip().lower()
        if not s or not t:
            rejected_dangling += 1
            continue
        if not _SINGLE_WORD.match(s) or not _SINGLE_WORD.match(t):
            rejected_multiword += 1
            continue
        type_map.setdefault(s, set()).add(t)
        used += 1
    census["is_a_edges_total"] = census["relation_types"].get("is_a", 0)
    census["is_a_edges_used"] = used
    census["is_a_rejected_multiword"] = rejected_multiword
    census["is_a_rejected_dangling"] = rejected_dangling

    # definitions
    row = cx.execute("SELECT value FROM metadata WHERE key='definitions'").fetchone()
    defs: Dict[str, str] = {}
    if row is not None:
        obj = pickle.loads(row[0])
        if isinstance(obj, dict):
            defs = {str(k): str(v) for k, v in obj.items()}
    census["definitions_total"] = len(defs)
    def_is_a = 0
    for word, text in defs.items():
        w = str(word).strip().lower()
        if not _SINGLE_WORD.match(w):
            continue
        prop = proposition_from_definition(text)
        if prop is not None and prop[0] == "is_a" and _SINGLE_WORD.match(str(prop[1])):
            type_map.setdefault(w, set()).add(str(prop[1]).strip().lower())
            def_is_a += 1
    census["definitions_recovered_is_a"] = def_is_a

    # coverage over the RSI vocabulary that actually appears in the 16 cases
    rsi_vocab = {w.lower() for dom in bank.DOMAINS for w in bank.OBJECTS[dom]}
    census["rsi_vocab_size"] = len(rsi_vocab)
    census["rsi_words_in_store_nodes"] = sorted(rsi_vocab & label_set)
    census["rsi_words_with_type_all_vocab"] = sorted(
        w for w in rsi_vocab if type_map.get(w))
    census["all_recovered_type_labels"] = sorted(
        {t for ts in type_map.values() for t in ts})
    census["rsi_typed_edges_touching_vocab"] = sorted(
        (s, sorted(type_map[s])) for s in type_map if s in rsi_vocab)
    cx.close()
    return type_map, census


# ── item bank (regenerate build_items() deterministically) ─────────────────
def load_item_bank():
    spec = importlib.util.spec_from_file_location("rsi_item_bank", str(ITEM_BANK_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    items = mod.build_items()
    return mod, {it["item_id"]: it for it in items}


def validate_join(row: dict, item: dict, bank) -> Tuple[bool, List[str]]:
    """Check that the regenerated item reproduces the recorded row exactly."""
    problems: List[str] = []
    cond = row["condition"]
    inp = row["input"]
    q = item.get(cond)
    if q is None:
        return False, ["missing condition %s" % cond]
    if item["item_id"] != row["id"]:
        problems.append("item id")
    if item["relation"] != row["relation"]:
        problems.append("relation")
    if q["subject"] != inp["query_subject"]:
        problems.append("subject")
    if list(q["candidates"]) != list(inp["candidates"]):
        problems.append("candidates")
    if row["gold"]["target"] != q["gold_target"]:
        problems.append("gold")
    if cond != "negative":
        demo_pairs = {canonical_pair(t, row["relation"])
                      for t in inp["demonstrations"]}
        demo_pairs.discard(None)
        expected = {tuple(p) for p in item["demo_pairs"][:8]}
        if demo_pairs != expected:
            problems.append("demo_pairs")
    # domain label recorded on the row must equal the item's condition domain
    expected_dom = item["far_domain"] if cond == "far" else item["domain"]
    if inp.get("domain") != expected_dom:
        problems.append("domain:%s!=%s" % (inp.get("domain"), expected_dom))
    return (not problems), problems


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


def replay_case(row: dict, item: dict, bank, lut, cause_edges, cause_count,
                sym_map, type_map) -> dict:
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

    prod_floor = round(min(iscores.values()), 4) if iscores else None
    baseline = si.rank_candidates_for_unseen_subject(
        subject, cands, hyps, traces, cscores, iscores)
    prod_match_sym = (baseline ==
                      sym_map[(family, condition)]["arms"]["prod_baseline"]["pick"])

    join_ok, join_problems = validate_join(row, item, bank)

    # demo r2 tokens = targets of the presented demonstrations
    demo_r2 = [b for _a, b in canonical_demo_pairs]
    demo_types: Set[str] = set()
    for b in demo_r2:
        demo_types |= type_map.get(b.lower(), set())

    arms: Dict[str, Any] = {}
    expect_reject = condition in ("foil", "negative")

    def record(name, pick, scores, floor, valid=True):
        if expect_reject:
            correct = pick is None if valid else None
        else:
            correct = pick == gold if valid else None
        arms[name] = {
            "pick": pick,
            "correct": correct,
            "scores": {k: round(float(v), 6) for k, v in sorted(scores.items())},
            "floor": round(float(floor), 6) if floor is not None else None,
            "valid": valid,
        }

    record("A0_prodbaseline", baseline, cscores, prod_floor)

    # typed congruence floors (rescaled, mirroring experiment 2's *_symfloor)
    a1_floor = min([1.0 if _pool_of(a, bank) == _pool_of(b, bank) == item["domain"]
                    else 0.0 for a, b in canonical_demo_pairs], default=0.0)
    wm = {(k[0], k[1]): set(v) for k, v in item["world_model"].items()}
    a2_floor = min([1.0 if (item["relation"], a) in wm and b in wm[(item["relation"], a)]
                    else 0.0 for a, b in canonical_demo_pairs], default=0.0)
    a3_floor = min([jaccard(type_map.get(a.lower(), set()),
                            type_map.get(b.lower(), set()))
                    for a, b in canonical_demo_pairs], default=0.0)

    if join_ok:
        s_a1 = oracle_domain_scores(item, condition, cands, bank)
        record("A1_oracle_domain_prodfloor",
               pick_for(hyps, traces, subject, cands, s_a1, iscores),
               s_a1, prod_floor)
        record("A1_oracle_domain_symfloor",
               pick_for(hyps, traces, subject, cands, s_a1, {"r2_lower": a1_floor}),
               s_a1, a1_floor)

        s_a2 = oracle_answerkey_scores(item, subject, cands)
        record("A2_oracle_answerkey_prodfloor",
               pick_for(hyps, traces, subject, cands, s_a2, iscores),
               s_a2, prod_floor)
        record("A2_oracle_answerkey_symfloor",
               pick_for(hyps, traces, subject, cands, s_a2, {"r2_lower": a2_floor}),
               s_a2, a2_floor)
    else:
        for name in ("A1_oracle_domain_prodfloor", "A1_oracle_domain_symfloor",
                     "A2_oracle_answerkey_prodfloor", "A2_oracle_answerkey_symfloor"):
            record(name, None, {}, None, valid=False)

    s_a3 = typed_jaccard_scores(cands, demo_types, type_map)
    record("A3_graph_type_prodfloor",
           pick_for(hyps, traces, subject, cands, s_a3, iscores), s_a3, prod_floor)
    record("A3_graph_type_symfloor",
           pick_for(hyps, traces, subject, cands, s_a3, {"r2_lower": a3_floor}),
           s_a3, a3_floor)

    return {
        "item_id": item_id, "family": family, "condition": condition,
        "subject": subject, "candidates": cands, "gold_target": gold,
        "demonstrations": inp["demonstrations"],
        "demo_count": len(canonical_demo_pairs),
        "demo_r2_tokens": demo_r2,
        "demo_type_union": sorted(demo_types),
        "trusted_edges_used": len(edges), "edge_source": edge_source,
        "traces": len(traces), "hypotheses": len(hyps),
        "prod_congruence_floor": prod_floor,
        "prod_baseline_matches_exp2": prod_match_sym,
        "join_ok": join_ok, "join_problems": join_problems,
        "item_domain": item["domain"], "item_far_domain": item["far_domain"],
        "arms": arms,
        "missing_glove_labels": missing,
    }


# ── F7 instrument validation ───────────────────────────────────────────────
def sanity_control() -> dict:
    """The typed scorer CAN move a production pick, and fails closed when the
    stored type labels are disjoint from the candidate type."""
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
    tmap = {"car": {"vehicle"}, "petal": {"plant"}}
    demo_types = {"vehicle"}
    s = typed_jaccard_scores(["car", "petal"], demo_types, tmap)
    inst = {"r2_lower": 1.0}
    pick = si.rank_candidates_for_unseen_subject(
        "thaw", ["car", "petal"], hyps, traces, {}, inst, role_scores=s)
    s2 = typed_jaccard_scores(["petal"], demo_types, tmap)
    pick2 = si.rank_candidates_for_unseen_subject(
        "thaw", ["petal"], hyps, traces, {}, inst, role_scores=s2)
    return {
        "overlap_scores": {k: round(v, 4) for k, v in s.items()},
        "overlap_pick": pick, "disjoint_scores": {k: round(v, 4) for k, v in s2.items()},
        "disjoint_pick": pick2,
        "passes": (s["car"] == 1.0 and s["petal"] == 0.0
                   and pick == "car" and pick2 is None),
    }


def main() -> int:
    missing = [str(p) for p in (ROWS_PATH, SYM_PATH, CENSUS_PATH, GLOVE_PATH,
                                DB_PATH, ITEM_BANK_PATH) if not p.exists()]
    if missing:
        print("MISSING INPUTS: " + ", ".join(missing), file=sys.stderr)
        return 2

    print("[typed] schema_induction from", si.__file__)
    print("[typed] response_gen  from", rg.__file__)
    print("[typed] item bank     from", ITEM_BANK_PATH)
    assert str(WORKTREE).replace("\\", "/") in si.__file__.replace("\\", "/")
    assert str(WORKTREE).replace("\\", "/") in rg.__file__.replace("\\", "/")
    assert "rank_with_roles" not in str(_ORIGINAL_RANK)

    sanity = sanity_control()
    print("[typed] F7 sanity control:", json.dumps(sanity))

    bank, items = load_item_bank()
    print("[typed] regenerated items:", len(items),
          "manifest:", bank.item_manifest_hash(list(items.values()))[:16])

    lut, dim = build_glove64_lookup(str(GLOVE_PATH))
    rows = json.loads(ROWS_PATH.read_text(encoding="utf-8"))
    sym = json.loads(SYM_PATH.read_text(encoding="utf-8"))
    sym_map = {(r["family"], r["condition"]): r for r in sym["rows"]}
    cause_edges, cause_count = recorded_cause_edges()

    selected = [r for r in rows if r.get("backend") == "live"
                and r.get("arm") == "structural" and r.get("k") == 8
                and r.get("condition") in CONDITIONS and r.get("relation") in FAMILIES]
    dedup: Dict[Tuple[str, str], dict] = {}
    for r in selected:
        key = (r["relation"], r["condition"])
        if key not in dedup or r.get("pool") == "primary":
            dedup[key] = r
    ordered = [dedup[k] for k in sorted(dedup)]

    type_map, census = build_type_map(DB_PATH, bank)
    print("[typed] store census:", json.dumps(census))

    join_failures: List[tuple] = []
    results = []
    for row in ordered:
        it = items[row["id"]]
        ok, probs = validate_join(row, it, bank)
        if not ok:
            join_failures.append((row["id"], row["condition"], probs))
        results.append(replay_case(row, it, bank, lut, cause_edges, cause_count,
                                   sym_map, type_map))
    print("[typed] item-bank join failures:", json.dumps(join_failures))

    case_vocab = set()
    for res in results:
        case_vocab |= {w.lower() for w in res["candidates"]}
        case_vocab |= {res["subject"].lower()}
        case_vocab |= {w.lower() for p in res["demonstrations"]
                       for w in (p[0], p[2])}
    usable_case = sorted(w for w in case_vocab if type_map.get(w))
    gold_case = {r["gold_target"].lower() for r in results if r["gold_target"]}
    usable_gold = sorted(w for w in gold_case if type_map.get(w))
    coverage_gate = {
        "case_vocab_size": len(case_vocab),
        "case_words_with_usable_type": usable_case,
        "case_usable_count": len(usable_case),
        "threshold": 5,
        "numeric_threshold_passed": len(usable_case) >= 5,
        "gold_candidates": sorted(gold_case),
        "gold_candidates_with_usable_type": usable_gold,
        "status": ("COVERAGE PRESENT" if len(usable_case) >= 5
                   else "INFORMATIVE NULL — substrate absent"),
        "interpretation": ("coverage count alone does not supply role slots: "
                           "check whether gold candidates carry a type and "
                           "whether A3 separates gold from foils"),
    }
    print("[typed] A3 coverage gate:", json.dumps(coverage_gate))

    reproduced = sum(1 for r in results if r["prod_baseline_matches_exp2"])
    f7_ok = sanity["passes"] and reproduced == 16 and not join_failures
    print("[typed] F7 baseline reproduced of 16:", reproduced,
          "join_ok:", not join_failures, "F7_OK:", f7_ok)

    # aggregate
    summary: Dict[str, Dict[str, dict]] = {}
    for family in FAMILIES:
        summary[family] = {}
        for cond in CONDITIONS:
            rs = [r for r in results if r["family"] == family and r["condition"] == cond]
            summary[family][cond] = {
                "n": len(rs), "gold": rs[0]["gold_target"] if rs else None,
                "arms": {a: {
                    "correct": sum(1 for r in rs if r["arms"][a]["correct"] is True),
                    "picks": [r["arms"][a]["pick"] for r in rs],
                } for a in ARM_NAMES},
            }
    totals = {a: {c: {
        "n": sum(summary[f][c]["n"] for f in FAMILIES),
        "correct": sum(summary[f][c]["arms"][a]["correct"] for f in FAMILIES),
    } for c in CONDITIONS} for a in ARM_NAMES}

    key = {"PART_OF_PRECEDES_near_far": {}, "far_all_families": {},
           "near_all_families": {}, "foil_reject": {}, "negative_reject": {}}
    for a in ARM_NAMES:
        pf = [r for r in results if r["family"] in ("PART_OF", "PRECEDES")
              and r["condition"] in ("near", "far")]
        key["PART_OF_PRECEDES_near_far"][a] = {
            "correct": sum(1 for r in pf if r["arms"][a]["correct"] is True),
            "n": len(pf)}
        for c in ("near", "far", "foil", "negative"):
            rs = [r for r in results if r["condition"] == c]
            key[{"near": "near_all_families", "far": "far_all_families",
                 "foil": "foil_reject", "negative": "negative_reject"}[c]][a] = {
                "correct": sum(1 for r in rs if r["arms"][a]["correct"] is True),
                "n": len(rs)}

    payload = {
        "metadata": {
            "experiment": "typed role slots on 16 recorded focused-live RSI cases",
            "worktree": str(WORKTREE), "main_repo": str(MAIN),
            "base_commit": "3028b4e", "production_files_edited": [],
            "engine_booted": False, "process_turn_calls": 0,
            "store_writes": 0, "network": False,
            "code_loaded_from": [si.__file__, rg.__file__],
            "rows_source": str(ROWS_PATH), "rows_total": len(rows),
            "rows_selected": len(results),
            "item_bank": str(ITEM_BANK_PATH),
            "sym_comparator": str(SYM_PATH),
            "census_source": str(CENSUS_PATH),
            "glove_cache": str(GLOVE_PATH), "glove_dim": int(dim),
            "store_source": str(DB_PATH),
            "strict_margin": "unmodified production ranker (min_margin=0)",
            "gold_use": "evaluation only, never passed to any scorer or ranker",
            "f7_note": ("INSTRUMENT INVALID — no score" if not f7_ok
                        else "instrument validated before scoring"),
        },
        "arm_status": ARM_STATUS,
        "instrument_validation": {
            "sanity_control": sanity,
            "prod_baseline_reproduced_of_16": reproduced,
            "join_failures": join_failures,
            "F7_OK": f7_ok,
        },
        "a3_store_census": census,
        "a3_coverage_gate": coverage_gate,
        "arm_names": ARM_NAMES,
        "totals": totals,
        "key_counts": key,
        "summary": summary,
        "rows": results,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")

    print("[typed] per-condition totals (correct/n):")
    for a in ARM_NAMES:
        t = totals[a]
        print("  %-34s near %d/%d far %d/%d foil %d/%d neg %d/%d" % (
            a, t["near"]["correct"], t["near"]["n"],
            t["far"]["correct"], t["far"]["n"],
            t["foil"]["correct"], t["foil"]["n"],
            t["negative"]["correct"], t["negative"]["n"]))
    print("[typed] PART_OF/PRECEDES near+far:",
          json.dumps(key["PART_OF_PRECEDES_near_far"]))
    print("[typed] far across all families:", json.dumps(key["far_all_families"]))
    print("[typed] wrote", OUT_PATH)
    print("[typed] EXIT=0" if f7_ok else "[typed] EXIT=0 (INSTRUMENT INVALID)")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
