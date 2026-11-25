#!/usr/bin/env python3
"""TRA experiment 2 -- scratch trusted is_a write + fail-closed slot-keyed read.

SCRATCH ONLY. No file under ravana/src, ravana_ml/src, scripts or tests is
edited. Every engine import below resolves to the UNMODIFIED production tree
at PROJ (main dbed80d) and that is asserted at runtime. All writes go to
separate COPIES of the weight store under SCRATCH_ROOT; production weights/
is only read from.

Two halves, both implemented locally in this file:

  WRITE  parse natural is_a / part-of assertions and push them through the
         SAME KnowledgeIntake.intake(Fact, Provenance, salience) API the
         production relation-family writer uses (chain_walker.py:1090-1105).
         Stamp a distinct type/category relation_type, a type_label and a
         role-slot label in source_metadata. Never touch the trust gate.

  READ   a pure, stateless forced-choice scorer that selects a slot ONLY
         when a cold-reloaded, provenance-trusted typed edge links the query
         subject to that slot. Congruence floor + strict margin + explicit
         fail-closed None for disjoint options (TRA C10). No GloVe cosine to
         the label word, no item gold, no world_model, no type_sets.

Controls: relation-only arm (no typed slots), typed arm, disjoint options,
untrusted forged typed edge, cold-reload retention, no cross-turn state.

Run:
  RAVANA_OFFLINE=1 PYTHONHASHSEED=0 PYTHONPATH=<production paths> \
    C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
    scratch/tra_slot_write_read_experiment.py > scratch/tra_slot_run.log 2>&1
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback
from collections import Counter

# ── Paths / environment ──────────────────────────────────────────────────────
PROJ = "C:/Users/Likhith/Documents/projects/ravana"          # unmodified production tree
WORKTREE = "C:/Users/Likhith/Documents/projects/ravana_slot_exp"
RESULTS = os.path.join(WORKTREE, "scratch", "tra_slot_results.json")
SCRATCH_ROOT = "C:/Users/Likhith/AppData/Local/hermes/cache/scratch/tra_slot_data"

os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ.pop("RAVANA_SCHEMA", None)                       # default production read path
os.environ.pop("RAVANA_RSI_DECLARATION_STORAGE", None)      # keep helpers default-OFF
for _p in (PROJ, os.path.join(PROJ, "scripts"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

R = {
    "experiment": "TRA experiment 2 -- scratch is_a write and fail-closed slot read",
    "project_tree": PROJ, "worktree": WORKTREE,
    "scratch_root": SCRATCH_ROOT,
    "production_code_modified": False,
    "arms": {}, "gates": {}, "errors": [],
}


def log(*a):
    print(*a, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Imports: production tree only ────────────────────────────────────────────
from ravana.chat.engine import CognitiveChatEngine            # noqa: E402
from ravana.chat import schema_induction as si                # noqa: E402
from ravana.chat import chain_walker as cw                    # noqa: E402
from ravana.learn.intake import Fact, Provenance              # noqa: E402

_resolved = {
    "engine": sys.modules[CognitiveChatEngine.__module__].__file__,
    "schema_induction": si.__file__,
    "chain_walker": cw.__file__,
    "intake": sys.modules[Fact.__module__].__file__,
}
R["module_resolution"] = _resolved
for _k, _v in _resolved.items():
    log("MODULE", _k, _v)
assert all(os.path.normcase(PROJ) in os.path.normcase(v) for v in _resolved.values()), \
    "all imports must resolve to the UNMODIFIED production tree"
R["trusted_edge_kinds"] = sorted(si.TRUSTED_EDGE_KINDS)


# ── Fresh copied store per arm (never write production weights) ──────────────
os.makedirs(SCRATCH_ROOT, exist_ok=True)


def make_arm_data_dir(name):
    d = os.path.join(SCRATCH_ROOT, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)
    man = {}
    for fn in ("ravana_weights.pkl", "ravana_weights.db",
               "junk_classifier.json", "junk_labels.jsonl"):
        s = os.path.join(PROJ, "weights", fn)
        t = os.path.join(d, fn)
        if os.path.exists(s):
            shutil.copy2(s, t)
        man[fn] = {
            "src_sha256": sha256(s) if os.path.exists(s) else None,
            "dst_sha256": sha256(t) if os.path.exists(t) else None,
        }
    return d, man


def boot(data_dir):
    t0 = time.time()
    eng = CognitiveChatEngine(data_dir=data_dir, user_suffix="")
    secs = round(time.time() - t0, 2)
    assert os.path.normcase(data_dir) in os.path.normcase(eng._save_path), \
        "engine must write ONLY to its own scratch arm data_dir"
    return eng, secs


# ── Census helpers (read-only; exact identity, never substring) ──────────────
TYPED_REL = "is_a"                      # distinct type/category relation_type
SLOT_META_KEYS = ("type_label", "slot_label")
TAUGHT_TYPED_WORDS = {
    "mist", "hail", "drizzle", "election", "ceremony", "contract",
    "weather", "phenomenon", "civic", "event",
}


def _label(g, nid):
    try:
        nd = g.get_node(nid)
        return str(getattr(nd, "label", "") or "")
    except Exception:
        return ""


def _meta(e):
    """READ-ONLY copy for censuses. Never use this to stamp an edge: mutating
    the copy silently loses the write (instrument defect found in run 1)."""
    m = getattr(e, "source_metadata", None)
    return dict(m) if isinstance(m, dict) else {}


def _writable_meta(e):
    """The LIVE source_metadata dict, so a stamp actually lands on the edge."""
    m = getattr(e, "source_metadata", None)
    if isinstance(m, dict):
        return m
    if hasattr(e, "_ensure_source_metadata"):
        m = e._ensure_source_metadata()
        if isinstance(m, dict):
            return m
    m = {}
    try:
        e.source_metadata = m
    except Exception:
        pass
    return m


def _rec(g, sid, tid, e):
    m = _meta(e)
    return {
        "source": _label(g, sid), "target": _label(g, tid),
        "relation_type": str(getattr(e, "relation_type", "") or "").lower(),
        "edge_kind": str(m.get("edge_kind") or ""),
        "edge_source": str(m.get("source") or ""),
        "source_type": str(m.get("source_type") or ""),
        "extractor": str(m.get("extractor") or ""),
        "asserted_relation": str(m.get("asserted_relation") or ""),
        "type_label": str(m.get("type_label") or ""),
        "slot_label": str(m.get("slot_label") or ""),
        "scratch_writer": str(m.get("scratch_writer") or ""),
        "url": str(m.get("url") or ""),
        "trusted": bool(si._is_trusted_edge(e)),
        "confidence": round(float(getattr(e, "confidence", 0) or 0), 4),
    }


def iter_edges(eng):
    g = eng.graph
    return (g.edges.items() if isinstance(getattr(g, "edges", None), dict) else [])


def census(eng, cap=40):
    g = eng.graph
    rel_hist, trusted_hist = Counter(), Counter()
    typed_slot_edges, is_a_edges, new_taught_incident = [], [], []
    n_edges = 0
    for (sid, tid), e in iter_edges(eng):
        n_edges += 1
        r = _rec(g, sid, tid, e)
        rel_hist[r["relation_type"]] += 1
        if r["trusted"]:
            trusted_hist[r["relation_type"]] += 1
        if r["relation_type"] == TYPED_REL and r["type_label"]:
            typed_slot_edges.append(r)
        if r["relation_type"] == TYPED_REL:
            is_a_edges.append(r)
        if r["source"].lower() in TAUGHT_TYPED_WORDS or r["target"].lower() in TAUGHT_TYPED_WORDS:
            if r["edge_source"] == "conversation":
                new_taught_incident.append(r)
    return {
        "n_edges": n_edges,
        "n_nodes": len(g.nodes) if hasattr(g, "nodes") else None,
        "relation_histogram": dict(sorted(rel_hist.items(), key=lambda x: -x[1])),
        "trusted_count": sum(trusted_hist.values()),
        "trusted_relation_histogram": dict(sorted(trusted_hist.items(), key=lambda x: -x[1])),
        "is_a_edge_count": len(is_a_edges),
        "typed_slot_edge_count": len(typed_slot_edges),
        "typed_slot_edges": typed_slot_edges[:cap],
        "conversation_incident_edges": new_taught_incident[:cap],
    }


def typed_slot_triples(eng):
    """FULL set of trusted (subject, type_label, target) slot triples."""
    out = set()
    g = eng.graph
    for (sid, tid), e in iter_edges(eng):
        r = _rec(g, sid, tid, e)
        if r["trusted"] and r["relation_type"] == TYPED_REL and r["type_label"]:
            out.add((r["source"].lower(), r["type_label"].lower(), r["target"].lower()))
    return out


def trusted_triples(eng):
    out = set()
    g = eng.graph
    for (sid, tid), e in iter_edges(eng):
        r = _rec(g, sid, tid, e)
        if r["trusted"] and r["source"] and r["target"]:
            out.add((r["source"].lower(), r["relation_type"], r["target"].lower()))
    return out


# ── SCRATCH WRITE half ───────────────────────────────────────────────────────
# Natural-language parsers. Hedges, negation and questions are refused at the
# top-level call: an uncertain or interrogative utterance is not a source.
IS_A_RE = re.compile(
    r"^\s*(?:(?:a|an|the)\s+)?(?P<s>[a-z][a-z\-]+)\s+is\s+(?:a|an)\s+"
    r"(?:(?:kind|type|sort|form)\s+of\s+)?"
    r"(?P<o>[a-z][a-z\-]+(?:\s+[a-z][a-z\-]+)?)\s*[.!?]?\s*$", re.I)
PART_OF_RE = re.compile(
    r"^\s*(?:(?:a|an|the)\s+)?(?P<s>[a-z][a-z\-]+)\s+is\s+part\s+of\s+"
    r"(?:(?:a|an|the)\s+)?(?P<o>[a-z][a-z\-]+(?:\s+[a-z][a-z\-]+)?)\s*[.!?]?\s*$", re.I)
REFUSE_RE = re.compile(
    r"\b(not|never|no|might|may|maybe|perhaps|possibly|probably|arguably|"
    r"seems?|appears?|could|i think|i guess|not sure|unclear)\b|\bn['\u2019]t\b", re.I)


def parse_typed_assertion(text):
    """-> list of (family, subject, object, type_label, slot_label)."""
    t = str(text or "").strip()
    if not t or t.endswith("?") or REFUSE_RE.search(t):
        return []
    out = []
    m = IS_A_RE.match(t)
    if m:
        s, o = m.group("s").lower(), " ".join(m.group("o").lower().split())
        if s != o:
            # type/category relation_type is distinct from relation families;
            # type_label is the domain-general slot, slot_label its role name.
            out.append(("IS_A", s, o, o, "type"))
    m = PART_OF_RE.match(t)
    if m:
        s, o = m.group("s").lower(), " ".join(m.group("o").lower().split())
        if s != o:
            out.append(("PART_OF", s, o, "structured whole", "whole"))
    return out


def scratch_write_typed_assertion(eng, text):
    """Write ONE natural assertion through the production intake API.

    Same call shape as chain_walker._consolidate_asserted_non_causal_relations:
    _ensure_assertion_node -> Fact -> Provenance(source='conversation') ->
    intake.intake(fact, prov, salience=1.0). Only the local post-write stamp
    (distinct relation_type + type_label + slot_label) is new, and it is a
    scratch-only metadata write on the resulting real ConceptEdge.
    """
    parsed = parse_typed_assertion(text)
    rows = []
    if not parsed:
        return rows
    try:
        intake = eng._get_knowledge_intake()
    except Exception as ex:
        return [{"error": "intake unavailable: %s" % ex}]
    if intake is None:
        return [{"error": "intake unavailable"}]
    for fam, subj, obj, typelabel, slotlabel in parsed:
        row = {"family": fam, "subject": subj, "object": obj,
               "type_label": typelabel, "slot_label": slotlabel,
               "assertion": text}
        try:
            sid = eng._ensure_assertion_node(subj)
            oid = eng._ensure_assertion_node(obj)
            if sid is None or oid is None or sid == oid:
                row["outcome"] = "node_resolution_failed"
                rows.append(row)
                continue
            relation = "is_a" if fam == "IS_A" else "part_of"
            fact = Fact(subject=subj, relation=relation, obj=obj,
                        confidence=0.9, source_sentence=text)
            prov = Provenance(
                source="conversation", source_type="user", url="",
                extractor="scratch-tra-typed-assertion",
                operations=("taught-in-conversation", "asserted-relation:" + fam))
            v = intake.intake(fact, prov, salience=1.0)
            row["route"] = v.route
            row["reason"] = v.reason
            edge = eng.graph.get_edge(sid, oid)
            if edge is None:
                row["outcome"] = "no_edge_after_intake"
                rows.append(row)
                continue
            meta = _writable_meta(edge)          # LIVE dict, not a census copy
            # Defensive provenance restore (chain_walker.py:1116-1121 does the
            # same); never lowers or widens the trust predicate.
            if not meta.get("edge_kind"):
                meta["edge_kind"] = "curated"
            edge.relation_type = relation
            meta["asserted_relation"] = fam
            meta["type_label"] = typelabel
            meta["slot_label"] = slotlabel
            meta["scratch_writer"] = "tra_slot_write_read_experiment.py"
            row["outcome"] = "written"
            row["edge"] = _rec(eng.graph, sid, oid, edge)
        except Exception as ex:
            row["outcome"] = "exception"
            row["error"] = "%s: %s" % (type(ex).__name__, ex)
            row["traceback"] = traceback.format_exc()[-1200:]
            R["errors"].append({"stage": "scratch_write", "assertion": text,
                                "error": row["error"]})
        rows.append(row)
    return rows


# ── SCRATCH READ half: slot-keyed, congruence-floored, fail-closed ────────────
def slot_keyed_forced_choice(eng, subject, options, *, congruence_floor=0.20,
                             strict_margin=0.10, typed_rel=TYPED_REL):
    """Pure stateless forced-choice over TYPE SLOTS.

    A candidate can be selected ONLY if a provenance-trusted typed edge
    (relation_type == typed_rel, dedicated type_label stamp present) goes from
    `subject` to that exact slot label. Score = stored edge confidence -- an
    intake write signal -- NOT cosine to the label word. No item gold,
    world_model, type_sets, domain or far_domain is read anywhere.

    congruence floor   : top score must be >= congruence_floor
    strict margin      : top - runner-up must be >= strict_margin
    fail-closed        : otherwise return pick=None (never an unlicensed pick)
    """
    s = str(subject or "").strip().lower()
    opt = {}
    for o in (options or ()):
        key = str(o).strip().lower()
        if key:
            opt[key] = str(o)
    scores = {}
    g = eng.graph
    for (sid, tid), e in iter_edges(eng):
        r = _rec(g, sid, tid, e)
        if r["relation_type"] != typed_rel or not r["type_label"]:
            continue
        if not r["trusted"]:
            continue
        if r["source"].lower() != s:
            continue
        slot = r["type_label"].strip().lower()
        # The stamped slot must agree with the edge target's exact identity.
        if slot != r["target"].strip().lower():
            continue
        if slot not in opt:
            continue
        cur = scores.setdefault(slot, {"score": 0.0, "support": 0, "edges": []})
        cur["score"] = max(cur["score"], r["confidence"])
        cur["support"] += 1
        cur["edges"].append({"source": r["source"], "target": r["target"],
                             "edge_kind": r["edge_kind"],
                             "edge_source": r["edge_source"],
                             "confidence": r["confidence"]})
    eligible = {k: v for k, v in scores.items() if v["score"] >= congruence_floor}
    out = {
        "subject": subject, "options": list(options or ()),
        "all_slot_scores": {k: round(v["score"], 4) for k, v in scores.items()},
        "eligible": sorted(eligible.keys()),
        "congruence_floor": congruence_floor, "strict_margin": strict_margin,
    }
    if not eligible:
        out.update({"pick": None, "margin": None,
                    "reason": "no_trusted_typed_slot_above_floor"})
        return out
    ranked = sorted(eligible.items(), key=lambda kv: (-kv[1]["score"], kv[0]))
    top_k, top = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else None
    margin = round(top["score"] - (second["score"] if second else 0.0), 4)
    out["margin"] = margin
    out["winner_evidence"] = {"slot": top_k, "support": top["support"],
                              "edges": top["edges"][:8]}
    if margin < strict_margin:
        out.update({"pick": None, "reason": "strict_margin_not_met"})
        return out
    out.update({"pick": opt[top_k], "reason": "trusted_typed_slot_strict_winner"})
    return out


# ═════════════════════════════════════════════════════════════════════════════
# ARM 1 -- main typed arm: teach, save, cold reload, read probes
# ═════════════════════════════════════════════════════════════════════════════
TYPE_TEACH = [
    "mist is a weather phenomenon.",
    "hail is a weather phenomenon.",
    "drizzle is a weather phenomenon.",
    "election is a civic event.",
    "ceremony is a civic event.",
    "contract is a civic event.",
    "a wheel is part of a vehicle.",
    "a petal is part of a flower.",
    "a page is part of a book.",
    "a chord is part of a guitar.",
]
TYPED_OPTS = ["weather phenomenon", "civic event", "structured whole", "similar form"]
DISJOINT_OPTS = ["apple", "banana", "chair", "table"]

data_type, man_type = make_arm_data_dir("type_arm")
arm = {"data_dir": data_type, "store_copy": man_type,
       "teaching_assertions": TYPE_TEACH}
R["arms"]["type_arm"] = arm

eng, s = boot(data_type)
arm["boot1_s"] = s
log("TYPE_ARM BOOT1 %.2fs path=%s" % (s, eng._save_path))
base = census(eng)
base_typed = typed_slot_triples(eng)
base_tr = trusted_triples(eng)
arm["baseline"] = base
log("TYPE_ARM BASE edges=%d trusted=%d is_a=%d typed_slots=%d" % (
    base["n_edges"], base["trusted_count"], base["is_a_edge_count"],
    base["typed_slot_edge_count"]))

teach_rows = []
for text in TYPE_TEACH:
    row = {"assertion": text}
    try:
        row["production_process_turn_response"] = str(eng.process_turn(text))[:220]
    except Exception as ex:
        row["production_error"] = "%s: %s" % (type(ex).__name__, ex)
        R["errors"].append({"stage": "type_arm_process_turn", "assertion": text,
                            "error": row["production_error"]})
    row["scratch_write"] = scratch_write_typed_assertion(eng, text)
    teach_rows.append(row)
    log("TYPE_ARM TEACH %-36s -> %s" % (
        text, [(w.get("family"), w.get("route"), w.get("outcome")) for w in row["scratch_write"]]))
arm["teaching"] = teach_rows

post = census(eng)
post_typed = typed_slot_triples(eng)
post_tr = trusted_triples(eng)
arm["post_teach_in_process"] = post
new_typed = sorted(post_typed - base_typed)
new_tr = sorted(post_tr - base_tr)
arm["new_trusted_triples"] = [{"source": a, "relation_type": b, "target": c}
                              for a, b, c in new_tr]
arm["new_typed_slot_triples"] = [{"source": a, "type_label": b, "target": c}
                                 for a, b, c in new_typed]
log("TYPE_ARM POST edges=%d trusted=%d is_a=%d typed_slots=%d new_typed=%d" % (
    post["n_edges"], post["trusted_count"], post["is_a_edge_count"],
    post["typed_slot_edge_count"], len(new_typed)))

try:
    arm["save"] = {"ok": True, "returned": str(eng.save())[:300]}
    log("TYPE_ARM SAVED")
except Exception as ex:
    arm["save"] = {"ok": False, "error": "%s: %s" % (type(ex).__name__, ex),
                   "traceback": traceback.format_exc()[-1200:]}
    log("TYPE_ARM SAVE FAILED", arm["save"]["error"])
del eng

eng, s = boot(data_type)
arm["boot2_s"] = s
cold = census(eng)
cold_typed = typed_slot_triples(eng)
arm["post_cold_reload"] = cold
survivors = sorted(set(new_typed) & cold_typed)
lost = sorted(set(new_typed) - cold_typed)
arm["cold_reload"] = {
    "n_new_typed": len(new_typed), "n_survivors": len(survivors),
    "retention": round(len(survivors) / len(new_typed), 4) if new_typed else None,
    "survivors": [{"source": a, "type_label": b, "target": c} for a, b, c in survivors],
    "lost": [{"source": a, "type_label": b, "target": c} for a, b, c in lost],
}
log("TYPE_ARM COLD edges=%d trusted=%d typed_slots=%d survivors=%d/%d" % (
    cold["n_edges"], cold["trusted_count"], cold["typed_slot_edge_count"],
    len(survivors), len(new_typed)))

# Read probes -- ALL inputs and outputs printed before any aggregate is read.
PROBES = [
    {"id": "positive_mist", "subject": "mist", "options": TYPED_OPTS},
    {"id": "positive_election", "subject": "election", "options": TYPED_OPTS},
    {"id": "permutation_A", "subject": "mist",
     "options": ["similar form", "structured whole", "civic event", "weather phenomenon"]},
    {"id": "permutation_B", "subject": "mist",
     "options": ["civic event", "weather phenomenon", "similar form", "structured whole"]},
    {"id": "part_of_subject_no_typed_slot", "subject": "wheel", "options": TYPED_OPTS},
    {"id": "disjoint_options_C10", "subject": "mist", "options": DISJOINT_OPTS},
    {"id": "unrelated_subject", "subject": "zeppelin", "options": TYPED_OPTS},
    {"id": "no_cross_turn_state_repeat", "subject": "mist", "options": TYPED_OPTS},
]
probe_rows = []
for p in PROBES:
    out = slot_keyed_forced_choice(eng, p["subject"], p["options"])
    row = {"id": p["id"], "subject": p["subject"], "options": p["options"], **out}
    probe_rows.append(row)
    log("READ_PROBE %-30s subject=%-9s scores=%s eligible=%s pick=%r margin=%s reason=%s" % (
        p["id"], p["subject"], out["all_slot_scores"], out["eligible"],
        out["pick"], out["margin"], out["reason"]))
arm["read_probes"] = probe_rows
_byid = {r["id"]: r for r in probe_rows}
_typed_picks = {k: _byid[k]["pick"] for k in
                ("positive_mist", "permutation_A", "permutation_B",
                 "no_cross_turn_state_repeat")}
arm["read_controls"] = {
    "permutation_stable": len(set(_typed_picks.values())) == 1,
    "picks": _typed_picks,
    "disjoint_pick_C10": _byid["disjoint_options_C10"]["pick"],
    "disjoint_fail_closed": _byid["disjoint_options_C10"]["pick"] is None,
    "unrelated_subject_pick": _byid["unrelated_subject"]["pick"],
    "unrelated_fail_closed": _byid["unrelated_subject"]["pick"] is None,
    "part_of_subject_pick": _byid["part_of_subject_no_typed_slot"]["pick"],
    "no_cross_turn_state": (
        json.dumps({k: v for k, v in _byid["positive_mist"].items() if k != "id"},
                   sort_keys=True, default=str) ==
        json.dumps({k: v for k, v in _byid["no_cross_turn_state_repeat"].items()
                    if k != "id"}, sort_keys=True, default=str)),
}

# Production comparison on the SAME cold engine: does the shipped opt-in path
# pass C10? (Read-only comparison; no production file or gate is edited.)
os.environ["RAVANA_SCHEMA"] = "1"
try:
    ans = eng.process_turn(
        "Considering the examples above, what is true of mist? "
        "Options: apple, banana, chair, table. Answer briefly.")
    txt = str(ans)
    t = txt.strip().lower().strip(" .;:!?\"'")
    pick = t if t in {o.lower() for o in DISJOINT_OPTS} else None
    arm["production_flag_on_disjoint_comparison"] = {
        "schema_flag": "1", "response": txt[:600], "graded_pick": pick,
        "C10_fail_closed": pick is None,
    }
except Exception as ex:
    arm["production_flag_on_disjoint_comparison"] = {
        "error": "%s: %s" % (type(ex).__name__, ex)}
finally:
    os.environ.pop("RAVANA_SCHEMA", None)
log("PRODUCTION_FLAG_ON_DISJOINT", arm["production_flag_on_disjoint_comparison"])
del eng


# ═════════════════════════════════════════════════════════════════════════════
# ARM 2 -- relation-only control: relation edges, NO typed slots
# ═════════════════════════════════════════════════════════════════════════════
REL_TEACH = [
    "drizzle causes mist.", "storm causes a blizzard.",
    "a wheel is part of a vehicle.", "a petal is part of a flower.",
    "dawn precedes sunrise.", "winter comes before spring.",
    "mist resembles fog.", "a wheel is similar to a gear.",
    "hail causes rain.", "a page is part of a book.",
]
data_rel, man_rel = make_arm_data_dir("relation_arm")
arm2 = {"data_dir": data_rel, "store_copy": man_rel,
        "teaching_assertions": REL_TEACH}
R["arms"]["relation_arm"] = arm2
eng, s = boot(data_rel)
arm2["boot_s"] = s
b2 = census(eng)
b2_typed = typed_slot_triples(eng)
b2_tr = trusted_triples(eng)
for text in REL_TEACH:
    try:
        eng.process_turn(text)
    except Exception as ex:
        R["errors"].append({"stage": "relation_arm_process_turn", "assertion": text,
                            "error": "%s: %s" % (type(ex).__name__, ex)})
p2 = census(eng)
p2_typed = typed_slot_triples(eng)
p2_tr = trusted_triples(eng)
arm2["baseline"] = b2
arm2["post_teach"] = p2
arm2["new_trusted_triple_count"] = len(p2_tr - b2_tr)
arm2["new_typed_slot_triples"] = [
    {"source": a, "type_label": b, "target": c} for a, b, c in sorted(p2_typed - b2_typed)]
arm2["new_typed_slot_count"] = len(p2_typed - b2_typed)
log("RELATION_ARM edges=%d trusted=%d new_trusted=%d new_typed_slots=%d" % (
    p2["n_edges"], p2["trusted_count"], arm2["new_trusted_triple_count"],
    arm2["new_typed_slot_count"]))
del eng


# ═════════════════════════════════════════════════════════════════════════════
# ARM 3 -- untrusted typed edge cannot seed
# ═════════════════════════════════════════════════════════════════════════════
data_u, man_u = make_arm_data_dir("untrusted_arm")
arm3 = {"data_dir": data_u, "store_copy": man_u}
R["arms"]["untrusted_arm"] = arm3
eng, s = boot(data_u)
arm3["boot_s"] = s
# (i) a real trusted typed edge proves the scorer is alive in this arm.
arm3["trusted_control_write"] = scratch_write_typed_assertion(
    eng, "hail is a weather phenomenon.")
# (ii) a forged typed edge with inferential provenance: correct relation_type,
# correct type_label, but edge_kind='inferred' and source!='conversation'.
_zsub = eng._ensure_assertion_node("zeppelin")
_zobj = eng._ensure_assertion_node("weather phenomenon")
_forged = {"outcome": "not_attempted"}
if _zsub is not None and _zobj is not None:
    intake = eng._get_knowledge_intake()
    fact = Fact(subject="zeppelin", relation="is_a", obj="weather phenomenon",
                confidence=0.95, source_sentence="forged untrusted control")
    prov = Provenance(source="inference", source_type="inference", url="",
                      extractor="scratch-forged-untrusted",
                      operations=("forged-control",))
    v = intake.intake(fact, prov, salience=1.0)
    edge = eng.graph.get_edge(_zsub, _zobj)
    if edge is not None:
        edge.relation_type = "is_a"
        m = _writable_meta(edge)
        m["type_label"] = "weather phenomenon"
        m["slot_label"] = "type"
        m["scratch_writer"] = "scratch-forged-untrusted"
        _forged = {"outcome": "written", "route": v.route,
                   "edge": _rec(eng.graph, _zsub, _zobj, edge),
                   "trusted_by_production_predicate": bool(si._is_trusted_edge(edge))}
    else:
        _forged = {"outcome": "no_edge_after_intake", "route": v.route}
arm3["forged_untrusted_edge"] = _forged
arm3["census"] = census(eng)
arm3["scorer_trusted_subject"] = slot_keyed_forced_choice(
    eng, "hail", TYPED_OPTS)
arm3["scorer_untrusted_subject"] = slot_keyed_forced_choice(
    eng, "zeppelin", TYPED_OPTS)
log("UNTRUSTED_ARM forged=%s trusted_pick=%r untrusted_pick=%r" % (
    _forged.get("outcome"), arm3["scorer_trusted_subject"]["pick"],
    arm3["scorer_untrusted_subject"]["pick"]))
del eng


# ═════════════════════════════════════════════════════════════════════════════
# ARM 4 -- instrument check: do the strict-margin and congruence-floor
# fail-closed branches actually fire? (scratch-store mutation check, analogous
# to M-A/M-B; this validates the SCORER, not a capability claim.)
# ═════════════════════════════════════════════════════════════════════════════
data_m, man_m = make_arm_data_dir("margin_arm")
arm4 = {"data_dir": data_m, "store_copy": man_m,
        "note": "metadata-only confidence mutation on a fresh scratch copy; "
                "instrument check, NOT a capability or learning result"}
R["arms"]["margin_arm"] = arm4
eng, s = boot(data_m)
arm4["boot_s"] = s
arm4["writes"] = (scratch_write_typed_assertion(eng, "mist is a weather phenomenon.")
                  + scratch_write_typed_assertion(eng, "mist is a civic event."))


def _mist_typed_edges(eng):
    out = []
    for nid in (eng._concept_keywords.get("mist", []) or []):
        for tid, e in (eng.graph.get_outgoing(nid) or []):
            r = _rec(eng.graph, nid, tid, e)
            if r["relation_type"] == "is_a" and r["type_label"]:
                out.append(e)
    return out


_marg_opts = ["weather phenomenon", "civic event", "structured whole", "similar form"]
_edges = _mist_typed_edges(eng)
arm4["n_typed_edges_found"] = len(_edges)
for e in _edges:
    e.confidence = 0.50
arm4["tie_probe"] = slot_keyed_forced_choice(eng, "mist", _marg_opts)
for e in _edges:
    e.confidence = 0.10
arm4["below_floor_probe"] = slot_keyed_forced_choice(eng, "mist", _marg_opts)
log("MARGIN_ARM n_edges=%d tie_pick=%r tie_reason=%s below_floor_pick=%r below_floor_reason=%s" % (
    len(_edges), arm4["tie_probe"]["pick"], arm4["tie_probe"]["reason"],
    arm4["below_floor_probe"]["pick"], arm4["below_floor_probe"]["reason"]))
del eng


# ── Gates / verdicts ─────────────────────────────────────────────────────────
R["gates"] = {
    "C9_cold_reload_retention_ge_0.80": {
        "value": arm["cold_reload"]["retention"],
        "pass": (arm["cold_reload"]["retention"] is not None
                 and arm["cold_reload"]["retention"] >= 0.80),
    },
    "C10_disjoint_control_returns_None": {
        "scratch_pick": arm["read_controls"]["disjoint_pick_C10"],
        "pass": arm["read_controls"]["disjoint_fail_closed"],
        "production_flag_on_pick": arm.get(
            "production_flag_on_disjoint_comparison", {}).get("graded_pick"),
    },
    "C12_provenance_integrity": {
        "new_typed_edges": [e for e in post["typed_slot_edges"]
                            if e["scratch_writer"] == "tra_slot_write_read_experiment.py"],
        "pass": all(
            e["edge_source"] == "conversation"
            and e["edge_kind"] in si.TRUSTED_EDGE_KINDS
            and e["source_type"] == "user"
            and e["trusted"]
            and not e["url"]
            for e in post["typed_slot_edges"]
            if e["scratch_writer"] == "tra_slot_write_read_experiment.py"),
    },
    "control_a_relation_only_no_typed_slots": {
        "new_trusted_relation_edges": arm2["new_trusted_triple_count"],
        "new_typed_slots": arm2["new_typed_slot_count"],
        "pass": arm2["new_trusted_triple_count"] > 0
                and arm2["new_typed_slot_count"] == 0,
    },
    "control_b_type_assertions_make_typed_slots": {
        "new_typed_slots": len(new_typed), "pass": len(new_typed) > 0,
    },
    "control_d_untrusted_typed_edge_cannot_seed": {
        "forged_trusted": _forged.get("trusted_by_production_predicate"),
        "untrusted_pick": arm3["scorer_untrusted_subject"]["pick"],
        "trusted_control_pick": arm3["scorer_trusted_subject"]["pick"],
        "pass": (not _forged.get("trusted_by_production_predicate", True)
                 and arm3["scorer_untrusted_subject"]["pick"] is None
                 and arm3["scorer_trusted_subject"]["pick"] is not None),
    },
    "control_e_cold_reload": {
        "before": len(new_typed), "after": len(survivors),
        "pass": len(survivors) == len(new_typed) and len(new_typed) > 0,
    },
    "control_f_no_cross_turn_state": {
        "identical": arm["read_controls"]["no_cross_turn_state"],
        "pass": bool(arm["read_controls"]["no_cross_turn_state"]),
    },
    "instrument_strict_margin_fires": {
        "tie_pick": arm4["tie_probe"]["pick"],
        "tie_reason": arm4["tie_probe"]["reason"],
        "pass": (arm4["n_typed_edges_found"] >= 2
                 and arm4["tie_probe"]["pick"] is None
                 and arm4["tie_probe"]["reason"] == "strict_margin_not_met"),
    },
    "instrument_congruence_floor_fires": {
        "below_floor_pick": arm4["below_floor_probe"]["pick"],
        "below_floor_reason": arm4["below_floor_probe"]["reason"],
        "pass": (arm4["below_floor_probe"]["pick"] is None
                 and arm4["below_floor_probe"]["reason"]
                 == "no_trusted_typed_slot_above_floor"),
    },
    "permutation_stability": {
        "picks": _typed_picks,
        "pass": arm["read_controls"]["permutation_stable"],
    },
}
R["summary"] = {
    "teaching_turns_type_arm": len(TYPE_TEACH),
    "teaching_turns_relation_arm": len(REL_TEACH),
    "engine_boots": 5,
    "type_arm_new_typed_slots": len(new_typed),
    "type_arm_cold_survivors": len(survivors),
    "relation_arm_new_typed_slots": arm2["new_typed_slot_count"],
    "relation_arm_new_trusted_edges": arm2["new_trusted_triple_count"],
    "scratch_disjoint_pick_C10": arm["read_controls"]["disjoint_pick_C10"],
    "production_flag_on_disjoint_pick": arm.get(
        "production_flag_on_disjoint_comparison", {}).get("graded_pick"),
    "positive_pick": _byid["positive_mist"]["pick"],
    "unrelated_pick": _byid["unrelated_subject"]["pick"],
    "all_gates_pass": all(v["pass"] for v in R["gates"].values()),
    "errors_count": len(R["errors"]),
}
log("GATES", json.dumps({k: v["pass"] for k, v in R["gates"].items()}, indent=2))
log("SUMMARY", json.dumps(R["summary"], indent=2))

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=str)
log("WROTE", RESULTS)
log("EXIT=0")
