#!/usr/bin/env python3
"""TRA experiment 1 — typed role write/read feasibility on the UNMODIFIED engine.

Scratch only. No file under ravana/src, ravana_ml/src, scripts or tests is
edited. The engine imported here is the production tree at PROJ (dbed80d);
this worktree holds only this script, its results and its doc.

Design (bounded): 1 copied scratch store, 2 engine boots, <=10 teaching turns.
  boot 1: cold baseline census -> teach via process_turn -> in-process census
          -> eng.save()
  boot 2: cold reload census -> production forced-choice typed-slot query
          -> scratch-only read-side probe (pure helpers on real edges)

Set RAVANA_OFFLINE=1, PYTHONHASHSEED=0. RAVANA_SCHEMA stays unset for the
default production query; the flag-on variant is a supported production
config (env var), not a monkeypatch. The ONLY monkeypatch in this file is an
explicitly scratch-only read-side adapter at the very end.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
import traceback
from collections import Counter

# ── Paths / environment ──────────────────────────────────────────────────────
PROJ = "C:/Users/Likhith/Documents/projects/ravana"          # unmodified production tree
WORKTREE = "C:/Users/Likhith/Documents/projects/ravana_tra_exp"
RESULTS = os.path.join(WORKTREE, "scratch", "tra_write_read_results.json")
DATA_DIR = os.environ.get(
    "TRA_DATA_DIR",
    "C:/Users/Likhith/AppData/Local/hermes/cache/scratch/tra_write_read_data",
)

os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ.pop("RAVANA_SCHEMA", None)          # default production read path
# Production imports from the UNMODIFIED tree, never from this worktree.
for _p in (PROJ, os.path.join(PROJ, "scripts"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

R = {"experiment": "TRA experiment 1 — typed role write/read feasibility",
     "project_tree": PROJ, "worktree": WORKTREE, "data_dir": DATA_DIR,
     "stages": {}, "errors": []}

def log(*a):
    print(*a, flush=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

# ── Make a real scratch copy of the store (never write to production weights) ─
src_weights = os.path.join(PROJ, "weights")
os.makedirs(DATA_DIR, exist_ok=True)
copy_manifest = {}
for name in ("ravana_weights.pkl", "ravana_weights.db",
             "junk_classifier.json", "junk_labels.jsonl"):
    s = os.path.join(src_weights, name)
    d = os.path.join(DATA_DIR, name)
    if os.path.exists(s) and not os.path.exists(d):
        shutil.copy2(s, d)
    copy_manifest[name] = {"src_exists": os.path.exists(s),
                           "dst_exists": os.path.exists(d),
                           "src_sha256": sha256(s) if os.path.exists(s) else None,
                           "dst_sha256": sha256(d) if os.path.exists(d) else None}
R["store_copy"] = copy_manifest
log("STORE_COPY", {k: (v["src_exists"], v["dst_exists"]) for k, v in copy_manifest.items()})

# ── Imports (assert they resolve to the production tree) ────────────────────
from ravana.chat.engine import CognitiveChatEngine            # noqa: E402
from ravana.chat import schema_induction as si                # noqa: E402
from ravana.chat import chain_walker as cw                     # noqa: E402
from ravana.chat import response_gen as rg                    # noqa: E402

resolved = {"engine": sys.modules[CognitiveChatEngine.__module__].__file__,
            "schema_induction": si.__file__,
            "chain_walker": cw.__file__,
            "response_gen": rg.__file__}
R["module_resolution"] = resolved
for k, v in resolved.items():
    log("MODULE", k, v)
assert all(os.path.normcase(PROJ) in os.path.normcase(v) for v in resolved.values()), \
    "imports must resolve to the UNMODIFIED production tree, not this worktree"

# Static write-path surface: which declarative phrases have a provenanced write?
_phrases = [str(p).lower()
            for _phs in cw.ChainWalkerMixin._ASSERTED_RELATION_PHRASES.values()
            for p, _ in _phs]
R["write_surface"] = {
    "consolidator_called_by_process_turn": "engine.py:10858-10865 _consolidate_asserted_relations",
    "relation_phrases": {fam: [p for p, _ in phs]
                         for fam, phs in cw.ChainWalkerMixin._ASSERTED_RELATION_PHRASES.items()},
    # EXACT phrase equality, not substring: 'is a component of' contains the
    # substring 'is a' but is a PART_OF phrase, not an is_a type assertion.
    "is_a_phrase_present": any(
        p in ("is", "is a", "is an", "is a kind of", "is a type of")
        for p in _phrases),
    "phrases_with_is_a_substring": [p for p in _phrases if "is a" in p],
    "note": "chain_walker.py:946-955 phrase map has PART_OF/PRECEDES/RESEMBLES "
            "and engine.py:10850-10865 wires only this consolidator; no standalone "
            "is_a type-assertion phrase is present.",
}
log("WRITE_SURFACE", json.dumps(R["write_surface"]))

# ── Census helpers ───────────────────────────────────────────────────────────
TYPE_RELS = ("is_a", "has_property", "property", "part_of", "located_in")
ROLE_RELS = ("causal", "temporal", "analogical")   # effect / later / similar proxies
SLOT_TOKENS = ("effect", "whole", "later", "similar", "earlier", "structured",
               "weather phenomenon", "civic event", "type_slot", "role_slot")
SLOT_META_KEYS = ("type_label", "slot", "slot_label", "type_slot", "role_slot")
TAUGHT_WORDS = {
    "mist", "weather", "phenomenon", "election", "civic", "event",
    "wheel", "vehicle", "petal", "flower", "drizzle", "storm", "blizzard",
    "dawn", "sunrise", "winter", "spring", "fog", "gear",
}

def _label(g, nid):
    try:
        nd = g.get_node(nid)
        return str(getattr(nd, "label", "") or "")
    except Exception:
        return ""

def census(eng):
    """Read-only census of type-bearing edges on the LIVE graph."""
    g = eng.graph
    rel_hist = Counter()
    trusted_hist = Counter()
    type_edges = []
    trusted_edges = []
    incident = []
    slot_evidence = []
    n_edges = 0
    for (sid, tid), e in (g.edges.items() if isinstance(getattr(g, "edges", None), dict) else []):
        n_edges += 1
        rel = str(getattr(e, "relation_type", "") or "").lower()
        rel_hist[rel] += 1
        meta = getattr(e, "source_metadata", None)
        meta = dict(meta) if isinstance(meta, dict) else {}
        trusted = bool(si._is_trusted_edge(e))
        a, b = _label(g, sid), _label(g, tid)
        conf = round(float(getattr(e, "confidence", 0) or 0), 3)
        rec = {"source": a, "target": b, "relation_type": rel,
               "edge_kind": str(meta.get("edge_kind") or ""),
               "edge_source": str(meta.get("source") or ""),
               "source_type": str(meta.get("source_type") or ""),
               "extractor": str(meta.get("extractor") or ""),
               "asserted_relation": str(meta.get("asserted_relation") or ""),
               "operations": list(meta.get("operations") or []),
               "trusted": trusted, "confidence": conf}
        if trusted:
            trusted_hist[rel] += 1
            if len(trusted_edges) < 60:
                trusted_edges.append(rec)
        if rel in TYPE_RELS:
            type_edges.append(rec)
        if a.lower() in TAUGHT_WORDS or b.lower() in TAUGHT_WORDS:
            incident.append(rec)
        # Slot evidence MUST come from a dedicated type-slot metadata field.
        # A substring scan over all metadata is a known false-positive probe
        # (e.g. a URL containing 'effects' matches token 'effect'): never use it.
        for key in SLOT_META_KEYS:
            val = meta.get(key)
            if val is None:
                continue
            vals = list(val) if isinstance(val, (list, tuple)) else [val]
            for v in vals:
                if str(v).strip().lower() in SLOT_TOKENS:
                    slot_evidence.append({"token": str(v).strip().lower(),
                                          "metadata_key": key, "edge": rec})
                    break
    return {
        "n_edges": n_edges,
        "n_nodes": len(g.nodes) if hasattr(g, "nodes") else None,
        "relation_histogram": dict(sorted(rel_hist.items(), key=lambda x: -x[1])),
        "trusted_count": sum(trusted_hist.values()),
        "trusted_relation_histogram": dict(sorted(trusted_hist.items(), key=lambda x: -x[1])),
        "type_relation_count": len(type_edges),
        "type_relation_edges": type_edges[:60],
        "trusted_edges_sample": trusted_edges,
        "taught_word_incident_edges": incident[:120],
        "taught_word_incident_count": len(incident),
        "slot_label_evidence_count": len(slot_evidence),
        "slot_label_evidence": slot_evidence[:40],
    }

def edge_triples(c):
    """Trusted triples keyed by labels (stable across save/reload); sample only."""
    return {(e["source"].lower(), e["relation_type"], e["target"].lower())
            for e in c["trusted_edges_sample"] if e["trusted"]}

def all_trusted_triples(eng):
    """FULL set of trusted (source_label, relation_type, target_label) triples.

    The JSON sample is capped for readability; before/after/survivor counts must
    use the full set or a taught edge beyond the sample cap is missed (skill §1:
    count what is stored, not a truncated view)."""
    out = set()
    g = eng.graph
    for (sid, tid), e in (g.edges.items() if isinstance(getattr(g, "edges", None), dict) else []):
        if not si._is_trusted_edge(e):
            continue
        a, b = _label(g, sid).lower(), _label(g, tid).lower()
        if a and b:
            out.add((a, str(getattr(e, "relation_type", "") or "").lower(), b))
    return out

# ── BOOT 1: baseline census, teach, in-process census, save ─────────────────
TEACH = [
    ("mist is a weather phenomenon.", "IS_A"),
    ("election is a civic event.", "IS_A"),
    ("a wheel is part of a vehicle.", "PART_OF"),
    ("a petal is part of a flower.", "PART_OF"),
    ("drizzle causes mist.", "CAUSE"),
    ("storm causes a blizzard.", "CAUSE"),
    ("dawn precedes sunrise.", "PRECEDES"),
    ("winter comes before spring.", "PRECEDES"),
    ("mist resembles fog.", "RESEMBLES"),
    ("a wheel is similar to a gear.", "RESEMBLES"),
]

t0 = time.time()
eng = CognitiveChatEngine(data_dir=DATA_DIR, user_suffix="")
boot1_s = round(time.time() - t0, 1)
R["stages"]["boot1_s"] = boot1_s
log("BOOT1 %.1fs save_path=%s" % (boot1_s, eng._save_path))
assert os.path.normcase(DATA_DIR) in os.path.normcase(eng._save_path), \
    "engine must write ONLY to the scratch data_dir"

base = census(eng)
base_tr = all_trusted_triples(eng)
R["stages"]["baseline"] = base
log("BASELINE edges=%d trusted=%d type_rel=%d hist=%s" % (
    base["n_edges"], base["trusted_count"], base["type_relation_count"],
    base["relation_histogram"]))

taught = []
for text, fam in TEACH:
    rec = {"turn": text, "declared_family": fam}
    try:
        ans = eng.process_turn(text)
        rec["response"] = str(ans)[:400]
    except Exception as ex:
        rec["error"] = "%s: %s" % (type(ex).__name__, ex)
        rec["traceback"] = traceback.format_exc()[-1500:]
        R["errors"].append({"stage": "teach", "turn": text, "error": rec["error"]})
    taught.append(rec)
    log("TEACH %-38s -> %s" % (text, str(rec.get("response") or rec.get("error") or "")[:120]))
R["stages"]["teaching"] = {"turns": taught, "n_turns": len(TEACH)}

post = census(eng)
post_tr = all_trusted_triples(eng)
R["stages"]["post_teach_in_process"] = post
log("POST_TEACH edges=%d trusted=%d type_rel=%d" % (
    post["n_edges"], post["trusted_count"], post["type_relation_count"]))
new_after_teach = sorted(post_tr - base_tr)
R["stages"]["new_trusted_after_teach"] = [
    {"source": s, "relation_type": r, "target": t} for s, r, t in new_after_teach]
log("NEW_TRUSTED_AFTER_TEACH %d %s" % (len(new_after_teach), new_after_teach))

try:
    saved_to = eng.save()
    R["stages"]["save"] = {"ok": True, "returned": str(saved_to)[:300]}
    log("SAVED", saved_to)
except Exception as ex:
    R["stages"]["save"] = {"ok": False, "error": "%s: %s" % (type(ex).__name__, ex),
                           "traceback": traceback.format_exc()[-1500:]}
    log("SAVE FAILED", R["stages"]["save"]["error"])
del eng

# ── BOOT 2: cold reload, census, production query, scratch-only probe ───────
t0 = time.time()
eng = CognitiveChatEngine(data_dir=DATA_DIR, user_suffix="")
boot2_s = round(time.time() - t0, 1)
R["stages"]["boot2_s"] = boot2_s
log("BOOT2 %.1fs save_path=%s" % (boot2_s, eng._save_path))

cold = census(eng)
cold_tr = all_trusted_triples(eng)
R["stages"]["post_cold_reload"] = cold
log("COLD edges=%d trusted=%d type_rel=%d" % (
    cold["n_edges"], cold["trusted_count"], cold["type_relation_count"]))
survivors = sorted(set(new_after_teach) & cold_tr)
R["stages"]["cold_reload_survivors"] = {
    "n_new_after_teach": len(new_after_teach),
    "n_survivors": len(survivors),
    "survivors": [{"source": s, "relation_type": r, "target": t} for s, r, t in survivors],
    "lost": [{"source": s, "relation_type": r, "target": t}
             for s, r, t in sorted(set(new_after_teach) - cold_tr)],
}
log("SURVIVORS %d/%d" % (len(survivors), len(new_after_teach)))

# ── Production read side — forced-choice typed-slot query, NO monkeypatch ───
SLOTS = ["effect", "whole", "later", "similar"]

def grade_emission(text, options):
    """Strict: accept only when the WHOLE (echo-stripped) emission is one option.
    This is the skill's 'bare option token IS a forced-choice answer' rule:
    never accept a candidate merely echoed inside a longer question."""
    raw = str(text or "")
    t = raw.strip().lower().strip(" .;:!?\"'")
    import re
    for frame in (r"^you mentioned[:\s]*", r"^i said[:\s]*", r"^as noted[:\s]*"):
        t = re.sub(frame, "", t).strip()
    quoted = re.findall(r"[\"'\u201c\u201d]([^\"'\u201c\u201d]+)[\"'\u201c\u201d]", raw)
    picked = None
    if t in {o.lower() for o in options}:
        picked = t
    return {"parseable": picked is not None, "picked": picked,
            "echoed_quotes": quoted[:5], "raw": raw[:800]}

queries = [
    {"id": "typed_slot_default",
     "text": "Considering the examples above, what is true of thaw? "
             "Options: effect, whole, later, similar. Answer briefly.",
     "options": SLOTS, "flag": None},
    {"id": "object_candidates_default",
     "text": "Considering the examples above, what is true of thaw? "
             "Options: mist, vehicle, spring, fog. Answer briefly.",
     "options": ["mist", "vehicle", "spring", "fog"], "flag": None},
    {"id": "typed_slot_schema_flag_on",
     "text": "Considering the examples above, what is true of thaw? "
             "Options: effect, whole, later, similar. Answer briefly.",
     "options": SLOTS, "flag": "1"},
    # Falsifier for the flag-on pick: if 'effect' is really a typed-slot
    # selection it must NOT follow option position. TRA spec C3 / P6.
    {"id": "typed_slot_flag_on_permuted_A",
     "text": "Considering the examples above, what is true of thaw? "
             "Options: similar, later, whole, effect. Answer briefly.",
     "options": ["similar", "later", "whole", "effect"], "flag": "1"},
    {"id": "typed_slot_flag_on_permuted_B",
     "text": "Considering the examples above, what is true of thaw? "
             "Options: later, effect, similar, whole. Answer briefly.",
     "options": ["later", "effect", "similar", "whole"], "flag": "1"},
    {"id": "object_flag_on_permuted",
     "text": "Considering the examples above, what is true of thaw? "
             "Options: fog, spring, vehicle, mist. Answer briefly.",
     "options": ["fog", "spring", "vehicle", "mist"], "flag": "1"},
    # C10 type-contrast sanity: an UNRELATED unseen subject must not keep
    # emitting 'effect'; a disjoint option set must fail closed.
    {"id": "typed_slot_flag_on_unrelated_subject",
     "text": "Considering the examples above, what is true of zeppelin? "
             "Options: effect, whole, later, similar. Answer briefly.",
     "options": SLOTS, "flag": "1"},
    {"id": "typed_slot_flag_on_disjoint_options",
     "text": "Considering the examples above, what is true of thaw? "
             "Options: apple, banana, chair, table. Answer briefly.",
     "options": ["apple", "banana", "chair", "table"], "flag": "1"},
]
qrows = []
for q in queries:
    if q["flag"] is None:
        os.environ.pop("RAVANA_SCHEMA", None)
    else:
        os.environ["RAVANA_SCHEMA"] = q["flag"]
    row = {"id": q["id"], "query": q["text"], "options": q["options"],
           "schema_flag": os.environ.get("RAVANA_SCHEMA", "unset")}
    try:
        ans = eng.process_turn(q["text"])
        row["response"] = str(ans)[:800]
    except Exception as ex:
        row["error"] = "%s: %s" % (type(ex).__name__, ex)
        row["traceback"] = traceback.format_exc()[-1200:]
    row["grade"] = grade_emission(row.get("response", ""), q["options"])
    qrows.append(row)
    log("QUERY %s parseable=%s picked=%s raw=%s" % (
        q["id"], row["grade"]["parseable"], row["grade"]["picked"],
        repr(row.get("response", ""))[:160]))
os.environ.pop("RAVANA_SCHEMA", None)
_typed = {q["id"]: q for q in qrows if q["id"].startswith("typed_slot")}
_perm = [q for q in qrows if "permuted" in q["id"] and q["id"].startswith("typed_slot")]
_picks = {q["id"]: q["grade"]["picked"] for q in _perm}
_first = {q["id"]: q["options"][0] for q in _perm}
_parsed = [pid for pid, p in _picks.items() if p is not None]
R["stages"]["production_query"] = {
    "path": "eng.process_turn (unmodified); typed-slot forced choice",
    "queries": qrows,
    "default_path_parseable": any(
        q["schema_flag"] == "unset" and q["id"].startswith("typed_slot")
        and q["grade"]["parseable"] for q in qrows),
    "flag_on_parseable": any(
        q["id"] == "typed_slot_schema_flag_on" and q["grade"]["parseable"] for q in qrows),
    "typed_slot_selection_expressible_on_default_path": any(
        q["schema_flag"] == "unset" and q["id"].startswith("typed_slot")
        and q["grade"]["parseable"] for q in qrows),
    "permutation_control": {
        "picks": _picks,
        "first_options": _first,
        "n_parseable": len(_parsed),
        "picks_follow_first_option": all(
            _picks[pid] == _first[pid] for pid in _parsed) if _parsed else None,
        "picks_position_invariant": (
            len({_picks[pid] for pid in _parsed}) == 1) if len(_parsed) > 1 else None,
        "verdict": ("POSITION_BIAS — pick follows option order, not a typed slot"
                    if _parsed and all(_picks[pid] == _first[pid] for pid in _parsed)
                    else "STABLE_ACROSS_PERMUTATIONS" if len(_parsed) > 1
                    else "INCONCLUSIVE — too few parseable decisions"),
    },
}
log("PERMUTATION_CONTROL", json.dumps(R["stages"]["production_query"]["permutation_control"]))
_byid = {q["id"]: q for q in qrows}
R["stages"]["production_query"]["type_contrast_control"] = {
    "unrelated_subject_pick": _byid["typed_slot_flag_on_unrelated_subject"]["grade"]["picked"],
    "disjoint_options_pick": _byid["typed_slot_flag_on_disjoint_options"]["grade"]["picked"],
    "verdict": (
        "CONSTANT ATTRACTOR — 'effect' emitted for an unrelated subject too"
        if (_byid["typed_slot_schema_flag_on"]["grade"]["picked"] or "") == "effect"
           and (_byid["typed_slot_flag_on_unrelated_subject"]["grade"]["picked"] or "") == "effect"
        else "TYPE CONTRAST OBSERVED — unrelated subject did not pick 'effect'"
    ),
}
log("TYPE_CONTRAST_CONTROL", json.dumps(R["stages"]["production_query"]["type_contrast_control"]))

# ── SCRATCH-ONLY read-side probe (never a production change) ─────────────────
# Calls the EXISTING pure helpers directly on EdgeViews adapted from the real
# recovered graph edges. This does not edit or monkeypatch any production code.
scratch_probe = {"label": "SCRATCH-ONLY probe — pure helpers on real recovered edges",
                 "production_code_modified": False}
try:
    views = []
    for e in cold["trusted_edges_sample"]:
        if not e["trusted"] or not e["source"] or not e["target"]:
            continue
        views.append(si.EdgeView(e["source"], e["relation_type"], e["target"],
                                 e["edge_kind"], e["edge_source"], e["confidence"]))
    rmap = {"causal": "CAUSE", "part_of": "PART_OF",
            "temporal": "PRECEDES", "analogical": "RESEMBLES"}
    traces = si.traces_from_edges(views, relation_map=rmap)
    hyps = si.induce_schema_hypotheses(traces, min_support=2)
    scratch_probe["n_edgeviews"] = len(views)
    scratch_probe["n_traces"] = len(traces)
    scratch_probe["n_hypotheses"] = len(hyps)
    scratch_probe["hypotheses"] = [
        {"relation": h.relation, "support": h.support,
         "instances": list(h.instances), "assertable": h.assertable} for h in hyps]
    scratch_probe["hypothesis_keys_are_type_slots"] = any(
        h.relation in {"effect", "whole", "later", "similar"}
        for h in hyps)
    # Would the recovered edges carry candidate signal? Feed graph cosines to
    # the unmodified ranker and see if a strict pick survives.
    pairs = [p for h in hyps for p in h.instances]
    # SCRATCH-ONLY: why did the production flag-on hook emit 'effect'? Compute
    # the exact scorer inputs it used. If the type labels carry no graph edge
    # and only GloVe cosine separates them, the pick is lexical, not a slot.
    gv = getattr(eng, "_glove_vector", None)
    ts_scores, ts_inst = rg.schema_association_scores_from_graph(
        eng.graph, eng._concept_keywords, "thaw", SLOTS, pairs, glove_vector=gv)
    zs_scores, _ = rg.schema_association_scores_from_graph(
        eng.graph, eng._concept_keywords, "zeppelin", SLOTS, pairs, glove_vector=gv)
    scratch_probe["typed_slot_scorer_inputs"] = {
        "subject_thaw_scores": {k: round(float(v), 5) for k, v in ts_scores.items()},
        "subject_zeppelin_scores": {k: round(float(v), 5) for k, v in zs_scores.items()},
        "n_instance_scores": len(ts_inst),
        "instance_floor": round(min(float(v) for v in ts_inst.values()), 5) if ts_inst else None,
        "type_labels_are_graph_nodes": {
            s: bool(eng._concept_keywords.get(s.lower())) for s in SLOTS},
    }
    cands = ["mist", "vehicle", "spring", "fog"]
    cscores, iscores = rg.schema_association_scores_from_graph(
        eng.graph, eng._concept_keywords, "thaw", cands, pairs,
        glove_vector=getattr(eng, "_glove_vector", None))
    pick = si.rank_candidates_for_unseen_subject(
        "thaw", cands, hyps, traces, cscores, iscores, min_margin=0.0)
    scratch_probe["ranker"] = {"candidates": cands,
                               "candidate_scores": {k: round(float(v), 5) for k, v in cscores.items()},
                               "n_instance_scores": len(iscores),
                               "strict_pick": pick}
    log("SCRATCH_PROBE traces=%d hyps=%d keys=%s pick=%s" % (
        len(traces), len(hyps), [h.relation for h in hyps], pick))
except Exception as ex:
    scratch_probe["error"] = "%s: %s" % (type(ex).__name__, ex)
    scratch_probe["traceback"] = traceback.format_exc()[-1500:]
    log("SCRATCH_PROBE ERROR", scratch_probe["error"])
R["stages"]["scratch_only_read_probe"] = scratch_probe

# ── Summary counts ───────────────────────────────────────────────────────────
R["summary"] = {
    "taught_turns": len(TEACH),
    "type_edges_before_teach": base["type_relation_count"],
    "type_edges_after_teach_in_process": post["type_relation_count"],
    "type_edges_after_cold_reload": cold["type_relation_count"],
    "trusted_edges_before_teach": base["trusted_count"],
    "trusted_edges_after_teach_in_process": post["trusted_count"],
    "trusted_edges_after_cold_reload": cold["trusted_count"],
    "new_trusted_edges_after_teach": len(new_after_teach),
    "cold_reload_survivors": len(survivors),
    "typed_slot_query_parseable_default_path": R["stages"]["production_query"]["default_path_parseable"],
    "typed_slot_query_parseable_flag_on": R["stages"]["production_query"]["flag_on_parseable"],
    "permutation_control_verdict": R["stages"]["production_query"]["permutation_control"]["verdict"],
    "type_contrast_control_verdict": R["stages"]["production_query"]["type_contrast_control"]["verdict"],
    "scratch_probe_hypotheses": scratch_probe.get("n_hypotheses"),
}
log("SUMMARY", json.dumps(R["summary"], indent=2))

with open(RESULTS, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2, default=str)
log("WROTE", RESULTS)
log("EXIT=0")
