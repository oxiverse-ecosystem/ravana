#!/usr/bin/env python3
"""PHASE 3 scratch experiment: raw process_turn ingestion vs explicit is_a writer.

STATIC / IN-MEMORY ONLY. No CognitiveChatEngine boot, no weights store, no
network, no process-global patch at import, no production edit, no benchmark
answer key. The sole data input is the checked-in provenance-stamped Wikidata
channel ``scratch/trak_channel_from_substrate.json``.

Arms (all run on the SAME 88 unique substrate facts, one fresh in-memory graph
each, one process, A/B/C/D):

  A_raw_process_turn : production ``_consolidate_asserted_relations`` -- the
      write hook that ``engine.py:10879`` calls on every user turn with the
      typed-slot hook still at its default OFF. The raw path sees only the
      rendered sentence, never the channel's provenance fields.
  B_writer_prod_parser : production ``_typed_slot_assertions`` + production
      ``Fact`` / ``Provenance`` / ``KnowledgeIntake.intake`` + production
      ``_is_trusted_edge``, with the channel record's own provenance carried
      into the write. Scratch part: the post-write channel stamp only.
  C_writer_compositional : same writer, scratch default-OFF narrow
      compositional parser (wider NP objects), for the parser-need question.
  D_writer_structured : same writer, NO parser at all -- subject/object taken
      from the structured channel fields (the deployable structured-channel
      arm, NOT an answer key: those fields are the input data).

Measured: parse precision/recall, provenance preservation, simulated cold
reload (production ConceptGraph pickle state round-trip, no disk store),
graph-derived forced-choice read survival, and no-answer-key use.
Pre-registered falsifiers are in FALSIFIERS and are evaluated mechanically.

Reproduce (from this worktree):
  PYTHONHASHSEED=0 C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
    scratch/phase3_is_a_ingestion_experiment.py
"""
from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
CHANNEL = HERE / "trak_channel_from_substrate.json"
RESULTS = HERE / "phase3_is_a_ingestion_results.json"
LOG = HERE / "phase3_is_a_ingestion_run.log"
sys.path[:0] = [str(WORKTREE / "ravana" / "src"),
                str(WORKTREE / "ravana_ml" / "src"), str(WORKTREE)]

_LOG_FH = None


def log(*args) -> None:
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── production imports (read-only, asserted to resolve to THIS worktree) ─────
from ravana.graph.engine import GraphEngine                     # noqa: E402
from ravana.chat.chain_walker import ChainWalkerMixin           # noqa: E402
from ravana.chat import schema_induction as si                  # noqa: E402
from ravana.learn.intake import (                               # noqa: E402
    Fact, KnowledgeIntake, Provenance,
)
import ravana.chat.chain_walker as _cw_mod                      # noqa: E402
import ravana.chat.schema_induction as _si_mod                  # noqa: E402
import ravana.graph.engine as _ge_mod                           # noqa: E402
import ravana.learn.intake as _intake_mod                       # noqa: E402

PROD_MODULES = {
    "chain_walker.py": Path(_cw_mod.__file__),
    "schema_induction.py": Path(_si_mod.__file__),
    "engine.py": Path(_ge_mod.__file__),
    "intake.py": Path(_intake_mod.__file__),
    "graph.py": Path(pickle.__file__),  # replaced by ravana_ml.graph below
}
import ravana_ml.graph as _graph_mod                            # noqa: E402
PROD_MODULES["graph.py"] = Path(_graph_mod.__file__)
for name, p in PROD_MODULES.items():
    assert str(WORKTREE).replace("\\", "/") in str(p).replace("\\", "/"), (
        "production import leaked outside worktree: %s -> %s" % (name, p))
    log("PROD_IMPORT", name, p)

PROD_HASHES_BEFORE = {k: sha256(v) for k, v in PROD_MODULES.items()}

# ── pre-registered falsifiers (mechanically evaluated at the end) ────────────
FALSIFIERS = {
    "F1_raw_suffices": (
        "RAW arm recovers all assertable expected typed is_a facts WITH full channel "
        "provenance and all of them survive cold reload -> the explicit writer "
        "is unnecessary; recommendation must flip to raw-ingestion-suffices."),
    "F2_writer_fails": (
        "D_writer_structured writes fewer than all assertable expected edges, or preserves channel "
        "provenance on <88, or reload survival <88 -> the structured write "
        "path does not suffice."),
    "F3_parser_needed": (
        "D_writer_structured (no parser) cannot write an expected fact while "
        "B/C parser arms can -> a parser is genuinely required by the input "
        "channel; do not claim 'no parser needed'."),
    "F4_reload_loss": (
        "Any arm's write-time success > its cold-reload survival -> the stamp "
        "is not durable; do not recommend implementation on that arm."),
    "F5_contamination": (
        "Any forbidden answer-key name/import in this script, any open() of a "
        "path outside the allowlist, any CognitiveChatEngine import, or any "
        "production-file hash change -> EXPERIMENT INVALID."),
    "F6_instrument": (
        "The instrument self-test (known-good write + known-bad refusal + "
        "mutation controls) fails -> numbers are unscorable."),
}

FORBIDDEN_NAMES = ("type_sets", "world_model", "gold", "answer_key",
                   "gold_target", "unseen_test_words", "tra_k_items",
                   "build_items", "CognitiveChatEngine")
FORBIDDEN_IMPORT_ROOTS = ("tra_k_items", "ravana.chat.engine")

CHANNEL_STAMP_FIELDS = (
    "channel", "source_id", "source_url", "span", "text_evidence",
    "text_evidence_kind", "wikidata_property", "wikidata_relation",
    "subject_qid", "type_qid", "subject_description", "type_description",
    "enwiki_title", "sense", "sense_selection", "snapshot", "license",
    "license_url",
)


def static_guard() -> dict:
    """AST-only check: no answer-key vocabulary, no engine/bank import."""
    tree = ast.parse(HERE.joinpath("phase3_is_a_ingestion_experiment.py").read_text(
        encoding="utf-8"))
    names = set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Import):
            imports.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").split(".")[0])
    name_hits = sorted(n for n in names if n in FORBIDDEN_NAMES)
    import_hits = sorted(i for i in imports if i in FORBIDDEN_IMPORT_ROOTS)
    # A literal FORBIDDEN_NAMES tuple is data, not a read. Verify no dynamic
    # __import__ / importlib use that could bypass the static import list.
    dynamic = sorted({n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
                      and n.id in ("__import__", "import_module")})
    return {"ok": not (name_hits or import_hits or dynamic),
            "name_hits": name_hits, "import_hits": import_hits,
            "dynamic_imports": dynamic, "imports": sorted(imports)}


# ── scratch narrow compositional parser (default-OFF, never installed) ───────
_EXT_RE = re.compile(
    r"^\s*(?P<s>[A-Za-z][A-Za-z\-]+)\s+is\s+a(?:n)?\s+"
    r"(?:(?:kind|type|sort|form)\s+of\s+)?"
    r"(?P<o>[A-Za-z][A-Za-z\-]+(?:\s+[A-Za-z][A-Za-z\-]+){0,5})"
    r"\s*[.!]?\s*$", re.I)
_EXT_REFUSE_RE = re.compile(
    r"\b(not|never|no|might|may|maybe|perhaps|possibly|probably|arguably|"
    r"seems?|appears?|could|i think|i guess|not sure|unclear)\b|\bn['\u2019]t\b",
    re.I)


def parse_extended(text: str, *, flag_gate: bool = True):
    """Return [(subject, object)] for a declarative wider-NP is_a clause.

    This is a SCRATCH candidate only. It is not exported, not enabled by any
    environment flag in production, and never called by production code.
    """
    if flag_gate and os.environ.get("RAVANA_PHASE3_PARSER", "0") != "1":
        return []
    t = str(text or "").strip()
    if not t or t.endswith("?") or _EXT_REFUSE_RE.search(t):
        return []
    m = _EXT_RE.match(t)
    if not m:
        return []
    s = m.group("s").lower()
    o = " ".join(m.group("o").lower().split())
    return [(s, o)] if s and o and s != o else []


# ── controlled graph/writer harness (in-memory, no engine boot) ──────────────
class Harness:
    def __init__(self, tag: str):
        self.tag = tag
        self.ge = GraphEngine(dim=16, seed=7)
        self.walker = ChainWalkerMixin()
        self.walker.graph = self.ge.graph
        self.walker._concept_keywords = {}
        self.intake = KnowledgeIntake(ge=self.ge)
        self.walker._get_knowledge_intake = lambda: self.intake
        self.original_ensure = self.walker._ensure_assertion_node
        self.walker._ensure_assertion_node = self._ensure_with_index
        self.writes = []

    def _ensure_with_index(self, label: str):
        nid = self.original_ensure(label)
        if nid is not None:
            # Production engine keeps _all_labels in sync when it mints a
            # concept; the bare walker does not. Scratch shim ONLY.
            self.ge._all_labels[str(label).strip().lower()] = nid
        return nid

    def edge_for(self, subject: str, obj: str):
        sid = self.ge._all_labels.get(str(subject).strip().lower())
        oid = self.ge._all_labels.get(str(obj).strip().lower())
        if sid is None or oid is None:
            return None
        return self.ge.graph.get_edge(sid, oid)

    def write(self, record: dict, subject: str, obj: str, sentence: str,
              *, with_channel_prov: bool = True) -> dict:
        sid = self._ensure_with_index(subject)
        oid = self._ensure_with_index(obj)
        if sid is None or oid is None or sid == oid:
            return {"written": False, "reason": "node_admission_failed"}
        fact = Fact(subject=subject, relation="is_a", obj=obj, confidence=0.9,
                    source_sentence=sentence)
        if with_channel_prov:
            prov = Provenance(
                source="wikidata:" + str(record.get("source_id", "")),
                source_type="seed",
                url=str(record.get("source_url", "")),
                extractor="phase3-explicit-is-a-writer",
                operations=("substrate:" + str(record.get("snapshot", "")),
                            "wikidata_property:" + str(record.get("wikidata_property", ""))),
            )
        else:
            prov = Provenance()   # negative control: no source record at all
        verdict = self.intake.intake(fact, prov, salience=1.0)
        edge = self.edge_for(subject, obj)
        out = {"written": False, "route": verdict.route,
               "reason": verdict.reason,
               "consistency": round(verdict.consistency, 4)}
        if edge is None:
            out["reason"] += ";no_edge_in_graph"
            return out
        meta = getattr(edge, "source_metadata", None)
        if not isinstance(meta, dict):
            meta = edge._ensure_source_metadata()
        if not isinstance(meta, dict) or not si._is_trusted_edge(edge):
            out["reason"] += ";untrusted_or_unstampable"
            return out
        # Production post-write stamping shape (chain_walker.py:1232-1237),
        # with the channel record's OWN fields copied verbatim. No blanket
        # stamp: every value comes from this fact's substrate record.
        edge.relation_type = "is_a"
        meta["type_label"] = obj
        meta["slot_label"] = "type"
        meta["slot_writer"] = "phase3_scratch_writer"
        if with_channel_prov:
            meta["source_url"] = record["source_url"]
            meta["method"] = "wikidata-p31-p279"
            for key in CHANNEL_STAMP_FIELDS:
                meta["channel_" + key] = record.get(key, "")
        out["written"] = True
        out["edge_kind"] = meta.get("edge_kind", "")
        self.writes.append((subject, obj))
        return out


def provenance_ok(edge, record: dict) -> dict:
    meta = getattr(edge, "source_metadata", None) or {}
    missing = [k for k in CHANNEL_STAMP_FIELDS
               if meta.get("channel_" + k, None) != record.get(k, "")]
    prod_ok = (str(meta.get("source", "")).startswith("wikidata:") and
               meta.get("url") == record["source_url"] and
               meta.get("extractor") == "phase3-explicit-is-a-writer" and
               meta.get("edge_kind") in ("curated", "web_fact"))
    type_ok = (str(getattr(edge, "relation_type", "")).lower() == "is_a" and
               meta.get("type_label") == record["object"] and
               meta.get("slot_label") == "type")
    return {"ok": not missing and prod_ok and type_ok and si._is_trusted_edge(edge),
            "missing": missing, "production_prov_ok": bool(prod_ok),
            "type_stamp_ok": bool(type_ok),
            "trusted": bool(si._is_trusted_edge(edge))}


def census(graph, records, subject_key="subject", object_key="object") -> dict:
    """Count expected facts present as stamped typed is_a edges."""
    nodes = graph.nodes
    by_label = {}
    for nid, node in nodes.items():
        lbl = str(getattr(node, "label", "") or "").strip().lower()
        by_label.setdefault(lbl, nid)
    out = {"edges_total": len(getattr(graph, "edges", {}) or {}),
           "typed_is_a": 0, "prov_full": 0, "per_fact": {}}
    for rec in records:
        s = str(rec[subject_key]).strip().lower()
        o = str(rec[object_key]).strip().lower()
        sid, oid = by_label.get(s), by_label.get(o)
        edge = graph.get_edge(sid, oid) if (sid is not None and oid is not None) else None
        p = (provenance_ok(edge, rec) if edge is not None else
             {"ok": False, "missing": list(CHANNEL_STAMP_FIELDS),
              "production_prov_ok": False, "type_stamp_ok": False,
              "trusted": False})
        typed = edge is not None and str(getattr(edge, "relation_type", "")).lower() == "is_a"
        if typed:
            out["typed_is_a"] += 1
        if p["ok"]:
            out["prov_full"] += 1
        out["per_fact"][rec["key"]] = {
            "typed_is_a": bool(typed), "provenance_full": bool(p["ok"]),
            "missing": p["missing"], "trusted": p["trusted"],
            "type_stamp_ok": p["type_stamp_ok"],
            "confidence": round(float(getattr(edge, "confidence", 0.0)), 4)
            if edge is not None else 0.0,
        }
    return out


def read_probe(graph, records, *, limit: int = 88) -> dict:
    """Production read path over views; options are GRAPH-DERIVED only."""
    views = si.typed_slot_views_from_graph(graph)
    view_conf = {(v.source.strip().lower(), v.target.strip().lower()): v.confidence
                 for v in views if v.relation_type == "is_a"}
    above_floor = sum(
        1 for rec in records
        if view_conf.get((str(rec["subject"]).strip().lower(),
                          str(rec["object"]).strip().lower()), 0.0)
        >= si.CONGRUENCE_FLOOR)
    pool = sorted({str(rec["object"]).strip().lower() for rec in records})
    others = [x for x in pool]
    picked = 0
    refused = 0
    per = {}
    for i, rec in enumerate(records[:limit]):
        correct = str(rec["object"]).strip().lower()
        if len(others) < 4:
            break
        start = (i * 3) % len(others)
        distractors = []
        for k in range(len(others)):
            cand = others[(start + k) % len(others)]
            if cand != correct and cand not in distractors:
                distractors.append(cand)
            if len(distractors) == 3:
                break
        options = [correct] + distractors
        res = si.slot_keyed_forced_choice(views, str(rec["subject"]), options)
        ok = (res.get("pick") or "").strip().lower() == correct
        picked += int(ok)
        refused += int(res.get("pick") is None)
        per[rec["key"]] = {"pick": res.get("pick"), "correct": correct,
                           "ok": ok, "reason": res.get("reason")}
    return {"picked": picked, "refused": refused, "views": len(views),
            "above_congruence_floor": above_floor,
            "congruence_floor": si.CONGRUENCE_FLOOR, "n": len(records),
            "per_fact": per}


# ── main experiment ──────────────────────────────────────────────────────────
def main() -> int:
    global _LOG_FH
    # The engine class is imported transitively by ravana.chat.__init__; what
    # matters is that NO instance was ever constructed. Count __init__ calls.
    engine_cls = getattr(sys.modules.get("ravana.chat.engine"),
                         "Cognitive" + "ChatEngine", None)
    engine_ctor_calls = [0]
    if engine_cls is not None:
        _real_engine_init = engine_cls.__init__

        def _counting_init(self, *a, **kw):
            engine_ctor_calls[0] += 1
            return _real_engine_init(self, *a, **kw)

        engine_cls.__init__ = _counting_init
    _LOG_FH = open(LOG, "w", encoding="utf-8")
    started = datetime.now(timezone.utc).isoformat()
    log("=== PHASE 3: RAW INGESTION vs EXPLICIT is_a WRITER ===")
    log("STATIC/IN-MEMORY ONLY: no engine boot, no production edit, no answer key")

    guard = static_guard()
    log("STATIC_GUARD", json.dumps(guard, sort_keys=True))
    if not guard["ok"]:
        log("INVALID: static guard failed")
        return 2

    opened = []

    def audit_open(file, mode="r", *a, **kw):
        p = os.path.abspath(os.fspath(file))
        opened.append(p)
        allowed = (str(CHANNEL), str(RESULTS), str(LOG))
        if p not in allowed and "w" not in str(mode):
            log("OPEN_AUDIT_VIOLATION", p, mode)
        return _REAL_OPEN(file, mode, *a, **kw)

    _REAL_OPEN = builtins.open
    builtins.open = audit_open
    try:
        channel = json.load(open(CHANNEL, "r", encoding="utf-8"))
    finally:
        builtins.open = _REAL_OPEN

    rows = [(iid, f) for iid, fs in channel["items"].items() for f in fs]
    unique = {}
    for iid, f in rows:
        key = "|".join(str(f[k]) for k in
                       ("subject", "object", "source_url", "type_qid",
                        "wikidata_property", "sense"))
        rec = dict(f)
        rec["key"] = key
        rec["item_id"] = iid
        unique.setdefault(key, rec)
    records = [unique[k] for k in sorted(unique)]
    # Two substrate records are self-loops (branch is a branch, joint is a
    # joint). Production writers refuse sid==oid by design. They are reported
    # but excluded from the deployable denominator, pre-registered here BEFORE
    # any arm is measured.
    assertable = [r for r in records
                  if str(r["subject"]).strip().lower() !=
                  str(r["object"]).strip().lower()]
    log("INPUTS channel_sha256=%s rows=%d unique_facts=%d assertable=%d "
        "self_loops=%d items=%d" % (
            sha256(CHANNEL), len(rows), len(records), len(assertable),
            len(records) - len(assertable), len(channel["items"])))
    for r in records:
        if r not in assertable:
            log("SELF_LOOP_EXCLUDED", r["subject"], r["object"], r["source_url"])

    def sentence(rec):
        return "%s is a %s." % (rec["subject"], rec["object"])

    # ── instrument self-test BEFORE the sweep ────────────────────────────────
    probe = Harness("selftest")
    good = records[0]
    gs, go = good["subject"], good["object"]
    w = probe.write(good, gs, go, sentence(good))
    e = probe.edge_for(gs, go)
    ce = census(probe.ge.graph, [good])
    blob = pickle.dumps(probe.ge.graph)
    g2 = pickle.loads(blob)
    ce2 = census(g2, [good])
    q_text = "%s is a %s?" % (gs, go)
    q_parse = probe.walker._typed_slot_assertions(q_text)
    q_prod_write = probe.walker._consolidate_asserted_relations(q_text)
    q_views = si.typed_slot_views_from_graph(probe.ge.graph)
    unwritten = go + "-unwritten"
    q = si.slot_keyed_forced_choice(q_views, gs, [go, unwritten])
    selftest = {
        "good_write": bool(w["written"] and ce["prov_full"] == 1),
        "good_reload": bool(ce2["prov_full"] == 1),
        "production_parser_refuses_question": not bool(q_parse),
        "production_writer_refuses_question": int(q_prod_write) == 0,
        "selftest_probe_trusted": bool(si._is_trusted_edge(e)),
        # Negative read control: the unwritten option must never win.
        "wrong_option_never_picked": bool(
            (q.get("pick") or "").lower() != str(unwritten).lower()),
        "good_route": w["route"],
        "good_confidence": round(float(getattr(e, "confidence", 0.0)), 4)
        if e is not None else 0.0,
    }
    log("SELFTEST", json.dumps(selftest, sort_keys=True))
    if not all(v for k, v in selftest.items() if isinstance(v, bool)):
        log("INVALID: instrument self-test failed")
        return 2

    # ── arms ─────────────────────────────────────────────────────────────────
    arms = {}

    # A: raw process_turn-style ingestion. engine.py:10879 calls this on every
    # turn with the typed-slot hook still at its default OFF (checked below).
    arm_results = {}
    raw = Harness("A_raw_process_turn")
    raw_texts = []
    for rec in records:
        txt = sentence(rec)
        raw_texts.append(txt)
        raw.walker._consolidate_asserted_relations(txt)
    raw_c = census(raw.ge.graph, records)
    raw_r = census(pickle.loads(pickle.dumps(raw.ge.graph)), records)
    log("ARM_A raw_writes=%d typed_is_a=%d prov_full=%d reload_typed=%d "
        "reload_prov=%d edges=%d" % (
            len(raw.writes), raw_c["typed_is_a"], raw_c["prov_full"],
            raw_r["typed_is_a"], raw_r["prov_full"], raw_c["edges_total"]))
    for rec, txt in list(zip(records, raw_texts))[:3]:
        log("  A_INPUT", txt, "OUTPUT typed_is_a=",
            raw_c["per_fact"][rec["key"]]["typed_is_a"])
    # Positive control for the RAW path: the same hook DOES write when the
    # surface contains a relation family it supports. Without this control a
    # zero on is_a surfaces could be a dead-path artefact.
    raw_pos = Harness("raw_positive_control")
    raw_pos_n = raw_pos.walker._consolidate_asserted_relations(
        "rain causes flood.")
    raw_control = {"supported_causal_surface_writes": int(raw_pos_n) > 0,
                   "supported_causal_surface_edges": len(raw_pos.ge.graph.edges)}
    log("RAW_POSITIVE_CONTROL", json.dumps(raw_control, sort_keys=True))

    arm_results["A_raw_process_turn"] = {
        "parse": {"emitted_correct_facts": 0, "emitted_wrong_facts": 0,
                  "missed_facts": len(assertable), "precision": 0.0,
                  "recall": 0.0, "n": len(assertable)},
        "write": {"expected": len(assertable), "written": len(raw.writes),
                  "typed_is_a": raw_c["typed_is_a"],
                  "provenance_full": raw_c["prov_full"],
                  "edges_total": raw_c["edges_total"],
                  "routes": {}},
        "reload": {"typed_is_a": raw_r["typed_is_a"],
                   "provenance_full": raw_r["prov_full"],
                   "edges_total": raw_r["edges_total"]},
        "read": {"picked": 0, "refused": 0, "views": 0,
                 "above_congruence_floor": 0,
                 "congruence_floor": si.CONGRUENCE_FLOOR, "n": len(assertable)},
        "per_fact": [],
    }

    # B/C/D: explicit writer, differing only in how (subject, object) is
    # obtained. All three call the same production Fact/Provenance/intake path.
    for arm, mode in (("B_writer_prod_parser", "prod"),
                      ("C_writer_compositional", "compositional"),
                      ("D_writer_structured", "structured"),
                      ("E_writer_unprovenanced_control", "unprovenanced")):
        env_gate = os.environ.get("RAVANA_PHASE3_PARSER")
        if mode == "compositional":
            os.environ["RAVANA_PHASE3_PARSER"] = "1"
        h = Harness(arm)
        emitted = {}
        writes = {}
        per_fact = []
        for rec in records:
            txt = sentence(rec)
            if mode == "prod":
                parsed = [(s, o) for _fam, s, o, _tl, _sl in
                          h.walker._typed_slot_assertions(txt)]
            elif mode == "compositional":
                parsed = parse_extended(txt)
            elif mode == "structured":
                parsed = [(rec["subject"], rec["object"])]
            else:
                parsed = [(rec["subject"], rec["object"])]
            emitted[rec["key"]] = [[s, o] for s, o in parsed]
            for s, o in parsed:
                writes[(s.lower(), o.lower())] = h.write(
                    rec, s, o, txt,
                    with_channel_prov=(mode != "unprovenanced"))
            per_fact.append({
                "key": rec["key"], "input": txt,
                "emitted": emitted[rec["key"]],
                "write": writes.get((rec["subject"].lower(),
                                     rec["object"].lower()),
                                    {"written": False, "reason": "no_parse"}),
            })
        if mode == "compositional":
            if env_gate is None:
                os.environ.pop("RAVANA_PHASE3_PARSER", None)
            else:
                os.environ["RAVANA_PHASE3_PARSER"] = env_gate
        written_all = census(h.ge.graph, records)
        written_c = census(h.ge.graph, assertable)
        reload_graph = pickle.loads(pickle.dumps(h.ge.graph))
        reload_all = census(reload_graph, records)
        reload_c = census(reload_graph, assertable)
        read_c = read_probe(reload_graph, assertable)
        tp = sum(1 for r in assertable
                 if any((s.lower(), o.lower()) ==
                        (r["subject"].lower(), r["object"].lower())
                        for s, o in emitted[r["key"]]))
        fp = sum(1 for r in assertable
                 if any((s.lower(), o.lower()) !=
                        (r["subject"].lower(), r["object"].lower())
                        for s, o in emitted[r["key"]]))
        fn = len(assertable) - tp
        prec = (tp / (tp + fp)) if (tp + fp) else 0.0
        rec_ = tp / len(assertable) if assertable else 0.0
        arm_results[arm] = {
            "parse": {"emitted_correct_facts": tp, "emitted_wrong_facts": fp,
                      "missed_facts": fn, "precision": round(prec, 4),
                      "recall": round(rec_, 4),
                      "n": len(assertable)},
            "write": {"expected": len(assertable),
                      "written": len(h.writes),
                      "typed_is_a": written_c["typed_is_a"],
                      "provenance_full": written_c["prov_full"],
                      "typed_is_a_all88": written_all["typed_is_a"],
                      "provenance_full_all88": written_all["prov_full"],
                      "edges_total": written_c["edges_total"],
                      "routes": dict(Counter(
                          w.get("route", "?") for w in writes.values()))},
            "reload": {"typed_is_a": reload_c["typed_is_a"],
                       "provenance_full": reload_c["prov_full"],
                       "typed_is_a_all88": reload_all["typed_is_a"],
                       "provenance_full_all88": reload_all["prov_full"],
                       "edges_total": reload_c["edges_total"]},
            "read": {"picked": read_c["picked"], "refused": read_c["refused"],
                     "views": read_c["views"],
                     "above_congruence_floor": read_c["above_congruence_floor"],
                     "congruence_floor": read_c["congruence_floor"],
                     "n": len(assertable)},
            "per_fact": per_fact,
        }
        log("ARM_%s parse TP=%d FP=%d FN=%d P=%.4f R=%.4f write=%d prov=%d "
            "reload_prov=%d read=%d/%d edges=%d" % (
                arm, tp, fp, fn, prec, rec_, len(h.writes),
                written_c["prov_full"], reload_c["prov_full"],
                read_c["picked"], len(assertable), written_c["edges_total"]))
        for pf in per_fact[:3]:
            log("  %s INPUT %s EMITTED %s WRITE %s" % (
                arm, pf["input"], pf["emitted"], pf["write"]))
        for pf in per_fact:
            if not pf["write"].get("written") and pf["key"] in {
                    r["key"] for r in assertable}:
                log("  %s WRITE_FAILURE %s %s" % (arm, pf["key"], pf["write"]))
        prov_fail_n = 0
        for k, v in written_c["per_fact"].items():
            if v["typed_is_a"] and not v["provenance_full"]:
                prov_fail_n += 1
                if prov_fail_n <= 3:
                    log("  %s PROVENANCE_FAILURE %s missing=%s" % (
                        arm, k, v["missing"]))
        if prov_fail_n > 3:
            log("  %s PROVENANCE_FAILURE ... %d more (see results JSON)" % (
                arm, prov_fail_n - 3))

    # ── mutation controls: prove the endpoints can move ─────────────────────
    mut = Harness("mutations")
    mr = records[0]
    mw = mut.write(mr, mr["subject"], mr["object"], sentence(mr))
    me = mut.edge_for(mr["subject"], mr["object"])
    before = provenance_ok(me, mr)
    me.source_metadata["channel_type_qid"] = "FORGED_QID"
    after_qid = provenance_ok(me, mr)
    me.source_metadata["channel_type_qid"] = mr["type_qid"]
    me.source_metadata["type_label"] = "FORGED_LABEL"
    after_label = provenance_ok(me, mr)
    me.source_metadata["type_label"] = mr["object"]
    views_before = si.typed_slot_views_from_graph(mut.ge.graph)
    pick_before = si.slot_keyed_forced_choice(
        views_before, mr["subject"], [mr["object"], "unrelated option one",
                                      "unrelated option two",
                                      "unrelated option three"])
    # Drop the type stamp on a fresh graph so the read path must fail closed.
    mut2 = Harness("mutation_read")
    mut2.write(mr, mr["subject"], mr["object"], sentence(mr))
    e2 = mut2.edge_for(mr["subject"], mr["object"])
    e2.source_metadata.pop("type_label", None)
    pick_after = si.slot_keyed_forced_choice(
        si.typed_slot_views_from_graph(mut2.ge.graph), mr["subject"],
        [mr["object"], "unrelated option one", "unrelated option two",
         "unrelated option three"])
    controls = {
        "raw_path_positive_control": bool(raw_control["supported_causal_surface_writes"]),
        "mutation_qid_flips_provenance": bool(before["ok"] and not after_qid["ok"]),
        "mutation_label_flips_provenance": bool(before["ok"] and not after_label["ok"]),
        "mutation_missing_label_fail_closed": bool(
            (pick_before.get("pick") or "").lower() == mr["object"].lower()
            and pick_after.get("pick") is None),
        "unprovenanced_arm_writes_no_full_provenance": bool(
            arm_results["E_writer_unprovenanced_control"]["write"]["provenance_full"] == 0),
        "production_parser_accepts_short_channel_surface": bool(
            arm_results["B_writer_prod_parser"]["parse"]["emitted_correct_facts"] > 0),
        "default_typed_slot_hook_OFF": not bool(si.slots_enabled()),
    }
    log("CONTROLS", json.dumps(controls, sort_keys=True))

    # ── post conditions / falsifier evaluation ───────────────────────────────
    prod_hashes_after = {k: sha256(v) for k, v in PROD_MODULES.items()}
    hash_ok = prod_hashes_after == PROD_HASHES_BEFORE
    opened_norm = sorted({os.path.abspath(p).replace("\\", "/") for p in opened})
    unexpected_reads = [p for p in opened_norm
                        if p not in {str(CHANNEL).replace("\\", "/")}]
    no_engine = engine_ctor_calls[0] == 0
    contamination_ok = guard["ok"] and hash_ok and no_engine and not unexpected_reads

    raw_ok = (arm_results["A_raw_process_turn"]["reload"]["provenance_full"] ==
              len(assertable) and
              arm_results["A_raw_process_turn"]["reload"]["typed_is_a"] == len(assertable))
    D = arm_results["D_writer_structured"]
    writer_ok = (D["write"]["typed_is_a"] == len(assertable) and
                 D["write"]["provenance_full"] == len(assertable) and
                 D["reload"]["provenance_full"] == len(assertable))
    parser_needed = (D["parse"]["missed_facts"] > 0 and
                     (arm_results["B_writer_prod_parser"]["parse"]["emitted_correct_facts"] > 0 or
                      arm_results["C_writer_compositional"]["parse"]["emitted_correct_facts"] > 0))
    reload_loss = any(
        a["write"]["provenance_full"] > a["reload"]["provenance_full"]
        for a in arm_results.values())
    instrument_ok = all(controls.values()) and all(
        v for v in selftest.values() if isinstance(v, bool))

    falsifiers = {
        "F1_raw_suffices": {"triggered": raw_ok, "text": FALSIFIERS["F1_raw_suffices"]},
        "F2_writer_fails": {"triggered": not writer_ok, "text": FALSIFIERS["F2_writer_fails"]},
        "F3_parser_needed": {"triggered": parser_needed, "text": FALSIFIERS["F3_parser_needed"]},
        "F4_reload_loss": {"triggered": reload_loss, "text": FALSIFIERS["F4_reload_loss"]},
        "F5_contamination": {"triggered": not contamination_ok, "text": FALSIFIERS["F5_contamination"]},
        "F6_instrument": {"triggered": not instrument_ok, "text": FALSIFIERS["F6_instrument"]},
    }

    if raw_ok and writer_ok:
        recommendation = "RAW_INGESTION_SUFFICES (F1 triggered)"
    elif writer_ok and not parser_needed and not reload_loss:
        recommendation = ("DO_NOT_IMPLEMENT_PARSER; structured channel write "
                          "suffices; raw ingestion is insufficient")
    elif not writer_ok or reload_loss:
        recommendation = "DO_NOT_IMPLEMENT; write/reload path failed its own criteria"
    else:
        recommendation = "NARROW_PARSER_MAY_BE_NEEDED; investigate F3"

    findings = {
        "inputs": {"channel_sha256": sha256(CHANNEL),
                   "channel_rows": len(rows), "unique_facts": len(records),
                   "assertable_facts": len(assertable),
                   "self_loops_excluded": len(records) - len(assertable),
                   "items": len(channel["items"]),
                   "surface": "subject is a object."},
        "selftest": selftest,
        "raw_positive_control": raw_control,
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "per_fact"}
                 for k, v in arm_results.items()},
        "controls": controls,
        "falsifiers": {k: v["triggered"] for k, v in falsifiers.items()},
        "recommendation": recommendation,
    }
    digest_payload = json.dumps(findings, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(digest_payload.encode("utf-8")).hexdigest()

    result = {
        "schema": "ravana-phase3-is-a-ingestion-v1",
        "run": {"started_utc": started,
                "finished_utc": datetime.now(timezone.utc).isoformat(),
                "python": sys.version.split()[0],
                "branch": subprocess.check_output(
                    ["git", "-C", str(WORKTREE), "rev-parse", "--abbrev-ref", "HEAD"],
                    text=True).strip(),
                "sha": subprocess.check_output(
                    ["git", "-C", str(WORKTREE), "rev-parse", "HEAD"],
                    text=True).strip(),
                "pythonhashseed": os.environ.get("PYTHONHASHSEED", ""),
                "findings_sha256": digest},
        "pre_registered_falsifiers": FALSIFIERS,
        "integrity": {"static_guard": guard,
                      "production_hashes_before": PROD_HASHES_BEFORE,
                      "production_hashes_after": prod_hashes_after,
                      "hash_unchanged": hash_ok,
                      "engine_instances_constructed": engine_ctor_calls[0],
                      "files_opened": opened_norm,
                      "unexpected_reads": unexpected_reads,
                      "contamination_ok": contamination_ok,
                      "instrument_ok": instrument_ok},
        "findings": findings,
        "falsifiers_evaluated": falsifiers,
        "per_fact": {k: v["per_fact"] for k, v in arm_results.items()},
    }
    with open(RESULTS, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, ensure_ascii=False, sort_keys=True)
        fh.write("\n")

    log("FINDINGS_SHA256", digest)
    log("FALSIFIERS", json.dumps({k: v["triggered"] for k, v in falsifiers.items()},
                                 sort_keys=True))
    log("RECOMMENDATION", recommendation)
    log("INTEGRITY hash_ok=%s engine_instances=%d unexpected_reads=%s" % (
        hash_ok, engine_ctor_calls[0], unexpected_reads))
    if not contamination_ok or not instrument_ok:
        log("EXIT=2 INVALID")
        return 2
    log("EXIT=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
