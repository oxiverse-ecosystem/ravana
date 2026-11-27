#!/usr/bin/env python3
"""M1+M2+M3 combined experiment runner (one process, one arm, one store copy).

CLI (exact contract from M123_COMBINED_PREREG.md section 7)::

  python scratch/m123_combined_runner.py --stage {trak,rsi,tra} \\
      --arm {OFF,OFF2,M1,M1+M2,M1+M2+M3,SHAM} --tag <tag> \\
      --base <isolated data root> --out <report.json> [--max-per-pool N]

Every invocation uses a FRESH store copy, the shipped harness's own
run_sweep()/summarize()/instrument_controls(), and the shared arm primitives in
m123_preflight.py.  It never reads gold/type_sets/world_model into staging or
scoring, and it fails closed (exit 2, no score) on any preflight or instrument
control violation.
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import hashlib
import json
import os
import re
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
PARENT = "C:/Users/Likhith/Documents/projects/ravana"
for _p in (WT, os.path.join(WT, "scripts"), os.path.join(WT, "ravana", "src"),
           os.path.join(WT, "ravana_ml", "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m123_preflight as pf           # noqa: E402  (boot-free shared primitives)
import front_gates                    # noqa: E402
import measure_tra_k as mtk           # noqa: E402
from ravana.chat import durable_provenance as dp   # noqa: E402
from ravana.chat.schema_induction import typed_slot_views_from_graph  # noqa: E402
from ravana.chat.response_gen import parse_forced_choice  # noqa: E402
from ravana.learn import subject_type_evidence as ste    # noqa: E402

SUBJECT_RE = re.compile(
    r"\b(?:what is true of|which option best fits|what is true about|"
    r"best fits)\s+([^?.!,]+)", re.IGNORECASE)
NEGATED_CLAIM_RE = re.compile(
    r"\b([A-Za-z][A-Za-z\-]*)\s+(?:is|are|was|were)\s+not\s+(?:a|an)\s+"
    r"([A-Za-z][A-Za-z \-]*?)\s*[.!?]", re.IGNORECASE)


def subject_from_prompt(question):
    """Resolve the displayed subject from the prompt only (no item metadata)."""
    m = SUBJECT_RE.search(str(question or ""))
    if m:
        return m.group(1).strip()
    words = str(question or "").strip().split()
    return words[-1].strip(" .!?,") if words else ""


def sha256_file(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def seed_data_dir(dst):
    """Fresh copy of the canonical store into an isolated data_dir."""
    shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    src_dir = os.path.join(PARENT, "weights")
    out = {}
    for name in sorted(os.listdir(src_dir)):
        if (name.startswith("ravana_weights.pkl") or name == "ravana_weights.db"
                or name.startswith("junk_")):
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dst, name))
                out[name] = sha256_file(src)
    return out


def allowed_facts(item):
    """Deployable channel provenance only: no answer-side fields."""
    out = []
    for f in (item.get("channel_evidence", []) or []):
        if not isinstance(f, dict):
            continue
        out.append({k: f.get(k) for k in (
            "subject", "predicate", "object", "source_id", "source",
            "source_url", "license", "license_url", "evidence_role")})
    return out


def graph_views(engine):
    out = []
    try:
        for v in typed_slot_views_from_graph(getattr(engine, "graph", None)):
            out.append({k: getattr(v, k, "") for k in (
                "source", "relation_type", "target", "type_label",
                "edge_kind", "edge_source", "source_type", "confidence")})
    except Exception:
        out = []
    return out


def negated_pairs(surfaces):
    """Pairs explicitly negated in THIS condition's teaching surfaces."""
    out = set()
    for s in surfaces or ():
        m = NEGATED_CLAIM_RE.search(str(s))
        if m:
            out.add((m.group(1).strip().casefold(), m.group(2).strip().casefold()))
    return out


def make_backend(stage, arm, arm_dir, counters):
    """Wrap the shipped RealBackend; install the arm on each engine INSTANCE."""
    if stage == "trak":
        import measure_tra_k as mod
    elif stage == "rsi":
        import measure_learn5_rsi as mod
    else:
        import measure_tra as mod

    class ArmBackend(mod.RealBackend):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.m123_arm = arm
            self.m123_counters = counters
            self.m123_ctx = {"facts": [], "surfaces": [], "condition": ""}
            self.m123_snapshot = ste.load_snapshot()
            self.m123_installed = 0

        def _boot(self):
            eng = super()._boot()
            if getattr(eng, "_m123_wrapped", False):
                return eng
            original = eng.process_turn

            def wrapped(raw_input, *args, **kwargs):
                counters["turns"] += 1
                question, options = parse_forced_choice(raw_input)
                is_query = bool(options)
                response = original(raw_input, *args, **kwargs)
                try:
                    if is_query:
                        counters["query_turns"] += 1
                        counters["explicit_option_prompts"] += 1
                        subject = subject_from_prompt(question)
                        self.m123_ctx["query_subject"] = subject
                        self.m123_ctx["query_options"] = list(options)
                        views = [v for v in graph_views(eng)
                                 if str(v.get("source", "")).strip().casefold()
                                 == str(subject).strip().casefold()]
                        store = dp.store_for_engine(eng)
                        ledger = store.load().to_dict().get("records", {})
                        records = [dict(v) for v in ledger.values()] if ledger else []
                        snap_ev, _ = ste.evidence_for_subject(subject, self.m123_snapshot)
                        bundle = pf.stage_arm_evidence(
                            subject, options, ledger_records=records,
                            graph_views=views, snapshot_evidence=snap_ev,
                            negated_pairs=negated_pairs(self.m123_ctx.get("surfaces")),
                            arm=arm, counters=counters)
                        pick = pf.reader_pick(subject, options, bundle, arm, counters)
                        if pick is not None:
                            counters["emitted_picks"] += 1
                            return str(pick)      # bare displayed source-native option
                    else:
                        surface = str(raw_input or "")
                        m = pf.CLAIM_RE.search(surface)
                        if m and (arm in ("M1", "M1+M2") or arm in pf.M3_ARMS):
                            label = pf.label_from_claim(surface)
                            fact = next((f for f in self.m123_ctx.get("facts", [])
                                         if str(f.get("subject", "")).strip().casefold()
                                         == m.group(1).strip().casefold()
                                         and str(f.get("object", "")).strip().casefold()
                                         == label.casefold()
                                         and str(f.get("predicate", "")).strip().casefold()
                                         in ("is_a", "type_of", "instance_of")), None)
                            if fact is not None:
                                pf.stage_m1_act(dp.store_for_engine(eng), surface,
                                                m.group(1), label, fact, arm, counters)
                except Exception as exc:
                    counters["hook_errors"] += 1
                    counters["last_error"] = repr(exc)
                return response

            eng.process_turn = wrapped
            eng._m123_wrapped = True
            self.m123_installed += 1
            return eng

        def reset_for_item(self, item):
            print("ITEM_START", getattr(item, "get", lambda _k: "")("item_id"),
                  flush=True)
            parent = getattr(super(), "reset_for_item", None)
            return parent(item) if callable(parent) else None

        def teach(self, *args, **kwargs):
            item = args[0] if args else kwargs.get("item")
            raw_surfaces = args[4] if len(args) > 4 else kwargs.get("surfaces")
            condition = args[2] if len(args) > 2 else kwargs.get("condition")
            # Scratch-only, arm-keyed far repair: the SAME list length and the
            # SAME per-turn token count as the shipped teaching surfaces, with
            # only the far misleading channel assertion neutralised.
            surfaces = pf.repair_far_surfaces(
                item, condition, raw_surfaces, arm, counters)
            if len(args) > 4:
                args = args[:4] + (surfaces,) + args[5:]
            elif "surfaces" in kwargs:
                kwargs["surfaces"] = surfaces
            if item is not None:
                self.m123_ctx["facts"] = allowed_facts(item)
            self.m123_ctx["surfaces"] = list(surfaces or [])
            self.m123_ctx["condition"] = str(condition or "")
            self.m123_ctx["item_id"] = str(getattr(item, "get", lambda _k: "")("item_id") or "")
            return super().teach(*args, **kwargs)

        def present(self, *args, **kwargs):
            item = args[1] if len(args) > 1 else kwargs.get("item")
            condition = args[3] if len(args) > 3 else kwargs.get("condition")
            if item is not None:
                self.m123_ctx["facts"] = allowed_facts(item)
            self.m123_ctx["surfaces"] = []
            self.m123_ctx["condition"] = str(condition or "")
            return super().present(*args, **kwargs)

    return ArmBackend(data_dir=arm_dir)


def select_items(stage, mod, items, max_per_pool):
    if stage == "trak":
        return pf.focused_items(items, max_per_pool or 2)
    if stage == "rsi":
        pick = []
        for pool in ("primary", "sealed"):
            pick += [it for it in items if it.get("pool") == pool][:2]
        return pick
    seen = collections.Counter()
    pick = []
    cap = max_per_pool or 20
    for it in items:
        if seen[it["pool"]] < cap:
            pick.append(it)
            seen[it["pool"]] += 1
    return pick


def offline_reject_probe(items, arm, arm_dir):
    """Boot-free fail-closed check on the SAVED stores after the sweep.

    Stage the same evidence the live wrapper staged for foil/negative/unrelated
    queries and require zero emitted picks (F5).  Uses only the engine's own
    stored graph/ledger and the independent subject channel.
    """
    import pickle
    out = {"applicable": arm in pf.M3_ARMS,
           "rows": 0, "picks": 0, "by_condition": {}, "examples": []}
    if not out["applicable"]:
        return out
    for it in items:
        p = os.path.join(arm_dir, it["item_id"], "ravana_weights.pkl")
        if not os.path.isfile(p):
            continue
        with open(p, "rb") as fh:
            state = pickle.load(fh)
        views = [{"source": str(getattr(v, "source", "") or ""),
                  "type_label": str(getattr(v, "type_label", "") or ""),
                  "confidence": getattr(v, "confidence", 0.0)}
                 for v in typed_slot_views_from_graph(state.get("graph"))]
        ledger = dp.DurableProvenanceStore(
            os.path.join(arm_dir, it["item_id"])).load().to_dict().get("records", {})
        records = [dict(v) for v in ledger.values()]
        for cond in ("foil", "negative", "unrelated"):
            q = mtk.query_for(it, cond)
            subject = str(q["subject"])
            options = [str(c) for c in q["candidates"]]
            surfaces = mtk.teaching_surfaces(it, it["k"], cond, mtk.CHANNEL_ARM)
            scoped = [v for v in views
                      if str(v["source"]).strip().casefold() == subject.casefold()]
            snap, _ = ste.evidence_for_subject(subject, ste.load_snapshot())
            cnt = collections.Counter()
            bundle = pf.stage_arm_evidence(
                subject, options, ledger_records=records, graph_views=scoped,
                snapshot_evidence=snap, negated_pairs=negated_pairs(surfaces),
                arm=arm, counters=cnt)
            pick = pf.reader_pick(subject, options, bundle, arm, cnt)
            out["rows"] += 1
            if pick is not None:
                out["picks"] += 1
                out["examples"].append({"item": it["item_id"], "condition": cond,
                                        "pick": pick})
            out["by_condition"][cond] = out["by_condition"].get(cond, 0) + int(
                pick is not None)
    return out


def endpoint_slice(summary, keys):
    out = {}
    for pool in ("primary", "sealed"):
        d = summary.get(pool, {})
        out[pool] = {k: d.get(k) for k in keys if k in d}
        out[pool]["_n"] = {k: d.get(k + "_n") for k in keys
                           if k + "_n" in d}
    return out


def run_stage(args, preflight):
    stage, arm, tag = args.stage, args.arm, args.tag
    os.environ["RAVANA_OFFLINE"] = "1"
    for flag in pf.CLEAR_FLAGS:
        os.environ.pop(flag, None)
    for key in tuple(os.environ):
        if key in ("RAVANA_DURABLE_PROVENANCE", "RAVANA_SUBJECT_TYPE_EVIDENCE",
                   "RAVANA_M3_GRADED_PROBE"):
            os.environ.pop(key, None)
    for key, value in pf.ARM_FLAGS[arm].items():
        os.environ[key] = value

    arm_dir = os.path.join(args.base, "%s_%s" % (stage, tag))
    os.makedirs(args.base, exist_ok=True)
    start_hashes = seed_data_dir(arm_dir)

    if stage == "trak":
        mod = mtk
        all_items, channel = pf.load_bank()
        items = select_items(stage, mod, all_items, args.max_per_pool or 2)
        for it in items:
            seed_data_dir(os.path.join(arm_dir, it["item_id"]))
    elif stage == "rsi":
        import measure_learn5_rsi as mod
        all_items = mod.build_items()
        items = select_items(stage, mod, all_items, args.max_per_pool)
    else:
        import measure_tra as mod
        all_items = mod.tra_items.build_items()
        items = select_items(stage, mod, all_items, args.max_per_pool)
        for it in items:
            seed_data_dir(os.path.join(arm_dir, it["item_id"]))

    counters = collections.Counter()
    backend = make_backend(stage, arm, arm_dir, counters)
    started = time.time()
    flag_ctx = (mod.tra_slots_flag() if stage == "tra"
                else mtk.trak_slots_flag() if stage == "trak"
                else contextlib.nullcontext())
    with flag_ctx:
        rows = mod.run_sweep(items, backend=backend)
        summary = mod.summarize(items, rows)
        if stage == "trak":
            audit = pf.delayed_option_audit(all_items)
            # Row-level audit on the actual delayed option sets.
            by_id = {it["item_id"]: it for it in all_items}
            bad = 0
            for r in rows:
                if r.get("phase") != "delayed":
                    continue
                it = by_id.get(r.get("id"))
                if it is None:
                    bad += 1
                    continue
                cond = r.get("condition")
                shown = [str(c) for c in (r.get("input") or {}).get("candidates", [])]
                if shown != [str(c) for c in mtk.query_for(it, cond)["candidates"]]:
                    bad += 1
            delayed_audit = {"ok": audit["ok"] and bad == 0,
                             "bank_audit": audit, "row_mismatches": bad,
                             "delayed_rows": sum(1 for r in rows
                                                 if r.get("phase") == "delayed")}
            live_checks = mod.instrument_controls(items, rows, summary, backend,
                                                 True, backend.api_error)
            # Instrument controls C1-C10/C12-C15 are certified on the shipped
            # full-bank fake/oracle arm (as harness main() does); C11 is the
            # live anti-fake/backend-live gate. Capability FAIL is a score,
            # never an INSTRUMENT INVALID.
            cert_backend = mod.FakeBackend()
            cert_rows = mod.run_sweep(all_items, backend=cert_backend)
            cert_summary = mod.summarize(all_items, cert_rows)
            cert_checks = mod.instrument_controls(
                all_items, cert_rows, cert_summary, cert_backend, False, None)
            checks = ([c for c in cert_checks if c["id"] not in ("C11", "C12")]
                      + [c for c in live_checks if c["id"] in ("C11", "C12")])
            sat = front_gates.gate_saturation(
                {"tra_k_near": round(100 * (summary["primary"]["near"] or 0)),
                 "tra_k_negative_reject": round(100 * (
                     summary["primary"]["negative_reject"] or 0))}, None, "tra_k")
            mutations = mod.mutation_control(items)
            rules, harness_pass = mod.decide(summary, checks, backend, True,
                                             backend.api_error, sat.status,
                                             mutations["moved"])
        elif stage == "rsi":
            delayed_audit = {"ok": True, "not_applicable": True}
            # Shape/selectivity controls certified on the shipped fake arm;
            # live arm supplies only capability endpoints + backend-liveness.
            cert_backend = mod.FakeBackend()
            cert_rows = mod.run_sweep(all_items, backend=cert_backend)
            cert_summary = mod.summarize(all_items, cert_rows)
            checks = mod.instrument_controls(all_items, cert_rows, cert_summary)
            sat = front_gates.gate_saturation(
                {"rsi_near": round(100 * (summary["primary"]["near"] or 0))},
                None, "rsi")
            mutations = {"moved": None, "not_applicable": True}
            rules, harness_pass = {}, None
        else:
            delayed_audit = {"ok": True, "not_applicable": True}
            live_checks = mod.instrument_controls(items, rows, summary, backend,
                                                 True, backend.api_error)
            cert_backend = mod.FakeBackend()
            cert_rows = mod.run_sweep(all_items, backend=cert_backend)
            cert_summary = mod.summarize(all_items, cert_rows)
            cert_checks = mod.instrument_controls(
                all_items, cert_rows, cert_summary, cert_backend, False, None)
            checks = ([c for c in cert_checks if c["id"] not in ("C11", "C12")]
                      + [c for c in live_checks if c["id"] in ("C11", "C12")])
            sat = front_gates.gate_saturation(
                {"tra_near": round(100 * (summary["primary"]["near"] or 0))},
                None, "tra")
            mutations = mod.mutation_control(items)
            rules, harness_pass = mod.decide(summary, checks, backend, True,
                                             backend.api_error, sat.status,
                                             mutations["moved"])
    # RSI's shipped controls are (name, ok) tuples; TRA/TRA-K use dicts.
    checks = [c if isinstance(c, dict) else
              {"id": str(c[0]), "name": str(c[0]), "ok": bool(c[1])}
              for c in checks]
    elapsed = round(time.time() - started, 1)

    end_hashes = seed_data_dir_hash_only(arm_dir, start_hashes)
    counters = dict(counters)
    counters["m1_decisions"] = dict(counters.get("m1_decisions", {}))
    counters["boot"] = getattr(backend, "engines_booted", None)
    counters["process_turn"] = getattr(backend, "api_calls", {}).get("process_turn")
    counters["api_error"] = getattr(backend, "api_error", None)
    counters["engine_response_unparsed"] = getattr(backend, "engine_response_unparsed", None)
    counters["installed_engines"] = backend.m123_installed

    keys = {
        "trak": ("transfer", "near", "far", "immediate", "delayed",
                 "delayed_retention", "growth", "foil", "negative",
                 "foil_reject", "negative_reject", "chance", "role_coverage",
                 "baseline", "monotonic_max_drop", "graded_max_drop",
                 "control_max_delta", "ostensive_transfer"),
        "rsi": ("near", "far", "foil", "negative", "chance",
                "negative_reject", "counterexample_reject", "growth_headroom",
                "monotonic_max_drop", "graded_max_drop", "relation_coverage"),
        "tra": ("transfer", "near", "far", "foil", "negative", "chance",
                "foil_reject", "negative_reject", "counterexample_reject",
                "growth", "monotonic_max_drop", "graded_max_drop",
                "role_coverage"),
    }[stage]
    endpoints = endpoint_slice(summary, keys)

    controls_ok = all(c["ok"] for c in checks)
    c11_ok = next((c["ok"] for c in checks if c["id"] == "C11"), None)
    reject_probe = (offline_reject_probe(items, arm, arm_dir)
                    if stage == "trak" else
                    {"applicable": False, "rows": 0, "picks": 0,
                     "by_condition": {}, "examples": []})
    guard = {
        "preflight_ok": bool(preflight.get("ok")),
        "sha_pins_match": all(v.get("match") for v in preflight["sha_pins"].values()),
        "delayed_option_audit": delayed_audit,
        "controls_ok": bool(controls_ok),
        "failed_controls": [c["id"] for c in checks if not c["ok"] and c["id"] != "C11"],
        "c11_anti_fake_ok": c11_ok,
        "backend_error": getattr(backend, "api_error", None),
        "mutation_control": mutations,
        "g6_status": sat.status,
        "g6_detail": sat.detail,
        "store_prefix_identical": all(
            start_hashes[k] == end_hashes[k] for k in start_hashes),
        "negation_control": {
            "negated_surface_vetoes": arm in ("M1", "M1+M2") or arm in pf.M3_ARMS,
            "wrong_tag_arm": arm == "SHAM",
            "reject_condition_probe": reject_probe,
            "admitted_on_wrong_subject": counters.get("staged_m1", 0) > 0
            and arm == "SHAM" and counters.get("picks", 0) > 0,
        },
    }
    failed = []
    if not guard["preflight_ok"] or not guard["sha_pins_match"]:
        failed.append("F1")
    if ((not guard["delayed_option_audit"]["ok"]) or not controls_ok
            or guard["backend_error"] is not None
            or (sat.status == "FLAG" and mutations.get("moved") is False)):
        failed.append("F2")
    if (guard["negation_control"]["admitted_on_wrong_subject"]
            or reject_probe["picks"] > 0):
        failed.append("F5")
    status = "INSTRUMENT INVALID" if failed else "OK"
    report = {
        "schema": "m123-combined-arm-v1",
        "stage": stage, "arm": arm, "tag": tag,
        "status": status, "scored": not bool(failed),
        "falsifiers_local": failed,
        "config": {
            "arm_flags": dict(pf.ARM_FLAGS[arm]),
            "env_flags": {k: os.environ.get(k) for k in pf.ARM_FLAGS[arm]},
            "base": args.base, "arm_dir": arm_dir,
            "max_per_pool": args.max_per_pool,
            "items": [it.get("item_id") for it in items],
            "item_count": len(items), "bank_count": len(all_items),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "ast_anti_leakage": preflight["ast_anti_leakage"],
        "preflight_result": os.path.join(pf.HERE, "m123_preflight_result.json"),
        "harness": {
            "summary": summary,
            "controls": checks,
            "decision_rules": rules,
            "harness_pass": harness_pass,
            "saturation": {"status": sat.status, "detail": sat.detail},
            "mutation_control": mutations,
        },
        "endpoints": endpoints,
        "counters": counters,
        "guard": guard,
        "store": {"start": start_hashes, "end": end_hashes,
                  "changed": {k: [start_hashes[k], end_hashes[k]]
                              for k in start_hashes
                              if start_hashes[k] != end_hashes[k]}},
        "elapsed_s": elapsed,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # Full per-row audit trail (scratch-only): needed for row-level far
    # immediate/delayed endpoints and falsifiers F5-F9. No score is computed
    # from these rows by the runner; summarize() remains authoritative.
    rows_path = os.path.abspath(args.out) + ".rows.json"
    with open(rows_path, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1, ensure_ascii=False, default=str)
    report["rows_path"] = rows_path
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=str)
    print("CONFIG", json.dumps({"stage": stage, "arm": arm, "tag": tag,
                                "flags": report["config"]["env_flags"],
                                "items": len(items), "base": args.base}, default=str),
          flush=True)
    print("RESULT", json.dumps({"status": status, "scored": report["scored"],
                                "falsifiers": failed,
                                "endpoints": endpoints,
                                "counters": {k: counters.get(k) for k in (
                                    "turns", "query_turns", "emitted_picks",
                                    "hook_calls", "scorer_calls", "m1_act_writes",
                                    "staged_m1", "staged_m2", "process_turn",
                                    "repair_misleading_replaced",
                                    "repair_selffloop_reaffirmed",
                                    "api_error")},
                                "controls_failed": guard["failed_controls"],
                                "g6": sat.status,
                                "elapsed_s": elapsed}, default=str), flush=True)
    print("DONE", flush=True)
    return 0 if not failed else 2


def seed_data_dir_hash_only(arm_dir, start_hashes):
    out = {}
    for name in start_hashes:
        p = os.path.join(arm_dir, name)
        out[name] = sha256_file(p) if os.path.isfile(p) else "MISSING"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=pf.STAGES, required=True)
    ap.add_argument("--arm", choices=pf.ARMS, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-pool", type=int, default=0)
    args = ap.parse_args(argv)
    if args.tag != pf.TAGS[args.arm]:
        print("CONFIG arm/tag mismatch: %s expects %s" % (args.arm, pf.TAGS[args.arm]))
        print("DONE")
        return 2
    preflight_path = os.path.join(pf.HERE, "m123_preflight_result.json")
    if not os.path.isfile(preflight_path):
        print("CONFIG missing preflight artifact -> INSTRUMENT INVALID")
        print("DONE")
        return 2
    with open(preflight_path, "r", encoding="utf-8") as fh:
        preflight = json.load(fh)
    if not preflight.get("ok"):
        print("CONFIG preflight not OK -> INSTRUMENT INVALID, no engine boot")
        print("DONE")
        return 2
    try:
        return run_stage(args, preflight)
    except Exception as exc:
        print("CONFIG runner exception: %r" % (exc,), flush=True)
        traceback.print_exc()
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"stage": args.stage, "arm": args.arm, "tag": args.tag,
                       "status": "INSTRUMENT INVALID", "scored": False,
                       "error": repr(exc), "traceback": traceback.format_exc()},
                      fh, indent=1)
        print("RESULT status=INSTRUMENT INVALID scored=false", flush=True)
        print("DONE", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
