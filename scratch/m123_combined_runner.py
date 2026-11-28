#!/usr/bin/env python3
"""M1+M2+M3 combined experiment runner (one process, one arm, one store copy).

CLI (exact contract from M123_COMBINED_PREREG.md section 7)::

  python scratch/m123_combined_runner.py --stage {trak,rsi,tra} \\
      --arm {OFF,OFF2,M1,M1+M2,M1+M2+M3,SHAM} --tag <tag> \\
      --base <isolated data root> --out <report.json> [--max-per-pool N]

With --taught-focus (TRA-K only) the focused subset is the k=8 item plus one
taught k>=1 item per pool; k=0 stays a cold-baseline control endpoint. This
is an INSTRUMENT-SELECTION REPAIR only: no gate, threshold, item, prompt,
gold key, scorer or production default path is changed.

Every invocation uses a FRESH store copy, the shipped harness's own
run_sweep()/summarize()/instrument_controls(), and the shared arm primitives in
m123_preflight.py.  It never reads gold/type_sets/world_model into staging or
scoring, and it fails closed (exit 2, no score) on any preflight or instrument
control violation.
"""
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import hashlib
import inspect
import json
import os
import re
import shutil
import sys
import textwrap
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
from ravana.learn import source_arbitration as sa        # noqa: E402  (pure, default-off)

ISA = frozenset({"is_a", "type_of", "instance_of"})
# Source-only provenance projection.  The benchmark's evidence_role and every
# answer-side field are deliberately absent: credibility never reads them.
SOURCE_FIELDS = ("subject", "predicate", "object", "source_id", "source",
                 "source_url", "license", "license_url", "sense",
                 "text_evidence_kind")
EXTRA_BANNED = frozenset({
    "evidence_role", "item_id", "pool", "display_gold", "unseen_test_words",
    "negative", "foil", "unrelated", "subject_type", "misleading",
    "condition", "k_levels", "max_per_pool", "role", "difficulty",
    "domain", "far_domain",
})

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


# ── default-off source arbitration (runner-side, caller-supplied provenance) ──
def norm(value):
    return str(value or "").strip().casefold()


def channel_source_facts(item):
    """Project ONLY licensed source provenance for arbitration staging.

    The benchmark's evidence_role and all answer-side fields are excluded, so
    no staging decision can depend on a gold key, domain or answer field.
    """
    out = []
    for f in (item.get("channel_evidence", []) or ()):
        if not isinstance(f, dict):
            continue
        out.append({k: f.get(k) for k in SOURCE_FIELDS})
    return out


def source_facts_for_teaching(item, k, taught_ever):
    """Stage source provenance only for an item taught at k>=1 in this arm.

    k=0 is the harness's neutral-exposure null and the cold baseline has not
    been taught yet; both must leave the overlay empty. A delayed paraphrase
    is taught=False but ``taught_ever`` is True, so it keeps the overlay.
    """
    if int(k or 0) < 1 or not taught_ever:
        return []
    return channel_source_facts(item)


def as_assertion(fact, corroboration=1):
    """Module input built from source fields alone; no item or answer value."""
    return sa.SourceAssertion(
        subject=str(fact.get("subject") or ""),
        relation=str(fact.get("predicate") or "is_a"),
        label=str(fact.get("object") or ""),
        license=str(fact.get("license") or ""),
        sense=str(fact.get("sense") or ""),
        evidence_kind=str(fact.get("text_evidence_kind") or ""),
        origin_id=str(fact.get("source_id") or ""),
        corroboration=int(corroboration))


def arbitrated_views(subject, facts, arm, counters):
    """Rebuild one subject's typed-slot views through the default-off module.

    Record order is fixed by source credibility alone. Every record costs one
    self-loop decision and every conflict one arbitrate call in BOTH the
    candidate and the wrong-tag control, so the control is compute-matched.
    Returns the engine-view shape the reader already consumes.
    """
    who = norm(subject)
    stream = [f for f in (facts or ())
              if norm(f.get("subject")) == who and norm(f.get("predicate")) in ISA]
    stream.sort(key=lambda f: (-sa.source_credibility(as_assertion(f)),
                               str(f.get("object") or ""),
                               str(f.get("source_id") or "")))
    traces, incumbent = {}, None
    for fact in stream:
        assertion = as_assertion(fact)
        cred = sa.source_credibility(assertion)
        counters["self_loop_decisions"] = counters.get("self_loop_decisions", 0) + 1
        loop = sa.self_loop_decision(subject, assertion.label, cred)
        label = str(fact.get("object") or "").strip()
        if incumbent is None:
            counters["slot_decisions"] = counters.get("slot_decisions", 0) + 1
            decision = sa.arbitrate(None, assertion)
            if decision.action in (sa.Action.ADMIT, sa.Action.REINFORCE):
                counters["slot_writes"] = counters.get("slot_writes", 0) + 1
                if loop is sa.Action.WRITE_IDENTITY_SLOT:
                    counters["identity_slots"] = counters.get("identity_slots", 0) + 1
                incumbent = {"label": label, "assertion": assertion, "cred": cred,
                             "conf": decision.slot_confidence, "kind": "slot"}
                traces[label] = dict(incumbent)
            continue
        counters["arbitration_calls"] = counters.get("arbitration_calls", 0) + 1
        decision = sa.arbitrate(incumbent["assertion"], assertion)
        action = decision.action
        if action is sa.Action.REINFORCE:
            continue
        if action is sa.Action.SUPERSEDE:
            counters["supersedes"] = counters.get("supersedes", 0) + 1
            incumbent["conf"] = 0.5 * float(incumbent["conf"])
            incumbent["kind"] = "demoted"
            incumbent = {"label": label, "assertion": assertion, "cred": cred,
                         "conf": decision.slot_confidence, "kind": "slot"}
            if loop is sa.Action.WRITE_IDENTITY_SLOT:
                counters["identity_slots"] = counters.get("identity_slots", 0) + 1
            traces[label] = dict(incumbent)
        elif action is sa.Action.SEPARATE:
            counters["separations"] = counters.get("separations", 0) + 1
            traces[label] = {"label": label, "cred": cred,
                             "conf": decision.alternative_confidence,
                             "kind": "separate_alternative"}
    if incumbent is not None:
        traces[incumbent["label"]] = dict(incumbent)
    views = []
    for label, trace in sorted(traces.items()):
        if trace.get("kind") in ("demoted", "skipped"):
            continue
        views.append({"source": subject, "relation_type": "is_a",
                      "target": label, "type_label": label,
                      "edge_kind": "curated", "edge_source": "conversation",
                      "source_type": "user",
                      "confidence": float(trace.get("conf") or 0.0)})
    return views


def ast_anti_leakage_arbitration():
    """AST scan of every NEW arbitration decision/staging function."""
    names = ("channel_source_facts", "source_facts_for_teaching",
             "as_assertion", "arbitrated_views")
    banned = frozenset(pf.BANNED) | EXTRA_BANNED
    problems, scanned = [], []
    for name in names:
        fn = globals().get(name)
        try:
            src = textwrap.dedent(inspect.getsource(fn))
            tree = ast.parse(src)
        except Exception as exc:
            problems.append("parse %s: %r" % (name, exc))
            continue
        body = tree.body[0].body if isinstance(tree.body[0], ast.FunctionDef) else tree.body
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            body = body[1:]
        hits = []
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Name) and node.id in banned:
                hits.append("name=" + node.id)
            if isinstance(node, ast.Attribute) and node.attr in banned:
                hits.append("attr=" + node.attr)
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value in pf.BANNED_SUBSCRIPT):
                hits.append("subscript=" + str(node.slice.value))
        scanned.append({"function": name, "hits": sorted(set(hits))})
        problems.extend("%s: %s" % (name, h) for h in sorted(set(hits)))
    return {"ok": not problems, "problems": problems, "scanned": scanned,
            "banned_tokens": sorted(banned)}


def boot_free_self_test_arbitration():
    """No engine boot: flag isolation, matched control, no-key input, AST."""
    import types as _types
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    ast = ast_anti_leakage_arbitration()
    add("arbitration_ast_anti_leakage", ast["ok"], ast["problems"])

    add("module_default_off", sa.ENV_FLAG == "RAVANA_SOURCE_ARBITRATION"
        and not sa.enabled(), sa.ENV_FLAG)
    add("only_arbitration_arm_sets_flag",
        pf.ARM_FLAGS["ARBITRATION"].get(sa.ENV_FLAG) == "1"
        and all(pf.ARM_FLAGS[a].get(sa.ENV_FLAG) != "1"
                for a in pf.ARM_FLAGS if a != "ARBITRATION"),
        str({a: pf.ARM_FLAGS[a].get(sa.ENV_FLAG) for a in pf.ARM_FLAGS}))

    fact = {"subject": "guild", "predicate": "is_a", "object": "association",
            "license": "CC0-1.0", "sense": "primary",
            "text_evidence_kind": "item_description", "source_id": "s1"}
    previous = os.environ.get(sa.ENV_FLAG)
    try:
        os.environ.pop(sa.ENV_FLAG, None)
        off = sa.integration_hook(None, as_assertion(fact))
        os.environ[sa.ENV_FLAG] = "1"
        on = sa.integration_hook(None, as_assertion(fact))
    finally:
        if previous is None:
            os.environ.pop(sa.ENV_FLAG, None)
        else:
            os.environ[sa.ENV_FLAG] = previous
    add("module_hook_inert_until_enabled",
        off is None and on is not None and on.action is sa.Action.ADMIT,
        str((off, getattr(on, "action", None))))

    # Credibility cannot see the asserted label or a benchmark role field.
    other = dict(fact, object="student association", evidence_role="misleading")
    add("credibility_ignores_label_and_role",
        sa.source_credibility(as_assertion(fact)) == 0.7
        and sa.source_credibility(as_assertion(other)) == 0.7
        and set(channel_source_facts({"channel_evidence": [other]})[0]) <= set(SOURCE_FIELDS),
        str(sorted(channel_source_facts({"channel_evidence": [other]})[0])))

    # k=0 neutral null and cold baseline must stage nothing (no untaught leak).
    item0 = {"channel_evidence": [fact]}
    gate = (source_facts_for_teaching(item0, 0, True) == []
            and source_facts_for_teaching(item0, 8, False) == []
            and len(source_facts_for_teaching(item0, 8, True)) == 1)
    cnull = collections.Counter()
    add("k0_and_cold_baseline_overlay_inert",
        gate and arbitrated_views("guild", [], "ARBITRATION", cnull) == []
        and not cnull, str({"gate": gate, "counters": dict(cnull)}))

    # Wrong-tag/permuted SHAM is compute-matched to ARBITRATION on the SAME
    # source-derived views, and its reader emits no pick.
    facts = [fact, dict(fact, object="student association", sense="alternate",
                        text_evidence_kind="type_description", source_id="s2")]
    ca, cs = collections.Counter(), collections.Counter()
    va = arbitrated_views("guild", facts, "ARBITRATION", ca)
    vs = arbitrated_views("guild", facts, "SHAM", cs)
    add("sham_views_and_decisions_matched", va == vs and ca == cs,
        str({"views_equal": va == vs, "counters_equal": ca == cs,
             "arbitration": dict(ca), "sham": dict(cs)}))

    options = ["association", "student association"]
    snap = [_types.SimpleNamespace(type_name="association", source="s1",
                                   relation="is_a", source_url="u1"),
            _types.SimpleNamespace(type_name="student association", source="s2",
                                   relation="is_a", source_url="u2")]
    ra_counts, rs_counts = collections.Counter(), collections.Counter()
    ba = pf.stage_arm_evidence("guild", options, ledger_records=[], graph_views=va,
                               snapshot_evidence=snap, negated_pairs=set(),
                               arm="ARBITRATION", counters=ra_counts)
    bs = pf.stage_arm_evidence("guild", options, ledger_records=[], graph_views=vs,
                               snapshot_evidence=snap, negated_pairs=set(),
                               arm="SHAM", counters=rs_counts)
    ra = pf.reader_pick("guild", options, ba, "ARBITRATION", ra_counts)
    rs = pf.reader_pick("guild", options, bs, "SHAM", rs_counts)
    call_keys = ("production_calls", "scorer_calls", "trust_calls",
                 "records_scanned", "m1_scanned", "m2_scanned", "hook_calls")
    subjects_ok = (all(str(r.get("subject")) == "guild"
                       for group in (ba.m1_records, ba.m2_subject_evidence)
                       for r in group)
                   and all(str(r.get("subject")) == pf.permuted_subject("guild")
                           for group in (bs.m1_records, bs.m2_subject_evidence)
                           for r in group))
    add("sham_wrong_tag_permuted_reader_matched",
        ra == "association" and rs is None and subjects_ok
        and all(ra_counts[k] == rs_counts[k] for k in call_keys),
        str({"read": ra, "sham": rs,
             "calls": {k: (ra_counts[k], rs_counts[k]) for k in call_keys}}))

    return {"ok": all(c["ok"] for c in checks),
            "passed": sum(c["ok"] for c in checks), "total": len(checks),
            "checks": checks, "ast": ast}


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
            self.m123_source_facts = []
            self.m123_taught_item = False
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
                        if arm in ("ARBITRATION", "SHAM"):
                            # Write-side overlay from licensed source provenance
                            # only; both arms pay identical decision calls.
                            views = arbitrated_views(
                                subject, self.m123_source_facts, arm, counters)
                            counters["arbitration_overlay_calls"] = (
                                counters.get("arbitration_overlay_calls", 0) + 1)
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
                        if m and arm in ("M1", "M1+M2", "M1+M2+M3", "SHAM"):
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
            # Fresh per-item context: a cold baseline query must not inherit the
            # previous item's teaching facts (which would seed a spurious pick).
            self.m123_ctx = {"facts": [], "surfaces": [], "condition": ""}
            self.m123_source_facts = []
            self.m123_taught_item = False
            parent = getattr(super(), "reset_for_item", None)
            return parent(item) if callable(parent) else None

        def teach(self, *args, **kwargs):
            item = args[0] if args else kwargs.get("item")
            surfaces = args[4] if len(args) > 4 else kwargs.get("surfaces")
            condition = args[2] if len(args) > 2 else kwargs.get("condition")
            k_val = args[1] if len(args) > 1 else kwargs.get("k")
            if item is not None:
                self.m123_ctx["facts"] = allowed_facts(item)
                # A k>=1 call with real surfaces marks the item as taught;
                # the cold baseline and the delayed paraphrase both pass no
                # surfaces, so this persistent flag decides the overlay.
                if int(k_val or 0) >= 1 and surfaces:
                    self.m123_taught_item = True
                self.m123_source_facts = source_facts_for_teaching(
                    item, k_val, self.m123_taught_item)
            self.m123_ctx["surfaces"] = list(surfaces or [])
            self.m123_ctx["condition"] = str(condition or "")
            self.m123_ctx["item_id"] = str(getattr(item, "get", lambda _k: "")("item_id") or "")
            return super().teach(*args, **kwargs)

        def present(self, *args, **kwargs):
            item = args[1] if len(args) > 1 else kwargs.get("item")
            condition = args[3] if len(args) > 3 else kwargs.get("condition")
            if item is not None:
                self.m123_ctx["facts"] = allowed_facts(item)
                self.m123_source_facts = channel_source_facts(item)
            self.m123_ctx["surfaces"] = []
            self.m123_ctx["condition"] = str(condition or "")
            return super().present(*args, **kwargs)

    return ArmBackend(data_dir=arm_dir)


# ── instrument-selection repair: taught-only focused subset ──────────────
SELECT_BANNED = frozenset(pf.BANNED) | frozenset({"display_gold", "gold_target"})


def focused_items_taught(items, max_per_pool):
    """Graded focus with ONLY k>=1 (taught) items; k=0 stays a baseline control.

    Per pool keep the round-robin k=8 item, then the first k>=1 non-8 item.
    Selection reads only 'pool' and 'k': no answer-side field can influence
    which items are graded. The unmodified harness summarize() still emits the
    cold k=0 baseline endpoint for every selected item.
    """
    if not max_per_pool:
        return list(items)
    keep = []
    for pool in ("primary", "sealed"):
        by_k = {k: [it for it in items if it["pool"] == pool and it["k"] == k]
                for k in mtk.K_LEVELS}
        if by_k.get(8):
            keep.append(by_k[8].pop(0))
        remaining = max_per_pool - 1
        for k in mtk.K_LEVELS:
            if k == 8 or k < 1 or not by_k.get(k):
                continue
            for _ in range(min(remaining, len(by_k[k]))):
                keep.append(by_k[k].pop(0))
                remaining -= 1
            if remaining <= 0:
                break
    return keep


def ast_anti_leakage_taught_focus():
    """AST scan of the selection repair against answer-side token names."""
    problems, scanned = [], []
    for name in ("focused_items_taught",):
        fn = globals().get(name)
        try:
            body = textwrap.dedent(inspect.getsource(fn))
            tree = ast.parse(body)
            body = tree.body[0].body if isinstance(tree.body[0], ast.FunctionDef) else tree.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]
        except Exception as exc:
            problems.append("parse %s: %r" % (name, exc))
            continue
        hits = []
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Name) and node.id in SELECT_BANNED:
                hits.append("name=" + node.id)
            if isinstance(node, ast.Attribute) and node.attr in SELECT_BANNED:
                hits.append("attr=" + node.attr)
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value in pf.BANNED_SUBSCRIPT):
                hits.append("subscript=" + str(node.slice.value))
        scanned.append({"function": name, "hits": sorted(set(hits))})
        problems.extend("%s: %s" % (name, h) for h in sorted(set(hits)))
    return {"ok": not problems, "problems": problems, "scanned": scanned,
            "banned_tokens": sorted(SELECT_BANNED)}


def select_items(stage, mod, items, max_per_pool, taught_focus=False):
    if stage == "trak":
        return (focused_items_taught(items, max_per_pool) if taught_focus
                else pf.focused_items(items, max_per_pool or 2))
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
    out = {"applicable": arm in ("M1+M2+M3", "SHAM", "ARBITRATION"),
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
            if arm in ("ARBITRATION", "SHAM") and int(it.get("k") or 0) >= 1:
                cnt_tmp = collections.Counter()
                scoped = arbitrated_views(subject, channel_source_facts(it),
                                          arm, cnt_tmp)
            elif arm in ("ARBITRATION", "SHAM"):
                scoped = []   # k=0 null: nothing was taught, so no slot views
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
        items = select_items(stage, mod, all_items, args.max_per_pool or 2,
                             args.taught_focus)
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
            "negated_surface_vetoes": arm in ("M1", "M1+M2", "M1+M2+M3",
                                              "SHAM", "ARBITRATION"),
            "wrong_tag_arm": arm == "SHAM",
            "reject_condition_probe": reject_probe,
            "admitted_on_wrong_subject": counters.get("staged_m1", 0) > 0
            and arm == "SHAM" and counters.get("picks", 0) > 0,
        },
        "arbitration": {
            "module": getattr(sa, "__file__", ""),
            "env_flag": {sa.ENV_FLAG: os.environ.get(sa.ENV_FLAG)},
            "flag_isolation_ok": (os.environ.get(sa.ENV_FLAG) == "1") ==
                                 (arm == "ARBITRATION"),
            "ast_anti_leakage": ast_anti_leakage_arbitration(),
            "overlay_calls": counters.get("arbitration_overlay_calls", 0),
            "module_decisions": {k: counters.get(k, 0) for k in (
                "self_loop_decisions", "slot_decisions", "slot_writes",
                "arbitration_calls", "supersedes", "separations",
                "identity_slots")},
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
    if not guard["arbitration"]["flag_isolation_ok"]:
        failed.append("F6")
    if not guard["arbitration"]["ast_anti_leakage"]["ok"]:
        failed.append("F7")
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
            "focus_mode": ("taught_only_k_ge_1" if args.taught_focus
                           else "legacy_round_robin"),
            "items": [it.get("item_id") for it in items],
            "item_count": len(items), "bank_count": len(all_items),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "arbitration": {
                "module": getattr(sa, "__file__", ""),
                "module_env_flag": sa.ENV_FLAG,
                "arm_uses_overlay": arm in ("ARBITRATION", "SHAM"),
                "sham_is_wrong_tag_permuted": arm == "SHAM",
            },
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
                                    "api_error", "arbitration_calls",
                                    "arbitration_overlay_calls", "separations",
                                    "identity_slots")},
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


def boot_free_self_test_taught_focus():
    """No engine boot: taught-only graded focus + k=0 baseline still reported."""
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    ast = ast_anti_leakage_taught_focus()
    add("taught_focus_ast_anti_leakage", ast["ok"], ast["problems"])

    bank, _channel = pf.load_bank()
    legacy = pf.focused_items(bank, 2)
    focused = focused_items_taught(bank, 2)
    per_pool = {p: [{"item_id": it["item_id"], "k": it["k"]}
                    for it in focused if it["pool"] == p]
                for p in ("primary", "sealed")}
    taught_ok = (len(focused) == 4
                 and all(int(it["k"]) >= 1 for it in focused)
                 and all(sum(1 for it in focused
                             if it["pool"] == p and it["k"] == 8) == 1
                         for p in ("primary", "sealed")))
    add("taught_focus_only_taught_items", taught_ok,
        {"focus": per_pool,
         "legacy": [(it["item_id"], it["k"]) for it in legacy]})

    legacy_pairs = [(it["item_id"], it["k"]) for it in legacy]
    focus_pairs = [(it["item_id"], it["k"]) for it in focused]
    add("taught_focus_replaces_untought_k0",
        focus_pairs != legacy_pairs
        and all(any(it["pool"] == p and it["k"] == 0 for it in legacy)
                for p in ("primary", "sealed")),
        {"legacy": legacy_pairs, "focus": focus_pairs})

    minimal = [{"pool": it["pool"], "k": it["k"]} for it in bank]
    minimal_pairs = [(it["pool"], it["k"])
                     for it in focused_items_taught(minimal, 2)]
    full_pairs = [(it["pool"], it["k"]) for it in focused]
    add("taught_focus_ignores_answer_side_fields",
        minimal_pairs == full_pairs,
        {"minimal": minimal_pairs, "full_bank": full_pairs})

    # Unmodified summarize() on the taught focus via the shipped fake backend:
    # no engine boot, no store write. The cold k=0 baseline must still appear.
    fake = mtk.FakeBackend()
    rows = mtk.run_sweep(focused, backend=fake)
    summary = mtk.summarize(focused, rows)
    baseline = {p: summary[p].get("baseline") for p in ("primary", "sealed")}
    far_n = {p: summary[p].get("far_n") for p in ("primary", "sealed")}
    k0_far = {p: summary[p]["by_k"]["0"]["far"]
              for p in ("primary", "sealed")}
    add("taught_focus_k0_baseline_reported",
        all(isinstance(baseline[p], (int, float)) for p in baseline)
        and all(far_n[p] == 2 for p in far_n)
        and all(k0_far[p] is None for p in k0_far),
        {"baseline": baseline, "far_n": far_n, "k0_far": k0_far,
         "k0_minus_baseline": {p: summary[p].get("k0_minus_baseline")
                               for p in ("primary", "sealed")}})

    return {"ok": all(c["ok"] for c in checks),
            "passed": sum(c["ok"] for c in checks), "total": len(checks),
            "checks": checks, "ast": ast}


def run_self_test(out_path):
    """Boot-free proof of arm wiring. No engine, no store write, no network."""
    for flag in pf.CLEAR_FLAGS:
        os.environ.pop(flag, None)
    preflight = pf.boot_free_self_test()
    arb = boot_free_self_test_arbitration()
    taught = boot_free_self_test_taught_focus()
    ast = pf.ast_anti_leakage()
    pins = pf.pin_hashes()
    gold = pf.gold_mutation_fingerprint()
    checks = {
        "preflight_self_test": preflight,
        "arbitration_self_test": arb,
        "taught_focus_self_test": taught,
        "preflight_ast_anti_leakage": ast,
        "arbitration_ast_anti_leakage": arb["ast"],
        "sha_pins": {k: v.get("match") for k, v in pins.items()},
        "gold_mutation_invariant": gold["ok"],
        "engine_booted": False,
        "stores_written": False,
        "production_files_edited": 0,
    }
    ok = (preflight["ok"] and arb["ok"] and taught["ok"] and ast["ok"]
          and all(v for v in checks["sha_pins"].values()) and gold["ok"])
    report = {"schema": "trak-arbitration-live-selftest-v1",
              "ok": bool(ok), "checks": checks,
              "module": getattr(sa, "__file__", ""),
              "env_flag": sa.ENV_FLAG}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=str)
    print("SELFTEST", "OK" if ok else "INVALID",
          json.dumps({"preflight": [c["name"] for c in preflight["checks"]
                                    if not c["ok"]],
                      "arbitration": [c["name"] for c in arb["checks"]
                                      if not c["ok"]],
                      "taught_focus": [c["name"] for c in taught["checks"]
                                       if not c["ok"]],
                      "pins_failed": [k for k, v in checks["sha_pins"].items()
                                      if not v]}, default=str), flush=True)
    print("DONE", flush=True)
    return 0 if ok else 2


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                    help="boot-free arbitration wiring self-test; no engine boot")
    ap.add_argument("--stage", choices=pf.STAGES)
    ap.add_argument("--arm", choices=pf.ARMS)
    ap.add_argument("--tag")
    ap.add_argument("--base")
    ap.add_argument("--out")
    ap.add_argument("--max-per-pool", type=int, default=0)
    ap.add_argument("--taught-focus", action="store_true",
                    help="TRA-K only: graded focus = k=8 + one taught k>=1 item/pool")
    args = ap.parse_args(argv)
    if args.taught_focus and args.stage != "trak":
        ap.error("--taught-focus is only valid with --stage trak")
    if args.self_test:
        return run_self_test(os.environ.get("TRAK_SELFTEST_OUT") or
                             os.path.join(pf.HERE, "trak_arbitration_live_selftest.json"))
    missing = [k for k in ("stage", "arm", "tag", "base", "out")
               if getattr(args, k) is None]
    if missing:
        ap.error("missing required for an arm run: %s" % ", ".join(missing))
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
