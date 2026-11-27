#!/usr/bin/env python3
"""BOOT-FREE source-arbitration experiment for TRA-K far (scratch only).

Reads saved stores/bank/snapshot read-only, constructs NO engine, makes NO
process_turn call, writes NO store, imports UNMODIFIED production/staging
primitives, and grades every arm with the UNMODIFIED M3 reader.

Pre-registration: docs/investigations/TRA_K_SOURCE_ARBITRATION_PREREG.md

Usage (from the isolated worktree):
  PYTHONHASHSEED=0 RAVANA_OFFLINE=1 \
  TRAK_CHANNEL_JSON=<wt>/scratch/trak_channel_from_substrate.json \
  <parent>/.venv/Scripts/python.exe scratch/trak_source_arbitration_experiment.py
"""
from __future__ import annotations

import ast
import collections
import copy
import hashlib
import inspect
import json
import os
import pickle
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
PARENT = "C:/Users/Likhith/Documents/projects/ravana"
RUN = "C:/Users/Likhith/Documents/projects/ravana_m123_run"
for p in (WT, os.path.join(WT, "scripts"), os.path.join(WT, "ravana", "src"),
          os.path.join(WT, "ravana_ml", "src"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import m123_preflight as pf                    # noqa: E402
import m3_graded_probe as m3                   # noqa: E402
import measure_tra_k as mtk                   # noqa: E402
from ravana.chat.schema_induction import (     # noqa: E402
    CONGRUENCE_FLOOR, STRICT_MARGIN, typed_slot_views_from_graph)
from ravana.learn.subject_type_evidence import (  # noqa: E402
    LICENSE_CREDIBILITY, load_snapshot)

ARMS = ("BASELINE", "BASELINE2", "ARBITRATION", "SHAM", "K0")
ITEM_IDS = ("tra_k_0020", "tra_k_0008", "tra_k_0000", "tra_k_0006")
STORE = os.path.join(RUN, "scratch", "m123_data", "trak_m123")
MARGIN = 0.05
SENSE_WEIGHT = {"primary": 1.0, "alternate": 0.5}
TIER_WEIGHT = {"item_description": 1.0, "type_description": 0.75,
               "label_only": 0.5}
ISA = frozenset({"is_a", "type_of", "instance_of"})

# Extra answer-side tokens, in addition to the preflight's BANNED set, must
# not appear in any new decision/staging function.
EXTRA_BANNED = frozenset({
    "evidence_role", "item_id", "pool", "display_gold", "unseen_test_words",
    "negative", "foil", "unrelated", "subject_type", "misleading",
    "condition", "k_levels", "max_per_pool",
})


def norm(value):
    return str(value or "").strip().casefold()


def source_credibility(fact):
    """Source-only credibility from licensed provenance; no answer-side field.

    Reads license, sense and evidence tier from the supplied source record.
    '' when the record is not source-admitted.
    """
    license_name = str(fact.get("license") or "CC0-1.0").strip()
    ceiling = LICENSE_CREDIBILITY.get(license_name, 0.0)
    sense = SENSE_WEIGHT.get(str(fact.get("sense") or "").strip().lower(), 0.0)
    tier = TIER_WEIGHT.get(str(fact.get("text_evidence_kind") or "").strip().lower(), 0.0)
    if not ceiling or not sense or not tier:
        return 0.0
    return round(ceiling * sense * tier, 12)


def slot_confidence(cred):
    """Write strength for an admitted slot trace (bounded, monotone in cred)."""
    return 0.5 + 0.5 * float(cred)


def alternative_confidence(cred):
    """Separate trace strength: stays below the production congruence floor."""
    return 0.2 * float(cred)


def arbitrate(incumbent, challenger):
    """Decide one typed-slot conflict from source credentials alone.

    ``incumbent`` and ``challenger`` are mappings with keys 'label' and
    'cred'. Returns (action, reason). No gate, floor or margin is redefined:
    MARGIN here is a property of the new, default-off arbitration module.
    """
    label_i = norm(incumbent.get("label"))
    label_c = norm(challenger.get("label"))
    cred_i = float(incumbent.get("cred") or 0.0)
    cred_c = float(challenger.get("cred") or 0.0)
    if not label_c or cred_c <= 0.0:
        return "REJECT", "unadmitted_source"
    if label_i == label_c:
        return "REINFORCE", "same_label"
    if cred_c > cred_i + MARGIN:
        return "SUPERSEDE", "higher_credibility_challenger"
    return "SEPARATE", "incumbent_source_not_inferior"


def self_loop_decision(subject, obj, cred):
    """Identity assertion handling, derived only from the two endpoints."""
    if norm(subject) != norm(obj):
        return "NORMAL", "different_endpoints"
    if float(cred or 0.0) <= 0.0:
        return "SKIP", "unadmitted_source"
    return "WRITE_IDENTITY_SLOT", "admitted_identity_assertion"


def source_fields(fact):
    """Source-only projection of a channel record; no role or answer field."""
    return {k: fact.get(k) for k in (
        "subject", "object", "predicate", "license", "sense",
        "text_evidence_kind", "source_id", "source_url")}


def simulate_stream(who, stream, arm):
    """Replay one recorded teaching stream through the candidate rule.

    Returns an audit dict. BASELINE reproduces production `_revise` semantics
    (halve incumbent on every conflict, skip sid==oid); ARBITRATION applies
    `arbitrate` and `self_loop_decision`. Decision-call counts are equal.
    """
    cnt = collections.Counter()
    conflicts = []
    incumbent = None
    traces = {}
    for fact in stream:
        label = str(fact.get("object") or "").strip()
        cred = source_credibility(fact)
        if norm(fact.get("subject")) != norm(who) or not label:
            continue
        cnt["events"] += 1
        loop = self_loop_decision(who, label, cred)
        cnt["self_loop_decisions"] += 1
        if norm(who) == norm(label) and arm in ("BASELINE", "BASELINE2"):
            traces[label] = {"label": label, "cred": cred, "conf": 0.0,
                             "kind": "skipped_self_loop"}
            continue
        if incumbent is None:
            incumbent = {"label": label, "cred": cred, "source": source_fields(fact),
                         "conf": slot_confidence(cred), "kind": "slot"}
            traces[label] = dict(incumbent)
            cnt["slot_writes"] += 1
            continue
        action, reason = arbitrate(incumbent, {"label": label, "cred": cred})
        cnt["arbitrate_calls"] += 1
        conflicts.append({"challenger": label, "cred_challenger": cred,
                          "incumbent": incumbent["label"],
                          "cred_incumbent": incumbent["cred"],
                          "action": action, "reason": reason,
                          "challenger_source": source_fields(fact),
                          "incumbent_source": dict(incumbent["source"])})
        if arm in ("BASELINE", "BASELINE2"):
            action = "SUPERSEDE"
        if action == "REINFORCE":
            continue
        if action == "SUPERSEDE":
            traces[incumbent["label"]] = {
                "label": incumbent["label"], "cred": incumbent["cred"],
                "conf": 0.5 * incumbent["conf"], "kind": "demoted"}
            incumbent = {"label": label, "cred": cred, "source": source_fields(fact),
                         "conf": slot_confidence(cred), "kind": "slot"}
            traces[label] = dict(incumbent)
            cnt["supersedes"] += 1
        else:
            traces[label] = {"label": label, "cred": cred,
                             "conf": alternative_confidence(cred),
                             "kind": "separate_alternative"}
            cnt["separations"] += 1
    if incumbent is not None:
        traces[incumbent["label"]] = dict(incumbent)
    return {"traces": traces, "conflicts": conflicts, "counters": dict(cnt),
            "incumbent": dict(incumbent) if incumbent else None,
            "self_loop_rule_fired": any(
                c["challenger"].casefold() == who.casefold()
                for c in conflicts) or any(
                norm(t.get("label")) == norm(who)
                and t.get("kind") == "slot" for t in traces.values())}


def load_saved_views(item_id):
    """Read-only census of the stored typed views from a saved arm store."""
    state = pickle.load(open(os.path.join(STORE, item_id, "ravana_weights.pkl"), "rb"))
    graph = state["graph"]
    return [{k: getattr(v, k, "") for k in (
        "source", "relation_type", "target", "type_label", "edge_kind",
        "edge_source", "source_type", "confidence")}
        for v in typed_slot_views_from_graph(graph)]


def load_saved_records(item_id):
    from ravana.chat import durable_provenance as dp
    d = os.path.join(STORE, item_id)
    ledger = dp.DurableProvenanceStore(d).load().to_dict().get("records", {})
    return [dict(v) for v in ledger.values()]


NEGATED_CLAIM_RE = __import__("re").compile(
    r"\b([A-Za-z][A-Za-z\-]*)\s+(?:is|are|was|were)\s+not\s+(?:a|an)\s+"
    r"([A-Za-z][A-Za-z \-]*?)\s*[.!?]", __import__("re").IGNORECASE)


def negated_from_surfaces(surfaces):
    """Explicit negation pairs from teaching text only (no answer-side field)."""
    out = set()
    for surface in surfaces or ():
        match = NEGATED_CLAIM_RE.search(str(surface))
        if match:
            out.add((match.group(1).strip().casefold(),
                     match.group(2).strip().casefold()))
    return out


def source_ordered_stream(bundle_facts):
    """Order channel records by SOURCE metadata only; never by answer fields."""
    def key(f):
        return (-source_credibility(f), str(f.get("object") or ""),
                str(f.get("source_id") or ""))
    return sorted(bundle_facts, key=key)


def build_arm_views(saved_views, simulated, who_set, arm):
    """Fresh, arm-local view list; saved evidence for other subjects unchanged."""
    views = copy.deepcopy(saved_views)
    views = [v for v in views
             if not (norm(v.get("source")) in who_set
                     and norm(v.get("relation_type")) in ISA)]
    for who, sim in simulated.items():
        for trace in sim["traces"].values():
            if trace.get("kind") in ("skipped_self_loop", "demoted"):
                continue
            views.append({
                "source": who, "relation_type": "is_a", "target": trace["label"],
                "type_label": trace["label"], "edge_kind": "curated",
                "edge_source": "conversation", "source_type": "user",
                "confidence": trace["conf"]})
    return views


def ast_anti_leakage():
    """AST scan of every NEW decision/staging function (oracle excluded)."""
    names = ("source_credibility", "slot_confidence", "alternative_confidence",
             "arbitrate", "self_loop_decision", "simulate_stream",
             "source_ordered_stream", "build_arm_views", "source_fields")
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
                    and node.slice.value in banned):
                hits.append("subscript=" + str(node.slice.value))
        scanned.append({"function": name, "hits": sorted(set(hits))})
        problems.extend("%s: %s" % (name, h) for h in sorted(set(hits)))
    return {"ok": not problems, "problems": problems, "scanned": scanned,
            "banned_tokens": sorted(banned)}


def module_equivalence(simulations):
    """Check the shipped pure module agrees with the scratch rule on every
    recorded conflict. The module adds a corroboration guard and an absolute
    floor on FIRST slot writes; both are neutral on this recorded stream.
    """
    from ravana.learn import source_arbitration as sa

    def as_assertion(d):
        return sa.SourceAssertion(
            subject=str(d.get("subject") or ""),
            relation=str(d.get("predicate") or "is_a"),
            label=str(d.get("object") or ""),
            license=str(d.get("license") or "CC0-1.0"),
            sense=str(d.get("sense") or ""),
            evidence_kind=str(d.get("text_evidence_kind") or ""),
            origin_id=str(d.get("source_id") or ""),
            corroboration=1)

    checked, mismatches = 0, []
    for iid, sims in sorted(simulations.items()):
        for who, sim in sorted(sims.items()):
            for c in sim["conflicts"]:
                decision = sa.arbitrate(as_assertion(c["incumbent_source"]),
                                        as_assertion(c["challenger_source"]))
                checked += 1
                if decision.action.value != c["action"]:
                    mismatches.append({"item_id": iid, "subject": who,
                                       "scratch": c["action"],
                                       "module": decision.action.value})
    return {"checked": checked, "mismatches": mismatches,
            "ok": bool(checked) and not mismatches}


def sha256_normalised(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def snapshot_consistency(items):
    """Cross-check each source record's sense against the frozen CC0 snapshot."""
    snap = load_snapshot()
    by_key = {}
    for row in snap.get("facts", ()):
        by_key.setdefault((norm(row.get("subject")), norm(row.get("type_name")),
                           str(row.get("source_url") or "")), set()).add(
            str(row.get("sense") or ""))
    checked = mismatches = 0
    for it in items:
        for f in it.get("channel_evidence", ()):
            key = (norm(f.get("subject")), norm(f.get("object")),
                   str(f.get("source_url") or ""))
            senses = by_key.get(key)
            if not senses:
                continue
            checked += 1
            if str(f.get("sense") or "") not in senses:
                mismatches += 1
    return {"checked": checked, "mismatches": mismatches,
            "snapshot_sha256": sha256_normalised(
                os.path.join(WT, "data", "subject_type_evidence", "snapshot.json"))}


def read_arm(items, arm, snapshot, records_by_item, views_by_item):
    rows = []
    for it in items:
        iid = it["item_id"]
        for cond in ("near", "far", "foil", "negative", "unrelated"):
            q = mtk.query_for(it, cond)
            subject = str(q["subject"])
            options = [str(c) for c in q["candidates"]]
            surfaces = mtk.teaching_surfaces(it, it["k"], cond, mtk.CHANNEL_ARM)
            counts = collections.Counter()
            bundle = pf.stage_arm_evidence(
                subject, options, ledger_records=records_by_item[iid],
                graph_views=views_by_item[iid],
                snapshot_evidence=snapshot.get(subject, ()),
                negated_pairs=negated_from_surfaces(surfaces),
                arm=arm if arm in ("SHAM",) else "M1+M2+M3", counters=counts)
            decision = m3.graded_probe_pick(
                pf.permuted_subject(subject) if arm == "SHAM" else subject,
                options, bundle, enabled=True,
                trust=m3.wrong_tag_trust if arm == "SHAM" else m3._is_trusted_edge,
                counters=counts)
            pick = decision.pick if decision is not None else None
            reason = decision.reason if decision is not None else "no_decision"
            scores = dict(decision.scores) if decision is not None else {}
            rows.append({"item_id": iid, "pool": it["pool"], "condition": cond,
                         "subject": subject, "options": options,
                         "pick": pick, "reason": reason, "scores": scores,
                         "counters": dict(counts)})
    return rows


def main():
    os.environ.setdefault("RAVANA_OFFLINE", "1")
    os.environ.setdefault("RAVANA_SUBJECT_TYPE_EVIDENCE", "1")
    os.environ.setdefault("TRAK_CHANNEL_JSON", os.path.join(
        WT, "scratch", "trak_channel_from_substrate.json"))
    os.environ["RAVANA_M3_GRADED_PROBE"] = "0"
    items, channel = pf.load_bank()
    items = [i for i in items if i["item_id"] in ITEM_IDS]
    items.sort(key=lambda i: ITEM_IDS.index(i["item_id"]))
    snap = load_snapshot()
    ste_snapshot = {}
    from ravana.learn import subject_type_evidence as ste
    seen = set()
    for it in items:
        for cond in ("near", "far", "negative"):
            who = str(mtk.query_for(it, cond)["subject"])
            if who in seen:
                continue
            seen.add(who)
    for who in sorted(seen):
        ev, _ = ste.evidence_for_subject(who, snap)
        ste_snapshot[who] = ev
    consistency = snapshot_consistency(items)
    pins = pf.pin_hashes()
    report = {
        "config": {"base": "cd711c8", "worktree": WT, "parent": PARENT,
                   "item_ids": list(ITEM_IDS), "channel": channel,
                   "margins": {"arbitration": MARGIN,
                               "congruence_floor": CONGRUENCE_FLOOR,
                               "strict_margin": STRICT_MARGIN},
                   "engine_booted": False, "stores_written": False},
        "preflight": {"pins": pins,
                      "pins_ok": all(v.get("match") for v in pins.values()),
                      "ast": ast_anti_leakage(),
                      "snapshot_consistency": consistency,
                      "boot_free_self_test": pf.boot_free_self_test()["checks"]},
        "arms": {},
    }
    if not (report["preflight"]["pins_ok"]
            and report["preflight"]["ast"]["ok"]
            and consistency["mismatches"] == 0):
        report["verdict"] = "INSTRUMENT INVALID — no score"
        json.dump(report, open(os.path.join(HERE, "trak_source_arbitration_result.json"),
                               "w", encoding="utf-8"), indent=1)
        print(report["verdict"])
        return 2

    records_by_item = {i["item_id"]: load_saved_records(i["item_id"]) for i in items}
    saved_by_item = {i["item_id"]: load_saved_views(i["item_id"]) for i in items}
    for arm in ARMS:
        simulated, audit = {}, {}
        for it in items:
            iid = it["item_id"]
            who_set = {norm(mtk.query_for(it, c)["subject"])
                       for c in ("near", "far", "negative")}
            if arm == "K0":
                sims = {w: {"traces": {}, "conflicts": [], "counters": {},
                            "incumbent": None, "self_loop_rule_fired": False}
                        for w in sorted(who_set)}
            else:
                sims = {}
                for who in sorted(who_set):
                    stream = source_ordered_stream([
                        f for f in it.get("channel_evidence", ())
                        if norm(f.get("subject")) == who])
                    sims[who] = simulate_stream(who, stream, arm)
            simulated[iid] = sims
            audit[iid] = sims
        views = {i["item_id"]: build_arm_views(
            saved_by_item[i["item_id"]], simulated[i["item_id"]],
            {norm(mtk.query_for(i, c)["subject"])
             for c in ("near", "far", "negative")}, arm)
            for i in items}
        # Deep-copy per-arm inputs so no arm can mutate another's state.
        rows = read_arm(items, arm, ste_snapshot,
                        copy.deepcopy(records_by_item),
                        {k: copy.deepcopy(v) for k, v in views.items()})
        report["arms"][arm] = {
            "rows": rows, "simulation": audit,
            "reader_call_totals": {k: sum(r["counters"].get(k, 0) for r in rows)
                                   for k in ("records_scanned", "trust_calls",
                                             "scorer_calls", "records_admitted",
                                             "production_calls", "hook_calls")},
            "decision_calls": sum(
                sum(s["counters"].get("arbitrate_calls", 0) for s in sim.values())
                for sim in simulated.values()),
        }

    report["module_equivalence"] = module_equivalence(
        report["arms"]["ARBITRATION"]["simulation"])
    # ORACLE (NON-DEPLOYABLE): unmodified reader + answer-side confidence
    # injection, exactly as in the root-cause probe; excluded from AST scan.
    oracle_rows = []
    for it in items:
        iid = it["item_id"]
        for cond in ("near", "far"):
            q = mtk.query_for(it, cond)
            subject, options = str(q["subject"]), [str(c) for c in q["candidates"]]
            counts = collections.Counter()
            bundle = pf.stage_arm_evidence(
                subject, options, ledger_records=records_by_item[iid],
                graph_views=saved_by_item[iid],
                snapshot_evidence=ste_snapshot.get(subject, ()),
                negated_pairs=set(), arm="M1+M2+M3", counters=counts)
            target = str(q["gold"]) if q.get("gold") else ""
            for rec in (bundle.m1_records + bundle.m2_subject_evidence):
                if norm(rec.get("label")) == norm(target):
                    rec["confidence"] = 1.0
            dec = m3.graded_probe_pick(subject, options, bundle, enabled=True,
                                       counters=counts)
            oracle_rows.append({"item_id": iid, "pool": it["pool"],
                                "condition": cond, "subject": subject,
                                "injected_label": target,
                                "pick": dec.pick if dec is not None else None,
                                "reason": dec.reason if dec is not None else "no_decision",
                                "deployable": False})
    report["oracle_non_deployable"] = oracle_rows

    def far_rows(arm, cond="far"):
        return [r for r in report["arms"][arm]["rows"] if r["condition"] == cond]

    def rate(rows, pool=None):
        sub = [r for r in rows if pool is None or r["pool"] == pool]
        return (sum(r["pick"] is not None for r in sub) / len(sub)) if sub else None

    endpoints = {}
    for pool in ("primary", "sealed", None):
        endpoints[str(pool)] = {
            arm: {c: rate([r for r in report["arms"][arm]["rows"]
                           if r["condition"] == c], pool)
                  for c in ("near", "far", "foil", "negative", "unrelated")}
            for arm in ARMS}
    report["endpoints"] = endpoints
    checks = {}
    checks["A1_instrument"] = bool(report["preflight"]["pins_ok"]
                                   and report["preflight"]["ast"]["ok"]
                                   and consistency["mismatches"] == 0
                                   and report["module_equivalence"]["ok"])
    checks["A2_repeat_identical"] = all(
        (a["pick"], a["scores"]) == (b["pick"], b["scores"])
        for a, b in zip(far_rows("BASELINE"), far_rows("BASELINE2")))
    sham = report["arms"]["SHAM"]
    arb = report["arms"]["ARBITRATION"]
    call_keys = ("records_scanned", "trust_calls", "scorer_calls",
                 "production_calls", "hook_calls")
    checks["A3_sham_no_pick_and_matched_calls"] = (
        all(r["pick"] is None for r in sham["rows"])
        and all(sham["reader_call_totals"].get(k) == arb["reader_call_totals"].get(k)
                for k in call_keys)
        and sham["decision_calls"] == arb["decision_calls"])
    checks["A4_oracle_4of4"] = all(r["pick"] == r["injected_label"]
                                   for r in oracle_rows if r["condition"] == "far")
    baseline_demoted = []
    arb_restored = []
    for iid, sims in report["arms"]["BASELINE"]["simulation"].items():
        for who, sim in sims.items():
            for c in sim["conflicts"]:
                victim = sim["traces"].get(c["incumbent"], {})
                baseline_demoted.append({
                    "item_id": iid, "subject": who,
                    "victim": c["incumbent"],
                    "baseline_victim_conf": victim.get("conf"),
                    "challenger": c["challenger"],
                    "cred_challenger": c["cred_challenger"],
                    "cred_incumbent": c["cred_incumbent"]})
    for iid, sims in arb["simulation"].items():
        for who, sim in sims.items():
            for tr in sim["traces"].values():
                if tr["kind"] == "slot" and tr["conf"] >= CONGRUENCE_FLOOR:
                    arb_restored.append({"item_id": iid, "subject": who,
                                         "label": tr["label"], "conf": tr["conf"]})
    checks["A5_mechanism_fired"] = bool(baseline_demoted) and bool(arb_restored)
    selfloop_rows = [r for r in far_rows("ARBITRATION")
                     if r["subject"].casefold() == "market"]
    checks["A9_self_loop_both_items"] = (
        len(selfloop_rows) == 2
        and all(r["pick"] is not None for r in selfloop_rows))
    checks["A6_far_gain_both_pools"] = all(
        rate(far_rows("ARBITRATION"), pool) - rate(far_rows("BASELINE"), pool) >= 0.50
        and rate(far_rows("ARBITRATION"), pool) >= 0.50
        for pool in ("primary", "sealed"))
    checks["A7_no_regression"] = (
        all(rate(far_rows("ARBITRATION", c), p) <= rate(far_rows("BASELINE", c), p)
            for p in ("primary", "sealed")
            for c in ("foil", "negative", "unrelated"))
        and all(rate(far_rows("ARBITRATION", "near"), p)
                >= rate(far_rows("BASELINE", "near"), p) - 0.05
                for p in ("primary", "sealed")))
    checks["A8_k0_null_and_pins"] = (
        all(r["pick"] is None for r in far_rows("K0"))
        and report["preflight"]["pins_ok"])
    report["falsifiers"] = checks
    report["mechanism_evidence"] = {"baseline_demotions": baseline_demoted,
                                    "arbitration_restorations": arb_restored}
    report["verdict"] = ("CLEARED — implement default-off module"
                         if all(checks.values()) else "NOT CLEARED — negative result")
    out = os.path.join(HERE, "trak_source_arbitration_result.json")
    json.dump(report, open(out, "w", encoding="utf-8"), indent=1)
    print("VERDICT", report["verdict"])
    for k, v in checks.items():
        print("  %-28s %s" % (k, v))
    for pool in ("primary", "sealed"):
        print("  %s far BASELINE=%s ARBITRATION=%s SHAM=%s" % (
            pool, endpoints[pool]["BASELINE"]["far"],
            endpoints[pool]["ARBITRATION"]["far"], endpoints[pool]["SHAM"]["far"]))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
