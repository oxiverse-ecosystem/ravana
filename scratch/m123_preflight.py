#!/usr/bin/env python3
"""M1+M2+M3 combined experiment -- BOOT-FREE preflight and arm primitives.

Implements docs/investigations/M123_COMBINED_PREREG.md section 7 (preflight)
and supplies the arm staging primitives imported by m123_combined_runner.py.

Hard contract: importing/running this module constructs NO engine, opens NO
weight store for write, makes NO network call, and edits NO production file,
gate, threshold, item, prompt, gold key or scorer.  It only reads files and
runs pure functions.  The runner imports the arm primitives from here so the
same AST anti-leakage proof covers the functions actually used in the arms.
"""
from __future__ import annotations

import ast
import collections
import copy
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.dirname(HERE)
PARENT = "C:/Users/Likhith/Documents/projects/ravana"
for _p in (WORKTREE, os.path.join(WORKTREE, "scripts"),
           os.path.join(WORKTREE, "ravana", "src"),
           os.path.join(WORKTREE, "ravana_ml", "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

import front_gates                                   # noqa: E402
import measure_tra_k as mtk                          # noqa: E402
import tra_k_items as ti                             # noqa: E402
import m3_graded_probe as m3                         # noqa: E402
from ravana.chat import durable_provenance as dp     # noqa: E402
from ravana.chat.source_evaluation import (          # noqa: E402
    ClaimContext, Decision, SourceStatus, TrustClass, evaluate_source)
from ravana.learn import subject_type_evidence as ste  # noqa: E402

ARMS = ("OFF", "OFF2", "M1", "M1+M2", "M1+M2+M3",
        "REPAIR", "REPAIR_SELFLOOP", "SHAM")
TAGS = {"OFF": "off1", "OFF2": "off2", "M1": "m1",
        "M1+M2": "m12", "M1+M2+M3": "m123",
        "REPAIR": "repair", "REPAIR_SELFLOOP": "repair_self", "SHAM": "sham"}
STAGES = ("trak", "rsi", "tra")

# Scratch-only far-repair arm keys.  REPAIR_SELFLOOP is a strict superset of
# REPAIR; neither touches production, gates, thresholds, items, prompts, the
# gold key or the scorer.  M3_ARMS are the arms for which the shipped M3 reader
# is enabled (SHAM uses wrong_tag_trust and admits nothing).
REPAIR_ARMS = frozenset(("REPAIR", "REPAIR_SELFLOOP"))
M3_ARMS = frozenset(("M1+M2+M3", "SHAM")) | REPAIR_ARMS
RESULT_PATH = os.path.join(HERE, "m123_preflight_result.json")

# Literal env contract: default-OFF production flags, set only per arm.
ARM_FLAGS = {
    "OFF": {},
    "OFF2": {},
    "M1": {"RAVANA_DURABLE_PROVENANCE": "1"},
    "M1+M2": {"RAVANA_DURABLE_PROVENANCE": "1",
              "RAVANA_SUBJECT_TYPE_EVIDENCE": "1"},
    "M1+M2+M3": {"RAVANA_DURABLE_PROVENANCE": "1",
                 "RAVANA_SUBJECT_TYPE_EVIDENCE": "1",
                 "RAVANA_M3_GRADED_PROBE": "1"},
    "SHAM": {"RAVANA_DURABLE_PROVENANCE": "1",
             "RAVANA_SUBJECT_TYPE_EVIDENCE": "1",
             "RAVANA_M3_GRADED_PROBE": "1"},
    "REPAIR": {"RAVANA_DURABLE_PROVENANCE": "1",
               "RAVANA_SUBJECT_TYPE_EVIDENCE": "1",
               "RAVANA_M3_GRADED_PROBE": "1"},
    "REPAIR_SELFLOOP": {"RAVANA_DURABLE_PROVENANCE": "1",
                        "RAVANA_SUBJECT_TYPE_EVIDENCE": "1",
                        "RAVANA_M3_GRADED_PROBE": "1"},
}
CLEAR_FLAGS = ("RAVANA_TRA_SLOTS", "RAVANA_TRA_K_SLOTS", "RAVANA_SOURCE_EVAL",
               "RAVANA_PRIOR_ACT_TRACE", "RAVANA_LEXICAL_SUPERORDINATE")

BANNED = frozenset({
    "gold", "type_sets", "world_model", "answer_key", "answer_key_leak",
    "typed_queries", "counterexample_target", "forbidden_channel_subjects",
    "oracle", "q_gold", "item_gold",
})
BANNED_SUBSCRIPT = frozenset({"gold", "type_sets", "world_model", "answer_key"})

PINNED = (
    "scripts/measure_tra_k.py", "scripts/measure_learn5_rsi.py",
    "scripts/measure_tra.py", "scripts/front_gates.py",
    "scripts/tra_k_items.py", "data/subject_type_evidence/snapshot.json",
    "scratch/trak_channel_from_substrate.json",
    "scratch/m3_graded_probe.py",
    "ravana/src/ravana/chat/durable_provenance.py",
    "ravana/src/ravana/chat/source_evaluation.py",
    "ravana/src/ravana/chat/schema_induction.py",
    "ravana/src/ravana/learn/subject_type_evidence.py",
    "docs/investigations/M123_COMBINED_PREREG.md",
)

NEGATION = re.compile(
    r"\b(?:is|are|was|were)\s+not\b|\bnot\s+(?:a|an)\b|\bnever\s+(?:a|an)\b",
    re.IGNORECASE)
CLAIM_RE = re.compile(r"\b([A-Za-z][A-Za-z\-]*)\s+is\s+(?:a|an)\s+"
                      r"([A-Za-z][A-Za-z \-]*?)\s*[.!?]", re.IGNORECASE)


# ── arm staging primitives (all AST-scanned; no answer-side identifiers) ────
def label_from_claim(claim):
    """Extract the source-native label from a teaching assertion, or ''."""
    m = CLAIM_RE.search(str(claim or ""))
    return m.group(2).strip() if m else ""


def view_confidence(views, subject, label):
    """Highest stored edge confidence for an exact (subject,label) pair."""
    who = str(subject or "").strip().casefold()
    want = str(label or "").strip().casefold()
    best = 0.0
    for v in views or ():
        src = str(v.get("source", v.get("subject", "")) or "").strip().casefold()
        tgt = str(v.get("type_label", v.get("target", "")) or "").strip().casefold()
        if src != who or tgt != want:
            continue
        try:
            conf = float(v.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        best = max(best, conf)
    return best


def permuted_subject(subject):
    """Fixed unrelated subject used by the compute-matched wrong-tag arm."""
    return "zeppelin" if str(subject or "").strip().casefold() != "zeppelin" else "blizzard"


def _neutral_like(surface):
    """A neutral, claim-free turn with EXACTLY the input's whitespace-token count.

    No subject, label, provenance or answer-side token is carried over.  Keeping
    the token count identical is what makes the arm compute-matched: the same
    number of process_turn calls and the same teaching-turn structure as M123.
    """
    toks = [str(t) for t in str(surface or "").split()]
    n = len(toks) or len(("things", "happen", "for", "many", "reasons"))
    base = ("things", "happen", "for", "many", "reasons")
    out = list(base) + ["also"] * (n - len(base)) if n >= len(base) else list(base[:n])
    return " ".join(out) + ("." if not out[-1].endswith(".") else "")


def repair_far_surfaces(item, condition, surfaces, arm, counters=None):
    """Scratch-only far-repair manipulation (the ONLY behavioural difference).

    REPAIR: in the FAR P2c teaching only, replace the planted `misleading`
    channel assertion about the far query subject with a length-matched neutral
    turn.  The positive subject assertion, all near/foil/negative teaching,
    M1/M2/M3 code, gates, thresholds, prompts and the scorer are unchanged.

    REPAIR_SELFLOOP: REPAIR plus ONE post-conflict re-affirmation of the SAME
    admitted positive subject fact, only on self-loop items (object == subject)
    at k>=1, so k=0 stays a null control.  A re-affirmation is the identical
    source-native assertion already admitted from the channel; no answer-side
    field is read.  ``counters`` records that the manipulation actually fired.
    """
    result = list(surfaces or ())
    if str(arm) not in REPAIR_ARMS or str(condition) != "far" or item is None:
        return result
    far_subject = str(mtk.query_for(item, "far").get("subject", "")).strip().casefold()
    misleading = set()
    for fact in ti.ordered_channel_facts(item):
        if str(fact.get("evidence_role", "")) != ti.MISLEADING_ROLE:
            continue
        if str(fact.get("subject", "")).strip().casefold() != far_subject:
            continue
        text = ti.channel_assertion(fact)
        if text:
            misleading.add(text)
    replaced = 0
    out = []
    for surface in result:
        text = str(surface)
        if text in misleading:
            out.append(_neutral_like(text))
            replaced += 1
        else:
            out.append(text)
    reaffirmed = 0
    if str(arm) == "REPAIR_SELFLOOP" and int(item.get("k", 0) or 0) >= 1:
        positive = ti.admitted_subject_fact(item, far_subject, ti.SUBJECT_TYPE_ROLE)
        if positive and str(positive.get("object", "")).strip().casefold() == far_subject:
            assertion = ti.channel_assertion(positive)
            if assertion:
                out.append(assertion)
                reaffirmed = 1
    if counters is not None:
        counters["repair_misleading_replaced"] = \
            counters.get("repair_misleading_replaced", 0) + replaced
        counters["repair_selffloop_reaffirmed"] = \
            counters.get("repair_selffloop_reaffirmed", 0) + reaffirmed
    return out


def stage_m1_act(store, surface, subject, label, fact, arm, counters):
    """Write ONE engine-owned, provenance-bearing act for a teaching surface.

    The decision comes from the UNMODIFIED source evaluator; the origin and
    licence come from the admitted independent channel record supplied by the
    harness.  Nothing here consults an answer-side field.
    """
    counters["m1_act_calls"] = counters.get("m1_act_calls", 0) + 1
    negated = bool(NEGATION.search(str(surface)))
    wrong = str(arm) == "SHAM"
    origin = "system" if wrong else str((fact or {}).get("source_id", "") or "")
    kind = "unknown" if wrong else "curated"
    stype = "system" if wrong else "channel"
    tclass = TrustClass.DISCREDITED if (negated or wrong) else TrustClass.CREDIBLE
    src = SimpleSource(origin_id=origin, source_type=stype, edge_kind=kind,
                       trust_class=tclass,
                       credibility_token=None if (negated or wrong) else "credible")
    ctx = ClaimContext(source=src, prior_decision=None, cue_present=True,
                       invalidated=bool(negated or wrong), trusted_source_ids=None,
                       familiarity=0.0)
    verdict = evaluate_source(str(surface), ctx)
    rec = dp.record_decision(
        store, dp.claim_key(str(surface)), verdict.decision,
        claim=str(surface),
        subject=permuted_subject(subject) if wrong else str(subject),
        source=src, negated=bool(negated or wrong), edge_kind=kind)
    counters["m1_act_writes"] = counters.get("m1_act_writes", 0) + 1
    counters.setdefault("m1_decisions", collections.Counter())
    counters["m1_decisions"][str(getattr(verdict.decision, "value", verdict.decision))] += 1
    return rec


def SimpleSource(origin_id, source_type, edge_kind, trust_class, credibility_token):
    """Real SourceStatus built ONLY from channel provenance; edge_kind is
    passed separately to record_decision (the production record owns it)."""
    return SourceStatus(origin_id=str(origin_id or ""),
                        source_type=str(source_type or ""),
                        trust_class=trust_class,
                        credibility_token=credibility_token)


def _ledger_records(store):
    out = []
    for rec in (store.ledger.to_dict().get("records", {}) if store else {}).values():
        out.append({k: v for k, v in rec.items() if not k.startswith("_")})
    return out


def stage_arm_evidence(subject, options, *, ledger_records, graph_views,
                       snapshot_evidence, negated_pairs, arm, counters):
    """Build the reader's evidence bundle from engine/OWN stores only.

    ``snapshot_evidence`` is the independently licensed subject-side channel;
    ``graph_views`` are the engine's stored typed edges; ``ledger_records`` are
    this engine's durable prior acts.  No item metadata is an argument.
    """
    wrong = str(arm) == "SHAM"
    who = permuted_subject(subject) if wrong else str(subject or "")
    denied = {(str(a).strip().casefold(), str(b).strip().casefold())
              for a, b in (negated_pairs or ())}
    m1 = []
    for rec in ledger_records or ():
        claim = str(rec.get("claim", "") or "")
        lab = label_from_claim(claim)
        if not lab:
            continue
        sub = str(rec.get("subject", "") or "")
        status = str(rec.get("trust_class", "") or "credible").strip().lower()
        if wrong:
            status = "discredited"
        prov = {
            "edge_kind": "unknown" if wrong else str(rec.get("edge_kind", "") or "curated"),
            "source": "system" if wrong else str(rec.get("origin_id", "") or ""),
            "source_type": "system" if wrong else str(rec.get("source_type", "") or "external"),
            "source_id": "system" if wrong else str(rec.get("origin_id", "") or ""),
        }
        neg = wrong or (sub.strip().casefold(), lab.casefold()) in denied \
            or bool(rec.get("negated", False))
        m1.append({
            "record_id": "m1:" + str(rec.get("claim_key", "") or claim)[:24],
            "claim": claim, "subject": sub if not wrong else who,
            "relation": "is_a", "label": lab, "provenance": prov,
            "confidence": view_confidence(graph_views, subject, lab),
            "source_status": status,
            "prior_decision": str(rec.get("prior_decision", "") or "ENDORSE"),
            "invalidated": bool(neg), "cue_present": True, "negated": bool(neg),
        })
    m2 = []
    for ev in snapshot_evidence or ():
        lab = str(getattr(ev, "type_name", "") or "")
        if not lab:
            continue
        neg = wrong or (str(subject).strip().casefold(), lab.casefold()) in denied
        m2.append({
            "record_id": "m2:" + hashlib.sha1(
                (str(getattr(ev, "source", "")) + "|" + lab).encode("utf-8")).hexdigest()[:16],
            "subject": who, "relation": str(getattr(ev, "relation", "is_a") or "is_a"),
            "label": lab,
            "provenance": {
                "edge_kind": "unknown" if wrong else "curated",
                "source": "system" if wrong else str(getattr(ev, "source", "") or ""),
                "source_type": "system" if wrong else "external",
                "source_id": "system" if wrong else str(getattr(ev, "source_url", "") or ""),
            },
            "confidence": view_confidence(graph_views, subject, lab),
            "stamp": lab, "negated": bool(neg),
        })
    counters["staged_m1"] = counters.get("staged_m1", 0) + len(m1)
    counters["staged_m2"] = counters.get("staged_m2", 0) + len(m2)
    return m3.EvidenceBundle.from_lists(m1, m2)


def reader_pick(subject, options, bundle, arm, counters):
    """M3 decision surface; OFF arms do zero work, SHAM matches M1+M2+M3."""
    enabled = str(arm) in M3_ARMS
    counters["hook_calls"] = counters.get("hook_calls", 0) + 1
    trust = m3.wrong_tag_trust if str(arm) == "SHAM" else None
    kwargs = {} if trust is None else {"trust": trust}
    decision = m3.graded_probe_pick(
        permuted_subject(subject) if str(arm) == "SHAM" else subject,
        options, bundle, enabled=enabled, counters=counters, **kwargs)
    if decision is not None and decision.pick is not None:
        return decision.pick
    return None


def evidence_fingerprint(bundle):
    """Content hash of staged evidence; no answer-side field is included."""
    payload = []
    for group in (bundle.m1_records, bundle.m2_subject_evidence):
        for rec in group:
            payload.append((rec.get("subject"), rec.get("label"),
                            rec.get("confidence"), bool(rec.get("negated")),
                            str(rec.get("provenance", {}).get("source_id", ""))))
    return hashlib.sha256(json.dumps(sorted(payload, key=repr),
                                     sort_keys=True).encode("utf-8")).hexdigest()


# ── static audits ────────────────────────────────────────────────────────────
def sha256_file(path):
    """Newline-normalised SHA-256 (git blob vs CRLF checkout invariant)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def ast_anti_leakage():
    """AST scan of every arm-specific decision/staging function."""
    targets = [(m3, n) for n in ("graded_probe_pick", "read_graded_probe",
                                 "_admit_m1", "wrong_tag_trust", "_fact_from")]
    targets += [(sys.modules[__name__], n) for n in (
        "stage_m1_act", "stage_arm_evidence", "reader_pick",
        "label_from_claim", "view_confidence", "permuted_subject",
        "_neutral_like", "repair_far_surfaces")]
    problems, scanned = [], []
    for module, name in targets:
        fn = getattr(module, name, None)
        if fn is None:
            problems.append("missing function: %s" % name)
            continue
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
            if isinstance(node, ast.Name) and node.id in BANNED:
                hits.append("name=%s" % node.id)
            if isinstance(node, ast.Attribute) and node.attr in BANNED:
                hits.append("attr=%s" % node.attr)
            if (isinstance(node, ast.Subscript)
                    and isinstance(node.slice, ast.Constant)
                    and node.slice.value in BANNED_SUBSCRIPT):
                hits.append("subscript=%s" % node.slice.value)
        scanned.append({"function": name, "module": module.__name__,
                        "hits": sorted(set(hits))})
        problems.extend("%s: %s" % (name, h) for h in sorted(set(hits)))
    return {"ok": not problems, "problems": problems, "scanned": scanned,
            "banned_tokens": sorted(BANNED)}


def pin_hashes():
    pins = {}
    for rel in PINNED:
        here = os.path.join(WORKTREE, rel.replace("/", os.sep))
        there = os.path.join(PARENT, rel.replace("/", os.sep))
        if not os.path.isfile(here):
            pins[rel] = {"missing": True}
            continue
        a = sha256_file(here)
        b = sha256_file(there) if os.path.isfile(there) and os.path.abspath(here) != os.path.abspath(there) else a
        pins[rel] = {"worktree": a, "parent": b, "match": a == b}
    return pins


def store_hashes():
    out = {}
    d = os.path.join(PARENT, "weights")
    for name in sorted(os.listdir(d)):
        if (name.startswith("ravana_weights.pkl") or name == "ravana_weights.db"
                or name.startswith("junk_")):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                out[name] = sha256_file(p)
    return out


def gold_mutation_fingerprint():
    """Anti-tautology: changing answer-side fields cannot change staging."""
    fact = {"subject": "blizzard", "predicate": "is_a", "object": "snow storm",
            "source_id": "abc123", "source": "https://example.test/Q1",
            "license": "CC0-1.0"}
    item = {"channel_evidence": [fact], "gold": {"target": "KEEP"},
            "type_sets": {"x": ["KEEP"]}, "world_model": {"x": ["KEEP"]}}
    mutated = copy.deepcopy(item)
    mutated["gold"]["target"] = "MUTATED"
    mutated["type_sets"]["x"] = ["MUTATED"]
    mutated["world_model"]["x"] = ["MUTATED"]

    def stage(obj):
        f = obj["channel_evidence"][0]
        src = SimpleSource(f["source_id"], "channel", "curated",
                           TrustClass.CREDIBLE, "credible")
        ctx = ClaimContext(source=src, prior_decision=None, cue_present=True,
                           invalidated=False, trusted_source_ids=None, familiarity=0.0)
        verdict = evaluate_source("blizzard is a snow storm.", ctx)
        return (f["subject"], f["object"], f["source_id"],
                str(getattr(verdict.decision, "value", verdict.decision)))

    base, after = stage(item), stage(mutated)
    return {"ok": base == after, "base": list(base), "mutated": list(after)}


def delayed_option_audit(items):
    """Static option-space check of the full bank and the focused subset."""
    errors = ti.validate_items(items)
    overlap = ti.channel_overlap(items)
    _, labels = ti.collect_channel_label_pools(items)
    lab = {str(x).strip().casefold() for x in labels}
    bad = []
    for it in items:
        for cond in ("near", "far", "foil", "negative"):
            q = mtk.query_for(it, cond)
            shown = [str(c) for c in q.get("candidates", [])]
            if not shown or any(str(c).strip().casefold() not in lab for c in shown):
                bad.append({"id": it["item_id"], "condition": cond, "options": shown})
            prompt = mtk.query_surface(it, cond, paraphrase=True)
            if any(c not in prompt for c in shown):
                bad.append({"id": it["item_id"], "condition": cond,
                            "reason": "prompt_missing_option"})
    return {"ok": not errors and overlap["ok"] and not bad,
            "validate_errors": len(errors), "channel_overlap_ok": overlap["ok"],
            "channel_overlap_totals": dict(overlap["totals"]),
            "bad_option_rows": len(bad), "examples": bad[:5],
            "channel_label_count": len(lab)}


def mutation_detection_control(items):
    """The option audit must reject a planted domain-word option set."""
    it = next(x for x in items if x["pool"] == "primary" and x["k"] == 8)
    q = mtk.query_for(it, "near")
    planted = dict(it)
    tq = dict(it["typed_queries"])
    near = dict(tq["near"])
    near["candidates"] = list(it["unseen_test_words"]["near"]["candidates"])
    tq["near"] = near
    planted["typed_queries"] = tq
    return {"planted": near["candidates"],
            "detected": not delayed_option_audit([planted])["ok"],
            "expected_options": q["candidates"]}


def focused_items(items, max_per_pool):
    """Same round-robin rule as harness main(), with k=8 forced into each pool."""
    if not max_per_pool:
        return list(items)
    keep = []
    for pool in ("primary", "sealed"):
        by_k = {k: [it for it in items if it["pool"] == pool and it["k"] == k]
                for k in mtk.K_LEVELS}
        if by_k.get(8):
            keep.append(by_k[8].pop(0))
        for idx in range(max_per_pool - 1):
            k = mtk.K_LEVELS[idx % len(mtk.K_LEVELS)]
            if by_k.get(k):
                keep.append(by_k[k].pop(0))
    return keep


def load_bank():
    base = ti.build_items()
    loaded = ti.load_channel_evidence(base)
    items = ti.build_items(channel_override=loaded["facts"])
    return items, {"path": loaded["path"], "external": loaded["external_items"],
                   "error": loaded["error"]}


# ── boot-free self-test ──────────────────────────────────────────────────────
def boot_free_self_test():
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    ast = ast_anti_leakage()
    add("ast_anti_leakage", ast["ok"], ast["problems"])
    add("arm_flags_default_off", all(
        not os.environ.get("RAVANA_DURABLE_PROVENANCE") for _ in (0,)))
    add("all_arms_declared", len(ARMS) == 8 and set(TAGS.values()) ==
        {"off1", "off2", "m1", "m12", "m123", "repair", "repair_self", "sham"},
        json.dumps({"arms": list(ARMS), "tags": TAGS}))

    fingerprint = gold_mutation_fingerprint()
    add("gold_mutation_invariant", fingerprint["ok"], fingerprint)

    # Default-off: OFF path performs zero evidence work.
    c0 = m3.new_counters()
    d0 = m3.graded_probe_pick("blizzard", ["snow storm", "process"],
                              m3.EvidenceBundle.from_lists(), enabled=False,
                              counters=c0)
    add("flag_off_inert", d0 is None and c0["records_scanned"] == 0
        and c0["flag_off"] == 1, c0)

    # READ vs SHAM on identical staged evidence, matched call counts.
    good = {"record_id": "r1", "subject": "blizzard", "label": "snow storm",
            "relation": "is_a", "provenance": {"edge_kind": "curated",
            "source": "https://example.test/Q1", "source_type": "external",
            "source_id": "src1"}, "confidence": 0.9, "negated": False,
            "source_status": "credible", "prior_decision": "ENDORSE"}
    bundle = m3.EvidenceBundle.from_lists((good,), ())
    cr, cs = m3.new_counters(), m3.new_counters()
    dr = m3.graded_probe_pick("blizzard", ["snow storm", "process"], bundle,
                              enabled=True, counters=cr)
    ds = m3.graded_probe_pick("blizzard", ["snow storm", "process"], bundle,
                              trust=m3.wrong_tag_trust, enabled=True, counters=cs)
    call_keys = ("production_calls", "scorer_calls", "trust_calls",
                 "records_scanned", "m1_scanned", "m2_scanned")
    add("read_picks_exact_option", dr is not None and dr.pick == "snow storm",
        getattr(dr, "reason", None))
    add("sham_admits_nothing", ds is not None and ds.pick is None
        and cs["records_admitted"] == 0, getattr(ds, "reason", None))
    add("read_sham_counters_matched",
        all(cr[k] == cs[k] for k in call_keys),
        {k: (cr[k], cs[k]) for k in call_keys})

    # The REAL arm primitive (not just m3 directly) must score and match SHAM.
    cr2, cs2 = m3.new_counters(), m3.new_counters()
    dread = reader_pick("blizzard", ["snow storm", "process"], bundle,
                        "M1+M2+M3", cr2)
    dsham = reader_pick("blizzard", ["snow storm", "process"], bundle,
                        "SHAM", cs2)
    add("reader_arm_picks_and_matches_sham",
        dread == "snow storm" and dsham is None and cr2["errors"] == 0
        and all(cr2[k] == cs2[k] for k in call_keys),
        {"read": dread, "sham": dsham, "errors": cr2["errors"]})
    cre, crs = m3.new_counters(), m3.new_counters()
    dre = reader_pick("blizzard", ["snow storm", "process"], bundle,
                      "REPAIR", cre)
    drs = reader_pick("blizzard", ["snow storm", "process"], bundle,
                      "REPAIR_SELFLOOP", crs)
    add("repair_arms_m3_enabled",
        dre == "snow storm" and drs == "snow storm"
        and all(cre[k] == cr2[k] and crs[k] == cr2[k] for k in call_keys),
        json.dumps({"repair": dre, "selfloop": drs,
                    "counters": {k: (cr2[k], cre[k], crs[k]) for k in call_keys}}))

    neg = dict(good, record_id="r2", negated=True)
    dn = m3.read_graded_probe("blizzard", ["snow storm", "process"],
                              m3.EvidenceBundle.from_lists((good, neg), ()))
    add("negation_veto", dn.pick is None and "snow storm" in dn.vetoed,
        dn.reason)
    dw = m3.read_graded_probe("zeppelin", ["snow storm", "process"], bundle)
    add("graph_view_confidence_shape",
        view_confidence([{"source": "blizzard", "type_label": "snow storm",
                          "confidence": 0.9}], "blizzard", "snow storm") == 0.9)
    add("subject_independence", dw.pick is None, dw.reason)

    # Durable act round-trip with a temp sidecar: no engine, no weight store.
    with tempfile.TemporaryDirectory() as td:
        s1 = dp.DurableProvenanceStore(td)
        src = SimpleSource("src1", "channel", "curated", TrustClass.CREDIBLE, "credible")
        verdict = evaluate_source("blizzard is a snow storm.", ClaimContext(
            source=src, prior_decision=None, cue_present=True, invalidated=False,
            trusted_source_ids=None, familiarity=0.0))
        rec = dp.record_decision(s1, dp.claim_key("blizzard is a snow storm."),
                                 verdict.decision,
                                 claim="blizzard is a snow storm.",
                                 subject="blizzard", source=src)
        s2 = dp.DurableProvenanceStore(td).load()
        loaded = s2.get(dp.claim_key("blizzard is a snow storm."))
        add("m1_durable_roundtrip",
            verdict.decision is Decision.ENDORSE and loaded is not None
            and loaded.claim == rec.claim and loaded.ever_endorsed,
            getattr(verdict, "rule", None))
        add("m1_key_answer_independent",
            dp.claim_key("blizzard is a snow storm.") ==
            "wfa_" + hashlib.sha1("blizzard is a snow storm.".encode("utf-8")
                                  ).hexdigest()[:12],
            "claim key is a plain claim digest")

    # Subject channel: default off, subject-specific, OOV empty.
    snap = ste.load_snapshot()
    facts = [f for f in snap["facts"]
             if str(f.get("subject", "")).strip().casefold() == "blizzard"]
    assertions = [ste.Assertion(subject=f["subject"], relation=f["relation"],
                                type_name=f["type_name"],
                                text=f.get("assertion") or f.get("text_evidence", ""),
                                source=f["source"], license=f["license"],
                                source_url=f["source_url"],
                                snapshot_sha256=snap["provenance"]["facts_sha256"],
                                credibility=0.0) for f in facts]
    admitted, _ = ste.admit("blizzard", assertions)
    add("m2_subject_specific", bool(admitted) and all(
        str(e.subject).casefold() == "blizzard" for e in admitted), len(admitted))
    add("m2_oov_empty", not ste.admit("zeppelin", assertions)[0])

    # Saturation gate must be recorded, never tuned.
    sat = front_gates.gate_saturation({"tra_k_near": 100, "tra_k_negative_reject": 100},
                                       None, "tra_k")
    add("g6_status_recorded", sat.status in ("PASS", "FLAG", "FAIL", "INVALID"),
        sat.status)

    items, info = load_bank()
    # Far-repair manipulation probes on the real frozen bank (boot-free).
    by_id = {it["item_id"]: it for it in items}
    guild = by_id["tra_k_0020"]      # primary k=8, far=guild, non-self-loop
    market = by_id["tra_k_0000"]     # sealed k=8, far=market, self-loop
    g_base = mtk.teaching_surfaces(guild, guild["k"], "far", mtk.CHANNEL_ARM)
    cg, cg2 = collections.Counter(), collections.Counter()
    g_rep = repair_far_surfaces(guild, "far", g_base, "REPAIR", cg)
    g_rep2 = repair_far_surfaces(guild, "far", g_base, "REPAIR_SELFLOOP", cg2)
    add("repair_replaces_only_far_misleading",
        cg["repair_misleading_replaced"] >= 1
        and cg2["repair_misleading_replaced"] == cg["repair_misleading_replaced"]
        and cg2["repair_selffloop_reaffirmed"] == 0
        and len(g_rep) == len(g_base) and len(g_rep2) == len(g_base),
        json.dumps({"replaced": cg["repair_misleading_replaced"],
                    "turns": (len(g_base), len(g_rep), len(g_rep2))}))
    add("repair_length_matched",
        [len(str(x).split()) for x in g_rep] ==
        [len(str(x).split()) for x in g_base],
        json.dumps({"base": [len(str(x).split()) for x in g_base],
                    "repair": [len(str(x).split()) for x in g_rep]}))
    g_near = mtk.teaching_surfaces(guild, guild["k"], "near", mtk.CHANNEL_ARM)
    g_k0 = mtk.teaching_surfaces(guild, 0, "far", mtk.CHANNEL_ARM)
    add("repair_leaves_near_and_k0_untouched",
        repair_far_surfaces(guild, "near", g_near, "REPAIR", None) == g_near
        and repair_far_surfaces(guild, "far", g_k0, "REPAIR", None) == g_k0
        and repair_far_surfaces(guild, "far", g_k0, "REPAIR_SELFLOOP", None) == g_k0,
        json.dumps({"near": len(g_near), "k0": len(g_k0)}))
    m_base = mtk.teaching_surfaces(market, market["k"], "far", mtk.CHANNEL_ARM)
    cm, cm2 = collections.Counter(), collections.Counter()
    m_rep = repair_far_surfaces(market, "far", m_base, "REPAIR", cm)
    m_self = repair_far_surfaces(market, "far", m_base, "REPAIR_SELFLOOP", cm2)
    add("repair_selffloop_reaffirms_once",
        cm["repair_selffloop_reaffirmed"] == 0
        and cm2["repair_selffloop_reaffirmed"] == 1
        and len(m_self) == len(m_rep) + 1,
        json.dumps({"repair": cm["repair_misleading_replaced"],
                    "selfloop": cm2["repair_misleading_replaced"],
                    "turns": (len(m_base), len(m_rep), len(m_self))}))
    add("repair_misleading_absent_from_output",
        all(str(x).strip() != "guild is a student association."
            for x in g_rep)
        and any(str(x).strip() == "guild is a student association."
                for x in g_base),
        json.dumps({"base_tail": g_base[-2:], "repair_tail": g_rep[-2:]}))
    audit = delayed_option_audit(items)
    add("full_bank_valid", not audit["validate_errors"] and audit["channel_overlap_ok"],
        {"errors": audit["validate_errors"], "overlap": audit["channel_overlap_totals"]})
    add("delayed_option_audit", audit["ok"], audit["examples"])
    mut = mutation_detection_control(items)
    add("audit_mutation_detects", mut["detected"], mut)
    add("channel_snapshot_loaded", bool(info["path"]) and info["external"] > 0, info)
    return {"ok": all(c["ok"] for c in checks),
            "passed": sum(c["ok"] for c in checks), "total": len(checks),
            "checks": checks}


def run():
    pins = pin_hashes()
    items, info = load_bank()
    audit = delayed_option_audit(items)
    ast = ast_anti_leakage()
    self_test = boot_free_self_test()
    config = {
        "worktree": WORKTREE, "parent": PARENT, "engine_booted": False,
        "production_files_edited": 0,
        "thresholds_or_items_changed": 0,
        "arms": list(ARMS), "tags": TAGS, "arm_flags": ARM_FLAGS,
        "clear_flags": list(CLEAR_FLAGS),
        "channel": info,
        "sha_pins": pins,
        "store_hashes": store_hashes(),
        "ast_anti_leakage": ast,
        "gold_mutation": gold_mutation_fingerprint(),
        "delayed_option_audit": audit,
        "audit_mutation_control": mutation_detection_control(items),
        "self_test": self_test,
    }
    ok = (ast["ok"] and all(v.get("match") for v in pins.values())
          and config["gold_mutation"]["ok"] and audit["ok"]
          and config["audit_mutation_control"]["detected"]
          and self_test["ok"] and self_test["total"] >= 15)
    config["ok"] = bool(ok)
    with open(RESULT_PATH, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=1, ensure_ascii=False, default=str)
    print("CONFIG", json.dumps({"pins": len(pins), "checks": self_test["total"],
                                "channel": info, "store_files": len(config["store_hashes"])}))
    if ok:
        print("PREFLIGHT OK")
        return 0
    print("PREFLIGHT INVALID", json.dumps({
        "ast": ast["problems"][:5],
        "pin_mismatch": [k for k, v in pins.items() if not v.get("match")],
        "self_test_failed": [c["name"] for c in self_test["checks"] if not c["ok"]],
        "audit": audit["examples"][:5]}, default=str))
    return 2


if __name__ == "__main__":
    sys.exit(run())
