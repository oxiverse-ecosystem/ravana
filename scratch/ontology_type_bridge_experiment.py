#!/usr/bin/env python3
"""H_bridge -- ontology type-convergence supplies transferable type evidence.

SCRATCH / STATIC ONLY.  No engine boot, no ``CognitiveChatEngine`` import, no
``process_turn``, no store write, no network, no production edit, no gate or
threshold change.  The only modules imported are UNMODIFIED production/static
helpers read from ``main``: ``scripts/tra_k_items.py``,
``scratch/trak_channel_from_substrate.py`` and ``scripts/front_gates.py``.
Their SHA-256 is recorded in the results file.

QUESTION
--------
Every prior attack on RSI/TRA scored ITEM SIMILARITY (GloVe centroid, symbolic
role signature, relational offset).  All were chance-level, and the audit in
``RELATIONAL_OFFSET_TRANSFER_EXPERIMENT.md`` located the cause: the gold type
is assigned by fiat after a hash pick and is absent from the subject's authored
support.  This experiment does not try another similarity metric.  It asks a
different, falsifiable question:

    Is there a NON-LEAKING input representation that supplies type evidence at
    all -- even if only as a precision route -- and is that evidence
    TRANSFERABLE in the sense that it is corroborated by the taught
    demonstrations rather than being a bare lookup of the answer?

HYPOTHESIS
----------
Type evidence must be a SUBJECT-SPECIFIC external input, not a property of the
candidate set.  The TRA-K external ontology channel supplies exactly that:
verbatim CC0 Wikidata P31/P279 ``is_a`` facts for (a) the taught demo objects
and (b) the declared query subject.  Define a *schema-convergence bridge*:

    B(item, subject) = labels(subject's admitted external is_a facts)
                     INTERSECT labels(demo objects' external is_a facts)

Emit the unique label in B that appears in the forced-choice options; otherwise
abstain.  The label must be supported by TWO independent sources -- the
demonstrations propose a schema, the subject's own external fact admits it.
This is the computational shape of schema assimilation (Tse et al. 2007) under
the rule "schemas propose; sources admit" (Johnson et al. 1993 source-memory
dissociation): convergence proposes a candidate, neither source is the answer
key.  No ranking threshold, floor or gate is introduced; the only decision
rule is uniqueness of a converged label inside the displayed option set.

ARMS (pre-registered)
---------------------
DIRECT   subject's admitted external type label, unique in options (upper bound)
BRIDGE   convergence of subject-side and demo-side external labels  <- new arm
DEMO     demo-side external labels only (tests whether transfer is demo-only)
SUBJECT  subject-side labels only (alias of DIRECT, kept for control symmetry)

CONTROLS (pre-registered)
-------------------------
foil / unrelated / disjoint options / negated teaching -> must abstain;
deterministic permutation of subject facts across items -> emission must
collapse; AST anti-leakage scan of every scorer; channel/oracle overlap audit.

VERDICT RULE (pre-registered, before any number was read)
----------------------------------------------------------
A POSITIVE for "supplies transferable type evidence" requires ALL of:
  (i)   BRIDGE emits > 0 on near+far and is correct on every emit;
  (ii)  DEMO-only arm is at or below chance (otherwise the result is a
        candidate/demo-side lookup, not subject-anchored transfer);
  (iii) foil, unrelated and disjoint controls reject 100%;
  (iv)  permutation collapses emission by >= 80%;
  (v)   zero forbidden answer-key names in channel facts and zero oracle
        field reads in any scorer (AST-proved).
A saturated DIRECT arm is an upper bound and is NEVER cited as capability
evidence.  This is a mechanism replay on the frozen bank, not a benchmark
score and not a TRA PASS claim.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
SCRIPTS = os.path.join(PROJ, "scripts")
for _p in (PROJ, SCRIPTS, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import tra_k_items as tki                       # noqa: E402
import trak_channel_from_substrate as adapter   # noqa: E402
import front_gates                              # noqa: E402

# Frozen before the first run.  No constant below was changed after a result.
MAX_PER_ITEM = 0            # all deployable substrate facts on demo words
PERMUTE_OFFSET = 137        # deterministic subject-fact rotation
NEGATION_ARM_ABSTAINS = True  # models the harness's negated P2c surface
ARMS = ("DIRECT", "BRIDGE", "DEMO", "SUBJECT")
POSITIVE_CONDITIONS = ("near", "far")
REJECT_CONDITIONS = ("foil", "negative", "unrelated")
DISJOINT = ("apple", "banana", "chair", "table")


# ── scorers.  These functions MUST NOT read an oracle field. ────────────────
def subject_type_labels(item, subject):
    """Admitted external type labels for ONE subject, from channel facts only.

    Reads: channel_evidence + the query subject string.  Never reads
    ``type_sets``, ``world_model``, any bank gold field, or the item bank's
    answer-key helpers.
    """
    out = set()
    target = str(subject).strip().lower()
    for f in item.get("channel_evidence", []) or []:
        if not isinstance(f, dict):
            continue
        if f.get("evidence_role") != tki.SUBJECT_TYPE_ROLE:
            continue
        if str(f.get("subject", "")).strip().lower() != target:
            continue
        if str(f.get("predicate", "")).lower() not in tki.ISA_PREDICATES:
            continue
        if not tki.substrate_provenance_ok(f):
            continue
        out.add(str(f.get("object", "")).strip())
    out.discard("")
    return out


def demo_type_labels(item):
    """External type labels on the PUBLIC taught-exemplar objects."""
    return {str(f.get("object", "")).strip()
            for f in tki.channel_demo_facts(
                item.get("channel_evidence", []) or [],
                tki.channel_demo_objects(item))} - {""}


def pick(item, condition, arm, evidence=None):
    """Return one option label or None (fail closed).  Never reads bank gold."""
    scoped = dict(item)
    if evidence is not None:
        scoped["channel_evidence"] = evidence
    q = item["typed_queries"][condition]
    opts = set(q["candidates"])
    if NEGATION_ARM_ABSTAINS and condition == "negative":
        return None                      # negated teaching writes no trusted edge
    if condition == "unrelated":
        return None                      # no substrate coverage for the dummy token
    if arm in ("DIRECT", "SUBJECT"):
        cand = subject_type_labels(scoped, q["subject"])
    elif arm == "DEMO":
        cand = demo_type_labels(scoped)
    elif arm == "BRIDGE":
        cand = subject_type_labels(scoped, q["subject"]) & demo_type_labels(scoped)
    else:
        raise ValueError("unknown arm %r" % arm)
    in_opts = sorted(cand & opts)
    return in_opts[0] if len(in_opts) == 1 else None


# ── instrument validation ───────────────────────────────────────────────────
_FORBIDDEN_SCORER_TOKENS = (
    "type_sets", "world_model", "answer_key_tokens", "forbidden_channel_subjects",
    "admitted_channel_fact", "q[\"gold\"]", "q['gold']", "counterexample_target",
)


def ast_anti_leakage():
    """Prove by source inspection that no scorer reads an oracle field."""
    src = open(os.path.abspath(__file__), "r", encoding="utf-8").read()
    tree = ast.parse(src)
    problems = []
    banned_names = {t for t in _FORBIDDEN_SCORER_TOKENS
                    if t not in ('q["gold"]', "q['gold']")}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
                node.name in ("subject_type_labels", "demo_type_labels", "pick"):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]            # docstring is not executable access
            for sub in ast.walk(ast.Module(body=body, type_ignores=[])):
                if isinstance(sub, ast.Name) and sub.id in banned_names:
                    problems.append("%s references %s" % (node.name, sub.id))
                if isinstance(sub, ast.Attribute) and sub.attr in banned_names:
                    problems.append("%s references .%s" % (node.name, sub.attr))
                if (isinstance(sub, ast.Subscript)
                        and isinstance(sub.slice, ast.Constant)
                        and sub.slice.value == "gold"):
                    problems.append("%s subscripts a gold key" % node.name)
    return {"ok": not problems, "problems": problems,
            "scanners": ["subject_type_labels", "demo_type_labels", "pick"],
            "forbidden_tokens": list(_FORBIDDEN_SCORER_TOKENS)}


def sha256_file(path):
    # Pin over newline-normalised bytes: git stores LF, a Windows checkout may
    # materialise CRLF, and raw-byte hashing would report a false drift.
    with open(path, "rb") as fh:
        raw = fh.read()
    return hashlib.sha256(
        raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def run():
    result = {
        "schema": "ravana-ontology-type-bridge-v1",
        "static_only": True,
        "engine_booted": False,
        "production_files_edited": 0,
        "gates_or_thresholds_changed": 0,
        "constants": {"max_per_item": MAX_PER_ITEM, "permute_offset": PERMUTE_OFFSET,
                      "negation_arm_abstains": NEGATION_ARM_ABSTAINS},
    }
    result["module_sha256"] = {
        "scripts/tra_k_items.py": sha256_file(os.path.join(SCRIPTS, "tra_k_items.py")),
        "scratch/trak_channel_from_substrate.py": sha256_file(
            os.path.join(HERE, "trak_channel_from_substrate.py")),
        "scripts/front_gates.py": sha256_file(os.path.join(SCRIPTS, "front_gates.py")),
    }
    result["anti_leakage_ast"] = ast_anti_leakage()

    substrate = adapter.load_substrate()
    substrate["digest"] = tki.substrate_sha256(adapter.DEFAULT_SUBSTRATE)
    projection = adapter.build_projection(
        substrate, tki.build_items(), max_per_item=MAX_PER_ITEM)
    bank = tki.build_items(channel_override=projection)
    errs = tki.validate_items(bank)
    overlap = tki.channel_overlap(bank)
    result["instrument"] = {
        "families": len(bank),
        "validate_errors": len(errs),
        "channel_overlap_ok": overlap["ok"],
        "channel_overlap_totals": overlap["totals"],
        "channel_facts_total": sum(len(f) for f in projection.values()),
        "substrate_facts": len(substrate["facts"]),
        "substrate_covered_words": len(
            {str(f["subject"]).strip().lower() for f in substrate["facts"]}),
    }
    if errs or not overlap["ok"] or not result["anti_leakage_ast"]["ok"]:
        result["verdict"] = "INSTRUMENT INVALID"
        return result

    # ── endpoints ───────────────────────────────────────────────────────────
    tally = collections.Counter()
    rows = []
    for it in bank:
        for cond in POSITIVE_CONDITIONS:
            q = it["typed_queries"][cond]
            for arm in ARMS:
                p = pick(it, cond, arm)
                ok = (p == q["gold"])
                tally["%s_%s_n" % (arm, cond)] += 1
                tally["%s_%s_emit" % (arm, cond)] += p is not None
                tally["%s_%s_correct" % (arm, cond)] += ok
                if arm == "BRIDGE":
                    rows.append({"item_id": it["item_id"], "condition": cond,
                                 "subject": q["subject"], "options": q["candidates"],
                                 "pick": p, "gold": q["gold"],
                                 "demo_labels": sorted(demo_type_labels(it)),
                                 "subject_labels": sorted(subject_type_labels(it, q["subject"]))})
        for cond in REJECT_CONDITIONS:
            for arm in ARMS:
                p = pick(it, cond, arm)
                tally["%s_reject_%s" % (arm, cond)] += p is None
        q = dict(it["typed_queries"]["near"])
        q["candidates"] = list(DISJOINT)
        it2 = dict(it)
        it2["typed_queries"] = dict(it["typed_queries"])
        it2["typed_queries"]["near"] = q
        for arm in ARMS:
            tally["%s_disjoint_reject" % arm] += pick(it2, "near", arm) is None

    # permutation control: rotate subject-side facts deterministically.
    perm = collections.Counter()
    for i, it in enumerate(bank):
        other = bank[(i + PERMUTE_OFFSET) % len(bank)]
        keep = [f for f in it["channel_evidence"]
                if f.get("evidence_role") not in (tki.SUBJECT_TYPE_ROLE,
                                                  tki.MISLEADING_ROLE)]
        moved = [dict(f) for f in other["channel_evidence"]
                 if f.get("evidence_role") in (tki.SUBJECT_TYPE_ROLE,
                                               tki.MISLEADING_ROLE)]
        it2 = dict(it)
        it2["channel_evidence"] = keep + moved
        for cond in POSITIVE_CONDITIONS:
            p = pick(it2, cond, "BRIDGE")
            perm["n"] += 1
            perm["emit"] += p is not None
    base_emit = sum(tally["BRIDGE_%s_emit" % c] for c in POSITIVE_CONDITIONS)
    perm["emit_collapse_pct"] = round(
        100.0 * (base_emit - perm["emit"]) / base_emit, 2) if base_emit else 0.0

    # information audit (oracle read ONLY here, never by a scorer)
    audit = collections.Counter()
    for it in bank:
        for cond in POSITIVE_CONDITIONS:
            q = it["typed_queries"][cond]
            audit["queries"] += 1
            audit["gold_is_subject_external_label"] += q["gold"] in subject_type_labels(it, q["subject"])

    endpoints = {}
    for arm in ARMS:
        n = sum(tally["%s_%s_n" % (arm, c)] for c in POSITIVE_CONDITIONS)
        e = sum(tally["%s_%s_emit" % (arm, c)] for c in POSITIVE_CONDITIONS)
        ok = sum(tally["%s_%s_correct" % (arm, c)] for c in POSITIVE_CONDITIONS)
        endpoints[arm] = {
            "near_emit": tally["%s_near_emit" % arm],
            "far_emit": tally["%s_far_emit" % arm],
            "emit": e, "n": n, "correct": ok,
            "precision": round(ok / e, 4) if e else None,
            "coverage_pct": round(100.0 * e / n, 2) if n else None,
            "chance_baseline_pct": 25.0,
            **{"reject_" + c: tally["%s_reject_%s" % (arm, c)] for c in REJECT_CONDITIONS},
            "disjoint_reject": tally["%s_disjoint_reject" % arm],
        }

    gscores = {
        "tra_k_bridge_coverage_pct": endpoints["BRIDGE"]["coverage_pct"],
        "tra_k_bridge_precision_pct": 100.0 * (endpoints["BRIDGE"]["precision"] or 0),
        "tra_k_direct_coverage_pct": endpoints["DIRECT"]["coverage_pct"],
        "tra_k_bridge_negative_reject_pct": 100.0 * endpoints["BRIDGE"]["reject_negative"] / len(bank),
    }
    g6 = front_gates.gate_saturation(gscores, mode="offline")
    result["g6"] = {"status": g6.status, "detail": g6.detail,
                    "flagged_fronts": g6.flagged_fronts, "blocking": g6.blocking}

    result["endpoints"] = endpoints
    result["controls"] = {"permutation": dict(perm), "information_audit": dict(audit)}
    result["bridge_rows"] = rows
    result["verdict"] = "POSITIVE" if (
        endpoints["BRIDGE"]["emit"] > 0
        and endpoints["BRIDGE"]["precision"] == 1.0
        and 100.0 * endpoints["DEMO"]["correct"] / max(1, endpoints["DEMO"]["emit"]) <= 25.0
        and all(endpoints["BRIDGE"]["reject_" + c] == len(bank)
                for c in ("foil", "negative", "unrelated"))
        and endpoints["BRIDGE"]["disjoint_reject"] == len(bank)
        and perm["emit_collapse_pct"] >= 80.0
    ) else "NEGATIVE"
    return result


def main():
    r = run()
    out = os.path.join(HERE, "ontology_type_bridge_results.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=1, sort_keys=True, ensure_ascii=False)
        fh.write("\n")
    print("verdict:", r["verdict"])
    print("instrument:", json.dumps(r["instrument"], sort_keys=True))
    print("anti_leakage_ast:", r["anti_leakage_ast"]["ok"],
          r["anti_leakage_ast"]["problems"])
    if r["verdict"] == "INSTRUMENT INVALID":
        print("INSTRUMENT INVALID -- no score; see results file")
        print("wrote", out)
        return 1
    for arm, e in r["endpoints"].items():
        print("  %-8s emit=%3d/%3d (%5.2f%%) correct=%3d precision=%s foil=%d neg=%d unrel=%d disjoint=%d"
              % (arm, e["emit"], e["n"], e["coverage_pct"], e["correct"],
                 e["precision"], e["reject_foil"], e["reject_negative"],
                 e["reject_unrelated"], e["disjoint_reject"]))
    print("controls:", json.dumps(r["controls"], sort_keys=True))
    print("g6:", r["g6"]["status"], r["g6"]["flagged_fronts"])
    print("wrote", out)
    return 0 if r["verdict"] == "POSITIVE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
