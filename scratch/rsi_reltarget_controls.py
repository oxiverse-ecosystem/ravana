#!/usr/bin/env python3
"""Scratch controls + coverage audit for the RSI relation-target channel.

BOOT-FREE. Imports the unmodified authored harness only to READ query rows
(subject / displayed options) and, SEPARATELY, the item-bank gold for SCORING
the pre-registered permutation control. The gold is NEVER passed to the channel
module or to relation_target_pick(): it lives only in this grader.

Runs the pre-registered criteria in docs/investigations/RSI_RELTARGET_PREREG.md
and writes scratch/rsi_reltarget_results.json.
"""
from __future__ import annotations

import collections
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "ravana", "src"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("RAVANA_OFFLINE", "1")

import measure_learn5_rsi as bank            # unmodified, boot-free
import rsi_relation_target_evidence as chan   # new pure module

RAW = os.path.join(HERE, "rsi_reltarget_wikidata_raw.json")
OUT = os.path.join(HERE, "rsi_reltarget_results.json")


def qrows():
    """(item_id, condition, relation, subject, options, gold) for exact+far."""
    out = []
    for it in bank.build_items():
        for cond in ("exact", "far"):
            q = it[cond]
            out.append({
                "item_id": it["item_id"], "condition": cond,
                "relation": it["relation"], "subject": q["subject"],
                "options": list(q["candidates"]),
                "gold": q["gold_target"],  # grader-only; never reaches chan
            })
    return out


def skey(x):
    return str(x or "").strip().casefold()


def main():
    records = chan.load_snapshot(RAW)
    rows = qrows()
    idx = collections.defaultdict(list)
    for r in records:
        idx[(skey(r.subject), skey(r.relation))].append(r)

    # ---- C1/C2 licence + real-assertion audit over the whole snapshot -------
    c1_ok = all(r.license == "CC0-1.0" and r.source_url and r.statement_url
                and r.provenance.get("source_id") for r in records)
    c2_ok = all(r.relation in chan.RELATION_FAMILIES for r in records)
    c5_ast = chan.assert_independence()

    # ---- C3 subject coverage over the 141-word query vocabulary ------------
    vocab = set()
    for r in rows:
        vocab.add(skey(r["subject"]))
    # all four conditions' subjects (exact/far/foil/negative)
    for it in bank.build_items():
        for c in ("exact", "far", "foil", "negative"):
            vocab.add(skey(it[c]["subject"]))
    covered = sum(1 for s in vocab if any(k[0] == s for k in idx))
    c3 = covered / len(vocab) if vocab else 0.0

    # ---- per-row channel read (gold NOT passed) -----------------------------
    picks = []
    for r in rows:
        rs = idx.get((skey(r["subject"]), skey(r["relation"])), [])
        c = chan.relation_target_pick(
            r["subject"], r["relation"], r["options"], rs,
            enabled_flag=True, counters=chan.new_counters())
        picks.append({"row": r, "decision": c})
    with_pick = [p for p in picks if p["decision"] and p["decision"]["pick"]]
    hit_rows = [p for p in picks if p["decision"] and p["decision"]["scores"]]
    c4 = len(hit_rows) / len(rows) if rows else 0.0
    accuracy = (sum(1 for p in with_pick
                    if skey(p["decision"]["pick"]) == skey(p["row"]["gold"]))
                / len(rows)) if rows else 0.0

    # ---- C5 permutation INSIDE each displayed option set -------------------
    perm = []
    rng = random.Random(0)
    for p in picks:
        r, d = p["row"], p["decision"]
        opts = r["options"]
        alt = [o for o in opts if skey(o) != skey(r["gold"])]
        if not alt:
            perm.append({"changed": False, "pick_before": (d or {}).get("pick"),
                         "pick_after": (d or {}).get("pick")})
            continue
        new_gold = rng.choice(alt)
        # The channel never receives new_gold; only the grader's key changes.
        perm.append({"changed": True, "gold_before": r["gold"],
                     "gold_after": new_gold,
                     "pick_before": (d or {}).get("pick"),
                     "pick_after": (d or {}).get("pick"),
                     "scores_before": (d or {}).get("scores"),
                     "scores_after": (d or {}).get("scores")})
    perm_changed = sum(1 for x in perm if x["changed"])
    pick_invariant = all(x["pick_before"] == x["pick_after"] for x in perm)
    perm_accuracy = (sum(1 for x in perm if x["changed"] and x["gold_after"]
                         and skey(x["pick_after"]) == skey(x["gold_after"]))
                     / perm_changed) if perm_changed else 0.0

    # ---- C6 negation veto + wrong-tag SHAM (20 fixed rows) ------------------
    # Exercise the NEGATION control on rows where the reader actually picked;
    # sampling the first 20 rows can miss the only pick and make the control
    # vacuous (a probe defect, not a mechanism result).
    pick_rows = [p for p in picks if p["decision"] and p["decision"]["pick"]]
    neg_rows = pick_rows[:20]
    veto_ok = 0
    sham_admitted = 0
    for p in neg_rows:
        r, d = p["row"], p["decision"]
        rs = idx.get((skey(r["subject"]), skey(r["relation"])), [])
        if not rs or not (d and d["pick"]):
            continue
        base = chan.relation_target_pick(
            r["subject"], r["relation"], r["options"], rs,
            enabled_flag=True, counters=chan.new_counters())
        target = base["pick"]
        neg = chan.EvidenceRecord(
            subject=r["subject"], relation=r["relation"], target=target,
            surface="%s is not %s." % (r["subject"], target),
            license="CC0-1.0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            source_url="https://www.wikidata.org/wiki/NEG_TEST",
            statement_url="urn:neg-test:%s" % r["item_id"],
            rank="preferred", credibility=1.0, negated=True,
            provenance={"edge_kind": "curated", "source": "wikidata",
                        "source_type": "external", "source_id": "neg-test"})
        after = chan.relation_target_pick(
            r["subject"], r["relation"], r["options"], rs + [neg],
            enabled_flag=True, counters=chan.new_counters())
        if after and skey(target) in set(after["vetoed"]) and after["pick"] != target:
            veto_ok += 1
        sham = chan.relation_target_pick(
            r["subject"], r["relation"], r["options"], rs,
            trust=chan.wrong_tag_trust, enabled_flag=True,
            counters=chan.new_counters())
        sham_admitted += len(sham["admitted"]) if sham else 0

    # ---- C7 candidate-order invariance -------------------------------------
    order_ok, order_n = 0, 0
    for p in with_pick:
        r, d = p["row"], p["decision"]
        rs = idx.get((skey(r["subject"]), skey(r["relation"])), [])
        shuffled = list(r["options"])
        rng.shuffle(shuffled)
        d2 = chan.relation_target_pick(
            r["subject"], r["relation"], shuffled, rs,
            enabled_flag=True, counters=chan.new_counters())
        order_n += 1
        if d2 and d2["pick"] == d["pick"]:
            order_ok += 1

    # ---- C8 default-off purity ---------------------------------------------
    off = chan.relation_target_pick(
        rows[0]["subject"], rows[0]["relation"], rows[0]["options"],
        idx.get((skey(rows[0]["subject"]), skey(rows[0]["relation"])), []),
        enabled_flag=False, counters=chan.new_counters())

    criteria = {
        "C1_licence_and_provenance": bool(c1_ok),
        "C2_real_relation_assertions": bool(c2_ok),
        "C3_subject_coverage": {"value": round(c3, 4), "pass": c3 >= 0.80},
        "C4_candidate_target_coverage": {"value": round(c4, 4), "pass": c4 >= 0.50},
        "C5_ast_independence": bool(c5_ast),
        "C5_permutation": {
            "keys_changed": perm_changed, "pick_invariant": pick_invariant,
            "permuted_key_accuracy": round(perm_accuracy, 4),
            "pass": pick_invariant and perm_accuracy <= 0.25},
        "C6_negation_veto": {
            "veto_ok": veto_ok, "sham_admitted": sham_admitted,
            "pass": veto_ok >= 1 and sham_admitted == 0},
        "C7_candidate_order_invariance": {
            "ok": order_ok, "n": order_n,
            "pass": order_n == 0 or order_ok == order_n},
        "C8_default_off": {"off_is_none": off is None, "pass": off is None},
    }
    verdict = "GO" if all(
        (v["pass"] if isinstance(v, dict) else v) for v in criteria.values()
    ) else "NO-GO"

    out = {
        "schema": "rsi-reltarget-results-v1",
        "snapshot_bindings": len(records),
        "rows": len(rows), "vocab_subjects": len(vocab),
        "subject_covered": covered,
        "rows_with_candidate_hit": len(hit_rows),
        "rows_with_pick": len(with_pick),
        "accuracy_original_key": round(accuracy, 4),
        "criteria": criteria, "verdict": verdict,
        "per_row": [{"item_id": p["row"]["item_id"], "condition": p["row"]["condition"],
                     "subject": p["row"]["subject"], "relation": p["row"]["relation"],
                     "options": p["row"]["options"], "gold_for_grader_only": p["row"]["gold"],
                     "decision": p["decision"]} for p in picks],
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in out.items() if k != "per_row"}, indent=1))


if __name__ == "__main__":
    main()
