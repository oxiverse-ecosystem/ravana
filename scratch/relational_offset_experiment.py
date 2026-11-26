#!/usr/bin/env python3
"""SCRATCH-ONLY: relational role transfer by analogical vector offset.

NO engine boot, NO process_turn, NO store write, NO gate/threshold/item edit.
Reads only: frozen item bank (regenerated from unmodified production code),
40 recorded live TRA rows (backend=live), one read-only GloVe cache, and the
unmodified shipped G6 saturation gate.

NEW HYPOTHESIS (H_offset) -- distinct from all prior negatives in this repo:
  Prior failures used ITEM-SIMILARITY surfaces: a GloVe centroid of taught r2
  tokens (TRA_CATEGORY_CENTRE_EXPERIMENT), surface/token signatures
  (RSI_SYMBOLIC_ROLE_EXPERIMENT), or candidate-side type lookup
  (GOAL_UNSEEN_TYPE_TRANSFER H_b). All were SUBJECT-INDEPENDENT (0/80 picks
  changed when the subject was replaced by `zeppelin`).

  H_offset instead learns a RELATIONAL TRANSFORMATION from the taught pairs and
  applies it to the QUERY SUBJECT -- the classic structure-mapping operation
  "map the relation, not the attributes":
      W     = unit( mean_i [ unit(v(r2_i)) - unit(v(r1_i)) ] )
      pred  = unit( unit(v(subject)) + ALPHA * W )
      score(o) = cos(unit(v(o)), pred)
  This is subject-dependent BY CONSTRUCTION and is a different computational
  primitive from a prototype/centroid or a token signature.

Brain grounding (verified in-repo ledger, RSI_ROLE_BINDING_DESIGN.md §2):
  Gentner 1983 (structure mapping: relations, not object attributes);
  Cohen & Eichenbaum 1993 / Eichenbaum 2004 (relational memory -> flexible
  inference in novel situations); McClelland, McNaughton & O'Reilly 1995
  + Yassa & Stark 2011 (keep episodes distinct: pattern separation);
  Bunge et al. 2005 (retrieval vs integration are separable stages).
  The offset is an EXTRAPOLATION (ledger 'A'/'E'), not a neural homology claim.

ARMS (all stateless, no learning, no writes):
  OFFSET       -- hypothesis above (new).
  INSTANCE_MAX -- pattern-separation control: retrieve the demonstrated pair
                  whose r1 is nearest the subject, rank by similarity of the
                  candidate to that pair's r2 (episodic nearest-neighbour).
  CENTROID     -- reproduce the known-negative prototype baseline.

FROZEN BEFORE ANY NUMBER IS READ (no tuning after results):
  FLOOR=0.20, MARGIN=0.10 (inherited from the merged primitive and the prior
  experiments), ALPHA=1.0, subject token='zeppelin', disjoint option set.
  The ALPHA sweep is reported only as a labelled sensitivity diagnostic,
  never used to select the primary number.

"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys

import numpy as np

# ── frozen constants ────────────────────────────────────────────────────────
FLOOR = 0.20
MARGIN = 0.10
ALPHA = 1.0
ALT_SUBJECT = "zeppelin"
DISJOINT = ["apple", "banana", "chair", "table"]
K_LEVELS = (0, 1, 2, 4, 8)
CONDITIONS = ("near", "far", "foil", "negative")

MAIN = "C:/Users/Likhith/Documents/projects/ravana"
WORKTREE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GLOVE = os.path.join(MAIN, "data", "ravana_glove_cache.npz")
ROWS = os.path.join(MAIN, "scratch", "tra_harness", "tra_rows.jsonl")

sys.path.insert(0, os.path.join(MAIN, "scripts"))
sys.path.insert(0, os.path.join(MAIN, "ravana", "src"))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v, dtype=np.float32)


def cos(a, b):
    return float(np.dot(unit(a), unit(b)))


# ── instrument validation ──────────────────────────────────────────────────
checks = {}
for rel in ("scripts/tra_items.py",
            "ravana/src/ravana/ontology/attribute_encoder.py",
            "scripts/front_gates.py"):
    m = sha256_file(os.path.join(MAIN, rel))
    w = sha256_file(os.path.join(WORKTREE, rel))
    checks[rel] = {"main": m, "worktree": w, "identical": m == w}
    assert m == w, "production module differs from main: %s" % rel

from tra_items import build_items, validate_items, manifest_hashes  # noqa: E402
from ravana.ontology.attribute_encoder import build_glove64_lookup  # noqa: E402
from front_gates import gate_saturation  # noqa: E402

bank = build_items()
bank_errors = validate_items(bank)
assert len(bank) == 400 and not bank_errors, (len(bank), bank_errors[:3])
bank_by_id = {it["item_id"]: it for it in bank}
manifests = manifest_hashes(bank)

rows_all = [json.loads(line) for line in open(ROWS, encoding="utf-8")]
rows_live = [r for r in rows_all if r.get("backend") == "live"]
fake_excluded = len(rows_all) - len(rows_live)

# one graded row per (item, condition): phase=query, arm=structural, non-permuted
graded = {}
for r in rows_live:
    if r["phase"] == "query" and r["arm"] == "structural" and not r.get("permuted"):
        graded[(r["id"], r["condition"])] = r
assert len(graded) == 160, len(graded)

# cross-check recorded subjects/candidates/gold against the regenerated bank
mismatch = []
for (iid, cond), r in graded.items():
    it = bank_by_id[iid]
    if cond in ("near", "far"):
        q = it["unseen_test_words"][cond]
        exp_c, exp_g = q["candidates"], q["gold"]
    elif cond == "foil":
        exp_c, exp_g = it["foil"]["candidates"], None
    else:
        exp_c, exp_g = it["negative"]["candidates"], None
    if (r["input"]["candidates"] != exp_c
            or r["gold"]["target"] != exp_g
            or r["gold"]["should_reject"] != (exp_g is None)):
        mismatch.append(iid)
assert not mismatch, mismatch[:5]

lut, dim = build_glove64_lookup(GLOVE)
all_words = set()
for (iid, cond), r in graded.items():
    all_words.add(r["input"]["query_subject"])
    all_words.update(r["input"]["candidates"])
    it = bank_by_id[iid]
    for ex in it["taught_exemplars"][:8]:
        all_words.update((ex["r1"], ex["r2"]))
missing_words = sorted(w for w in all_words if w not in lut)

# ── the mechanism (stateless) ──────────────────────────────────────────────
NEG_WORDS = ("does not", "not ")


def demo_pairs(item, condition, k):
    """Regenerate the teaching pairs deterministically from the frozen bank.

    Recorded live rows carry empty `teaching_surfaces` (the recorder dropped
    them), so the surfaces are reconstructed from `taught_exemplars[:k]`,
    exactly as in GOAL_UNSEEN_TYPE_TRANSFER.md §8. The NEGATIVE condition
    negates each surface; a real negation channel must detect that and abstain.
    """
    exs = item["taught_exemplars"][:k]
    pairs = []
    for ex in exs:
        surface = ex["surface"]
        negated = condition == "negative"
        if negated:
            surface = surface.replace(" causes ", " does not cause ") \
                             .replace(" is part of ", " is not part of ") \
                             .replace(" is preceded by ", " is not preceded by ") \
                             .replace(" resembles ", " does not resemble ") \
                             .replace(" supports ", " does not support ") \
                             .replace(" contrasts with ", " does not contrast with ")
        pairs.append({"r1": ex["r1"], "r2": ex["r2"],
                      "surface": surface,
                      "negated": any(n in surface for n in NEG_WORDS)})
    return pairs


def score_arm(arm, subject, candidates, pairs, alpha=ALPHA):
    """Return (pick, scores, reason). Fail-closed on every defect."""
    if not candidates or len(set(candidates)) != len(candidates):
        return None, {}, "bad_option_set"
    if any(c not in lut for c in candidates) or subject not in lut:
        return None, {}, "oov"
    usable = [p for p in pairs if not p["negated"]]
    if not usable:
        return None, {}, "no_admitted_trace_negated"
    u_s = unit(lut[subject])
    if arm == "OFFSET":
        diffs = [unit(lut[p["r2"]]) - unit(lut[p["r1"]]) for p in usable]
        W = unit(np.mean(np.stack(diffs), axis=0).astype(np.float32))
        pred = unit(u_s + alpha * W)
        scores = {c: cos(lut[c], pred) for c in candidates}
    elif arm == "INSTANCE_MAX":
        # pattern separation: pick the single demonstrated episode whose r1
        # best matches the subject; rank by fit to that episode's r2.
        i_star = max(range(len(usable)),
                     key=lambda i: cos(u_s, lut[usable[i]["r1"]]))
        anchor = unit(lut[usable[i_star]["r2"]])
        scores = {c: cos(lut[c], anchor) for c in candidates}
    elif arm == "CENTROID":
        centre = unit(np.mean(np.stack([unit(lut[p["r2"]]) for p in usable]),
                              axis=0).astype(np.float32))
        scores = {c: cos(lut[c], centre) for c in candidates}
    elif arm == "SUBJECT_COS":
        scores = {c: cos(lut[c], u_s) for c in candidates}
    else:
        raise ValueError(arm)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    top, second = ranked[0], ranked[1] if len(ranked) > 1 else (None, -1e9)
    if top[1] < FLOOR:
        return None, scores, "no_category_above_floor"
    if top[1] - second[1] < MARGIN:
        return None, scores, "strict_margin_not_met"
    return top[0], scores, "strict_winner"


# ── run all arms on the 160 graded cases ───────────────────────────────────
results = []
subject_dep = {a: {"emitted": 0, "changed": 0, "same": 0}
               for a in ("OFFSET", "INSTANCE_MAX", "CENTROID", "SUBJECT_COS")}
for (iid, cond), r in sorted(graded.items()):
    it = bank_by_id[iid]
    subj = r["input"]["query_subject"]
    cands = r["input"]["candidates"]
    gold = r["gold"]["target"]
    pairs = demo_pairs(it, cond, r["k"])
    for arm in subject_dep:
        pick, scores, reason = score_arm(arm, subj, cands, pairs)
        correct = (pick is None) if gold is None else (pick == gold)
        row = {"item": iid, "condition": cond, "pool": r["pool"], "k": r["k"],
               "role": it["role"], "arm": arm, "subject": subj,
               "candidates": cands, "gold": gold, "pick": pick,
               "reason": reason, "correct": correct,
               "scores": {c: round(float(s), 6) for c, s in scores.items()}}
        if gold is not None and scores:
            order = sorted(scores, key=lambda c: -scores[c])
            row["raw_top1"] = order[0]
            row["raw_top1_correct"] = order[0] == gold
            row["gold_rank"] = order.index(gold) + 1
        # subject-independence control on the positive conditions
        if cond in ("near", "far"):
            p2, s2, _ = score_arm(arm, ALT_SUBJECT, cands, pairs)
            row["alt_pick"] = p2
            if pick is not None:
                subject_dep[arm]["emitted"] += 1
                if p2 != pick:
                    subject_dep[arm]["changed"] += 1
                else:
                    subject_dep[arm]["same"] += 1
        results.append(row)

# ── aggregate ──────────────────────────────────────────────────────────────
def tally(arm):
    rs = [x for x in results if x["arm"] == arm]
    out = {}
    for cond in CONDITIONS:
        sub = [x for x in rs if x["condition"] == cond]
        out[cond] = {"n": len(sub), "correct": sum(x["correct"] for x in sub),
                     "picks": sum(x["pick"] is not None for x in sub)}
        out[cond + "_by_pool"] = {
            p: {"n": sum(x["pool"] == p for x in sub),
                "correct": sum(x["correct"] and x["pool"] == p for x in sub)}
            for p in ("primary", "sealed")}
    out["reasons"] = dict(collections.Counter(x["reason"] for x in rs))
    positive = [x for x in rs if x["condition"] in ("near", "far")]
    emitted = [x for x in positive if x["pick"] is not None]
    out["raw_top1_correct"] = sum(x.get("raw_top1_correct", False) for x in positive)
    out["gold_rank_hist"] = dict(collections.Counter(
        x.get("gold_rank") for x in positive))
    out["emitted_n"] = len(emitted)
    out["emitted_correct"] = sum(x["correct"] for x in emitted)
    return out

summary = {arm: tally(arm) for arm in subject_dep}

# fail-closed boundary controls (fresh, deterministic)
controls = {}
for arm in subject_dep:
    pairs = demo_pairs(bank_by_id["tra_0000"], "near", 4)
    p_dis, _, r_dis = score_arm(arm, "thaw", DISJOINT, pairs)
    p_oov, _, r_oov = score_arm(arm, "notaword_zz", ["glow", "chill", "dew", "sleet"], pairs)
    # negation channel: all pairs negated
    p_neg, _, r_neg = score_arm(arm, "thaw", ["glow", "chill", "dew", "sleet"],
                               [dict(p, negated=True) for p in pairs])
    controls[arm] = {"disjoint_pick": p_dis, "disjoint_reason": r_dis,
                     "oov_pick": p_oov, "oov_reason": r_oov,
                     "all_negated_pick": p_neg, "all_negated_reason": r_neg}

g6 = {}
for arm in subject_dep:
    t = summary[arm]
    near_rate = 100.0 * t["near"]["correct"] / t["near"]["n"]
    neg_rate = 100.0 * t["negative"]["correct"] / t["negative"]["n"]
    gr = gate_saturation({"tra_near": near_rate,
                          "tra_negative_reject": neg_rate}, mode="offline")
    g6[arm] = {"tra_near": near_rate, "tra_negative_reject": neg_rate,
               "status": gr.status, "detail": gr.detail}

# information-plane audit: is the gold separable from distractors by ANY
# deployable surface? Also count k=0 (no teaching) cases explicitly.
subject_typed = gold_typed = 0
positive_cases = 0
for (iid, cond), r in graded.items():
    if cond not in ("near", "far"):
        continue
    positive_cases += 1
    it = bank_by_id[iid]
    subj = r["input"]["query_subject"]
    gold = r["gold"]["target"]
    if it["type_label"] in it["type_sets"].get(subj, []):
        subject_typed += 1
    if it["type_label"] in it["type_sets"].get(gold, []):
        gold_typed += 1
info_audit = {"k0_items": sum(1 for (iid, c), r in graded.items()
                              if c == "near" and r["k"] == 0),
              "gold_in_glove": sum(1 for (iid, c), r in graded.items()
                                   if c in ("near", "far") and r["gold"]["target"] in lut),
              "gold_cases": positive_cases,
              "subject_type_label_present": subject_typed,
              "gold_type_label_present": gold_typed,
              "note": "type_sets is ORACLE-only; used here to AUDIT information availability, never scored."}

# ALPHA sensitivity (diagnostic only, never used to pick the headline)
alpha_sens = {}
for arm in ("OFFSET",):
    for a in (0.0, 0.5, 1.0, 2.0, 4.0):
        picked = good = 0
        for (iid, cond), r in graded.items():
            if cond not in ("near", "far"):
                continue
            it = bank_by_id[iid]
            p, _, _ = score_arm(arm, r["input"]["query_subject"],
                                r["input"]["candidates"],
                                demo_pairs(it, cond, r["k"]), alpha=a)
            picked += p is not None
            good += p == r["gold"]["target"]
        alpha_sens[str(a)] = {"emitted": picked, "correct": good, "n": 80}

# hand-check a few cases (probe input/output, ravana-debugging §1)
handcheck = []
for (iid, cond) in [("tra_0000", "near"), ("tra_0001", "far"),
                    ("tra_0000", "foil"), ("tra_0000", "negative")]:
    for arm in ("OFFSET", "CENTROID"):
        row = next(x for x in results
                   if x["item"] == iid and x["condition"] == cond and x["arm"] == arm)
        handcheck.append({k: row[k] for k in
                          ("item", "condition", "arm", "subject", "candidates",
                           "gold", "pick", "reason", "scores")})

out = {
    "hypothesis": "H_offset: relational vector offset (structure mapping) -> subject-dependent role transfer",
    "status": "scratch-only, no engine boot, no store write, no gate/threshold/item edit",
    "instrument": {
        "module_sha256": checks,
        "bank_sha256": manifests,
        "bank_families": len(bank), "bank_validation_errors": len(bank_errors),
        "rows_total": len(rows_all), "rows_fake_excluded": fake_excluded,
        "rows_live_used": len(rows_live), "graded_rows": len(graded),
        "bank_crosscheck_mismatch": len(mismatch),
        "glove_dim": dim, "glove_words": len(lut),
        "oov_words": missing_words,
        "frozen": {"FLOOR": FLOOR, "MARGIN": MARGIN, "ALPHA": ALPHA,
                   "ALT_SUBJECT": ALT_SUBJECT, "DISJOINT": DISJOINT},
    },
    "summary": summary,
    "subject_independence": subject_dep,
    "controls": controls,
    "g6": g6,
    "info_audit": info_audit,
    "alpha_sensitivity_diagnostic": alpha_sens,
    "handcheck": handcheck,
    "rows": results,
}
json.dump(out, open(os.path.join(WORKTREE, "scratch", "relational_offset_results.json"),
                    "w", encoding="utf-8"), indent=1)

# ── report ─────────────────────────────────────────────────────────────────
print("H_offset scratch experiment (no engine boot)")
print("graded rows:", len(graded), " fake excluded:", fake_excluded)
print("\nARM                 near     far     foil(rej) neg(rej)  near+far")
for arm in subject_dep:
    t = summary[arm]
    nf = t["near"]["correct"] + t["far"]["correct"]
    print("%-18s %2d/%-2d  %2d/%-2d   %2d/%-2d     %2d/%-2d    %2d/80"
          % (arm, t["near"]["correct"], t["near"]["n"],
             t["far"]["correct"], t["far"]["n"],
             t["foil"]["correct"], t["foil"]["n"],
             t["negative"]["correct"], t["negative"]["n"], nf))
print("\nSUBJECT-INDEPENDENCE (picks emitted -> pick changed when subject -> zeppelin):")
for arm, d in subject_dep.items():
    print("  %-18s emitted=%-3d changed=%-3d same=%-3d" %
          (arm, d["emitted"], d["changed"], d["same"]))
print("\nCONTROLS:")
for arm, c in controls.items():
    print("  %-18s disjoint=%s(%s) oov=%s allneg=%s(%s)" %
          (arm, c["disjoint_pick"], c["disjoint_reason"], c["oov_pick"],
           c["all_negated_pick"], c["all_negated_reason"]))
print("\nG6 (non-blocking FLAG):")
for arm, g in g6.items():
    print("  %-18s near=%.1f negrej=%.1f -> %s" %
          (arm, g["tra_near"], g["tra_negative_reject"], g["status"]))
print("\nALPHA sensitivity (diagnostic only):", alpha_sens)
print("\nRAW RANK (gate removed) over 80 near+far cases:")
for arm in subject_dep:
    t = summary[arm]
    print("  %-18s raw_top1=%2d/80  emitted=%2d  emitted_correct=%2d  gold_rank=%s"
          % (arm, t["raw_top1_correct"], t["emitted_n"],
             t["emitted_correct"], t["gold_rank_hist"]))
print("\nHANDCHECK (probe input -> output):")
for h in handcheck:
    print("  %s %-8s %-10s subj=%-9s cands=%s gold=%s -> pick=%s reason=%s" %
          (h["item"], h["condition"], h["arm"], h["subject"],
           h["candidates"], h["gold"], h["pick"], h["reason"]))
print("\nEXIT=0")
