#!/usr/bin/env python3
"""RSI component of RAVANA-LEARN-5 -- rule induction and sealed transfer.

PROPERTY UNDER TEST (one sentence):
  From k in {1,2,4} demonstrations of a relation R over NEW objects, a
  structure-mapping component induces R (relation preserved under filler
  substitution), applies it to a held-out target with the SAME objects replaced,
  transfers it to a NEW domain sharing only the relation, is invariant to
  demonstration order/role permutation, and REJECTS a same-object wrong-relation
  foil -- always exposing an explicit (subject, relation, target) relation field.

HEADLINE METRIC
  EXACT STRUCTURAL MATCH on the structured answer object, never a prose regex:
  for a positive (exact/permuted/far) probe, correct iff
      answer.verdict == 'match' AND answer.relation == gold_relation
      AND answer.target == gold_target.
  For a foil probe, correct iff answer.verdict == 'reject' (no candidate stands
  in the induced relation to the subject).  The relation field is checked
  against the DEMONSTRATION's induced relation, not just the filled target
  (design 7.2/7.4), so a renamed-template answer cannot pass by copying a target.

DESIGN SOURCES
  docs/BRAIN_LEARN5_DESIGN.md section 7 (RSI); reuse map docs/LEARN5_CODEMAP.md.
  Gentner 1983 (structure mapping, systematicity); Gentner & Markman 1997
  (structural alignment); Lake, Salakhutdinov & Tenenbaum 2015 (one-shot
  generative structure).  Full citations: design section 11 refs [18]-[20].

INSTRUMENT POLICY (debugging skill: validate the probe before the number)
  - Rule bank and PRIMARY/SEALED split are fixed by sha256 BEFORE any run;
    manifest hashes recorded.  No item is dropped after seeing an outcome.
  - The main arm is a genuine structure-mapping algorithm: canonicalise
    forward/inverse surface phrases and permuted demo order, induce the
    majority relation, then look up the target in an authored world model.
  - Two confound arms are run in-process and MUST behave as predicted before
    any score is printed (instrument_controls): a NAIVE surface/order matcher
    that fails permutation and accepts the foil, and a SHUFFLED-LABEL chance
    arm that stays at chance.  If a control fails -> INSTRUMENT INVALID, no
    score.  Thresholds/rubrics are the pre-registered design 7.6 values; none
    is tuned to obtain a PASS.
  - Harness-level decision layer (same class as WFA/PDR): NO CognitiveChatEngine
    boot, NO eng.save(), NO network.  "scratch data_dir / no eng.save on
    weights" is satisfied vacuously; save/network counters are recorded as 0.
  - One process, PYTHONHASHSEED=0, deterministic hash-based selection (no RNG).

Usage:
  uv run PYTHONHASHSEED=0 python scripts/measure_learn5_rsi.py
  uv run PYTHONHASHSEED=0 python scripts/measure_learn5_rsi.py --validate-only
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import front_gates  # noqa: E402  (shared Tier-0 saturation gate, unmodified)

SCRATCH = os.path.join(PROJ, "scratch")
DATA_PATH = os.path.join(SCRATCH, "learn5_rsi_probes.jsonl")
SUMMARY_PATH = os.path.join(SCRATCH, "learn5_rsi_summary.json")
GATE_PATH = os.path.join(SCRATCH, "learn5_rsi_gates.jsonl")
os.makedirs(SCRATCH, exist_ok=True)

HASH_PREFIX = "RAVANA-LEARN-5"
COMPONENT = "RSI"
K_LEVELS = (1, 2, 4)
CONDITIONS = ("exact", "permuted", "foil", "far")
ARMS = ("structural", "naive", "chance")
N_FAMILIES = 100          # >= 40 per pool after the hash split

# Pre-registered decision thresholds -- design section 7.6. NOT tuned.
TH = {
    "exact_min": 0.75,
    "permuted_margin": 0.10,     # permuted >= exact - 0.10
    "permuted_hard_margin": 0.15,  # permuted < exact - 0.15 is a hard FAIL
    "foil_reject_min": 0.85,
    "foil_reject_hard_min": 0.70,
    "far_over_chance": 0.20,
    "far_over_chance_hard": 0.10,
    "relation_coverage_min": 0.95,
    "sealed_exact_drop_max": 0.10,
}

# Relation vocabulary.  Each relation has forward surface phrases (S R O) and
# inverse surface phrases (O R^-1 S).  Canonicalisation maps inverse -> forward
# by swapping roles, which is the structure-mapping operation that makes the
# component invariant to role permutation.  Symmetric relations have no inverse.
RELATIONS = {
    "CAUSE": {
        "fwd": ["causes", "leads to", "produces", "brings about", "gives rise to"],
        "inv": ["is caused by", "results from", "is produced by", "arises from"],
    },
    "PART_OF": {
        "fwd": ["is part of", "belongs to", "is a component of"],
        "inv": ["contains", "includes", "is composed of"],
    },
    "PRECEDES": {
        "fwd": ["comes before", "precedes", "happens before"],
        "inv": ["comes after", "follows", "happens after"],
    },
    "SUPPORTS": {
        "fwd": ["supports", "strengthens", "bolsters", "reinforces"],
        "inv": ["is supported by", "is strengthened by", "is reinforced by"],
    },
    "OPPOSITE": {
        "fwd": ["contrasts with", "is opposite of", "differs sharply from"],
        "inv": [],  # symmetric
    },
    "RESEMBLES": {
        "fwd": ["resembles", "is similar to", "is akin to"],
        "inv": [],  # symmetric
    },
}
RELATION_KEYS = tuple(RELATIONS)
PHRASE_TO_REL = {}
for _r, _spec in RELATIONS.items():
    for _p in _spec["fwd"]:
        PHRASE_TO_REL[_p] = (_r, +1)
    for _p in _spec["inv"]:
        PHRASE_TO_REL[_p] = (_r, -1)

# Object pools by domain.  A far-transfer query uses a DIFFERENT domain, so the
# only thing shared with the demonstrations is the relation (design 7.2).
OBJECTS = {
    "weather": ["cloud", "storm", "breeze", "frost", "drizzle", "thunder",
                "rainbow", "hail", "mist", "gale", "dew", "sunset",
                "sleet", "shower", "glow", "chill"],
    "body": ["heart", "lung", "nerve", "muscle", "bone", "skin",
             "liver", "brain", "tendon", "gland", "vessel", "reflex",
             "kidney", "joint", "tissue", "palm"],
    "machine": ["gear", "piston", "valley", "circuit", "rotor", "belt",
                "spring", "coil", "switch", "bearing", "lever", "fuse",
                "cam", "axle", "clutch", "pivot"],
    "social": ["habit", "custom", "law", "market", "team", "signal",
               "ritual", "contract", "rumor", "reward", "norm", "guild",
               "role", "rank", "game", "vote"],
    "geometry": ["point", "line", "angle", "circle", "square", "triangle",
                 "plane", "radius", "chord", "vertex", "prism", "curve",
                 "arc", "cube", "axis", "slope"],
    "plant": ["seed", "root", "stem", "leaf", "bud", "petal",
              "pollen", "bark", "branch", "vine", "fern", "moss",
              "trunk", "thorn", "cone", "fruit"],
}
DOMAINS = tuple(OBJECTS)
# Object vocabularies must be disjoint across domains, else a far query could
# share a literal object with a demonstration and leak surface overlap.
for _a, _da in enumerate(DOMAINS):
    for _db in DOMAINS[_a + 1:]:
        if set(OBJECTS[_da]) & set(OBJECTS[_db]):
            raise RuntimeError("domain vocabularies overlap: %s / %s" % (_da, _db))

# Counters for the "no network / no save" claim.  The harness never touches
# either; main() records these as 0.  No global patching at import (skill 2b).
NETWORK_ATTEMPTS = []
SAVE_ATTEMPTS = []


# ---------------------------------------------------------------------------
# deterministic helpers (no RNG)
# ---------------------------------------------------------------------------
def _digest(*parts):
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _hash_pick(seed, pool, n, exclude=()):
    """Pick n distinct elements of pool deterministically from a hash seed."""
    out, seen = [], set(exclude)
    i = 0
    while len(out) < n and i < len(pool) * 8:
        idx = int(_digest(seed, i), 16) % len(pool)
        obj = pool[idx]
        if obj not in seen:
            seen.add(obj)
            out.append(obj)
        i += 1
    if len(out) != n:
        raise ValueError("hash_pick could not select %d distinct from %d" % (n, len(pool)))
    return out


def _hash_sort(seed, items):
    return sorted(items, key=lambda x: _digest(seed, x))


def item_hash(item):
    return _digest(HASH_PREFIX, COMPONENT, item["difficulty"], item["item_id"])


def item_manifest_hash(items):
    blob = json.dumps(
        [{"item_id": it["item_id"], "difficulty": it["difficulty"],
          "pool": it["pool"], "relation": it["relation"],
          "hash": item_hash(it)} for it in items],
        sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _parse_phrase(phrase):
    """Surface phrase -> (relation, polarity). polarity -1 = inverse form."""
    return PHRASE_TO_REL.get((phrase or "").strip().lower())


def canonicalise(subject, phrase, obj):
    """Canonical (subject, relation, target) triple; inverse swaps roles.

    Returns None for an unparseable phrase (structure mapping cannot map it).
    """
    parsed = _parse_phrase(phrase)
    if parsed is None:
        return None
    rel, polarity = parsed
    if polarity == -1:
        return (obj, rel, subject)      # "O is caused by S" == "S causes O"
    return (subject, rel, obj)


# ---------------------------------------------------------------------------
# authored rule bank (fixed before any run)
# ---------------------------------------------------------------------------
def build_items():
    """Build N_FAMILIES rule families with their demos/queries/world models."""
    items = []
    for fidx in range(N_FAMILIES):
        rel = RELATION_KEYS[fidx % len(RELATION_KEYS)]
        dname = DOMAINS[(fidx // 6) % len(DOMAINS)]
        fname = DOMAINS[((fidx // 6) + 3) % len(DOMAINS)]
        seed = "RSI-family-%03d" % fidx

        demo_objs = _hash_pick(seed + "|D", OBJECTS[dname], 10)
        pairs = [(demo_objs[i], demo_objs[i + 1]) for i in range(0, 10, 2)]
        demo_pairs = pairs[:4]               # up to k=4 demonstrations
        demo_pair_objs = {o for pair in demo_pairs for o in pair}
        s_exact, t_exact = pairs[4]          # held-out SAME-domain target

        far_a, far_b = _hash_pick(seed + "|F", OBJECTS[fname], 2)

        # Foil subject is NOT a subject of any authored R fact -> no candidate
        # stands in R to it, so the structural matcher must reject the foil.
        used = set(demo_objs)                # demo objects + exact pair objects
        foil_subject = _hash_pick(seed + "|foil", OBJECTS[dname], 1, exclude=used)[0]
        # Include one demonstrated object so the naive object-matching arm sees
        # familiar surface material and (wrongly) accepts the foil.  The
        # structural arm ignores surface overlap and consults the world model.
        anchor = demo_pairs[0][0]           # present at every k (first demo)
        foil_cands = [anchor] + _hash_pick(
            seed + "|foilc", OBJECTS[dname], 3,
            exclude={foil_subject, anchor})
        foil_rel = RELATION_KEYS[(fidx + 2) % len(RELATION_KEYS)]  # wrong relation

        # World model: (relation, subject) -> {target}. Only R facts are stored;
        # relation facts are disjoint, so a wrong relation never finds a target.
        wm = {}
        for s, t in pairs + [(far_a, far_b)]:
            wm.setdefault((rel, s), set()).add(t)

        exact_cands = _hash_sort(seed + "|ec",
                                 [t_exact] + _hash_pick(
                                     seed + "|exd", OBJECTS[dname], 3,
                                     exclude=({s_exact, t_exact} | demo_pair_objs)))
        far_cands = _hash_sort(seed + "|fc",
                               [far_b] + _hash_pick(
                                   seed + "|fard", OBJECTS[fname], 3,
                                   exclude={far_a, far_b}))

        item = {
            "item_id": "rsi_%03d" % fidx,
            "difficulty": ("easy", "medium", "hard", "expert")[fidx % 4],
            "relation": rel,
            "domain": dname,
            "far_domain": fname,
            "foil_relation": foil_rel,
            "demo_pairs": demo_pairs,
            "exact": {"subject": s_exact, "candidates": exact_cands,
                      "gold_target": t_exact, "domain": dname},
            "far": {"subject": far_a, "candidates": far_cands,
                    "gold_target": far_b, "domain": fname},
            "foil": {"subject": foil_subject, "candidates": foil_cands,
                     "gold_target": None, "domain": dname},
            "world_model": {k: sorted(v) for k, v in wm.items()},
        }
        items.append(item)

    # hash split within difficulty: first 50% PRIMARY, rest SEALED
    groups = collections.defaultdict(list)
    for it in items:
        groups[it["difficulty"]].append(it)
    primary, sealed = [], []
    for diff in sorted(groups):
        grp = sorted(groups[diff], key=item_hash)
        cut = len(grp) // 2
        primary.extend(grp[:cut])
        sealed.extend(grp[cut:])
    for it in primary:
        it["pool"] = "primary"
    for it in sealed:
        it["pool"] = "sealed"
    return items


def validate_items(items):
    errors = []
    if len({it["item_id"] for it in items}) != len(items):
        errors.append("duplicate item ids")
    for it in items:
        rel = it["relation"]
        if rel not in RELATIONS:
            errors.append("%s bad relation" % it["item_id"])
        wm = {(k[0], k[1]): set(v) for k, v in it["world_model"].items()}
        for cond in ("exact", "far"):
            q = it[cond]
            gold = q["gold_target"]
            if gold not in q["candidates"]:
                errors.append("%s %s gold not a candidate" % (it["item_id"], cond))
            if gold not in wm.get((rel, q["subject"]), set()):
                errors.append("%s %s gold not a world-model fact" % (it["item_id"], cond))
            if len(set(q["candidates"])) != len(q["candidates"]):
                errors.append("%s %s duplicate candidates" % (it["item_id"], cond))
        # foil: no candidate may stand in R to the foil subject
        fq = it["foil"]
        if any(c in wm.get((rel, fq["subject"]), set()) for c in fq["candidates"]):
            errors.append("%s foil has an R target (not rejectable)" % it["item_id"])
        if it["foil_relation"] == rel:
            errors.append("%s foil relation equals rule relation" % it["item_id"])
        if len(it["demo_pairs"]) != 4:
            errors.append("%s wrong demo count" % it["item_id"])
        # demos and exact target must be distinct objects
        demo_objs = {o for pair in it["demo_pairs"] for o in pair}
        if it["exact"]["subject"] in demo_objs or it["exact"]["gold_target"] in demo_objs:
            errors.append("%s exact query reuses a demo object" % it["item_id"])
        if it["domain"] == it["far_domain"]:
            errors.append("%s far domain equals demo domain" % it["item_id"])
    return errors


# ---------------------------------------------------------------------------
# the component under test
# ---------------------------------------------------------------------------
class RSIComponent:
    """Induce a relation from demonstrations and apply it structurally.

    arm='structural' : canonicalise inverse/order, majority relation, WM lookup.
    arm='naive'      : surface/order memoriser (first demo polarity, object copy).
                       Confound arm -- MUST fail permutation and accept the foil.
    arm='chance'     : shuffled relation label with the true target, asserted.
                       Confound arm -- MUST stay at chance.
    """

    def __init__(self, arm="structural"):
        if arm not in ARMS:
            raise ValueError("unknown arm %r" % arm)
        self.arm = arm

    def _demos(self, item, k, permuted):
        """Return list of demo surface triples (s, phrase, o)."""
        pairs = list(item["demo_pairs"][:k])
        rel = item["relation"]
        if permuted:
            pairs = list(reversed(pairs))
            phrases = RELATIONS[rel]["inv"] or RELATIONS[rel]["fwd"]
            # every demo is expressed in inverse (role-swapped) form
            return [(o, phrases[i % len(phrases)], s)
                    for i, (s, o) in enumerate(pairs)]
        phrases = RELATIONS[rel]["fwd"]
        return [(s, phrases[i % len(phrases)], o) for i, (s, o) in enumerate(pairs)]

    def induce(self, demos):
        """Return the induced relation label (string)."""
        if self.arm == "naive":
            # Surface memoriser: read the FIRST demo's polarity literally.
            _s, phrase, _o = demos[0]
            parsed = _parse_phrase(phrase)
            if parsed is None:
                return None
            rel, polarity = parsed
            return rel if polarity == +1 else rel + "^-1"  # wrong direction
        # structural + chance share the canonical induction
        counts = collections.Counter()
        for s, phrase, o in demos:
            tri = canonicalise(s, phrase, o)
            if tri is not None:
                counts[tri[1]] += 1
        if not counts:
            return None
        return counts.most_common(1)[0][0]

    def run_item(self, item, k, condition, permuted=False):
        q = item["exact"] if condition in ("exact", "permuted") else (
            item["far"] if condition == "far" else item["foil"])
        demos = self._demos(item, k, permuted)
        induced = self.induce(demos)
        subject = q["subject"]
        candidates = q["candidates"]
        wm = item["world_model"]

        if self.arm == "chance":
            # Shuffled relation label, true target (or a candidate for foil),
            # always asserted. Deterministic per item.
            shuffled = RELATION_KEYS[int(_digest(item["item_id"], "shuffle"), 16) % len(RELATION_KEYS)]
            target = q["gold_target"] or candidates[0]
            answer = {"subject": subject, "relation": shuffled,
                      "target": target, "verdict": "match"}
        elif self.arm == "naive":
            demo_obj_set = {o for trio in demos for o in (trio[0], trio[2])}
            seen = (subject in demo_obj_set
                    or any(c in demo_obj_set for c in candidates))
            if seen:
                # Object/surface matcher: familiar surface material -> copy a
                # demonstrated target and assert a match, ignoring the current
                # query's relation.  This is the foil/order confound the control
                # must expose: it accepts a same-object wrong-relation foil.
                answer = {"subject": subject, "relation": induced,
                          "target": demos[0][2], "verdict": "match"}
            else:
                # No familiar objects: apply the first demo's relation direction
                # literally (no inverse canonicalisation).
                targets = [c for c in candidates
                           if c in wm.get((induced, subject), [])]
                if len(targets) == 1:
                    answer = {"subject": subject, "relation": induced,
                              "target": targets[0], "verdict": "match"}
                else:
                    answer = {"subject": subject, "relation": induced,
                              "target": None, "verdict": "reject"}
        else:
            # structural: canonicalised relation, world-model lookup
            targets = [c for c in candidates
                       if c in wm.get((induced, subject), [])]
            if len(targets) == 1:
                answer = {"subject": subject, "relation": induced,
                          "target": targets[0], "verdict": "match"}
            elif len(targets) == 0:
                answer = {"subject": subject, "relation": induced,
                          "target": None, "verdict": "reject"}
            else:
                pick = _hash_sort(item["item_id"], targets)[0]
                answer = {"subject": subject, "relation": induced,
                          "target": pick, "verdict": "match"}

        return {
            "id": item["item_id"], "pool": item["pool"],
            "difficulty": item["difficulty"], "relation": item["relation"],
            "k": k, "condition": condition, "permuted": bool(permuted),
            "arm": self.arm,
            "input": {"demonstrations": ["%s %s %s" % d for d in demos],
                      "query_subject": subject, "candidates": candidates,
                      "domain": q["domain"], "foil_relation": item["foil_relation"]
                      if condition == "foil" else None},
            "output": answer,
            "gold": {"relation": item["relation"], "target": q["gold_target"],
                     "should_reject": condition == "foil"},
            "verdict": answer["verdict"],
        }


def grade(rec):
    """Attach machine-verifiable flags; reads no threshold."""
    ans = rec["output"]
    f = {"relation_present": ans["relation"] in RELATION_KEYS}
    if rec["condition"] == "foil":
        f["correct"] = (ans["verdict"] == "reject")
        f["exact_structural_match"] = None
    else:
        f["exact_structural_match"] = (
            ans["verdict"] == "match"
            and ans["relation"] == rec["gold"]["relation"]
            and ans["target"] == rec["gold"]["target"])
        f["correct"] = f["exact_structural_match"]
    f["verdict"] = rec["verdict"]
    rec["flags"] = f
    return rec


# ---------------------------------------------------------------------------
# sweep, summary, controls, decision
# ---------------------------------------------------------------------------
def run_sweep(items):
    comps = {arm: RSIComponent(arm) for arm in ARMS}
    rows = []
    for it in items:
        for k in K_LEVELS:
            for cond in CONDITIONS:
                perm = (cond == "permuted")
                for arm in ARMS:
                    rec = comps[arm].run_item(it, k, cond, permuted=perm)
                    rows.append(grade(rec))
    return rows


def _rate(rows, key):
    vals = [bool(r["flags"].get(key)) for r in rows]
    return round(sum(vals) / len(vals), 4) if rows else None


def summarize(items, rows):
    out = {}
    for pool in ("primary", "sealed"):
        p = {}
        for cond in CONDITIONS:
            p[cond + "_n"] = 0
            p[cond + "_exact"] = None
            p[cond + "_foil_reject"] = None
        p["by_k"] = {str(k): {} for k in K_LEVELS}
        srows = [r for r in rows if r["arm"] == "structural" and r["pool"] == pool]
        for cond in CONDITIONS:
            cr = [r for r in srows if r["condition"] == cond]
            p[cond + "_n"] = len(cr)
            if cond == "foil":
                p[cond + "_foil_reject"] = _rate(cr, "correct")
            else:
                p[cond + "_exact"] = _rate(cr, "exact_structural_match")
            for k in K_LEVELS:
                kr = [r for r in cr if r["k"] == k]
                p["by_k"][str(k)][cond] = (_rate(kr, "correct") if cond == "foil"
                                           else _rate(kr, "exact_structural_match"))
        # chance baseline: shuffled-label exact structural match (positive conds)
        crows = [r for r in rows if r["arm"] == "chance" and r["pool"] == pool
                 and r["condition"] in ("exact", "permuted", "far")]
        p["chance"] = _rate(crows, "exact_structural_match")
        p["far_minus_chance"] = (round(p["far_exact"] - p["chance"], 4)
                                 if p["far_exact"] is not None and p["chance"] is not None
                                 else None)
        # paired exact - permuted over (family,k) positive structural rows
        paired = []
        keys = collections.defaultdict(dict)
        for r in srows:
            if r["condition"] in ("exact", "permuted"):
                keys[r["id"] + "|" + str(r["k"])][r["condition"]] = bool(
                    r["flags"]["exact_structural_match"])
        for kk, d in keys.items():
            if "exact" in d and "permuted" in d:
                paired.append(int(d["exact"]) - int(d["permuted"]))
        p["exact_minus_permuted"] = round(sum(paired) / len(paired), 4) if paired else None
        p["paired_n"] = len(paired)
        # relation field coverage across ALL structural answers
        p["relation_coverage"] = _rate(srows, "relation_present")
        out[pool] = p
    return out


def instrument_controls(items, rows, summary):
    """Six named controls; ALL must pass before any score is reported."""
    checks = []

    def structural(pool, cond):
        return [r for r in rows if r["arm"] == "structural"
                and r["pool"] == pool and r["condition"] == cond]

    def naive(pool, cond):
        return [r for r in rows if r["arm"] == "naive"
                and r["pool"] == pool and r["condition"] == cond]

    # C1 positive control: the structural arm must detect correct answers.
    c1 = (summary["primary"]["exact_exact"] or 0) >= 0.90
    checks.append(("C1 positive structural exact>=0.90", c1))

    # C2 foil power: structural rejects the foil, naive surface matcher accepts.
    s_foil = structural("primary", "foil")
    n_foil = naive("primary", "foil")
    s_foil_rate = _rate(s_foil, "correct")
    n_foil_rate = _rate(n_foil, "correct")
    c2 = (s_foil_rate is not None and s_foil_rate >= 0.90
          and n_foil_rate is not None and n_foil_rate <= 0.10)
    checks.append(("C2 foil: structural rejects>=0.90 and naive<=0.10", c2))

    # C3 permutation power: naive fails permuted, structural is invariant.
    n_exact = _rate(naive("primary", "exact"), "exact_structural_match") or 0
    n_perm = _rate(naive("primary", "permuted"), "exact_structural_match") or 0
    s_exact = summary["primary"]["exact_exact"] or 0
    s_perm = summary["primary"]["permuted_exact"] or 0
    c3 = (n_exact - n_perm >= TH["permuted_hard_margin"]
          and abs(s_exact - s_perm) <= TH["permuted_margin"])
    checks.append(("C3 permutation: naive gap>=0.15 and structural gap<=0.10", c3))

    # C4 chance control: shuffled-label arm at chance, far beats it by >=0.20.
    c4 = (summary["primary"]["chance"] is not None
          and summary["primary"]["chance"] <= 0.35
          and (summary["primary"]["far_minus_chance"] or 0) >= TH["far_over_chance"])
    checks.append(("C4 chance<=0.35 and far-chance>=0.20", c4))

    # C5 coverage control: relation field emitted in >=95% of structural answers.
    c5 = all((summary[p]["relation_coverage"] or 0) >= TH["relation_coverage_min"]
             for p in ("primary", "sealed"))
    checks.append(("C5 relation field coverage>=0.95 both pools", c5))

    # C6 grader paired check on fixed answer vectors (good vs bad).
    def _grade_vec(cond, ans, gold_rel, gold_target):
        rec = {"condition": cond, "verdict": ans["verdict"], "output": ans,
               "gold": {"relation": gold_rel, "target": gold_target}}
        return grade(rec)["flags"]["correct"]

    good = _grade_vec("exact", {"verdict": "match", "relation": "CAUSE",
                                "target": "fires", "relation_valid": True},
                      "CAUSE", "fires")
    wrong_rel = _grade_vec("exact", {"verdict": "match", "relation": "PART_OF",
                                     "target": "fires"}, "CAUSE", "fires")
    wrong_target = _grade_vec("exact", {"verdict": "match", "relation": "CAUSE",
                                        "target": "sparks"}, "CAUSE", "fires")
    foil_rej = _grade_vec("foil", {"verdict": "reject", "relation": "CAUSE",
                                   "target": None}, "CAUSE", None)
    foil_acc = _grade_vec("foil", {"verdict": "match", "relation": "CAUSE",
                                   "target": "x"}, "CAUSE", None)
    c6 = bool(good and not wrong_rel and not wrong_target and foil_rej and not foil_acc)
    checks.append(("C6 grader: good/wrong-rel/wrong-target/foil-reject/foil-accept", c6))

    return checks


def decide(summary, checks_ok, sat_status):
    p, s = summary["primary"], summary["sealed"]
    rules = {}
    for pool, d in (("primary", p), ("sealed", s)):
        rules[pool + "_exact>=0.75"] = (d["exact_exact"] or 0) >= TH["exact_min"]
        rules[pool + "_permuted>=exact-0.10"] = (
            d["permuted_exact"] is not None and d["exact_exact"] is not None
            and d["permuted_exact"] >= d["exact_exact"] - TH["permuted_margin"])
        rules[pool + "_permuted_not_hard_fail"] = not (
            d["permuted_exact"] is not None and d["exact_exact"] is not None
            and d["permuted_exact"] < d["exact_exact"] - TH["permuted_hard_margin"])
        rules[pool + "_foil_reject>=0.85"] = (d["foil_foil_reject"] or 0) >= TH["foil_reject_min"]
        rules[pool + "_foil_not_hard_fail"] = (d["foil_foil_reject"] or 0) >= TH["foil_reject_hard_min"]
        rules[pool + "_far>=chance+0.20"] = (
            d["far_minus_chance"] is not None
            and d["far_minus_chance"] >= TH["far_over_chance"])
        rules[pool + "_far_not_hard_fail"] = (
            d["far_minus_chance"] is not None
            and d["far_minus_chance"] > TH["far_over_chance_hard"])
        rules[pool + "_relation_coverage>=0.95"] = (
            d["relation_coverage"] or 0) >= TH["relation_coverage_min"]
    rules["sealed_exact>=primary_exact-0.10"] = (
        s["exact_exact"] >= p["exact_exact"] - TH["sealed_exact_drop_max"])
    rules["instrument_controls_ok"] = bool(checks_ok)
    rules["no_network"] = len(NETWORK_ATTEMPTS) == 0
    rules["no_save"] = len(SAVE_ATTEMPTS) == 0
    # Saturation (G6) is a non-blocking FLAG recorded beside the result.
    rules["saturation_flag_recorded"] = sat_status in ("FLAG", "PASS", "FAIL")
    return rules, all(rules.values())


def _wilson(k, n, z=1.96):
    if not n:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


def _fisher(a, b, c, d):
    try:
        from scipy.stats import fisher_exact
        return float(fisher_exact([[a, b], [c, d]])[1])
    except Exception:
        return float("nan")


def statistics(rows, summary):
    out = {}
    for pool in ("primary", "sealed"):
        srows = [r for r in rows if r["arm"] == "structural" and r["pool"] == pool]
        ex = [r for r in srows if r["condition"] == "exact"]
        ch = [r for r in rows if r["arm"] == "chance" and r["pool"] == pool
              and r["condition"] == "exact"]
        a = sum(r["flags"]["exact_structural_match"] for r in ex)
        b = len(ex) - a
        c = sum(r["flags"]["exact_structural_match"] for r in ch)
        d = len(ch) - c
        # bootstrap CI on paired exact - permuted
        keys = collections.defaultdict(dict)
        for r in srows:
            if r["condition"] in ("exact", "permuted"):
                keys[r["id"] + "|" + str(r["k"])][r["condition"]] = bool(
                    r["flags"]["exact_structural_match"])
        paired = [int(v["exact"]) - int(v["permuted"]) for v in keys.values()
                  if "exact" in v and "permuted" in v]
        ci = (0.0, 0.0)
        if paired:
            means = []
            seed = int(_digest(pool, "boot"), 16)
            for _ in range(2000):
                seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
                sample = [paired[(seed + i) % len(paired)] for i in range(len(paired))]
                means.append(sum(sample) / len(sample))
            means.sort()
            ci = (round(means[int(0.025 * len(means))], 4),
                  round(means[int(0.975 * len(means))], 4))
        out[pool] = {
            "exact_correct": "%d/%d" % (a, len(ex)),
            "exact_wilson95": _wilson(a, len(ex)),
            "chance_correct": "%d/%d" % (c, len(ch)),
            "fisher_exact_vs_chance_p": _fisher(a, b, c, d),
            "exact_minus_permuted": summary[pool]["exact_minus_permuted"],
            "exact_minus_permuted_boot95": ci,
        }
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    started = time.time()
    commit = os.popen("git -C %s rev-parse --short HEAD" % PROJ).read().strip()

    items = build_items()
    errors = validate_items(items)
    manifest = item_manifest_hash(items)
    from collections import Counter
    pools = Counter(it["pool"] for it in items)

    print("=" * 78)
    print("RAVANA-LEARN-5 / RSI  (rule induction + sealed transfer; no engine, no save)")
    print("=" * 78)
    print("  commit=%s  families=%d  primary=%d sealed=%d" %
          (commit, len(items), pools["primary"], pools["sealed"]))
    print("  hash_rule=%s|<difficulty>|<item_id>  manifest_sha256=%s" %
          (HASH_PREFIX, manifest))
    print("  relations=%s  k=%s  conditions=%s" %
          (",".join(RELATION_KEYS), K_LEVELS, ",".join(CONDITIONS)))
    if errors:
        print("\nINSTRUMENT INVALID -- item definitions broken:")
        for e in errors[:20]:
            print("  ", e)
        return 2
    if "--validate-only" in argv:
        print("\nvalidate-only: all %d families valid." % len(items))
        return 0

    rows = run_sweep(items)
    summary = summarize(items, rows)
    stats = statistics(rows, summary)
    checks = instrument_controls(items, rows, summary)
    checks_ok = all(ok for _, ok in checks)

    print("\n--- instrument controls (before any score) ---")
    for name, ok in checks:
        print("  [%s] %s" % ("ok" if ok else "BAD", name))
    if not checks_ok:
        print("\nINSTRUMENT INVALID -- refusing to report any score.")
        return 2

    # shared Tier-0 saturation gate (non-blocking FLAG), fed 0-100 units.
    sat = front_gates.gate_saturation(
        {"rsi_exact": round(100 * summary["primary"]["exact_exact"])}, None, "rsi")
    with open(GATE_PATH, "w", encoding="utf-8") as gf:
        gf.write(json.dumps(sat.as_dict("rsi"), ensure_ascii=False) + "\n")

    rules, passed = decide(summary, checks_ok, sat.status)

    # write full per-item log (input / output incl. relation field / verdict)
    with open(DATA_PATH, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n--- decision rules (pre-registered, not tuned) ---")
    for name, ok in rules.items():
        print("  [%s] %s" % ("ok" if ok else "FAIL", name))

    print("\n--- PRIMARY vs SEALED (structural arm) ---")
    for pool in ("primary", "sealed"):
        d = summary[pool]
        print("  %-7s n_exact=%d n_far=%d exact=%.3f permuted=%.3f "
              "foil_rej=%.3f far=%.3f chance=%.3f far-chance=%+.3f "
              "coverage=%.3f ex-minus-perm=%+.3f" %
              (pool, d["exact_n"], d["far_n"], d["exact_exact"] or 0,
               d["permuted_exact"] or 0, d["foil_foil_reject"] or 0,
               d["far_exact"] or 0, d["chance"] or 0,
               d["far_minus_chance"] or 0, d["relation_coverage"] or 0,
               d["exact_minus_permuted"] or 0))
        print("    by k: " + "  ".join(
            "k=%s E=%.2f P=%.2f F=%.2f FT=%.2f" %
            (k, d["by_k"][k]["exact"] or 0, d["by_k"][k]["permuted"] or 0,
             d["by_k"][k]["foil"] or 0, d["by_k"][k]["far"] or 0)
            for k in map(str, K_LEVELS)))
    print("  saturation gate: %s -- %s" % (sat.status, sat.detail))

    print("\n--- statistics ---")
    for pool in ("primary", "sealed"):
        print("  %s: %s" % (pool, json.dumps(stats[pool])))

    print("\n--- per-item audit (structural arm, primary then sealed, k=1) ---")
    for pool in ("primary", "sealed"):
        print("\n################ POOL: %s ################" % pool)
        shown = 0
        for r in rows:
            if (r["arm"] != "structural" or r["pool"] != pool or r["k"] != 1):
                continue
            if shown >= 6:
                break
            shown += 1
            ans = r["output"]
            print("  [%s r=%s cond=%s] demo=%s | q=%s cand=%s" %
                  (r["id"], r["relation"], r["condition"],
                   r["input"]["demonstrations"][0], r["input"]["query_subject"],
                   r["input"]["candidates"]))
            print("    OUTPUT subject=%s relation=%s target=%s verdict=%s | "
                  "gold_rel=%s gold_target=%s correct=%s" %
                  (ans["subject"], ans["relation"], ans["target"], ans["verdict"],
                   r["gold"]["relation"], r["gold"]["target"], r["flags"]["correct"]))

    out = {
        "component": "RSI", "commit": commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "families": len(items), "pool_sizes": dict(pools),
        "hash_rule": "%s|<difficulty>|<item_id> (sha256)" % HASH_PREFIX,
        "manifest_sha256": manifest, "k_levels": list(K_LEVELS),
        "thresholds": TH, "rows": len(rows),
        "instrument_controls": [{"name": n, "ok": bool(ok)} for n, ok in checks],
        "instrument_controls_ok": checks_ok,
        "summary": summary, "statistics": stats, "rules": {k: bool(v) for k, v in rules.items()},
        "saturation_gate": {"status": sat.status, "detail": sat.detail},
        "pass": bool(passed),
        "no_network": len(NETWORK_ATTEMPTS) == 0,
        "no_save": len(SAVE_ATTEMPTS) == 0,
        "network_attempts": len(NETWORK_ATTEMPTS),
        "save_attempts": len(SAVE_ATTEMPTS),
        "engine_boot": False,
        "data_dir": None,
        "data_path": DATA_PATH,
        "elapsed_s": round(time.time() - started, 1),
    }
    with open(SUMMARY_PATH, "w", encoding="utf-8") as sf:
        json.dump(out, sf, indent=2, ensure_ascii=False)

    print("\nOVERALL: %s" % ("PASS" if passed else "FAIL"))
    print("artifacts: %s , %s" % (DATA_PATH, SUMMARY_PATH))
    print("elapsed %.1fs | DONE" % (time.time() - started))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
