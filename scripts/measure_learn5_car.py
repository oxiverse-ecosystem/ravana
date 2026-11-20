#!/usr/bin/env python3
"""CAR component of RAVANA-LEARN-5 — measured learning curve.

PROPERTY UNDER TEST (one sentence):
  After k spaced, provenance-trusted 2-hop causal teaching episodes about a
  subject X, a FRESH COLD-RELOADED engine yields a provenance-trusted >=2-hop
  causal chain for X on a held-out paraphrase probe at a higher rate than an
  untaught (k=0) item, and this does not happen for definition-only teaching.

HEADLINE METRIC
  chain_i(stage) = 1 iff trusted_chain([node_ids(X)], graph, "backward")
  returns an assertable chain (grade_chain verdict 'assert' AND
  assertable_chain == every hop provenance-trusted). This is the STRUCTURED
  CHAIN OBJECT, never a prose regex (debugging skill: "measure the decision,
  not the prose"). The held-out paraphrase question is asked at every stage so
  the retrieval path is exercised and raw input/output are logged; the emitted
  answer reach is reported separately as a secondary endpoint.

  Delta_CAR(k) = mean_i chain_i(k, cold) - mean_i chain_i(baseline).
  Primary test: risk difference (k=8 minus k=0) on the delayed cold-reload
  probe, with Wilson 95% CIs and Fisher exact.

DESIGN SOURCES
  docs/BRAIN_LEARN5_DESIGN.md section 3 (CAR); docs/LEARN5_CODEMAP.md 3.1;
  docs/BENCHMARKING_SYSTEMS_2026.md 3 / 6.0 / 6.1.
  Refs [1] Ebbinghaus 1885, [2] Murre & Dros 2015 PLOS ONE, [3] Roediger &
  Karpicke 2006 Psych Science, [4] Karpicke & Blunt 2011 Science, [5] Cepeda
  et al. 2006 Psych Bulletin.

INSTRUMENT POLICY
  - Hash split (RAVANA-LEARN-5|CAR|role|difficulty|item_id) frozen BEFORE any
    engine boot; manifest hashes recorded. 50 primary + 50 sealed.
  - Reuses measure_learning_curve helpers verbatim (run_detector_controls,
    prepare_data_dir, causal_snapshot, trusted_chain, trusted_chain_summary,
    ask, answer_verdict, teach_conversational, teach_web, install_trace,
    label_map, node_ids_for).
  - ONE process, scratch data_dir copied from weights/, NEVER weights/ itself.
    eng.save() is called ONCE into the isolated scratch data_dir to enable the
    delayed cold-reload endpoint; weights/ is never a save target.
  - No threshold/rubric is tuned to obtain a pass; run_detector_controls must
    pass before any curve is reported.

Usage:
  uv run PYTHONHASHSEED=0 python scripts/measure_learn5_car.py
  uv run PYTHONHASHSEED=0 python scripts/measure_learn5_car.py --quick   # iteration
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import shutil
import sys
import time

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ["RAVANA_OFFLINE"] = "1"

# Import the existing CAR harness as a MODULE and reuse its helpers verbatim.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_learning_curve as mlc

SCRATCH = os.path.join(PROJ, "scratch")
CAR_DATA_DIR = os.path.join(SCRATCH, "learn5_car_data")
OUT_PATH = os.path.join(SCRATCH, "learn5_car_measured.json")

HASH_PREFIX = "RAVANA-LEARN-5"
COMPONENT = "CAR"
K_LEVELS = (0, 1, 2, 4, 8)
MAX_K = 8
# Traversal budget for the trusted-chain detector. The reused prototype helper
# uses limit=8. An invalid pilot run showed that after k=8 teaching the graph
# holds >64 chains/subject and the one trusted curated chain ranks below 8,
# so a top-8 truncation reports false negatives after cold reload. Raising the
# SEARCH budget changes no trust threshold: grade_chain + assertable_chain are
# byte-identical gate functions. Evidence: scratch/car_cold_inspect.py showed a
# curated, grade='assert' chain for tornado/limestone/callus/blush at limit=64
# that limit=8 missed. Documented in docs/LEARN5_CAR_MEASURED.md.
TRAVERSAL_LIMIT = 64

# Item bank. Mechanism taught for (x,a,b): a causes b, and b causes x.
# Single-word x/a/b so WebToGraph's OpenIE + node admission can create nodes.
# Subjects are unique. FIXED here before any engine boot; no item is removed
# because of an observed outcome.
_CANDIDATES = [
    ("fog", "chill", "condensation"), ("dew", "cooling", "condensation"),
    ("rust", "moisture", "oxidation"), ("echo", "reflection", "repetition"),
    ("rainbow", "refraction", "dispersion"), ("shadow", "blockage", "darkness"),
    ("hiccup", "spasm", "interruption"), ("blush", "arousal", "flushing"),
    ("sunburn", "radiation", "damage"), ("callus", "friction", "thickening"),
    ("bruise", "impact", "rupture"), ("sweat", "heat", "secretion"),
    ("sneeze", "irritation", "reflex"), ("shiver", "chill", "contraction"),
    ("yawn", "fatigue", "inhalation"), ("tear", "irritant", "secretion"),
    ("cough", "tickling", "expulsion"), ("frost", "chill", "deposition"),
    ("mist", "cooling", "condensation"), ("tarnish", "moisture", "oxidation"),
    ("reverb", "reflection", "repetition"), ("prism", "refraction", "dispersion"),
    ("glow", "excitation", "emission"), ("marble", "rock", "mineral"),
    ("velvet", "fabric", "texture"), ("amber", "resin", "fossil"),
    ("copper", "metal", "conductor"), ("denim", "cloth", "cotton"),
    ("ivory", "material", "tusk"), ("hail", "freezing", "ice"),
    ("sleet", "freezing", "slush"), ("snowfall", "cooling", "crystals"),
    ("thunder", "expansion", "shock"), ("lightning", "discharge", "flash"),
    ("tornado", "rotation", "vortex"), ("hurricane", "warming", "rotation"),
    ("drought", "dryness", "shortage"), ("flood", "rainfall", "overflow"),
    ("erosion", "weathering", "wear"), ("canyon", "erosion", "cutting"),
    ("stalactite", "dripping", "deposition"), ("stalagmite", "dripping", "accumulation"),
    ("geyser", "heating", "eruption"), ("volcano", "magma", "eruption"),
    ("earthquake", "shift", "tremor"), ("tsunami", "displacement", "wave"),
    ("tide", "gravity", "rising"), ("eclipse", "alignment", "shadow"),
    ("aurora", "radiation", "glow"), ("comet", "ice", "vapor"),
    ("meteor", "friction", "burning"), ("crater", "impact", "collision"),
    ("fossil", "burial", "mineralization"), ("coal", "burial", "pressure"),
    ("oil", "decomposition", "pressure"), ("diamond", "pressure", "crystal"),
    ("graphite", "pressure", "layers"), ("quartz", "cooling", "crystal"),
    ("granite", "cooling", "solidification"), ("basalt", "lava", "solidification"),
    ("sandstone", "cementation", "compaction"), ("shale", "compaction", "layers"),
    ("limestone", "deposition", "sediment"), ("coral", "secretion", "reef"),
    ("pearl", "irritation", "secretion"), ("shell", "secretion", "hardening"),
    ("horn", "keratin", "growth"), ("antler", "growth", "bone"),
    ("tusk", "ivory", "growth"), ("feather", "growth", "keratin"),
    ("scale", "overlap", "covering"), ("hide", "tanning", "leather"),
    ("silk", "spinning", "thread"), ("webbing", "spinning", "thread"),
    ("cocoon", "spinning", "silkworm"), ("hive", "wax", "comb"),
    ("honey", "nectar", "evaporation"), ("gum", "resin", "hardening"),
    ("nectar", "sugar", "secretion"), ("pollen", "transfer", "fertilization"),
    ("seed", "germination", "sprout"), ("root", "absorption", "water"),
    ("leaf", "photosynthesis", "sugar"), ("bark", "protecting", "covering"),
    ("fruit", "ripening", "sweetening"), ("flavor", "taste", "sensation"),
    ("aroma", "odor", "sensation"), ("color", "reflection", "wavelength"),
    ("mirror", "reflection", "image"), ("lens", "refraction", "focus"),
    ("magnify", "refraction", "enlargement"), ("resonance", "vibration", "amplification"),
    ("vibration", "oscillation", "tremor"), ("music", "rhythm", "sound"),
    ("voice", "vibration", "sound"), ("whisper", "breath", "sound"),
    ("snore", "vibration", "breath"), ("sigh", "breath", "exhale"),
    ("boil", "heating", "bubbles"), ("steam", "heating", "vapor"),
    ("condense", "cooling", "droplets"), ("freeze", "chill", "solidification"),
    ("melt", "warming", "liquefaction"), ("evaporate", "heating", "vaporization"),
]
# Definition-only control bank. Subjects MUST be disjoint from every teach
# subject: a shared subject string is the same graph node, so a definition-only
# item whose subject was also taught causally inherits the taught chain and the
# control is confounded (diagnosed in the invalid pilot run, 17/20 overlapped).
_DEF_CANDIDATES = [
    ("texture", "quality", "property"), ("shape", "quality", "form"),
    ("weight", "mass", "measure"), ("length", "distance", "measure"),
    ("width", "distance", "measure"), ("height", "distance", "measure"),
    ("depth", "distance", "measure"), ("speed", "rate", "motion"),
    ("temperature", "heat", "measure"), ("density", "mass", "volume"),
    ("hardness", "resistance", "quality"), ("roughness", "texture", "quality"),
    ("smoothness", "texture", "quality"), ("brightness", "light", "quality"),
    ("warmth", "heat", "quality"), ("coldness", "chill", "quality"),
    ("sweetness", "taste", "quality"), ("sourness", "taste", "quality"),
    ("bitterness", "taste", "quality"), ("saltiness", "taste", "quality"),
]


def _build_bank(candidates, role, n):
    seen, out = set(), []
    for x, a, b in candidates:
        x = x.strip().lower()
        if x in seen:
            continue
        seen.add(x)
        out.append({"item_id": "%s_%03d" % (role, len(out)), "role": role,
                    "x": x, "a": a.strip().lower(), "b": b.strip().lower()})
        if len(out) == n:
            break
    return out


def item_hash(item):
    key = "%s|%s|%s|%s|%s" % (HASH_PREFIX, COMPONENT, item["role"],
                               item["difficulty"], item["item_id"])
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def canonical_hash(records):
    blob = json.dumps(records, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def assign_difficulty(items, band_sizes):
    idx = 0
    for band, size in band_sizes:
        for it in items[idx:idx + size]:
            it["difficulty"] = band
        idx += size
    assert idx == len(items), (idx, len(items))
    return items


def assign_pools(items):
    groups = collections.defaultdict(list)
    for it in items:
        groups[(it["role"], it["difficulty"])].append(it)
    primary, sealed = [], []
    for key in sorted(groups):
        grp = sorted(groups[key], key=item_hash)
        cut = len(grp) // 2
        primary.extend(grp[:cut])
        sealed.extend(grp[cut:])
    return primary, sealed


def assign_k(pool):
    ordered = sorted(pool, key=item_hash)
    for i, it in enumerate(ordered):
        it["k"] = K_LEVELS[i % len(K_LEVELS)]
    return ordered


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def fisher_p(a, b, c, d):
    from scipy.stats import fisher_exact
    try:
        return float(fisher_exact([[a, b], [c, d]])[1])
    except Exception:
        return float("nan")


def trusted_chain_detected(nids, g):
    """Same trust gate as the prototype's trusted_chain, with a larger search
    budget so a trusted chain ranked below top-8 is not missed. Gate functions
    (grade_chain, assertable_chain) are unchanged."""
    from ravana.learn.explanation import enumerate_chains, grade_chain, assertable_chain
    if not nids:
        return None, None
    try:
        chains = enumerate_chains(nids, g, direction="backward",
                                  min_steps=2, max_steps=4, limit=TRAVERSAL_LIMIT)
    except Exception:
        return None, None
    for c in chains:
        try:
            gr = grade_chain(c)
            if gr.get("verdict") == "assert" and assertable_chain(c):
                return c, gr
        except Exception:
            continue
    return None, None


def gate_census(nids, g):
    from ravana.learn.explanation import enumerate_chains, grade_chain, assertable_chain
    out = {"raw_chains": 0, "graded_assert": 0, "trusted_all_hops": 0,
           "rejected_by_trust": 0, "hop_gates": [], "search_limit": TRAVERSAL_LIMIT}
    if not nids:
        return out
    try:
        chains = enumerate_chains(nids, g, direction="backward",
                                  min_steps=2, max_steps=4, limit=TRAVERSAL_LIMIT)
    except Exception:
        return out
    out["raw_chains"] = len(chains)
    for ch in chains:
        try:
            gr = grade_chain(ch)
            trusted = assertable_chain(ch)
            if gr.get("verdict") == "assert":
                out["graded_assert"] += 1
            if trusted:
                out["trusted_all_hops"] += 1
            else:
                out["rejected_by_trust"] += 1
            for h in ch.hops:
                meta = h.source_metadata if isinstance(h.source_metadata, dict) else {}
                out["hop_gates"].append({"from": h.from_label, "to": h.to_label,
                                         "edge_kind": str(meta.get("edge_kind") or "")})
        except Exception:
            continue
    return out


def teach_definition(eng, spec):
    x, a, b = spec["x"], spec["a"], spec["b"]
    stmt = "%s is a kind of %s and refers to a %s with a distinctive form." % (x, a, b)
    turn = mlc.ask(eng, stmt, mlc.TRACE)
    web = mlc.teach_web(eng, spec, definition=True)
    return {"assertion": stmt, "response": turn["answer"], "web": web}


def do_exposure(eng, spec):
    conv = mlc.teach_conversational(eng, spec)
    web = mlc.teach_web(eng, spec, definition=False)
    return {"conversation": conv, "web": web}


PARAPHRASES = (
    "why does {x} occur?",
    "how is {x} produced?",
    "what is the mechanism behind {x}?",
    "explain how {x} comes about.",
)
IDENTICAL = "why does {x} happen? is it because {a} causes {b}, and {b} causes {x}?"


def probe(eng, item, paraphrase_idx=0):
    q = PARAPHRASES[paraphrase_idx % len(PARAPHRASES)].format(x=item["x"])
    turn = mlc.ask(eng, q, mlc.TRACE)
    g = eng.graph
    nids = mlc.node_ids_for(item, g, mlc.label_map(g))
    chain, grade = trusted_chain_detected(nids, g)
    verdict = mlc.answer_verdict(turn)
    return {
        "question": q, "answer": turn["answer"], "secs": turn["secs"],
        "chain_present": chain is not None,
        "chain": mlc.trusted_chain_summary(chain, grade) if chain else None,
        "answer_verdict": verdict, "gate": gate_census(nids, g),
    }


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="halve k and item counts (iteration only)")
    args = ap.parse_args()

    t_start = time.time()
    commit = os.popen("git -C %s rev-parse --short HEAD" % PROJ).read().strip()

    # FIX THE BANK AND SPLIT BEFORE ANY ENGINE BOOT
    bank = assign_difficulty(_build_bank(_CANDIDATES, "subject", 100),
                             [("easy", 34), ("medium", 34), ("hard", 32)])
    defbank = assign_difficulty(_build_bank(_DEF_CANDIDATES, "defonly", 20),
                                [("easy", 10), ("hard", 10)])
    assert len({it["x"] for it in bank}) == 100
    assert len({it["x"] for it in defbank}) == 20

    primary, sealed = assign_pools(bank)
    def_primary, def_sealed = assign_pools(defbank)
    for pool in (primary, sealed):
        assign_k(pool)
    for pool in (def_primary, def_sealed):
        for it in pool:
            it["k"] = MAX_K

    def seal_record(it):
        return {"item_id": it["item_id"], "role": it["role"],
                "difficulty": it["difficulty"], "x": it["x"], "hash": item_hash(it)}
    manifest = {
        "hash_rule": "%s|<component>|<role>|<difficulty>|<item_id> (sha256, sorted bytes)" % HASH_PREFIX,
        "split_rule": "per (role,difficulty) group: sort by hash, first 50%% PRIMARY, rest SEALED",
        "k_assign": "sorted by hash, cycle (0,1,2,4,8); each item one k",
        "seed": "PYTHONHASHSEED=0", "commit": commit,
        "primary_n": len(primary), "sealed_n": len(sealed),
        "primary_manifest_hash": canonical_hash([seal_record(i) for i in primary]),
        "sealed_manifest_hash": canonical_hash([seal_record(i) for i in sealed]),
        "defonly_primary_n": len(def_primary), "defonly_sealed_n": len(def_sealed),
        "defonly_manifest_hash": canonical_hash([seal_record(i) for i in def_sealed]),
        "primary": [seal_record(i) for i in primary],
        "sealed": [seal_record(i) for i in sealed],
    }
    print("=== MANIFEST (frozen before boot) ===", flush=True)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("primary", "sealed")},
                     indent=2), flush=True)

    # scratch data_dir: isolated COPY; override the reused helper's global
    if os.path.exists(CAR_DATA_DIR):
        shutil.rmtree(CAR_DATA_DIR)
    mlc.DATA_DIR = CAR_DATA_DIR
    print("\n=== scratch data_dir copy ===", flush=True)
    print("  copied:", mlc.prepare_data_dir(), flush=True)

    from ravana.chat.engine import CognitiveChatEngine
    print("\n=== cold boot 1 (one process) ===", flush=True)
    t0 = time.time()
    eng = CognitiveChatEngine(data_dir=CAR_DATA_DIR, user_suffix="")
    mlc.TRACE = mlc.install_trace()
    eng._web_direct_answer = lambda ctx: None
    boot1_s = round(time.time() - t0, 2)
    print("  booted in %.2fs" % boot1_s, flush=True)

    det = mlc.run_detector_controls()
    det_ok = (det["positive_trusted_2hop_detected"] and
              det["negative_definition_rejected"] and
              det["direction_forward_ok"])
    print("\n=== INSTRUMENT SELF-CHECK ===", flush=True)
    print(json.dumps(det, indent=2), flush=True)
    print("  detector_ok =", det_ok, flush=True)
    if not det_ok:
        print("INSTRUMENT INVALID — refusing to report a curve", flush=True)
        return 2

    teach_items, def_items = [], []
    for name, p, dp in (("primary", primary, def_primary),
                        ("sealed", sealed, def_sealed)):
        teach_items += [(name, it) for it in p]
        def_items += [(name, it) for it in dp]
    if args.quick:
        teach_items = [(nm, it) for nm, it in teach_items if it["k"] in (0, 2, 8)]
        def_items = def_items[:6]

    all_items = teach_items + def_items
    records = {}
    for nm, it in all_items:
        records[it["item_id"]] = {
            "pool": nm, "role": it["role"], "difficulty": it["difficulty"],
            "k": it["k"], "subject": it["x"], "a": it["a"], "b": it["b"],
            "hash": item_hash(it), "stages": {},
        }

    print("\n=== T0 baseline probes (%d items) ===" % len(all_items), flush=True)
    for i, (nm, it) in enumerate(all_items):
        records[it["item_id"]]["stages"]["baseline"] = probe(eng, it, 0)
        if (i + 1) % 25 == 0:
            print("  baseline %d/%d" % (i + 1, len(all_items)), flush=True)

    print("\n=== spaced teaching (rounds 1..%d) ===" % MAX_K, flush=True)
    for r in range(1, MAX_K + 1):
        rnd = {"round": r, "exposures": 0, "neutral": 0}
        for nm, it in all_items:
            if it["role"] == "defonly":
                teach_definition(eng, it)             # matched 8 definition exposures
                rnd["exposures"] += 1
            elif it["k"] == 0:
                # familiarity control: same subject name, same turn count at
                # every round (8 neutral turns, matching the k=8 arm).
                mlc.ask(eng, "what do you know about %s?" % it["x"], mlc.TRACE)
                rnd["neutral"] += 1
            elif it["k"] < r:
                continue                              # this item is done teaching
            else:
                do_exposure(eng, it)
                rnd["exposures"] += 1
        mlc.ask(eng, "things happen for many reasons, don't they?", mlc.TRACE)
        print("  round %d: causal/def exposures=%d matched-neutral=%d" %
              (r, rnd["exposures"], rnd["neutral"]), flush=True)

    print("\n=== T1 immediate held-out paraphrase + identical-surface probes ===", flush=True)
    for i, (nm, it) in enumerate(all_items):
        rec = records[it["item_id"]]
        rec["stages"]["immediate"] = probe(eng, it, 1)
        qid = IDENTICAL.format(x=it["x"], a=it["a"], b=it["b"])
        t_id = mlc.ask(eng, qid, mlc.TRACE)
        rec["stages"]["identical"] = {
            "question": qid, "answer": t_id["answer"],
            "answer_verdict": mlc.answer_verdict(t_id),
            "mechanism_words_in_answer": (it["a"] in t_id["answer"].lower() and
                                          it["b"] in t_id["answer"].lower()),
        }
        if (i + 1) % 25 == 0:
            print("  immediate %d/%d" % (i + 1, len(all_items)), flush=True)

    print("\n=== save to isolated scratch data_dir (NEVER weights/) ===", flush=True)
    save_path = eng.save()
    print("  saved:", save_path, flush=True)
    del eng
    t0 = time.time()
    eng2 = CognitiveChatEngine(data_dir=CAR_DATA_DIR, user_suffix="")
    eng2._web_direct_answer = lambda ctx: None
    boot2_s = round(time.time() - t0, 2)
    print("  cold boot 2 in %.2fs" % boot2_s, flush=True)

    print("\n=== T2 delayed cold-reload held-out paraphrase probes ===", flush=True)
    for i, (nm, it) in enumerate(all_items):
        records[it["item_id"]]["stages"]["cold"] = probe(eng2, it, 2)
        if (i + 1) % 25 == 0:
            print("  cold %d/%d" % (i + 1, len(all_items)), flush=True)

    def rate(items, stage):
        vals = [records[it["item_id"]]["stages"][stage]["chain_present"] for nm, it in items]
        return sum(bool(v) for v in vals), len(vals)

    def arm_summary(items):
        out = {}
        for stage in ("baseline", "immediate", "cold"):
            k, n = rate(items, stage)
            out[stage + "_chain_rate"] = round(k / n, 3) if n else 0.0
            out[stage + "_chain_n"] = k
            out[stage + "_n"] = n
        base = out["baseline_chain_rate"]
        out["delta_immediate"] = round(out["immediate_chain_rate"] - base, 3)
        out["delta_cold"] = round(out["cold_chain_rate"] - base, 3)
        ans_reach = sum(1 for nm, it in items
                        if records[it["item_id"]]["stages"]["immediate"]
                        ["answer_verdict"]["has_chain"])
        out["immediate_answer_reach"] = round(ans_reach / len(items), 3) if items else 0
        echo = sum(1 for nm, it in items
                   if records[it["item_id"]]["stages"]["identical"]
                   ["mechanism_words_in_answer"])
        out["identical_surface_echo"] = round(echo / len(items), 3) if items else 0
        return out

    summary = {}
    for pool in ("primary", "sealed"):
        p_items = [(nm, it) for nm, it in teach_items if nm == pool]
        d_items = [(nm, it) for nm, it in def_items if nm == pool]
        summary[pool] = {"k_levels": {}}
        for kv in K_LEVELS:
            if args.quick and kv not in (0, 2, 8):
                continue
            lvl = [(nm, it) for nm, it in p_items if it["k"] == kv]
            summary[pool]["k_levels"][str(kv)] = arm_summary(lvl)
        summary[pool]["definition_only"] = arm_summary(d_items)

    criteria = []
    deltas = {}
    for pool in ("primary", "sealed"):
        d8 = summary[pool]["k_levels"].get("8", {}).get("delta_cold", 0.0)
        d0 = summary[pool]["k_levels"].get("0", {}).get("delta_cold", 0.0)
        dd = summary[pool]["definition_only"]["delta_cold"]
        deltas[pool] = {"d8": d8, "d0": d0, "ddef": dd}
        criteria.append({"criterion": "%s: delta_cold(k=8) >= 0.25" % pool,
                         "measured": d8, "pass": d8 >= 0.25})
        criteria.append({"criterion": "%s: k=0 delta <= 0.05" % pool,
                         "measured": d0, "pass": d0 <= 0.05})
        criteria.append({"criterion": "%s: definition-only delta <= 0.05" % pool,
                         "measured": dd, "pass": dd <= 0.05})
    for pool in ("primary", "sealed"):
        ks = [k for k in K_LEVELS if str(k) in summary[pool]["k_levels"]]
        curve = [summary[pool]["k_levels"][str(k)]["delta_cold"] for k in ks]
        drops = [round(curve[i + 1] - curve[i], 3) for i in range(len(curve) - 1)]
        max_drop = min(drops) if drops else 0.0
        criteria.append({"criterion": "%s: no adjacent-k reversal > 0.10" % pool,
                         "measured": max_drop, "pass": max_drop >= -0.10,
                         "adjacent_deltas": dict(zip(
                             ["%d->%d" % (ks[i], ks[i + 1]) for i in range(len(ks) - 1)],
                             drops))})
    for pool in ("primary", "sealed"):
        l8 = summary[pool]["k_levels"].get("8", {})
        gc, gi = l8.get("delta_cold", 0.0), l8.get("delta_immediate", 0.0)
        survives = gc > 0.0 and gc >= 0.5 * gi
        criteria.append({"criterion": "%s: gain survives cold reload" % pool,
                         "measured": {"cold": gc, "immediate": gi},
                         "pass": bool(survives)})

    p8 = summary["primary"]["k_levels"].get("8", {})
    p0 = summary["primary"]["k_levels"].get("0", {})
    a = p8.get("cold_chain_n", 0); n8 = p8.get("cold_n", 0); b = n8 - a
    c = p0.get("cold_chain_n", 0); n0 = p0.get("cold_n", 0); d = n0 - c
    stats = {
        "primary_k8_vs_k0": {
            "k8_cold": "%d/%d" % (a, n8), "k8_wilson95": wilson(a, n8),
            "k0_cold": "%d/%d" % (c, n0), "k0_wilson95": wilson(c, n0),
            "risk_difference": round(a / n8 - c / n0, 3) if n8 and n0 else None,
            "fisher_p": fisher_p(a, b, c, d),
        },
        "deltas": deltas,
    }

    all_pass = all(x["pass"] for x in criteria)
    result = {
        "component": "CAR", "commit": commit, "quick": bool(args.quick),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "boot1_s": boot1_s, "boot2_s": boot2_s,
        "elapsed_s": round(time.time() - t_start, 1),
        "data_dir": CAR_DATA_DIR, "weights_dir_untouched": True,
        "save_path": save_path, "detector_controls": det, "detector_ok": det_ok,
        "manifest": manifest, "summary": summary, "criteria": criteria,
        "statistics": stats, "pass": all_pass, "records": records,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)

    print("\n\n================ PER-ITEM INPUT/OUTPUT (primary then sealed) ================", flush=True)
    for pool in ("primary", "sealed"):
        print("\n################ POOL: %s ################" % pool, flush=True)
        for nm, it in [(nm, it) for nm, it in all_items if nm == pool]:
            rec = records[it["item_id"]]
            b, im, cd = (rec["stages"][s] for s in ("baseline", "immediate", "cold"))
            print("  [%s k=%d] %s (a=%s,b=%s)" %
                  (it["role"], it["k"], it["x"], it["a"], it["b"]), flush=True)
            print("    BASE Q: %r -> A: %r | chain=%s" %
                  (b["question"], b["answer"][:90], b["chain_present"]), flush=True)
            print("    POST Q: %r -> A: %r | chain=%s" %
                  (im["question"], im["answer"][:90], im["chain_present"]), flush=True)
            print("    COLD Q: %r -> A: %r | chain=%s" %
                  (cd["question"], cd["answer"][:90], cd["chain_present"]), flush=True)
            print("    IDENTICAL: %r -> echo=%s" %
                  (rec["stages"]["identical"]["answer"][:90],
                   rec["stages"]["identical"]["mechanism_words_in_answer"]), flush=True)
            print("    gate(immediate): raw=%d assert=%d trusted=%d rejected=%d" %
                  (im["gate"]["raw_chains"], im["gate"]["graded_assert"],
                   im["gate"]["trusted_all_hops"], im["gate"]["rejected_by_trust"]), flush=True)

    print("\n\n================ SUMMARY (primary | sealed) ================", flush=True)
    for pool in ("primary", "sealed"):
        print("\nPOOL %s" % pool, flush=True)
        for k in K_LEVELS:
            s = summary[pool]["k_levels"].get(str(k))
            if s:
                print("  k=%d  base=%.3f post=%.3f cold=%.3f  d_post=%+.3f d_cold=%+.3f" %
                      (k, s["baseline_chain_rate"], s["immediate_chain_rate"],
                       s["cold_chain_rate"], s["delta_immediate"], s["delta_cold"]), flush=True)
        s = summary[pool]["definition_only"]
        print("  DEF   base=%.3f post=%.3f cold=%.3f  d_cold=%+.3f" %
              (s["baseline_chain_rate"], s["immediate_chain_rate"],
               s["cold_chain_rate"], s["delta_cold"]), flush=True)
    print("\nSTATS:", json.dumps(stats, indent=2), flush=True)
    print("\nCRITERIA:", flush=True)
    for c in criteria:
        print("  [%s] %s : %s" % ("PASS" if c["pass"] else "FAIL",
                                   c["criterion"], c["measured"]), flush=True)
    print("\nOVERALL: %s" % ("PASS" if all_pass else "FAIL"), flush=True)
    print("OUTPUT -> %s" % OUT_PATH, flush=True)
    print("elapsed %.1fs | DONE" % (time.time() - t_start), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(run())
