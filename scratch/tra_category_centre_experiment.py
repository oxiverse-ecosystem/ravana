#!/usr/bin/env python3
"""SCRATCH-ONLY: category-centre transfer on the recorded TRA cases.

Tests ONE untried brain-plausible route for unseen-subject typed transfer:
a CATEGORY CENTRE built from several provenance-taught exemplar slots
(mean unit GloVe vector of the taught ``r2`` objects carrying a slot),
instead of a single label vector, a token signature, or a gold-key oracle.

Brain grounding (verified elsewhere, re-cited in the doc):
  Carey & Bartlett 1978; Dollaghan 1985 doi:10.1044/jshr.2803.454;
  Medina et al. 2011 doi:10.1073/pnas.1105040108  (fast mapping / sparse
  lexical type acquisition);
  Xu & Tenenbaum 2007 doi:10.1037/0033-295X.114.2.245  (Bayesian category
  induction); Tse 2007 doi:10.1126/science.1135935  (schema assimilation).

HARD CONSTRAINTS honoured by this script:
  * NO engine boot, NO process_turn, NO store write. Static + recorded only.
  * NO production edit: imports the UNMODIFIED production GloVe loader
    (ravana.ontology.attribute_encoder.build_glove64_lookup) from the main
    tree, read-only; asserts module resolution and source hashes.
  * NO type_sets / world_model / gold / query-subject type at scoring time.
    Only ``type_label`` and ``taught_exemplars`` (teaching material) and the
    stored confidences from scratch/tra_slot_results.json are inputs.
  * Emits scratch/ JSON + log only.

Thresholds are fixed BEFORE any score is read (designer choices mirroring
the earlier slot experiment, NOT TRA-pre-registered): floor 0.20, margin 0.10.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
MAIN = WORKTREE.parent / "ravana"

# Resolve the UNMODIFIED production tree first (no worktree ravana package).
for _p in (str(MAIN / "ravana" / "src"), str(MAIN / "ravana_ml" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.ontology.attribute_encoder import build_glove64_lookup  # noqa: E402

GLOVE_PATH = MAIN / "data" / "ravana_glove_cache.npz"
ROWS_PATH = MAIN / "scratch" / "tra_harness" / "tra_rows.jsonl"
SLOT_RESULTS_PATH = MAIN / "scratch" / "tra_slot_results.json"
ITEM_BANK_PATH = WORKTREE / "scripts" / "tra_items.py"
OUT_PATH = HERE / "tra_category_centre_results.json"

# ---- pre-registered constants (frozen before the run) --------------------
CONGRUENCE_FLOOR = 0.20
STRICT_MARGIN = 0.10
PERM_SEEDS = (11, 29)
DISJOINT_OPTIONS = ["apple", "banana", "chair", "table"]
UNRELATED_SUBJECT = "zeppelin"
RECORDED_IDS = [f"tra_{i:04d}" for i in range(40)]  # the 40 live-focused items
EPS = 1e-9

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > EPS else np.zeros_like(v)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- instrument preconditions ---------------------------------------------
def instrument() -> dict:
    loader_file = Path(build_glove64_lookup.__module__.replace(".", "/"))
    loaded_from = Path(sys.modules["ravana.ontology.attribute_encoder"].__file__).resolve()
    expected = (MAIN / "ravana" / "src" / "ravana" / "ontology" / "attribute_encoder.py").resolve()
    assert loaded_from == expected, f"production loader resolved to {loaded_from}, want {expected}"
    prod_sha = sha256(expected)
    wt_sha = sha256(WORKTREE / "ravana" / "src" / "ravana" / "ontology" / "attribute_encoder.py")
    assert prod_sha == wt_sha, "production loader differs between main and worktree"
    bank_sha_main = sha256(MAIN / "scripts" / "tra_items.py")
    bank_sha_wt = sha256(ITEM_BANK_PATH)
    assert bank_sha_main == bank_sha_wt, "item bank differs between main and worktree"

    spec = importlib.util.spec_from_file_location("tra_items_scratch", str(ITEM_BANK_PATH))
    ti = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(ti)
    bank = ti.build_items()
    errs = ti.validate_items(bank)
    assert not errs, f"item bank invalid: {errs[:3]}"
    by_id = {it["item_id"]: it for it in bank}

    # Load recorded rows; use ONLY backend=='live' (fake rows are fake).
    live_rows = []
    fake_rows = 0
    with open(ROWS_PATH, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("backend") == "live":
                live_rows.append(r)
            else:
                fake_rows += 1
    live_ids = sorted({r["id"] for r in live_rows})
    assert live_ids == RECORDED_IDS, f"unexpected recorded ids: {live_ids[:5]}... n={len(live_ids)}"

    # Cross-check every recorded live query against the regenerated bank.
    checked = 0
    for r in live_rows:
        it = by_id[r["id"]]
        inp = r["input"]
        cond = r["condition"]
        if cond == "near":
            q = it["unseen_test_words"]["near"]
            exp = (q["subject"], q["candidates"], q["gold"])
        elif cond == "far":
            q = it["unseen_test_words"]["far"]
            exp = (q["subject"], q["candidates"], q["gold"])
        elif cond == "foil":
            exp = (it["foil"]["subject"], it["foil"]["candidates"], None)
        else:
            exp = (it["negative"]["subject"], it["negative"]["candidates"], None)
        assert inp["query_subject"] == exp[0], (r["id"], cond, "subject")
        assert set(inp["candidates"]) == set(exp[1]), (r["id"], cond, "candidates")
        assert r["gold"]["target"] == exp[2], (r["id"], cond, "gold")
        checked += 1

    lut, dim = build_glove64_lookup(str(GLOVE_PATH))
    # Recorded live production baseline from the focused run (doc citation).
    baseline = {
        "production_live_focused_transfer": 0.000,
        "source": "docs/investigations/TRA_CONTROL_SCOPE_AUDIT.md E3; "
                  "focused live, 40 families, RAVANA_TRA_SLOTS=1",
        "vector_role_centre_recorded": {"near": "1/4", "far": "1/4", "foil": "0/4", "negative": "4/4",
                                        "source": "RSI_ROLE_BINDING_EXPERIMENT.md (n=1/family)"},
        "symbolic_signature_recorded": {"near": "0/4 V1-V2; 2/4 V3/V4 sym-floor (already-correct items)",
                                        "far": "0/4 all arms", "source": "RSI_SYMBOLIC_ROLE_EXPERIMENT.md"},
    }
    return {
        "ti": ti, "bank": bank, "by_id": by_id, "live_rows": live_rows,
        "fake_rows": fake_rows, "live_ids": live_ids, "checked": checked,
        "lut": lut, "dim": dim, "baseline": baseline,
        "hashes": {"production_loader_sha256": prod_sha, "worktree_loader_sha256": wt_sha,
                   "item_bank_sha256": bank_sha_wt,
                   "glove_cache_path": str(GLOVE_PATH), "glove_cache_sha256": sha256(GLOVE_PATH)},
    }


# ---- teaching material -> category centres --------------------------------
def build_centres(inst: dict) -> dict:
    """Mean unit vector of taught r2 objects carrying each type-label slot.

    For each recorded item, its first ``k`` taught exemplars contribute to
    the centre of that item's ``type_label`` (one-k-per-item, as the bank
    assigns). A k=0 item contributes nothing -> fail-closed by construction.
    No ``type_sets``, ``world_model``, gold or query-subject type is read.
    """
    lut = inst["lut"]
    by_id = inst["by_id"]
    labels = sorted(set(inst["ti"].ROLE_TO_TYPE.values()))
    contrib: dict[str, list[np.ndarray]] = collections.defaultdict(list)
    contrib_words: dict[str, list[str]] = collections.defaultdict(list)
    per_item: dict[str, dict[str, list[np.ndarray]]] = {}
    for iid in inst["live_ids"]:
        it = by_id[iid]
        k = it["k"]
        label = it["type_label"]  # teaching material (stamped on each demo)
        own = []
        for ex in it["taught_exemplars"][:k]:
            v = lut.get(ex["r2"].strip().lower())
            if v is None:
                continue
            uv = unit(v)
            own.append(uv)
            contrib[label].append(uv)
            contrib_words[label].append(ex["r2"])
        per_item[iid] = {label: own}
    pooled = {L: (np.mean(vs, axis=0) if vs else None)
              for L, vs in contrib.items()}
    # confidence-weighted variant, stored confidences where available
    with open(SLOT_RESULTS_PATH, encoding="utf-8") as fh:
        sr = json.load(fh)
    conf_map: dict[str, float] = {}
    for e in sr["gates"]["C12_provenance_integrity"]["new_typed_edges"]:
        if e.get("trusted") and e.get("confidence") is not None:
            conf_map[str(e["source"]).strip().lower()] = float(e["confidence"])
    weighted: dict[str, object] = {}
    conf_only: dict[str, object] = {}
    covered = 0
    for L in labels:
        wv, wsum, conf_vals = [], 0.0, []
        for w in contrib_words.get(L, []):
            c = conf_map.get(w.strip().lower())
            if c is not None:
                covered += 1
                conf_vals.append(c)
            wv.append(lut[w.strip().lower()] if w.strip().lower() in lut else np.zeros(64))
            wsum += c if c is not None else 1.0
        if wv and wsum > EPS:
            vec = np.zeros(64, dtype=np.float64)
            for w, wt in zip(wv, [conf_map.get(w.strip().lower(), 1.0) for w in contrib_words[L]]):
                vec += unit(w) * wt
            weighted[L] = unit(vec / wsum)
        else:
            weighted[L] = None
        conf_only[L] = (float(np.mean(conf_vals)) if conf_vals else None)
    return {
        "labels": labels, "pooled": pooled, "per_item": per_item,
        "contrib_counts": {L: len(contrib[L]) for L in labels},
        "contrib_words": dict(contrib_words), "weighted": weighted, "conf_only": conf_only,
        "confidence_map": conf_map, "confidence_exemplar_coverage": covered,
        "confidence_note": "Recorded typed edges are for 'weather phenomenon'/'civic event' "
                           "(6 words), NOT for any TRA type label; coverage on TRA slots is "
                           "reported, not assumed.",
    }


# ---- fail-closed scorer (no gold / type_sets / world_model in scope) ------
def choose(subject_vec, options, centres, rng_seed, label_vecs=None,
           tiebreak=False, floor=CONGRUENCE_FLOOR, margin=STRICT_MARGIN,
           permutation=True):
    """Pick an option by cosine(subject, slot centre), fail-closed.

    Inputs are ONLY a subject vector, an option list, centre vectors, and
    (optionally) label vectors for a tie-break. No answer key is in scope.
    """
    if subject_vec is None:
        return {"pick": None, "reason": "subject_vector_missing", "scores": {},
                "eligible": [], "permutation_picks": []}
    eligible = [o for o in options if o in centres and centres[o] is not None]
    disjoint = [o for o in options if o not in centres or centres[o] is None]
    if not eligible:
        return {"pick": None, "reason": "disjoint_or_unknown_option_set",
                "scores": {}, "eligible": [], "disjoint_options": disjoint,
                "permutation_picks": []}
    scores = {o: cosine(subject_vec, centres[o]) for o in eligible}

    def _one(opts):
        order = sorted(opts, key=lambda o: (-scores[o], o))
        best, second = order[0], (order[1] if len(order) > 1 else None)
        if scores[best] < floor:
            return None, "no_category_above_floor"
        if second is not None and (scores[best] - scores[second]) < margin:
            if tiebreak and label_vecs and best in label_vecs and second in label_vecs \
                    and label_vecs[best] is not None and label_vecs[second] is not None:
                tb = {o: cosine(centres[o], label_vecs[o]) for o in (best, second)}
                win = sorted(tb, key=lambda o: (-tb[o], o))[0]
                if abs(tb[best] - tb[second]) > EPS:
                    return win, "label_vector_tiebreak"
            return None, "strict_margin_not_met"
        return best, "trusted_centre_strict_winner"

    pick, reason = _one(eligible)
    perm_picks = [pick]
    if permutation and pick is not None:
        for sd in PERM_SEEDS:
            shuffled = list(eligible)
            # deterministic shuffle by stable hash, no RNG state dependency
            shuffled.sort(key=lambda o: hashlib.sha256(f"{sd}|{o}".encode()).hexdigest())
            p, _ = _one(shuffled)
            perm_picks.append(p)
        if any(p != pick for p in perm_picks[1:]):
            return {"pick": None, "reason": "permutation_unstable", "scores": scores,
                    "eligible": eligible, "permutation_picks": perm_picks,
                    "disjoint_options": disjoint}
    return {"pick": pick, "reason": reason, "scores": scores, "eligible": eligible,
            "permutation_picks": perm_picks, "disjoint_options": disjoint}


def choose_conf(options, scalar_scores, floor=CONGRUENCE_FLOOR,
                margin=STRICT_MARGIN):
    """Confidence-only arm: stored edge confidence IS the score (no vectors).

    Fixed after the run-1 instrument defect (scalars fed to cosine); the
    run-1 log is retained as retracted. Thresholds are unchanged.
    """
    eligible = [o for o in options if scalar_scores.get(o) is not None]
    if not eligible:
        return {"pick": None, "reason": "no_recorded_confidence_for_any_slot",
                "scores": {}, "eligible": [], "permutation_picks": [],
                "disjoint_options": [o for o in options if o not in eligible]}
    scores = {o: float(scalar_scores[o]) for o in eligible}
    order = sorted(eligible, key=lambda o: (-scores[o], o))
    best, second = order[0], (order[1] if len(order) > 1 else None)
    if scores[best] < floor:
        return {"pick": None, "reason": "no_category_above_floor", "scores": scores,
                "eligible": eligible, "permutation_picks": [None], "disjoint_options": []}
    if second is not None and (scores[best] - scores[second]) < margin:
        return {"pick": None, "reason": "strict_margin_not_met", "scores": scores,
                "eligible": eligible, "permutation_picks": [None], "disjoint_options": []}
    return {"pick": best, "reason": "recorded_confidence_strict_winner",
            "scores": scores, "eligible": eligible,
            "permutation_picks": [best], "disjoint_options": []}


def choose_domain(subject_vec, candidates, centre, taught_r2, rng_seed=0,
                  floor=CONGRUENCE_FLOOR, margin=STRICT_MARGIN):
    """Membership surface: cosine(candidate, slot centre), teaching-material excluded."""
    if centre is None or subject_vec is None:
        return {"pick": None, "reason": "no_centre_or_subject", "scores": {}}
    eligible = [c for c in candidates
                if c.strip().lower() not in taught_r2 and c.strip().lower() in GLOBAL_LUT]
    if not eligible:
        return {"pick": None, "reason": "no_unseen_candidate", "scores": {}}
    scores = {c: cosine(GLOBAL_LUT[c.strip().lower()], centre) for c in eligible}
    order = sorted(eligible, key=lambda c: (-scores[c], c))
    best, second = order[0], (order[1] if len(order) > 1 else None)
    if scores[best] < floor:
        # fail-closed: a below-floor candidate must NOT be emitted.
        return {"pick": None, "reason": "no_candidate_above_floor", "scores": scores}
    if second is not None and (scores[best] - scores[second]) < margin:
        return {"pick": None, "reason": "strict_margin_not_met", "scores": scores}
    return {"pick": best, "reason": "trusted_centre_strict_winner", "scores": scores}


# ---- case runner -----------------------------------------------------------
GLOBAL_LUT = {}


def run(inst: dict, cen: dict) -> dict:
    global GLOBAL_LUT
    GLOBAL_LUT = inst["lut"]
    lut = inst["lut"]
    labels = cen["labels"]
    # label vectors: 5 of 6 labels are OOV in GloVe; report, do not fabricate.
    label_vecs = {L: lut.get(L) for L in labels}
    label_vec_coverage = {L: (label_vecs[L] is not None) for L in labels}

    arm_names = ["C_pooled_cos", "C_local_cos", "C_label_tiebreak",
                 "C_conf_weighted", "C_conf_only", "C_domain_membership"]
    rows = []

    for iid in inst["live_ids"]:
        it = inst["by_id"][iid]
        k = it["k"]
        own_label = it["type_label"]
        own_c = cen["per_item"][iid][own_label]
        local_centres = dict(cen["pooled"])
        # Own item's slot centre from its OWN k taught exemplars (k=0 -> absent).
        local_centres[own_label] = (np.mean(own_c, axis=0) if own_c else None)

        for cond in ("near", "far", "foil", "negative"):
            if cond == "near":
                q = it["unseen_test_words"]["near"]; subject = q["subject"]
                dom_cands, dom_gold = q["candidates"], q["gold"]
                expected_label, expected_dom = own_label, dom_gold
            elif cond == "far":
                q = it["unseen_test_words"]["far"]; subject = q["subject"]
                dom_cands, dom_gold = q["candidates"], q["gold"]
                expected_label, expected_dom = own_label, dom_gold
            elif cond == "foil":
                subject = it["foil"]["subject"]
                dom_cands = it["foil"]["candidates"]; dom_gold = None
                expected_label, expected_dom = None, None
            else:
                subject = it["negative"]["subject"]
                dom_cands = it["negative"]["candidates"]; dom_gold = None
                expected_label, expected_dom = None, None
            sv = lut.get(subject.strip().lower())

            # primary literal surface: score the 6 type-label options
            pooled = choose(sv, labels, cen["pooled"], 11)
            local = choose(sv, labels, local_centres, 11)
            tbreak = choose(sv, labels, cen["pooled"], 11,
                            label_vecs=label_vecs, tiebreak=True)
            confw = choose(sv, labels, cen["weighted"], 11)
            confo = choose_conf(labels, cen["conf_only"])
            # secondary comparability surface: domain-word options
            taught_r2 = {ex["r2"].strip().lower() for ex in it["taught_exemplars"][:k]}
            dom = choose_domain(sv, dom_cands, cen["pooled"].get(own_label), taught_r2)

            arm_picks = {"C_pooled_cos": pooled["pick"], "C_local_cos": local["pick"],
                         "C_label_tiebreak": tbreak["pick"], "C_conf_weighted": confw["pick"],
                         "C_conf_only": confo["pick"], "C_domain_membership": dom["pick"]}
            # grade AFTER scoring only
            graded = {}
            for a in arm_names[:5]:
                exp = expected_label
                graded[a] = (arm_picks[a] is None) if exp is None else (arm_picks[a] == exp)
            graded["C_domain_membership"] = (
                (arm_picks["C_domain_membership"] is None) if expected_dom is None
                else (arm_picks["C_domain_membership"] == expected_dom))

            rows.append({
                "id": iid, "pool": it["pool"], "role": it["role"], "k": k,
                "type_label": own_label, "condition": cond, "subject": subject,
                "option_space": {"label_options": labels,
                                 "domain_options": dom_cands,
                                 "label_vector_coverage": label_vec_coverage},
                "expected": {"label_pick": expected_label, "domain_pick": expected_dom,
                             "source": "item bank type_label / gold; used ONLY to grade after scoring"},
                "arms": {a: {"pick": arm_picks[a],
                             "reason": (pooled if a == "C_pooled_cos" else
                                        local if a == "C_local_cos" else
                                        tbreak if a == "C_label_tiebreak" else
                                        confw if a == "C_conf_weighted" else
                                        confo if a == "C_conf_only" else dom)["reason"],
                             "scores": (pooled if a == "C_pooled_cos" else
                                        local if a == "C_local_cos" else
                                        tbreak if a == "C_label_tiebreak" else
                                        confw if a == "C_conf_weighted" else
                                        confo if a == "C_conf_only" else dom)["scores"],
                             "permutation_picks": (pooled if a == "C_pooled_cos" else
                                                   local if a == "C_local_cos" else
                                                   tbreak if a == "C_label_tiebreak" else
                                                   confw if a == "C_conf_weighted" else
                                                   confo)["permutation_picks"]
                             if a != "C_domain_membership" else [],
                             "correct": graded[a]} for a in arm_names},
            })
    return rows


def controls(inst: dict, cen: dict) -> dict:
    lut = inst["lut"]
    labels = cen["labels"]
    disjoint = choose(lut.get(UNRELATED_SUBJECT), DISJOINT_OPTIONS, cen["pooled"], 11)
    unrelated = choose(lut.get(UNRELATED_SUBJECT), labels, cen["pooled"], 11)
    # instrument sanity: set subject vector exactly to a known slot centre
    alive_label = next((L for L in labels if cen["pooled"].get(L) is not None), None)
    sanity = (choose(cen["pooled"][alive_label], labels, cen["pooled"], 11)
              if alive_label else {"pick": None, "reason": "no_centre"})
    # k=0 control: those items have no own exemplars
    k0 = [i for i in inst["live_ids"] if inst["by_id"][i]["k"] == 0]

    # Diagnose the exact-centre sanity failure: top-1 identity vs margin.
    sep = {}
    wiring = {}
    for L in labels:
        cv = cen["pooled"].get(L)
        if cv is None:
            continue
        scored = sorted(((M, cosine(cv, cen["pooled"][M]))
                         for M in labels if cen["pooled"].get(M) is not None),
                        key=lambda x: (-x[1], x[0]))
        top1, top2 = scored[0], (scored[1] if len(scored) > 1 else None)
        sep[L] = {
            "top1_label": top1[0], "top1_cosine": round(top1[1], 4),
            "top2_label": (top2[0] if top2 else None),
            "top2_cosine": (round(top2[1], 4) if top2 else None),
            "margin": (round(top1[1] - top2[1], 4) if top2 else None),
            "margin_pass": (bool(top2 and top1[1] - top2[1] >= STRICT_MARGIN)),
            "top1_identity_correct": (top1[0] == L),
        }
        # INSTRUMENT-ONLY wiring check: margin disabled solely for this probe.
        # This is NOT a capability arm and NOT a threshold change to any arm.
        w = choose(cv, labels, cen["pooled"], 11, margin=0.0)
        wiring[L] = {"pick": w["pick"], "pass": w["pick"] == L}

    return {
        "C10_disjoint_options": {"options": DISJOINT_OPTIONS, "pick": disjoint["pick"],
                                 "reason": disjoint["reason"],
                                 "pass": disjoint["pick"] is None},
        "unrelated_subject": {"subject": UNRELATED_SUBJECT, "pick": unrelated["pick"],
                              "reason": unrelated["reason"], "scores": unrelated["scores"],
                              "pass": unrelated["pick"] is None},
        "instrument_sanity_centre_as_subject": {"label": alive_label, "pick": sanity["pick"],
                                                "reason": sanity["reason"],
                                                "pass": sanity["pick"] == alive_label,
                                                "note": "FAILS at the pre-registered 0.10 margin; "
                                                        "see centre_separation for the geometry"},
        "wiring_sanity_centre_as_subject_no_margin": {
            "detail": wiring,
            "n_correct": sum(1 for v in wiring.values() if v["pass"]),
            "n": len(wiring),
            "pass": all(v["pass"] for v in wiring.values()),
            "note": "instrument-only: margin=0.0 ONLY in this probe; proves the scorer "
                    "is connected, never used as a capability result"},
        "centre_separation": sep,
        "k0_items": {"ids": k0, "n": len(k0),
                     "note": "these items contribute no taught exemplars to their own slot centre"},
    }


def summarize(rows: list[dict], arm_names: list[str]) -> dict:
    out = {}
    for a in arm_names:
        per = {}
        for cond in ("near", "far", "foil", "negative"):
            rr = [r for r in rows if r["condition"] == cond]
            per[cond] = {
                "n": len(rr),
                "correct": sum(1 for r in rr if r["arms"][a]["correct"]),
                "picks_none": sum(1 for r in rr if r["arms"][a]["pick"] is None),
            }
        pool = {}
        for p in ("primary", "sealed"):
            rr = [r for r in rows if r["pool"] == p and r["condition"] in ("near", "far")]
            pool[p] = {"n": len(rr),
                       "correct": sum(1 for r in rr if r["arms"][a]["correct"])}
        out[a] = {"per_condition": per, "near_far_by_pool": pool,
                  "reasons": dict(collections.Counter(
                      r["arms"][a]["reason"] for r in rows))}
    return out


def main() -> int:
    t0 = time.time()
    log("TRA CATEGORY-CENTRE EXPERIMENT — scratch, static, no engine boot")
    log("worktree=" + str(WORKTREE) + "  branch=experiment/tra-category-centre")
    log("constraints: no production edit, no boot, no store write; teaching material only")
    try:
        inst = instrument()
        log(f"instrument OK: loader={sys.modules['ravana.ontology.attribute_encoder'].__file__}")
        log(f"loader_sha256={inst['hashes']['production_loader_sha256'][:16]} "
            f"bank_sha256={inst['hashes']['item_bank_sha256'][:16]}")
        log(f"recorded live rows={len(inst['live_rows'])} fake rows excluded={inst['fake_rows']} "
            f"items cross-checked={inst['checked']} dim={inst['dim']} lut={len(inst['lut'])}")
        cen = build_centres(inst)
        log("slot centre sizes (taught r2 contributions): " + str(cen["contrib_counts"]))
        log("label vector coverage: " + str({L: (inst['lut'].get(L) is not None)
                                              for L in cen['labels']}))
        log("recorded-confidence exemplar coverage on TRA slots: "
            f"{cen['confidence_exemplar_coverage']} "
            f"({cen['confidence_note']})")
        rows = run(inst, cen)
        ctr = controls(inst, cen)
        log("controls: " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk in
                                           ("pick", "pass", "reason", "n", "n_correct")}
                                       for k, v in ctr.items()}, ensure_ascii=False))
        log("centre separation (top1 always the exact label at 1.0; margins 0.020-0.087 < 0.10):")
        for L, d in ctr["centre_separation"].items():
            log(f"  {L:20s} top1={d['top1_label']:20s} ({d['top1_cosine']:.4f})  "
                f"top2={str(d['top2_label']):20s} ({d['top2_cosine']:.4f})  "
                f"margin={d['margin']:.4f} margin_pass={d['margin_pass']}")
        log(f"wiring sanity (margin=0, instrument-only): "
            f"{ctr['wiring_sanity_centre_as_subject_no_margin']['n_correct']}/"
            f"{ctr['wiring_sanity_centre_as_subject_no_margin']['n']} correct")
        arm_names = ["C_pooled_cos", "C_local_cos", "C_label_tiebreak",
                     "C_conf_weighted", "C_conf_only", "C_domain_membership"]
        summ = summarize(rows, arm_names)
        log("\n=== PER-ARM CORRECT COUNTS (n=40 items x 4 conditions) ===")
        for a in arm_names:
            per = summ[a]["per_condition"]
            log(f"{a:22s} near {per['near']['correct']}/{per['near']['n']}  "
                f"far {per['far']['correct']}/{per['far']['n']}  "
                f"foil {per['foil']['correct']}/{per['foil']['n']}  "
                f"negative {per['negative']['correct']}/{per['negative']['n']}")
        log("\n=== near+far by pool ===")
        for a in arm_names:
            p = summ[a]["near_far_by_pool"]
            log(f"{a:22s} primary {p['primary']['correct']}/{p['primary']['n']}  "
                f"sealed {p['sealed']['correct']}/{p['sealed']['n']}")
        # G6 exactly as shipped (scripts/front_gates.py, unmodified, read-only).
        spec_g6 = importlib.util.spec_from_file_location(
            "front_gates_scratch", str(MAIN / "scripts" / "front_gates.py"))
        fg = importlib.util.module_from_spec(spec_g6)
        assert spec_g6 and spec_g6.loader
        spec_g6.loader.exec_module(fg)
        g6 = {}
        for a in arm_names:
            per = summ[a]["per_condition"]
            near_pct = 100.0 * per["near"]["correct"] / per["near"]["n"]
            neg_pct = 100.0 * per["negative"]["correct"] / per["negative"]["n"]
            res = fg.gate_saturation({"tra_near": near_pct,
                                      "tra_negative_reject": neg_pct})
            g6[a] = {"tra_near": round(near_pct, 3),
                     "tra_negative_reject": round(neg_pct, 3),
                     "status": res.status, "detail": res.detail,
                     "flagged_fronts": res.flagged_fronts, "blocking": res.blocking}
        log("G6 saturation (shipped gate, non-blocking FLAG):")
        for a in arm_names:
            log(f"  {a:22s} {g6[a]['status']}: {g6[a]['detail']}")
        # Audit every correct near/far pick: was the item's OWN teaching used?
        hit_audit = {}
        for a in arm_names:
            hits = []
            for row in rows:
                if row["condition"] not in ("near", "far") or not row["arms"][a]["correct"]:
                    continue
                hits.append({"id": row["id"], "k": row["k"], "condition": row["condition"],
                             "subject": row["subject"], "picked": row["arms"][a]["pick"],
                             "own_item_contributed_exemplars": row["k"] > 0})
            hit_audit[a] = {"n": len(hits), "with_k0_no_own_teaching":
                            sum(1 for h in hits if not h["own_item_contributed_exemplars"]),
                            "hits": hits}
        log("near/far hit audit (k=0 hits are cross-item artefacts, not this item's teaching):")
        for a in arm_names:
            log(f"  {a:22s} hits={hit_audit[a]['n']} k0_hits={hit_audit[a]['with_k0_no_own_teaching']}")
        result = {
            "experiment": "TRA category-centre transfer on recorded cases (scratch, static)",
            "branch": "experiment/tra-category-centre", "base_commit": "bfe99a6",
            "production_code_modified": False, "engine_booted": False,
            "store_written": False, "gold_or_type_sets_at_scoring_time": False,
            "inputs": {
                "item_bank": "scripts/tra_items.py (build_items), teaching material: "
                             "type_label + taught_exemplars only",
                "recorded_rows": str(ROWS_PATH),
                "recorded_rows_used": "backend=='live' only; fake rows excluded",
                "recorded_live_rows": len(inst["live_rows"]), "fake_rows_excluded": inst["fake_rows"],
                "items_covered": len(inst["live_ids"]),
                "item_ids": inst["live_ids"],
                "pool_split": dict(collections.Counter(inst["by_id"][i]["pool"] for i in inst["live_ids"])),
                "k_distribution": dict(collections.Counter(inst["by_id"][i]["k"] for i in inst["live_ids"])),
                "rows_cross_checked_against_bank": inst["checked"],
                "stored_confidences": str(SLOT_RESULTS_PATH),
                "hashes": inst["hashes"],
            },
            "method": {
                "centre": "mean of unit GloVe vectors of the first k taught r2 objects "
                          "per slot (one-k-per-item), pooled by type_label",
                "primary_score": "cosine(unit(vec(query_subject)), unit(slot_centre))",
                "tiebreak_score": "cosine(unit(slot_centre), unit(vec(type_label))) "
                                  "only when top-2 margin < 0.10; 5/6 labels OOV",
                "confidence_variant": "unit r2 vectors weighted by stored confidence "
                                      "where a recorded typed edge exists; "
                                      "separate confidence-only arm has TRA-label coverage 0",
                "domain_arm": "cosine(unit(vec(candidate)), slot_centre) with taught "
                              "r2 excluded; foil/negative correct == None",
                "floor": CONGRUENCE_FLOOR, "strict_margin": STRICT_MARGIN,
                "permutation": "2 deterministic hash shuffles; any changed pick -> None",
                "no_oracle": "no type_sets, world_model, gold, or query-subject type "
                             "in any scorer input; gold used only to grade after scoring",
                "instrument_caveat": "exact-centre sanity FAILS the 0.10 margin (6/6 top-1 "
                                     "correct at score 1.0 but margins 0.020-0.087): the "
                                     "centres are not separable at the frozen gate. A "
                                     "margin=0 wiring sanity (instrument-only) passes 6/6. "
                                     "Near/far nulls are fail-closed non-separability, "
                                     "not a demonstrated capability failure of a working gate.",
            },
            "controls": ctr,
            "g6_saturation": g6,
            "hit_audit": hit_audit,
            "arm_summary": summ,
            "production_baseline_references": inst["baseline"],
            "rows": rows,
            "elapsed_s": round(time.time() - t0, 2),
            "errors": [],
        }
        OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\nwrote {OUT_PATH}")
        log("EXIT=0")
        return 0
    except Exception as exc:  # surface swallowed exceptions (skill §6)
        import traceback
        tb = traceback.format_exc()
        log("ERROR: " + repr(exc))
        log(tb)
        log("EXIT=2")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
