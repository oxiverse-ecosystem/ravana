#!/usr/bin/env python3
"""SCRATCH-ONLY: is there any deployable teaching-side signal for unseen subjects?

Question: on the 40 recorded live TRA items, can ANY signal actually present in
the teaching turns classify an unseen subject (near/far) while still rejecting
foil and negative cases?  No engine boot, no store write, no production edit,
no answer key at scoring time.

Arms
  A0  production baseline (0.000 focused live) -- recomputed from recorded
      flags where possible, cited otherwise. NOT a teaching-side arm.
  A1  GloVe ITEM-OWN category centre: mean unit vector of this item's first-k
      taught r2 words; score candidates by cos(candidate, centre).  Prior
      category-centre work already failed; reproduced here as a control.
  A2  TEACHING-CORPUS CO-OCCURRENCE (deployable): per-word context vectors built
      ONLY from words co-occurring in the actual teaching surfaces (regenerated
      with the item bank's own teaching_surface()).  Rank options by cosine
      between the QUERY subject's teaching-corpus context and each option's.
      Pooled corpus is primary; item-local scope is a reported diagnostic.
  A3  Morphological / character n-gram fast-mapping cue (deployable): choose the
      taught r2 word most similar to the QUERY subject by char-3gram Jaccard;
      rank options by their n-gram Jaccard to that exemplar.  Subject admission
      uses the same floor as option ranking.  Two diagnostics are reported.
  A4  ORACLE upper bound (NON-DEPLOYABLE contamination ceiling): item-bank
      world_model / type_sets answer key.  Reported to size the gap, never
      eligible for implementation.

Fail-closed ranker (frozen before the run): congruence floor 0.20, strict
margin 0.10, two deterministic permutation shuffles (any changed pick ->
None), option absent from the scorer's known universe -> None.  Gold is read
ONLY to grade a pick after all scoring is complete.  A4 is the sole exception
and is explicitly marked as reading the key.

Brain grounding (inherited and re-cited in the companion doc): fast mapping /
sparse lexical type acquisition (Carey & Bartlett 1978; Dollaghan 1985,
doi:10.1044/jshr.2803.454; Medina et al. 2011, doi:10.1073/pnas.1105040108);
category induction from scattered exemplars (Xu & Tenenbaum 2007,
doi:10.1037/0033-295X.114.2.245); schema assimilation (Tse 2007,
doi:10.1126/science.1135935).
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
MAIN = WORKTREE.parent / "ravana"

# Import the UNMODIFIED production tree read-only; no worktree ravana package.
for _p in (str(MAIN / "ravana" / "src"), str(MAIN / "ravana_ml" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.ontology.attribute_encoder import build_glove64_lookup  # noqa: E402

GLOVE_PATH = MAIN / "data" / "ravana_glove_cache.npz"
ROWS_PATH = MAIN / "scratch" / "tra_harness" / "tra_rows.jsonl"
ITEM_BANK_PATH = WORKTREE / "scripts" / "tra_items.py"
OUT_PATH = HERE / "type_acquisition_results.json"

# ---- frozen constants (chosen before any score was read) ------------------
CONGRUENCE_FLOOR = 0.20
STRICT_MARGIN = 0.10
PERM_SEEDS = (11, 29)
DISJOINT_OPTIONS = ["apple", "banana", "chair", "table"]
UNRELATED_SUBJECT = "zeppelin"
RECORDED_IDS = [f"tra_{i:04d}" for i in range(40)]
CONDITIONS = ("near", "far", "foil", "negative")
NGRAM_N = 3
EPS = 1e-9

# English function words stripped from teaching surfaces for co-occurrence.
STOP = set((
    "a an the of in on at to by for is are was were be been being it its "
    "this that these those with and or from as into than then there their "
    "they them he she we you your our i am do does did have has had will "
    "would can could should may might must not no nor so if but while about "
    "over under again further once here when where why how all any both each "
    "few more most other some such only own same too very s t just don now"
).split())

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
    loaded_from = Path(sys.modules["ravana.ontology.attribute_encoder"].__file__).resolve()
    expected = (MAIN / "ravana" / "src" / "ravana" / "ontology" / "attribute_encoder.py").resolve()
    assert loaded_from == expected, f"GloVe loader resolved to {loaded_from}, want {expected}"
    prod_sha = sha256(expected)
    wt_sha = sha256(WORKTREE / "ravana" / "src" / "ravana" / "ontology" / "attribute_encoder.py")
    assert prod_sha == wt_sha, "production loader differs between main and worktree"
    bank_sha_main = sha256(MAIN / "scripts" / "tra_items.py")
    bank_sha_wt = sha256(ITEM_BANK_PATH)
    assert bank_sha_main == bank_sha_wt, "item bank differs between main and worktree"
    assert "CognitiveChatEngine" not in sys.modules, "an engine module was imported: no boot allowed"

    spec = importlib.util.spec_from_file_location("tra_items_scratch", str(ITEM_BANK_PATH))
    ti = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(ti)
    bank = ti.build_items()
    errs = ti.validate_items(bank)
    assert not errs, f"item bank invalid: {errs[:3]}"
    by_id = {it["item_id"]: it for it in bank}

    # Recorded rows: use ONLY backend=='live'; fake rows are a different oracle.
    live_rows, fake_rows = [], 0
    with open(ROWS_PATH, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("backend") == "live":
                live_rows.append(r)
            else:
                fake_rows += 1
    live_ids = sorted({r["id"] for r in live_rows})
    assert live_ids == RECORDED_IDS, f"unexpected recorded ids, n={len(live_ids)}"

    # One graded row per (item, condition): structural, query, non-permuted.
    sel = [r for r in live_rows
           if r.get("arm") == "structural" and r.get("phase") == "query"
           and not r.get("permuted")]
    assert len(sel) == 160, f"expected 160 graded rows, got {len(sel)}"
    assert collections.Counter(r["condition"] for r in sel) == {
        c: 40 for c in CONDITIONS}
    assert len({r["id"] for r in sel}) == 40

    # Cross-check recorded inputs and gold against the regenerated bank.
    checked = 0
    for r in sel:
        it = by_id[r["id"]]
        cond = r["condition"]
        if cond in ("near", "far"):
            q = it["unseen_test_words"][cond]
            exp = (q["subject"], q["candidates"], q["gold"])
        elif cond == "foil":
            exp = (it["foil"]["subject"], it["foil"]["candidates"], None)
        else:
            exp = (it["negative"]["subject"], it["negative"]["candidates"], None)
        assert r["input"]["query_subject"] == exp[0], (r["id"], cond, "subject")
        assert set(r["input"]["candidates"]) == set(exp[1]), (r["id"], cond, "candidates")
        assert r["gold"]["target"] == exp[2], (r["id"], cond, "gold")
        checked += 1

    # Recorded production endpoints (no engine boot to recompute capability).
    prod_near = sum(1 for r in sel if r["condition"] == "near" and r["flags"]["correct"])
    prod_far = sum(1 for r in sel if r["condition"] == "far" and r["flags"]["correct"])
    prod_foil = sum(1 for r in sel if r["condition"] == "foil" and r["flags"]["correct"])
    prod_neg = sum(1 for r in sel if r["condition"] == "negative" and r["flags"]["correct"])
    lut, dim = build_glove64_lookup(str(GLOVE_PATH))

    return {
        "ti": ti, "bank": bank, "by_id": by_id, "live_rows": live_rows,
        "fake_rows": fake_rows, "live_ids": live_ids, "selected": sel,
        "checked": checked, "lut": lut, "dim": dim,
        "production_recorded": {"near": [prod_near, 40], "far": [prod_far, 40],
                                "foil": [prod_foil, 40], "negative": [prod_neg, 40]},
        "hashes": {
            "production_loader_sha256": prod_sha, "worktree_loader_sha256": wt_sha,
            "item_bank_sha256": bank_sha_wt,
            "glove_cache_path": str(GLOVE_PATH), "glove_cache_sha256": sha256(GLOVE_PATH),
            "recorded_rows_path": str(ROWS_PATH),
            "recorded_rows_sha256": sha256(ROWS_PATH),
        },
    }


# ---- teaching material -> reusable surfaces / contexts --------------------
def surface_tokens(teaching_surface_text: str, label_tokens: set[str]) -> list[str]:
    toks = [w for w in re.findall(r"[a-z]+", teaching_surface_text.lower())
            if w not in STOP and len(w) > 1]
    return [w for w in toks if w not in label_tokens]


def build_teaching(inst: dict) -> dict:
    """Everything below is derived from type_label + taught_exemplars only.

    No type_sets / world_model / domain / gold word is ever used to build a
    scorer input.  ``domain`` appears in the recorded row but is not consulted.
    """
    ti, by_id = inst["ti"], inst["by_id"]
    all_label_tokens = set()
    for label in ti.ROLE_TO_TYPE.values():
        all_label_tokens |= set(re.findall(r"[a-z]+", label.lower()))

    per_item: dict[str, dict] = {}
    pooled_tok_lists: list[list[str]] = []
    for iid in inst["live_ids"]:
        it = by_id[iid]
        k = int(it["k"])
        surfaces = [ti.teaching_surface(it["role"], ex["r1"], ex["r2"])
                    for ex in it["taught_exemplars"][:k]]
        tok_lists = [surface_tokens(s, all_label_tokens) for s in surfaces]
        r2 = [ex["r2"].strip().lower() for ex in it["taught_exemplars"][:k]]
        per_item[iid] = {
            "k": k, "surfaces": surfaces, "tok_lists": tok_lists,
            "r2": r2, "label": it["type_label"],
        }
        pooled_tok_lists.extend(tok_lists)

    def contexts(tok_lists: list[list[str]]) -> tuple[dict, dict, int]:
        co: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for toks in tok_lists:
            words = sorted(set(toks))
            for w in words:
                for v in words:
                    if w != v:
                        co[w][v] += 1
        vocab = sorted(set(co) | {v for c in co.values() for v in c})
        idx = {w: i for i, w in enumerate(vocab)}
        mat = np.zeros((len(vocab), len(vocab)), dtype=np.float64)
        for w, c in co.items():
            for v, n in c.items():
                mat[idx[w], idx[v]] = float(n)
        ctx = {w: unit(mat[idx[w]]) for w in co}
        return ctx, idx, len(vocab)

    pooled_ctx, pooled_idx, pooled_vocab_n = contexts(pooled_tok_lists)
    local_ctx = {iid: contexts(d["tok_lists"])[0] for iid, d in per_item.items()}
    return {
        "per_item": per_item,
        "all_label_tokens": sorted(all_label_tokens),
        "pooled_ctx": pooled_ctx, "pooled_idx": pooled_idx,
        "pooled_vocab_n": pooled_vocab_n,
        "pooled_words_n": len(pooled_ctx),
        "local_ctx": local_ctx,
        "surfaces_n": len(pooled_tok_lists),
    }


# ---- fail-closed ranker (no gold / type_sets / world_model in scope) -------
def rank(scores: dict[str, float], options: list[str], floor: float = CONGRUENCE_FLOOR,
         margin: float = STRICT_MARGIN, permutation: bool = True) -> dict:
    """Strict winner; floor, margin and permutation stability all fail closed."""
    if not scores:
        return {"pick": None, "reason": "no_scored_option", "scores": {},
                "permutation_picks": []}
    eligible = list(scores)

    def _one(opts: list[str]):
        order = sorted(opts, key=lambda o: (-scores[o], o))
        best, second = order[0], (order[1] if len(order) > 1 else None)
        if scores[best] < floor:
            return None, "no_option_above_floor"
        if second is not None and (scores[best] - scores[second]) < margin:
            return None, "strict_margin_not_met"
        return best, "strict_winner"

    pick, reason = _one(eligible)
    perm_picks = [pick]
    if permutation and pick is not None:
        for sd in PERM_SEEDS:
            shuffled = sorted(eligible, key=lambda o: hashlib.sha256(
                f"{sd}|{o}".encode()).hexdigest())
            p, _ = _one(shuffled)
            perm_picks.append(p)
        if any(p != pick for p in perm_picks[1:]):
            return {"pick": None, "reason": "permutation_unstable", "scores": scores,
                    "eligible": eligible, "permutation_picks": perm_picks}
    return {"pick": pick, "reason": reason, "scores": {k: round(v, 4)
                                                       for k, v in scores.items()},
            "eligible": eligible, "permutation_picks": perm_picks}


def a1_item_own_centre(subject_vec, candidates, centre, taught_r2, lut):
    if subject_vec is None or centre is None:
        return {"pick": None, "reason": "no_subject_vector_or_own_centre", "scores": {}}
    eligible = [c for c in candidates if c.strip().lower() in lut]
    if not eligible:
        return {"pick": None, "reason": "disjoint_or_unknown_option_set", "scores": {}}
    scores = {c: cosine(lut[c.strip().lower()], centre) for c in eligible}
    return rank(scores, eligible)


def a2_corpus(subject, candidates, ctx, scope_name):
    sv = ctx.get(subject.strip().lower())
    if sv is None:
        return {"pick": None, "reason": f"subject_absent_from_{scope_name}_teaching_corpus",
                "scores": {}}
    eligible = [c for c in candidates if c.strip().lower() in ctx]
    if not eligible:
        return {"pick": None, "reason": "disjoint_or_unknown_option_set", "scores": {}}
    scores = {c: cosine(sv, ctx[c.strip().lower()]) for c in eligible}
    return rank(scores, eligible)


def ngrams(word: str, n: int = NGRAM_N) -> set[str]:
    s = "^" + word.strip().lower() + "$"
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    u = a | b
    return float(len(a & b) / len(u)) if u else 0.0


def a3_morph(subject, candidates, taught_r2, subject_gate=True):
    if not taught_r2:
        return {"pick": None, "reason": "no_taught_r2_word_at_k0", "scores": {}}
    prof = {r: ngrams(r) for r in taught_r2}
    s_prof = ngrams(subject)
    r_star = max(taught_r2, key=lambda r: (jaccard(s_prof, prof[r]), r))
    subject_fit = jaccard(s_prof, prof[r_star])
    if subject_gate and subject_fit < CONGRUENCE_FLOOR:
        return {"pick": None, "reason": "subject_not_fast_mapped_to_taught_form",
                "scores": {}, "subject_fit": round(subject_fit, 4),
                "nearest_taught_r2": r_star}
    scores = {c: jaccard(ngrams(c), prof[r_star]) for c in candidates}
    out = rank(scores, list(candidates))
    out["subject_fit"] = round(subject_fit, 4)
    out["nearest_taught_r2"] = r_star
    return out


def a4_oracle(item, cond, candidates):
    """CONTAMINATION CEILING -- never deployable. Reads the answer key."""
    if cond in ("near", "far"):
        gold = item["unseen_test_words"][cond]["gold"]
        key = f"{item['role']}|{item['unseen_test_words'][cond]['subject']}"
        wm_has = gold in item["world_model"].get(key, [])
        return {"pick": gold if gold in candidates else None,
                "reason": "oracle_world_model_gold", "source": "world_model",
                "in_world_model": wm_has}
    return {"pick": None, "reason": "oracle_should_reject_no_licensed_role",
            "source": "world_model"}


# ---- case runner -----------------------------------------------------------
def case_spec(item, cond):
    if cond in ("near", "far"):
        q = item["unseen_test_words"][cond]
        return q["subject"], q["candidates"], q["gold"]
    if cond == "foil":
        return item["foil"]["subject"], item["foil"]["candidates"], None
    return item["negative"]["subject"], item["negative"]["candidates"], None


def run(inst: dict, teach: dict) -> list[dict]:
    lut = inst["lut"]
    rows = []
    for r in inst["selected"]:
        iid, cond = r["id"], r["condition"]
        it = inst["by_id"][iid]
        d = teach["per_item"][iid]
        subject, candidates, gold = case_spec(it, cond)
        sv = lut.get(subject.strip().lower())
        centre = None
        if d["r2"]:
            vs = [unit(lut[w]) for w in d["r2"] if w in lut]
            if vs:
                centre = np.mean(vs, axis=0)

        a1 = a1_item_own_centre(sv, candidates, centre, set(d["r2"]), lut)
        a2 = a2_corpus(subject, candidates, teach["pooled_ctx"], "pooled")
        a2_local = a2_corpus(subject, candidates, teach["local_ctx"][iid], "local")
        a3 = a3_morph(subject, candidates, d["r2"], subject_gate=True)
        a3_diag = a3_morph(subject, candidates, d["r2"], subject_gate=False)
        a3_subjopt = rank({c: jaccard(ngrams(subject), ngrams(c)) for c in candidates},
                          list(candidates))
        a4 = a4_oracle(it, cond, candidates)

        # Grade AFTER all scoring; gold is not passed to A0-A3.
        expect = None if cond in ("foil", "negative") else gold
        graded = {
            "A0": r["flags"]["correct"],
            "A1": (a1["pick"] is None) if expect is None else (a1["pick"] == expect),
            "A2": (a2["pick"] is None) if expect is None else (a2["pick"] == expect),
            "A3": (a3["pick"] is None) if expect is None else (a3["pick"] == expect),
            "A4": (a4["pick"] is None) if expect is None else (a4["pick"] == expect),
        }
        diag_graded = {
            "A2_local": ((a2_local["pick"] is None) if expect is None
                         else (a2_local["pick"] == expect)),
            "A3_no_subject_gate": ((a3_diag["pick"] is None) if expect is None
                                   else (a3_diag["pick"] == expect)),
            "A3_subject_to_option": ((a3_subjopt["pick"] is None) if expect is None
                                     else (a3_subjopt["pick"] == expect)),
        }
        rows.append({
            "id": iid, "pool": it["pool"], "role": it["role"], "k": d["k"],
            "condition": cond, "subject": subject, "candidates": candidates,
            "gold": gold, "expected_pick": expect,
            "taught_r2": d["r2"], "teaching_surfaces": d["surfaces"],
            "arms": {
                "A0": {"pick": r["engine_answer"]["target"] if r["engine_answer"] else None,
                       "verdict": (r["engine_answer"] or {}).get("verdict"),
                       "correct": graded["A0"], "reason": "recorded_live_structural_query"},
                "A1": {**a1, "correct": graded["A1"]},
                "A2": {**a2, "correct": graded["A2"]},
                "A3": {**a3, "correct": graded["A3"]},
                "A4": {**a4, "correct": graded["A4"], "deployable": False},
            },
            "diagnostics": {
                "A2_local": {**a2_local, "correct": diag_graded["A2_local"]},
                "A3_no_subject_gate": {**a3_diag, "correct": diag_graded["A3_no_subject_gate"]},
                "A3_subject_to_option": {**a3_subjopt, "correct": diag_graded["A3_subject_to_option"]},
            },
        })
    return rows


# ---- controls and summary ---------------------------------------------------
def controls(inst: dict, teach: dict, rows: list[dict]) -> dict:
    lut = inst["lut"]
    first = rows[0]
    subject, candidates = first["subject"], first["candidates"]
    d = teach["per_item"][first["id"]]
    centre_vs = [unit(lut[w]) for w in d["r2"] if w in lut]
    centre = np.mean(centre_vs, axis=0) if centre_vs else None

    out = {}
    out["A1_disjoint_options"] = {
        "options": DISJOINT_OPTIONS,
        **a1_item_own_centre(lut.get(subject), DISJOINT_OPTIONS, centre, set(d["r2"]), lut)}
    out["A1_unrelated_subject"] = {
        "subject": UNRELATED_SUBJECT,
        **a1_item_own_centre(lut.get(UNRELATED_SUBJECT), candidates, centre, set(d["r2"]), lut)}
    out["A2_disjoint_options"] = {"options": DISJOINT_OPTIONS,
                                  **a2_corpus(subject, DISJOINT_OPTIONS, teach["pooled_ctx"], "pooled")}
    out["A2_unrelated_subject"] = {"subject": UNRELATED_SUBJECT,
                                   **a2_corpus(UNRELATED_SUBJECT, candidates, teach["pooled_ctx"], "pooled")}
    out["A3_disjoint_options"] = {"options": DISJOINT_OPTIONS,
                                  **a3_morph(subject, DISJOINT_OPTIONS, d["r2"])}
    out["A3_unrelated_subject"] = {"subject": UNRELATED_SUBJECT,
                                   **a3_morph(UNRELATED_SUBJECT, candidates, d["r2"])}
    for k, v in out.items():
        v["pass_fail_closed"] = v.get("pick") is None

    # Coverage: how often did the subject / options actually exist in the corpus?
    subj_cov = sum(1 for rr in rows if rr["subject"] in teach["pooled_ctx"])
    opt_cov = sum(1 for rr in rows
                  if any(c in teach["pooled_ctx"] for c in rr["candidates"]))
    k0 = [iid for iid in inst["live_ids"] if inst["by_id"][iid]["k"] == 0]
    out["coverage"] = {
        "cases": len(rows), "subject_in_pooled_teaching_corpus": subj_cov,
        "any_candidate_in_pooled_teaching_corpus": opt_cov,
        "k0_items_no_own_teaching": k0,
        "pooled_vocab_words": teach["pooled_vocab_n"],
        "pooled_words_with_context": teach["pooled_words_n"],
        "teaching_surfaces_used": teach["surfaces_n"],
    }
    # Oracle audit: the subject's OWN authored type_set never carries the slot.
    subj_type_examples = {}
    for rr in rows[:6]:
        it = inst["by_id"][rr["id"]]
        subj_type_examples[rr["id"]] = {
            "subject": rr["subject"], "subject_type_set": it["type_sets"].get(rr["subject"]),
            "taught_type_label": it["type_label"],
            "subject_type_contains_slot_label": it["type_label"] in
                                                 it["type_sets"].get(rr["subject"], []),
        }
    subj_has_slot = sum(1 for rr in rows
                        if inst["by_id"][rr["id"]]["type_label"] in
                        inst["by_id"][rr["id"]]["type_sets"].get(rr["subject"], []))
    gold_has_slot = sum(1 for rr in rows if rr["condition"] in ("near", "far")
                        and inst["by_id"][rr["id"]]["type_label"] in
                        inst["by_id"][rr["id"]]["type_sets"].get(rr["gold"], []))
    out["oracle_audit"] = {
        "subject_own_type_set_examples": subj_type_examples,
        "cases": len(rows),
        "subject_type_set_contains_taught_slot_label": subj_has_slot,
        "near_far_gold_type_set_contains_taught_slot_label": [gold_has_slot, 80],
        "note": "A4 reads world_model (gold) because the query subject's own "
                "type_set deliberately excludes the taught slot label; a "
                "candidate-side membership key is what solves the ceiling.",
    }
    return out


def summarize(rows: list[dict], arm_names: list[str], diag_names: list[str]) -> dict:
    out = {}
    for a in arm_names + diag_names:
        per = {}
        for cond in CONDITIONS:
            rr = [r for r in rows if r["condition"] == cond]
            per[cond] = {
                "n": len(rr),
                "correct": sum(1 for r in rr if (r["arms"].get(a) or r["diagnostics"].get(a, {}))["correct"]),
                "picks_none": sum(1 for r in rr
                                  if (r["arms"].get(a) or r["diagnostics"].get(a, {}))["pick"] is None),
            }
        pool = {}
        for p in ("primary", "sealed"):
            rr = [r for r in rows if r["pool"] == p and r["condition"] in ("near", "far")]
            pool[p] = {"n": len(rr),
                       "correct": sum(1 for r in rr
                                      if (r["arms"].get(a) or r["diagnostics"].get(a, {}))["correct"])}
        src = lambda r: (r["arms"].get(a) or r["diagnostics"].get(a, {}))
        out[a] = {
            "per_condition": per, "near_far_by_pool": pool,
            "foil_leakage": sum(1 for r in rows if r["condition"] == "foil"
                                and src(r)["pick"] is not None),
            "negative_leakage": sum(1 for r in rows if r["condition"] == "negative"
                                    and src(r)["pick"] is not None),
            "reasons": dict(collections.Counter(src(r)["reason"] for r in rows)),
        }
    return out


def g6_for(summ: dict, arm_names: list[str]) -> dict:
    spec = importlib.util.spec_from_file_location(
        "front_gates_scratch", str(MAIN / "scripts" / "front_gates.py"))
    fg = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(fg)
    out = {}
    for a in arm_names:
        per = summ[a]["per_condition"]
        near_pct = 100.0 * per["near"]["correct"] / per["near"]["n"]
        neg_pct = 100.0 * per["negative"]["correct"] / per["negative"]["n"]
        res = fg.gate_saturation({"tra_near": near_pct, "tra_negative_reject": neg_pct})
        out[a] = {"tra_near": round(near_pct, 3), "tra_negative_reject": round(neg_pct, 3),
                  "status": res.status, "detail": res.detail,
                  "flagged_fronts": res.flagged_fronts, "blocking": res.blocking}
    return out


def main() -> int:
    t0 = time.time()
    log("TYPE-ACQUISITION SIGNAL SEARCH — scratch, static, NO engine boot")
    log(f"worktree={WORKTREE}  branch=experiment/type-acquisition-signal  base=bfe99a6")
    log("constraints: no production edit, no boot, no store write, "
        "no answer key in deployable scorers")
    try:
        inst = instrument()
        log(f"instrument OK: loader={sys.modules['ravana.ontology.attribute_encoder'].__file__}")
        log(f"loader_sha={inst['hashes']['production_loader_sha256'][:16]} "
            f"bank_sha={inst['hashes']['item_bank_sha256'][:16]} "
            f"glove_sha={inst['hashes']['glove_cache_sha256'][:16]}")
        log(f"recorded live rows={len(inst['live_rows'])} fake excluded={inst['fake_rows']} "
            f"graded rows={len(inst['selected'])} cross-checked={inst['checked']} "
            f"dim={inst['dim']} lut={len(inst['lut'])}")
        log(f"recorded production baseline (flags.correct): {inst['production_recorded']}")
        teach = build_teaching(inst)
        log(f"teaching surfaces built={teach['surfaces_n']} "
            f"pooled vocab={teach['pooled_vocab_n']} words with ctx={teach['pooled_words_n']}")
        rows = run(inst, teach)
        ctr = controls(inst, teach, rows)
        log("controls: " + json.dumps({k: {"pick": v.get("pick"), "pass_fail_closed":
                                          v.get("pass_fail_closed")} for k, v in ctr.items()
                                       if isinstance(v, dict) and "pick" in v}))
        log("coverage: " + json.dumps(ctr["coverage"]))
        arm_names = ["A0", "A1", "A2", "A3", "A4"]
        diag_names = ["A2_local", "A3_no_subject_gate", "A3_subject_to_option"]
        summ = summarize(rows, arm_names, diag_names)
        log("\n=== CORRECT COUNTS (n = 40 items x 4 conditions per arm) ===")
        for a in arm_names + diag_names:
            p = summ[a]["per_condition"]
            tag = " [NON-DEPLOYABLE ORACLE]" if a == "A4" else ""
            log(f"{a:20s} near {p['near']['correct']:2d}/{p['near']['n']}  "
                f"far {p['far']['correct']:2d}/{p['far']['n']}  "
                f"foil {p['foil']['correct']:2d}/{p['foil']['n']}  "
                f"negative {p['negative']['correct']:2d}/{p['negative']['n']}"
                f"  foil_leak={summ[a]['foil_leakage']}{tag}")
        log("\n=== near+far by pool ===")
        for a in arm_names:
            p = summ[a]["near_far_by_pool"]
            log(f"{a:20s} primary {p['primary']['correct']}/{p['primary']['n']}  "
                f"sealed {p['sealed']['correct']}/{p['sealed']['n']}")
        g6 = g6_for(summ, arm_names)
        log("\n=== G6 saturation (shipped front_gates, non-blocking FLAG) ===")
        for a in arm_names:
            log(f"  {a:20s} {g6[a]['status']}: {g6[a]['detail']}")

        # Honest deployable verdict.
        deploy = {}
        for a in ("A1", "A2", "A3"):
            nf = (summ[a]["per_condition"]["near"]["correct"] +
                  summ[a]["per_condition"]["far"]["correct"])
            deploy[a] = {
                "near_far_correct": nf, "near_far_n": 80,
                "foil_leakage": summ[a]["foil_leakage"],
                "moves_above_zero_without_foil_leakage": nf > 0 and summ[a]["foil_leakage"] == 0,
            }
        # Audit every near/far hit: did this item's own teaching produce it?
        hit_audit = {}
        for a in arm_names:
            hits = [{"id": rr["id"], "k": rr["k"], "condition": rr["condition"],
                     "subject": rr["subject"], "picked": rr["arms"][a]["pick"],
                     "gold": rr["gold"]}
                    for rr in rows if rr["condition"] in ("near", "far")
                    and rr["arms"][a]["correct"]]
            repeated = collections.Counter(h["picked"] for h in hits)
            hit_audit[a] = {
                "n": len(hits), "k0_hits": sum(1 for h in hits if h["k"] == 0),
                "repeated_pick_counts": dict(repeated), "hits": hits,
            }
        log("near/far hit audit (k=0 or repeated-token hits are not within-item transfer):")
        for a in arm_names:
            log(f"  {a:20s} hits={hit_audit[a]['n']} k0_hits={hit_audit[a]['k0_hits']} "
                f"picks={hit_audit[a]['repeated_pick_counts']}")

        any_signal = any(v["moves_above_zero_without_foil_leakage"] for v in deploy.values())
        log("\nDEPLOYABLE VERDICT: " + ("AT LEAST ONE ARM MOVES near/far > 0 "
                                       "WITHOUT FOIL LEAKAGE" if any_signal
                                       else "NO DEPLOYABLE TEACHING-SIDE SIGNAL MOVES "
                                            "near/far ABOVE 0 WITHOUT FOIL LEAKAGE"))
        result = {
            "experiment": "search for a deployable teaching-side unseen-word type signal",
            "branch": "experiment/type-acquisition-signal", "base_commit": "bfe99a6",
            "production_code_modified": False, "engine_booted": False,
            "store_written": False, "answer_key_in_deployable_scorers": False,
            "question": "Can any signal legitimately present in the teaching turns "
                        "classify an unseen subject (near/far) while rejecting foil/negative?",
            "arms": {
                "A0": "production baseline (recorded focused live; recomputed near/far from flags)",
                "A1": "GloVe item-own category centre over candidate words (deployable control)",
                "A2": "teaching-corpus co-occurrence, pooled (deployable)",
                "A3": "char-3gram fast-mapping between query subject and taught r2 (deployable)",
                "A4": "ORACLE world_model/type_sets ceiling (NON-DEPLOYABLE contamination)",
            },
            "method": {
                "floor": CONGRUENCE_FLOOR, "strict_margin": STRICT_MARGIN,
                "permutation": "2 deterministic hash shuffles; any changed pick -> None",
                "disjoint_control": DISJOINT_OPTIONS,
                "unrelated_subject": UNRELATED_SUBJECT,
                "ngram_n": NGRAM_N,
                "gold_use": "graded only, after A0-A3 scoring; A4 reads the key by design",
                "teaching_source": "regenerated with scripts/tra_items.py::teaching_surface "
                                   "(recorded rows carry empty teaching_surfaces)",
            },
            "inputs": {
                "item_bank": str(ITEM_BANK_PATH), "bank_sha256": inst["hashes"]["item_bank_sha256"],
                "families": len(inst["bank"]),
                "recorded_rows": inst["hashes"]["recorded_rows_path"],
                "recorded_rows_sha256": inst["hashes"]["recorded_rows_sha256"],
                "live_rows": len(inst["live_rows"]), "fake_excluded": inst["fake_rows"],
                "graded_rows": len(inst["selected"]), "items": len(inst["live_ids"]),
                "item_ids": inst["live_ids"],
                "pool_split": dict(collections.Counter(inst["by_id"][i]["pool"]
                                                       for i in inst["live_ids"])),
                "k_distribution": dict(collections.Counter(inst["by_id"][i]["k"]
                                                           for i in inst["live_ids"])),
                "rows_cross_checked": inst["checked"], "hashes": inst["hashes"],
            },
            "recorded_production_baseline": inst["production_recorded"],
            "controls": ctr, "g6_saturation": g6, "hit_audit": hit_audit,
            "deployable_verdict": deploy, "any_deployable_signal": any_signal,
            "arm_summary": summ, "rows": rows,
            "elapsed_s": round(time.time() - t0, 2), "errors": [],
        }
        OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"\nwrote {OUT_PATH}")
        log("EXIT=0")
        return 0
    except Exception as exc:  # surface swallowed exceptions (skill section 6)
        import traceback
        log("ERROR: " + repr(exc))
        log(traceback.format_exc())
        log("EXIT=2")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
