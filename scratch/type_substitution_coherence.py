#!/usr/bin/env python3
"""SCRATCH-ONLY: test the user's substitution-plus-coherence typing hypothesis.

Hypothesis (verbatim intent): "I will learn new words (unseen) through replacing
the word with already known word, and then checking the para when study wholly
makes sense and consistent."

Mechanisation on the 40 recorded live TRA cases:
  * build a PARAGRAPH from teaching surfaces + the query surface (only
    type_label / taught_exemplars / domain and the bare query token -- NEVER
    the query subject's type from type_sets, world_model or the gold target);
  * substitute each taught exemplar into the query SUBJECT position;
  * for each candidate option form the reading "<subject> is a <option>" with
    the exemplar as evidence and score COHERENCE, not similarity:
       S1 graph-chain consistency (provenance-trusted typed edge + role kept
          across all k exemplars),
       S2 query-context <-> teaching-context overlap,
       S3 whole-paragraph entailment,
       S4/S5 the same over the six TYPE LABELS (the literal "candidate type
          option" / "<subject> is a <type>" reading),
       S6 an ORACLE using type_sets (NON-DEPLOYABLE ceiling);
  * fail-closed gates: congruence floor, strict margin, permutation stability,
    disjoint options -> None, unrelated subject -> None, shipped G6 saturation.

No engine boot, no store write, no production edit, no answer key in any
deployable scorer.  No word embeddings are used: the hypothesis is a
COHERENCE test, and the embedding-centroid route is already a measured failure
(TRA_CATEGORY_CENTRE_EXPERIMENT.md).

Brain grounding (verified in TYPE_LEARNING_BRAIN.md, re-cited in the doc):
Carey & Bartlett 1978; Dollaghan 1985 doi:10.1044/jshr.2803.454; Medina 2011
doi:10.1073/pnas.1105040108; Xu & Tenenbaum 2007
doi:10.1037/0033-295X.114.2.245; Tse 2007 doi:10.1126/science.1135935.
"""
from __future__ import annotations

import collections
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent
MAIN = WORKTREE.parent / "ravana"

# Read-only production inputs.  The engine package is NEVER imported.
ROWS_PATH = MAIN / "scratch" / "tra_harness" / "tra_rows.jsonl"
SLOT_RESULTS_PATH = MAIN / "scratch" / "tra_slot_results.json"
ITEM_BANK_PATH = WORKTREE / "scripts" / "tra_items.py"
FRONT_GATES_PATH = MAIN / "scripts" / "front_gates.py"
OUT_PATH = HERE / "type_substitution_results.json"

# ---- frozen constants (chosen before any score was read) -------------------
CONGRUENCE_FLOOR = 0.20
STRICT_MARGIN = 0.10
PERM_SEEDS = (11, 29)
DISJOINT_OPTIONS = ["apple", "banana", "chair", "table"]
UNRELATED_SUBJECT = "zeppelin"
RECORDED_IDS = [f"tra_{i:04d}" for i in range(40)]
CONDITIONS = ("near", "far", "foil", "negative")

# TRA_BENCHMARK_SPEC.md:125 -- the relation-neutral forced-choice frame.
QUERY_TEMPLATE = ("Considering the examples above, what is true of {subject}? "
                  "Options: {options}. Answer briefly.")

# Function words stripped before co-occurrence is counted.
STOP = set((
    "a an the of in on at to by for is are was were be been being it its "
    "this that these those with and or from as into than then there their "
    "they them he she we you your our i am do does did have has had will "
    "would can could should may might must not no nor so if but while about "
    "over under again further once here when where why how all any both each "
    "few more most other some such only own same too very s t just don now "
    "considering true options answer briefly what above examples"
).split())

LOG: list[str] = []


def log(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.append(str(msg))


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def article(word: str) -> str:
    return "an" if word.strip().lower()[:1] in "aeiou" else "a"


def tokens(text: str, label_tokens: set[str]) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if w not in STOP and len(w) > 1 and w not in label_tokens]


# ---- instrument preconditions ----------------------------------------------
def instrument() -> dict:
    assert "CognitiveChatEngine" not in sys.modules, "an engine module was imported: no boot allowed"
    bank_sha_main = sha256(MAIN / "scripts" / "tra_items.py")
    bank_sha_wt = sha256(ITEM_BANK_PATH)
    assert bank_sha_main == bank_sha_wt, "item bank differs between main and worktree"

    spec = importlib.util.spec_from_file_location("tra_items_substitution", str(ITEM_BANK_PATH))
    assert spec is not None and spec.loader is not None
    ti = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ti)
    bank = ti.build_items()
    errs = ti.validate_items(bank)
    assert not errs, f"item bank invalid: {errs[:3]}"
    by_id = {it["item_id"]: it for it in bank}

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

    sel = [r for r in live_rows
           if r.get("arm") == "structural" and r.get("phase") == "query"
           and not r.get("permuted")]
    assert len(sel) == 160, f"expected 160 graded rows, got {len(sel)}"
    assert collections.Counter(r["condition"] for r in sel) == {c: 40 for c in CONDITIONS}
    assert len({r["id"] for r in sel}) == 40

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

    # Recorded provenance-trusted typed edges (read-only, from the trusted-edge
    # scratch experiment).  No engine boot; this is the only graph evidence.
    sr = json.loads(SLOT_RESULTS_PATH.read_text(encoding="utf-8"))
    typed = sr["arms"]["type_arm"]["post_cold_reload"]["typed_slot_edges"]
    trusted = [e for e in typed
               if e.get("trusted") and e.get("relation_type") == "is_a"
               and e.get("edge_kind") == "curated"
               and e.get("edge_source") == "conversation"]
    trusted_targets = sorted({e["target"] for e in trusted})

    prod = {
        c: sum(1 for r in sel if r["condition"] == c and r["flags"]["correct"])
        for c in CONDITIONS
    }

    return {
        "ti": ti, "bank": bank, "by_id": by_id, "live_rows": live_rows,
        "fake_rows": fake_rows, "live_ids": live_ids, "selected": sel,
        "checked": checked, "trusted_typed_edges": trusted,
        "trusted_typed_targets": trusted_targets,
        "production_recorded": prod,
        "hashes": {
            "item_bank_sha256": bank_sha_wt,
            "recorded_rows_path": str(ROWS_PATH),
            "recorded_rows_sha256": sha256(ROWS_PATH),
            "slot_results_path": str(SLOT_RESULTS_PATH),
            "slot_results_sha256": sha256(SLOT_RESULTS_PATH),
            "front_gates_sha256": sha256(FRONT_GATES_PATH),
        },
    }


# ---- teaching-only paragraph construction ---------------------------------
def build_teaching(inst: dict) -> dict:
    """Paragraph evidence.  Inputs: type_label, taught_exemplars, domain only."""
    ti = inst["ti"]
    label_tokens: set[str] = set()
    for label in ti.ROLE_TO_TYPE.values():
        label_tokens |= set(re.findall(r"[a-z]+", label.lower()))

    per_item: dict[str, dict] = {}
    for iid in inst["live_ids"]:
        it = inst["by_id"][iid]
        k = int(it["k"])
        ex = it["taught_exemplars"][:k]
        r1s = [e["r1"].strip().lower() for e in ex]
        r2s = [e["r2"].strip().lower() for e in ex]
        # teaching_surface() uses only r1, r2 and the type label.
        surfaces = [ti.teaching_surface(it["role"], e["r1"], e["r2"]) for e in ex]
        per_item[iid] = {
            "k": k, "role": it["role"], "label": it["type_label"],
            "domain": it["domain"], "r1s": r1s, "r2s": r2s,
            "surfaces": surfaces,
            "surface_tokens": [tokens(s, label_tokens) for s in surfaces],
        }

    def co_context(lists: list[list[str]]) -> dict[str, collections.Counter]:
        co: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        for toks in lists:
            words = sorted(set(toks))
            for w in words:
                for v in words:
                    if w != v:
                        co[w][v] += 1
        return co

    pooled_lists: list[list[str]] = []
    for d in per_item.values():
        pooled_lists.extend(d["surface_tokens"])
    return {
        "per_item": per_item, "label_tokens": label_tokens,
        "type_labels": sorted(set(ti.ROLE_TO_TYPE.values())),
        "pooled_co": co_context(pooled_lists),
        "local_co": {iid: co_context(d["surface_tokens"]) for iid, d in per_item.items()},
        "surfaces_n": len(pooled_lists),
    }


def case_spec(item: dict, cond: str):
    if cond in ("near", "far"):
        q = item["unseen_test_words"][cond]
        return q["subject"], q["candidates"], q["gold"]
    if cond == "foil":
        return item["foil"]["subject"], item["foil"]["candidates"], None
    return item["negative"]["subject"], item["negative"]["candidates"], None


def query_surface(subject: str, candidates: list[str]) -> str:
    return QUERY_TEMPLATE.format(subject=subject, options=", ".join(candidates))


def substitute(exemplars: list[str], candidates: list[str]) -> list[str]:
    """Literal task reading: a TAUGHT EXEMPLAR OBJECT goes into the query
    subject position (role-violating for this bank, since the teaching asserts
    r1 <relation> r2 and the unseen subject occupies r1's role).

    The teaching half of every substituted paragraph is identical and never
    names the original unseen `subject`.
    """
    return [f"{{TEACH}} " + QUERY_TEMPLATE.format(
        subject=e, options=", ".join(candidates)) for e in exemplars]


# ---- fail-closed ranker (gold / type_sets never in scope) ------------------
def rank(scores: dict[str, float]) -> dict:
    eligible = list(scores)
    if not eligible:
        return {"pick": None, "reason": "no_candidate_has_evidence", "scores": {},
                "permutation_picks": []}

    def _one(order: list[str]):
        best = order[0]
        second = order[1] if len(order) > 1 else None
        if scores[best] < CONGRUENCE_FLOOR:
            return None, "no_option_above_floor"
        if second is not None and (scores[best] - scores[second]) < STRICT_MARGIN:
            return None, "strict_margin_not_met"
        return best, "strict_winner"

    base_order = sorted(eligible, key=lambda o: (-scores[o], o))
    pick, reason = _one(base_order)
    perm = [pick]
    if pick is not None:
        # Permute the TIE-BREAK order, not the score order: a winner that only
        # wins on alphabetical ordering is unstable.  Scores are never changed.
        for sd in PERM_SEEDS:
            alt = sorted(eligible, key=lambda o: (
                -scores[o], hashlib.sha256(f"{sd}|{o}".encode()).hexdigest()))
            p, _ = _one(alt)
            perm.append(p)
        if any(p != pick for p in perm[1:]):
            return {"pick": None, "reason": "permutation_unstable",
                    "scores": {k: round(v, 4) for k, v in scores.items()},
                    "permutation_picks": perm}
    return {"pick": pick, "reason": reason,
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "permutation_picks": perm}


def ctx_cos(a: collections.Counter, b: collections.Counter) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ---- coherence scorers -----------------------------------------------------
def score_chain(inst, d, options: list[str]) -> dict:
    """S1/S4: does a provenance-trusted typed edge reach the option, and is the
    role preserved across ALL k exemplars?  Strict recorded-edge evidence.

    `options` is explicit so the same scorer serves the candidate-word space
    (S1) and the six-type-label space (S4), and so the disjoint control really
    substitutes disjoint options instead of silently reverting to a default.
    """
    targets = set(inst["trusted_typed_targets"])
    universe = list(options)
    if not d["r2s"]:
        return {"pick": None, "reason": "k0_no_taught_exemplar", "scores": {}}
    if not any(x in targets for x in universe):
        # census result, reported as a reason not hidden as an exception
        return {"pick": None, "reason": "no_trusted_typed_edge_to_any_option",
                "scores": {c: 0.0 for c in universe},
                "trusted_targets": sorted(targets)}
    scores = {}
    for c in universe:
        support = sum(1 for e in d["r2s"] if c in targets) / len(d["r2s"])
        # role check: substituting r2 (object) into the subject position breaks
        # the r1->r2 assertion direction; strict mode therefore requires r1s.
        role_ok = sum(1 for e in d["r1s"] if c in targets) / len(d["r1s"])
        scores[c] = min(support, role_ok)
    return rank(scores)


def score_context(inst, teach, d, subject, candidates) -> dict:
    """S2: words co-occurring with the unseen SUBJECT in the QUERY context
    versus words co-occurring with a taught exemplar in TEACHING contexts."""
    qt = tokens(query_surface(subject, candidates), teach["label_tokens"])
    q_co: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for w in sorted(set(qt)):
        for v in sorted(set(qt)):
            if w != v:
                q_co[w][v] += 1
    subj_ctx = q_co.get(subject.strip().lower(), collections.Counter())
    scores: dict[str, float] = {}
    overlaps = []
    for c in candidates:
        cl = c.strip().lower()
        best = 0.0
        for e in d["r2s"]:
            e_ctx = teach["pooled_co"].get(e, collections.Counter())
            if subj_ctx and e_ctx:
                best = max(best, ctx_cos(subj_ctx, e_ctx))
        overlaps.append(round(best, 4))
        scores[c] = best
    out = rank(scores)
    out["subject_query_context_words"] = sorted(subj_ctx)
    out["per_option_overlap"] = dict(zip(candidates, overlaps))
    return out


def score_para(d, options: list[str]) -> dict:
    """S3/S5: whole-paragraph coherence.  An option is coherent only if it is
    licensed by the TEACHING paragraph itself, under the literal reading
    '<substituted exemplar> is a <option>'.

    Candidate words (S3) are never named in teaching -> no reading is licensed.
    Type labels (S5) ARE asserted verbatim by each teaching surface ('r2 is an
    effect') -> that reading is licensed -- and since the unseen subject was
    substituted away, it is licensed for EVERY subject alike.  That is the
    user's hypothesis in its strongest, most charitable deployable form.

    `options` is explicit so the disjoint control substitutes real disjoint
    options rather than silently reverting to the default universe.
    """
    if not d["surfaces"]:
        return {"pick": None, "reason": "k0_no_teaching_paragraph", "scores": {}}
    assert_ = " ".join(d["surfaces"]).lower()
    scores = {}
    for c in options:
        cl = c.strip().lower()
        licensed = bool(re.search(r"\b" + re.escape(cl) + r"\b", assert_))
        scores[c] = 1.0 if licensed else 0.0
    out = rank(scores)
    out["paragraph_is_subject_invariant"] = True
    return out


def score_oracle(item, cond, candidates) -> dict:
    """S6 ORACLE -- CONTAMINATION CEILING, NON-DEPLOYABLE.  Reads type_sets."""
    if cond in ("near", "far"):
        gold = item["unseen_test_words"][cond]["gold"]
        return {"pick": gold if gold in candidates else None,
                "reason": "oracle_type_sets_gold", "source": "type_sets",
                "deployable": False}
    return {"pick": None, "reason": "oracle_should_reject_no_licensed_role",
            "source": "type_sets", "deployable": False}


# ---- case runner -----------------------------------------------------------
def run(inst: dict, teach: dict) -> list[dict]:
    rows = []
    for r in inst["selected"]:
        iid, cond = r["id"], r["condition"]
        it = inst["by_id"][iid]
        d = teach["per_item"][iid]
        subject, candidates, gold = case_spec(it, cond)

        s1 = score_chain(inst, d, candidates)
        s2 = score_context(inst, teach, d, subject, candidates)
        s3 = score_para(d, candidates)
        s4 = score_chain(inst, d, teach["type_labels"])
        s5 = score_para(d, teach["type_labels"])
        s6 = score_oracle(it, cond, candidates)

        # grade AFTER all scoring; deployable scorers never saw gold/type_sets
        word_expect = None if cond in ("foil", "negative") else gold
        label_expect = None if cond in ("foil", "negative") else it["type_label"]
        arms = {
            "S0": {"pick": (r["engine_answer"] or {}).get("target"),
                   "correct": bool(r["flags"]["correct"]),
                   "reason": "recorded_live_structural_query"},
            "S1": {**s1, "correct": (s1["pick"] is None) if word_expect is None
                   else (s1["pick"] == word_expect)},
            "S2": {**s2, "correct": (s2["pick"] is None) if word_expect is None
                   else (s2["pick"] == word_expect)},
            "S3": {**s3, "correct": (s3["pick"] is None) if word_expect is None
                   else (s3["pick"] == word_expect)},
            "S4": {**s4, "correct": (s4["pick"] is None) if label_expect is None
                   else (s4["pick"] == label_expect)},
            "S5": {**s5, "correct": (s5["pick"] is None) if label_expect is None
                   else (s5["pick"] == label_expect)},
            "S6": {**s6, "correct": (s6["pick"] is None) if word_expect is None
                   else (s6["pick"] == word_expect)},
        }
        rows.append({
            "id": iid, "pool": it["pool"], "k": d["k"], "condition": cond,
            "subject": subject, "candidates": candidates, "gold": gold,
            "type_label": it["type_label"],
            "expected_word_pick": word_expect, "expected_label_pick": label_expect,
            "teaching_surfaces": d["surfaces"],
            "query_surface": query_surface(subject, candidates),
            "substitution_obj_paragraphs": substitute(d["r2s"], candidates),
            "trusted_typed_targets": sorted(set(inst["trusted_typed_targets"])),
            "arms": arms,
        })
    return rows


# ---- controls, summary, G6 -------------------------------------------------
def controls(inst, teach, rows) -> dict:
    d = teach["per_item"][rows[0]["id"]]
    subject, candidates = rows[0]["subject"], rows[0]["candidates"]
    out = {}
    word_arms = {
        "S1": lambda s, c: score_chain(inst, d, c),
        "S2": lambda s, c: score_context(inst, teach, d, s, c),
        "S3": lambda s, c: score_para(d, c),
    }
    label_arms = {
        "S4": lambda s, c: score_chain(inst, d, c),
        "S5": lambda s, c: score_para(d, c),
    }
    for name, fn in word_arms.items():
        out[f"{name}_disjoint_options"] = {
            "options": DISJOINT_OPTIONS, **fn(subject, DISJOINT_OPTIONS)}
        out[f"{name}_unrelated_subject"] = {
            "subject": UNRELATED_SUBJECT, "options": candidates,
            **fn(UNRELATED_SUBJECT, candidates)}
    for name, fn in label_arms.items():
        out[f"{name}_disjoint_options"] = {
            "options": DISJOINT_OPTIONS, **fn(subject, DISJOINT_OPTIONS)}
        out[f"{name}_unrelated_subject"] = {
            "subject": UNRELATED_SUBJECT, "options": teach["type_labels"],
            **fn(UNRELATED_SUBJECT, teach["type_labels"])}
    for v in out.values():
        v["pass_fail_closed"] = v.get("pick") is None

    # Information audit: is the unseen token (or its gold) anywhere in teaching?
    subj_hits = 0
    gold_hits = 0
    for rr in rows:
        p = " ".join(rr["teaching_surfaces"]).lower()
        if re.search(r"\b" + re.escape(rr["subject"].lower()) + r"\b", p):
            subj_hits += 1
        if rr["gold"] and re.search(r"\b" + re.escape(rr["gold"].lower()) + r"\b", p):
            gold_hits += 1
    label_targets = set(inst["trusted_typed_targets"]) & set(teach["type_labels"])
    out["information_audit"] = {
        "cases": len(rows),
        "query_subject_in_own_teaching_surfaces": subj_hits,
        "gold_candidate_in_own_teaching_surfaces": [gold_hits, 80],
        "trusted_typed_edges_targeting_a_TRA_type_label": sorted(label_targets),
        "teaching_surfaces_used": teach["surfaces_n"],
        "substitution_note": "substituting the exemplar for the subject removes "
                             "the only token that differs between near/far and "
                             "foil/negative; the teaching half is identical.",
    }
    # Substitution invariance: teaching text is item-level, not condition-level.
    per_item_teach = {}
    for rr in rows:
        per_item_teach.setdefault(rr["id"], hashlib.sha256(
            " ".join(rr["teaching_surfaces"]).encode()).hexdigest())
    out["substitution_invariance"] = {
        "items": len(per_item_teach),
        "distinct_teaching_paragraph_hashes": len(set(per_item_teach.values())),
        "note": "one teaching paragraph per item, shared by all four conditions",
    }
    return out


def summarize(rows, arms) -> dict:
    out = {}
    for a in arms:
        per = {}
        for cond in CONDITIONS:
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
        out[a] = {
            "per_condition": per, "near_far_by_pool": pool,
            "foil_leakage": sum(1 for r in rows if r["condition"] == "foil"
                                and r["arms"][a]["pick"] is not None),
            "negative_leakage": sum(1 for r in rows if r["condition"] == "negative"
                                    and r["arms"][a]["pick"] is not None),
            "reasons": dict(collections.Counter(
                r["arms"][a]["reason"] for r in rows)),
        }
    return out


def g6_for(summ, arms) -> dict:
    spec = importlib.util.spec_from_file_location("front_gates_sub", str(FRONT_GATES_PATH))
    assert spec is not None and spec.loader is not None
    fg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fg)
    out = {}
    for a in arms:
        p = summ[a]["per_condition"]
        near = 100.0 * p["near"]["correct"] / p["near"]["n"]
        neg = 100.0 * p["negative"]["correct"] / p["negative"]["n"]
        res = fg.gate_saturation({"tra_near": near, "tra_negative_reject": neg})
        out[a] = {"tra_near": round(near, 3), "tra_negative_reject": round(neg, 3),
                  "status": res.status, "detail": res.detail,
                  "flagged_fronts": res.flagged_fronts, "blocking": res.blocking}
    return out


def main() -> int:
    t0 = time.time()
    log("SUBSTITUTION + COHERENCE TEST of unseen-word typing — scratch/static, NO engine boot")
    log(f"worktree={WORKTREE} branch=experiment/substitution-coherence base=bfe99a6")
    log("constraints: no production edit, no boot, no store write, "
        "no key in deployable scorers, no embeddings (coherence not similarity)")
    try:
        inst = instrument()
        log(f"instrument OK: bank={inst['hashes']['item_bank_sha256'][:16]} "
            f"rows={sha256(ROWS_PATH)[:16]} slot={inst['hashes']['slot_results_sha256'][:16]}")
        log(f"recorded live rows={len(inst['live_rows'])} fake excluded={inst['fake_rows']} "
            f"graded={len(inst['selected'])} cross-checked={inst['checked']}")
        log(f"production baseline recomputed: {inst['production_recorded']}")
        log(f"trusted typed edge targets (all recorded): {inst['trusted_typed_targets']}")
        teach = build_teaching(inst)
        log(f"teaching surfaces={teach['surfaces_n']} type labels={teach['type_labels']}")

        rows = run(inst, teach)
        ctr = controls(inst, teach, rows)
        arm_names = ["S0", "S1", "S2", "S3", "S4", "S5", "S6"]
        summ = summarize(rows, arm_names)
        g6 = g6_for(summ, arm_names)

        log("\n=== one worked example: the paragraph and the substitution ===")
        ex = rows[0]
        log(f"item={ex['id']} cond={ex['condition']} k={ex['k']} subject={ex['subject']}")
        log("TEACH PARAGRAPH: " + " | ".join(ex["teaching_surfaces"]))
        log("QUERY SURFACE : " + ex["query_surface"])
        log("SUBSTITUTED(obj): " + " | ".join(ex["substitution_obj_paragraphs"]))
        log("S1 scores=" + json.dumps(ex["arms"]["S1"].get("scores")) +
            " pick=" + str(ex["arms"]["S1"]["pick"]))
        log("S2 query-context=" + json.dumps(ex["arms"]["S2"].get("subject_query_context_words")) +
            " overlaps=" + json.dumps(ex["arms"]["S2"].get("per_option_overlap")))
        log("S3 scores=" + json.dumps(ex["arms"]["S3"].get("scores")))
        log("S5 scores=" + json.dumps(ex["arms"]["S5"].get("scores")))

        log("\n=== CORRECT COUNTS (n = 40 per condition; foil/negative correct = None) ===")
        for a in arm_names:
            p = summ[a]["per_condition"]
            tag = " [NON-DEPLOYABLE ORACLE]" if a == "S6" else ""
            tag += " [TYPE-LABEL OPTION SPACE]" if a in ("S4", "S5") else ""
            log(f"{a:3s} near {p['near']['correct']:2d}/{p['near']['n']}  "
                f"far {p['far']['correct']:2d}/{p['far']['n']}  "
                f"foil {p['foil']['correct']:2d}/{p['foil']['n']}  "
                f"negative {p['negative']['correct']:2d}/{p['negative']['n']}"
                f"  foil_leak={summ[a]['foil_leakage']:2d}{tag}")
        log("\n=== near+far by pool ===")
        for a in arm_names:
            p = summ[a]["near_far_by_pool"]
            log(f"{a:3s} primary {p['primary']['correct']}/{p['primary']['n']}  "
                f"sealed {p['sealed']['correct']}/{p['sealed']['n']}")
        log("\n=== fail-closed controls ===")
        for k, v in ctr.items():
            if isinstance(v, dict) and "pass_fail_closed" in v:
                log(f"  {k:34s} pick={str(v.get('pick')):24s} "
                    f"reason={v.get('reason')} pass={v['pass_fail_closed']}")
        log("\n=== G6 saturation (shipped front_gates, non-blocking FLAG) ===")
        for a in arm_names:
            log(f"  {a:3s} {g6[a]['status']}: {g6[a]['detail']}")

        deployable = [a for a in ("S1", "S2", "S3", "S4", "S5")]
        verdict = {}
        for a in deployable:
            p = summ[a]["per_condition"]
            nf = p["near"]["correct"] + p["far"]["correct"]
            verdict[a] = {
                "near_far_correct": nf, "near_far_n": 80,
                "foil_leakage": summ[a]["foil_leakage"],
                "moves_above_zero_without_foil_leakage": nf > 0 and summ[a]["foil_leakage"] == 0,
                "passes_unrelated_subject_control":
                    ctr[f"{a}_unrelated_subject"]["pass_fail_closed"],
            }
        any_signal = any(v["moves_above_zero_without_foil_leakage"] for v in verdict.values())
        log("\nDEPLOYABLE VERDICT: " + (
            "AT LEAST ONE ARM MOVES near/far > 0 WITH ZERO FOIL LEAKAGE"
            if any_signal else
            "NO SUBSTITUTION+COHERENCE ARM MOVES near/far > 0 WITH ZERO FOIL LEAKAGE"))
        log("INFORMATION AUDIT: " + json.dumps(ctr["information_audit"]))

        result = {
            "experiment": "substitution-plus-coherence test of unseen-word typing",
            "branch": "experiment/substitution-coherence", "base_commit": "bfe99a6",
            "production_code_modified": False, "engine_booted": False,
            "store_written": False, "answer_key_in_deployable_scorers": False,
            "embeddings_used": False,
            "user_hypothesis": "learn unseen words by replacing the word with an "
                               "already-known word and checking whether the paragraph "
                               "wholly makes sense and is consistent",
            "arms": {
                "S0": "production baseline (recorded focused live; comparator)",
                "S1": "graph-chain coherence, trusted typed edge + role, candidate words",
                "S2": "query-context <-> teaching-context co-occurrence overlap",
                "S3": "whole-paragraph entailment coherence, candidate words",
                "S4": "graph-chain coherence over the 6 type labels (label option space)",
                "S5": "teaching-surface assertion as trusted typed evidence over type "
                      "labels (strongest charitable form of the hypothesis)",
                "S6": "ORACLE type_sets ceiling (NON-DEPLOYABLE contamination)",
            },
            "method": {
                "floor": CONGRUENCE_FLOOR, "strict_margin": STRICT_MARGIN,
                "permutation": "2 deterministic hash shuffles; any changed pick -> None",
                "disjoint_control": DISJOINT_OPTIONS,
                "unrelated_subject": UNRELATED_SUBJECT,
                "gold_use": "graded only after all deployable scoring; S6 reads the key by design",
                "query_surface": QUERY_TEMPLATE,
                "substitution": "taught r2 object into query subject position (literal); "
                                "r1 used in the role-consistency term of S1",
            },
            "inputs": {
                "item_bank": str(ITEM_BANK_PATH), "families": len(inst["bank"]),
                "recorded_rows": str(ROWS_PATH),
                "live_rows": len(inst["live_rows"]), "fake_excluded": inst["fake_rows"],
                "graded_rows": len(inst["selected"]), "items": len(inst["live_ids"]),
                "item_ids": inst["live_ids"],
                "k_distribution": dict(collections.Counter(
                    inst["by_id"][i]["k"] for i in inst["live_ids"])),
                "rows_cross_checked": inst["checked"], "hashes": inst["hashes"],
            },
            "trusted_typed_targets": inst["trusted_typed_targets"],
            "type_labels": teach["type_labels"],
            "recorded_production_baseline": inst["production_recorded"],
            "controls": ctr, "g6_saturation": g6,
            "deployable_verdict": verdict, "any_deployable_signal": any_signal,
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
