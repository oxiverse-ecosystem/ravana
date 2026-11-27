#!/usr/bin/env python3
"""H_lang -- cross-source fast mapping: real-language evidence, independent ontology key.

SCRATCH / STATIC ONLY. No engine boot, no ``CognitiveChatEngine`` import, no
``process_turn``, no store write, no network, no production edit, no gate or
threshold change. The only imported project modules are the UNMODIFIED
``scripts/tra_k_items.py`` (item bank, public fields only) and
``scripts/front_gates.py`` (shipped saturation gate), plus the checked-in CC0
Wikidata substrate ``scratch/trak_substrate.json`` and NLTK WordNet data
(used in the pre-flight audit only; not by the deployable scorer).

WHY THIS DESIGN
---------------
The prior ontology bridge scored 176/176 but was partly tautological: TRA-K
DEFINES the typed gold as the subject's own highest-credibility substrate fact,
so any route reading that fact is right by construction when it emits. Here the
scorer NEVER reads the subject's type label. It reads only the subject's
Wikidata *item description* -- real natural-language text -- and learns a
word -> type cue map from taught demonstration words. The grader's answer key
is a RELATION (which candidate shares the subject's structural type), computed
from facts the scorer cannot see. No item-bank ``gold``, ``type_sets``,
``world_model``, ``typed_queries`` or ``role``/``type_label`` enters any scorer.

HYPOTHESIS (pre-registered)
---------------------------
A subject-specific language cue, assimilated into a schema learned from
demonstrations, emits the type-sharing candidate on a low-coverage subset with
precision materially above a demo-only schema lookup. Correctness must depend
on the subject's own text, not on a candidate-side or answer-key lookup.

FROZEN CONSTANTS (set before any run; see PRE_REGISTERED below)
--------------------------------------------------------------
MIN_DEMO_SUPPORT = 2, MIN_CUE_SUPPORT = 2, PERMUTE_OFFSET = 137.
No threshold, floor, margin, weight or gate is tuned after a result.

Verdict rule: POSITIVE only if TEXT emits >= 10, precision >= 0.60, precision
exceeds DEMO_ONLY by >= 0.15 on the same gradeable cases, the subject-text
permutation changes >= 80% of TEXT picks, the gold permutation leaves picks
byte-identical while accuracy collapses to <= 0.40, and all abstention controls
pass. Otherwise the result is reported as negative / no-go, never softened.
"""
from __future__ import annotations

import ast
import collections
import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_k_items  # noqa: E402  (unmodified, read-only)
import front_gates  # noqa: E402  (unmodified, read-only)

MIN_DEMO_SUPPORT = 2
MIN_CUE_SUPPORT = 2
PERMUTE_OFFSET = 137
HASH_PREFIX = "RAVANA-LANG-FASTMAP-v1"
STOPWORDS = frozenset(
    "a an the of in on at to for by with is are was were be been being that "
    "which who whom whose from as or and not no it its this these those one "
    "two into about over under than then there their they them he she his her "
    "we our you your i me my can may will would should could has have had do "
    "does did very more most some any each other another same such only also "
    "term used refers referring type kind class something someone".split()
)
BANNED_SCORER_NAMES = frozenset(
    ("gold", "type_sets", "world_model", "typed_queries", "unseen_test_words",
     "type_label", "item_id", "role", "front_gates", "tra_k_items")
)


# ── provenance of inputs ────────────────────────────────────────────────────
def sha_file(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read().replace(b"\r\n", b"\n")).hexdigest()


def sha_git_blob(rel):
    out = subprocess.check_output(
        ["git", "show", "main:" + rel], cwd=ROOT, stderr=subprocess.DEVNULL)
    return hashlib.sha256(out.replace(b"\r\n", b"\n")).hexdigest()


def main():
    module_shas = {}
    for rel in ("scripts/tra_k_items.py", "scripts/front_gates.py",
                "scratch/trak_substrate.json"):
        work = sha_file(os.path.join(ROOT, *rel.split("/")))
        base = sha_git_blob(rel)
        module_shas[rel] = {"worktree": work, "main": base, "identical": work == base}
        if work != base:
            raise SystemExit("UNMODIFIED-INPUT CHECK FAILED: " + rel)

    substrate = json.load(open(os.path.join(ROOT, "scratch", "trak_substrate.json"),
                               encoding="utf-8"))
    facts = substrate["facts"]
    by_subject = collections.defaultdict(list)
    for fact in facts:
        by_subject[fact["subject"]].append(fact)

    # ── deployable read surfaces: text and TYPES, never the item answer key ──
    def norm(text):
        return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).strip()

    def primary(word):
        return [f for f in by_subject.get(word, ()) if f.get("sense") == "primary"]

    def subject_text(word):
        """REAL LANGUAGE: verbatim CC0 Wikidata item description."""
        for f in primary(word):
            if (f.get("wikidata_property") == "P279"
                    and f.get("text_evidence_kind") == "item_description"
                    and f.get("text_evidence")):
                return norm(f["text_evidence"])
        return ""

    def label_set(word):
        """Structural type facts -- GRADER side for the subject, channel for candidates."""
        return frozenset(
            norm(f["type_label"]) for f in primary(word)
        ) - {norm(word)}

    def content_words(text):
        return tuple(sorted({w for w in re.findall(r"[a-z]+", text)
                             if len(w) > 3 and w not in STOPWORDS}))

    def demo_teaching(demo_words):
        """Public teaching material in the new variant: description + typed assertion."""
        out = []
        for word in demo_words:
            t = subject_text(word)
            if t:
                out.append((t, label_set(word)))
        return tuple(out)

    # ── SCORER (AST-scanned below). Never sees subject structural types. ─────
    def learn_cue_map(demo_facts):
        """Word -> type counts from taught demonstrations (fast mapping schema)."""
        mapping = collections.defaultdict(collections.Counter)
        for text, taught_types in demo_facts:
            words = content_words(text)
            for w in words:
                for ty in taught_types:
                    mapping[w][ty] += 1
        return {w: dict(c) for w, c in mapping.items()}

    def cue_predict(text, cue_map):
        """Assimilate the subject's OWN language into the demo-learned schema."""
        counts = collections.Counter()
        for w in content_words(text):
            for ty, n in cue_map.get(w, {}).items():
                counts[ty] += n
        return frozenset(ty for ty, n in counts.items() if n >= MIN_CUE_SUPPORT)

    def schema_from_demos(demo_facts):
        counts = collections.Counter()
        for _text, taught_types in demo_facts:
            for ty in taught_types:
                counts[ty] += 1
        return frozenset(ty for ty, n in counts.items() if n >= MIN_DEMO_SUPPORT)

    def pick(predicted_types, candidate_types):
        """Emit the unique candidate whose admitted type matches; else abstain."""
        hits = [c for c, tys in candidate_types.items() if tys & predicted_types]
        return hits[0] if len(hits) == 1 else None

    # ── instrument control 1: AST anti-leakage of every scorer function ─────
    scorer_sources = {}
    ast_problems = []
    import inspect
    import textwrap
    for fn in (learn_cue_map, cue_predict, schema_from_demos, pick):
        src = textwrap.dedent(inspect.getsource(fn))
        scorer_sources[fn.__name__] = src
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in BANNED_SCORER_NAMES:
                ast_problems.append((fn.__name__, node.id))
            if isinstance(node, ast.Attribute) and node.attr in BANNED_SCORER_NAMES:
                ast_problems.append((fn.__name__, node.attr))
    ast_ok = not ast_problems

    bank = tra_k_items.build_items()
    bank_errors = tra_k_items.validate_items(bank)
    all_subjects = sorted({it["subject"] for it in bank})

    # ── fixed-by-construction answer key: independent of every scorer input ──
    def graded_key(subject, candidates):
        subj_types = label_set(subject)  # GRADER ONLY; never passed to scorer
        if not subj_types:
            return None
        hits = [c for c in candidates if label_set(c) & subj_types]
        return hits[0] if len(hits) == 1 else None

    cases = []
    for it in bank:
        for cond in ("near", "far"):
            query = it["unseen_test_words"][cond]  # public question: subject + options
            subject, candidates = query["subject"], tuple(query["candidates"])
            cases.append({
                "item_id": it["item_id"], "pool": it["pool"], "condition": cond,
                "subject": subject, "candidates": candidates,
                "text": subject_text(subject),
                "demo_facts": demo_teaching([e["r2"] for e in it["taught_exemplars"]]),
                "key": graded_key(subject, candidates),
            })
    permuted_text = [subject_text(all_subjects[(i + PERMUTE_OFFSET) % len(all_subjects)])
                     for i in range(len(cases))]

    rows = []
    for i, case in enumerate(cases):
        demo = case["demo_facts"]
        cue_map = learn_cue_map(demo)
        cand_types = {c: label_set(c) for c in case["candidates"]}
        text_types = cue_predict(case["text"], cue_map)
        demo_types = schema_from_demos(demo)
        p_text = pick(text_types, cand_types)
        p_demo = pick(demo_types, cand_types)
        p_perm = pick(cue_predict(permuted_text[i], cue_map), cand_types)
        p_neg = pick(cue_predict(case["text"], learn_cue_map(())), cand_types)
        p_empty = pick(cue_predict("", cue_map), cand_types)
        p_disjoint = pick(text_types, {c: frozenset() for c in
                                       ("apple", "banana", "chair", "table")})
        rows.append({
            "item_id": case["item_id"], "pool": case["pool"],
            "condition": case["condition"], "subject": case["subject"],
            "subject_text": case["text"],
            "candidates": list(case["candidates"]),
            "key": case["key"],  # grader-side only, never fed to a scorer
            "text_types": sorted(text_types),
            "demo_types": sorted(demo_types),
            "text": p_text, "demo": p_demo,
            "text_on_permuted_text": p_perm,
            "text_on_negated_teaching": p_neg,
            "text_on_empty_subject_text": p_empty,
            "text_on_disjoint_options": p_disjoint,
        })

    graded = [r for r in rows if r["key"] is not None]
    primary = [r for r in graded if r["pool"] == "primary"]

    def tally(subset, arm):
        emitted = [r for r in subset if r[arm] is not None]
        correct = sum(1 for r in emitted if r[arm] == r["key"])
        return {"n": len(subset), "emit": len(emitted), "correct": correct,
                "precision": round(correct / len(emitted), 4) if emitted else None,
                "coverage": round(len(emitted) / len(subset), 4) if subset else None}

    arms = {arm: {"all": tally(graded, arm), "primary": tally(primary, arm)}
            for arm in ("text", "demo")}

    # permutation of the answer key: scorer output MUST be byte-identical.
    # Replace every key with a DIFFERENT candidate from its own displayed set,
    # so a real leak would change accuracy (rotating across items can coincide
    # because the bank reuses the same candidate words).
    def permute_key(case):
        cands = list(case["candidates"])
        old = case["key"]
        for step in (1, 2, 3):
            new = cands[(cands.index(old) + step) % len(cands)] if old in cands else cands[0]
            if new != old:
                return new
        return None

    keys_before = [(r["text"], r["key"]) for r in graded]
    keys_after = [(r["text"], permute_key({"candidates": r["candidates"], "key": r["key"]}))
                  for r in graded]
    gold_perm = {
        "n": len(graded),
        "all_keys_changed": all(a[1] != b[1] and b[1] is not None
                                for a, b in zip(keys_before, keys_after)),
        "pick_rows_identical": all(a[0] == b[0] for a, b in zip(keys_before, keys_after)),
        "accuracy_before": round(sum(p == k for p, k in keys_before if p is not None)
                                 / max(1, sum(p is not None for p, _ in keys_before)), 4),
        "accuracy_after": round(sum(p == k for (p, _), (_, k) in zip(keys_before, keys_after)
                                    if p is not None)
                                / max(1, sum(p is not None for p, _ in keys_before)), 4),
    }
    emitted_text = [r for r in graded if r["text"] is not None]
    changed = sum(1 for r in emitted_text if r["text_on_permuted_text"] != r["text"])
    controls = {
        "subject_text_permutation": {
            "emits": len(emitted_text), "picks_changed": changed,
            "rate": round(changed / len(emitted_text), 4) if emitted_text else None,
            "required": ">= 0.80"},
        "gold_key_permutation": gold_perm,
        "negated_teaching_emits": sum(r["text_on_negated_teaching"] is not None
                                      for r in graded),
        "empty_subject_text_emits": sum(r["text_on_empty_subject_text"] is not None
                                        for r in graded),
        "disjoint_option_emits": sum(r["text_on_disjoint_options"] is not None
                                     for r in graded),
        "ast_anti_leakage": {"ok": ast_ok, "problems": ast_problems},
        "bank_validate_errors": len(bank_errors),
    }

    scores = {"xsrc_text_precision_pct": 100 * (arms["text"]["all"]["precision"] or 0),
              "xsrc_demo_precision_pct": 100 * (arms["demo"]["all"]["precision"] or 0),
              "xsrc_text_coverage_pct": 100 * (arms["text"]["all"]["coverage"] or 0)}
    g6 = front_gates.gate_saturation(scores, mode="offline")

    handcheck = [{
        "item_id": r["item_id"], "subject": r["subject"],
        "subject_text": r["subject_text"], "text_types": r["text_types"],
        "candidates": r["candidates"], "pick": r["text"], "key": r["key"],
        "correct": r["text"] == r["key"],
    } for r in emitted_text[:12]]

    t_all = arms["text"]["all"]
    d_all = arms["demo"]["all"]
    positive = bool(
        ast_ok and not bank_errors and t_all["emit"] >= 10
        and (t_all["precision"] or 0) >= 0.60
        and (t_all["precision"] or 0) - (d_all["precision"] or 0) >= 0.15
        and controls["subject_text_permutation"]["rate"] is not None
        and controls["subject_text_permutation"]["rate"] >= 0.80
        and gold_perm["pick_rows_identical"] and gold_perm["all_keys_changed"]
        and gold_perm["accuracy_after"] <= 0.40
        and controls["negated_teaching_emits"] == 0
        and controls["empty_subject_text_emits"] == 0
        and controls["disjoint_option_emits"] == 0)

    out = {
        "hypothesis": "H_lang: subject-specific real-language evidence + demo-learned "
                      "schema emits the type-sharing candidate above demo-only lookup",
        "frozen_constants": {"MIN_DEMO_SUPPORT": MIN_DEMO_SUPPORT,
                             "MIN_CUE_SUPPORT": MIN_CUE_SUPPORT,
                             "PERMUTE_OFFSET": PERMUTE_OFFSET,
                             "HASH_PREFIX": HASH_PREFIX,
                             "STOPWORDS": sorted(STOPWORDS)},
        "module_shas": module_shas,
        "scorer_sources": scorer_sources,
        "n_cases": len(rows), "n_graded": len(graded), "n_primary": len(primary),
        "arms": arms, "controls": controls,
        "g6": {"status": g6.status, "detail": g6.detail, "flags": g6.flagged_fronts},
        "handcheck": handcheck,
        "verdict": "POSITIVE" if positive else "NEGATIVE / NO-GO",
    }
    results_path = os.path.join(ROOT, "scratch", "cross_source_fast_mapping_results.json")
    with open(results_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("module SHAs identical to main:", all(m["identical"] for m in module_shas.values()))
    print("AST anti-leakage:", ast_ok, ast_problems)
    print("cases", len(rows), "graded", len(graded), "primary", len(primary),
          "bank_errors", len(bank_errors))
    print("TEXT  all:", arms["text"]["all"], " primary:", arms["text"]["primary"])
    print("DEMO  all:", arms["demo"]["all"], " primary:", arms["demo"]["primary"])
    print("controls:", json.dumps(controls, sort_keys=True)[:1200])
    print("G6:", g6.status, g6.detail)
    print("hand-check rows:")
    for r in handcheck:
        print(" ", r["item_id"], r["subject"], "text=", (r["subject_text"] or "")[:90],
              "cues=", r["text_types"][:4], "opts=", r["candidates"],
              "pick=", r["pick"], "key=", r["key"], "OK" if r["correct"] else "WRONG")
    print("VERDICT:", out["verdict"])
    print("results:", results_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
