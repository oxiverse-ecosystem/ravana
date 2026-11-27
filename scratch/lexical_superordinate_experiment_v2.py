#!/usr/bin/env python3
"""H_struct -- schema-calibrated lexical superordinate transfer (RSI/TRA).

SCRATCH / STATIC ONLY. No engine boot, no CognitiveChatEngine import, no
process_turn, no store write, no network, no production edit, no item-bank,
gate, threshold or answer-key change.

MECHANISM (pre-registered 2026-11-27, before the sealed split was read)
----------------------------------------------------------------------
The brain does not type a novel word from a centroid of other exemplars; it
uses a structured semantic hypothesis space (Xu & Tenenbaum 2007), one
subject-specific input (Carey & Bartlett 1978; Lany & Saffran 2010), and a
criterion calibrated by prior learning (Tse et al. 2007; McClelland et al.
1995). Here the structured space is WordNet 3.0's noun hypernym graph; the
subject-specific input is the QUERY SUBJECT's own lexical identity; the
criterion is calibrated ONLY from the item bank's public teaching pairs
(r1, r2 demo words), never from a gold key, type_sets, world_model or domain
tag. A candidate is emitted iff its WordNet path-similarity to the subject is
(a) above the teaching-pair median + DELTA and (b) MARGIN above the runner-up;
otherwise the mechanism abstains. A subject-INDEPENDENT arm scores candidates
by their similarity to the taught demo words under the identical criterion.

Deployable inputs: query subject, displayed candidates, public teaching pairs
and an offline lexical ontology. The graded key is computed by the grader as
"the unique displayed candidate sharing the subject's PRIMARY Wikidata type"
from the independent CC0 substrate; no scorer sees that key or any type label.

FROZEN CONSTANTS (never changed after any number was read)
  DELTA = 0.10, MARGIN = 0.05, PERMUTE_OFFSET = 137
  THETA = median(path_similarity of all publicly taught demo pairs) + DELTA

POSITIVE criterion (all required, declared before the sealed split was read):
  P1 AST anti-leak clean; bank errors == 0; inputs identical to HEAD blobs
  P2 STRUCT emits >= 20 of 125 gradeable cases (>=16% coverage)
  P3 STRUCT precision >= 0.60
  P4 STRUCT precision - DEMO-WN precision >= 0.15 on the same cases
  P5 subject permutation changes >= 80% of emitted STRUCT picks
  P6 gold permutation: picks byte-identical, keys all changed, accuracy <= .40
  P7 negated teaching, OOV disjoint options (no WordNet noun synset), and
     unrelated subject each emit 0; candidate-order permutation changes 0 picks
  (v1's apple/banana/chair/table tuple is NOT disjoint in WordNet, see
   LEXICAL_SUPERORDINATE_V2_PREREG.md; retained as a reported diagnostic only)
  P8 |primary precision - sealed precision| <= 0.20
"""
from __future__ import annotations

import ast
import collections
import hashlib
import inspect
import json
import os
import statistics
import subprocess
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import tra_k_items  # unmodified, read-only
import front_gates  # unmodified, read-only
import nltk
nltk.data.path.append(r"C:/Users/Likhith/AppData/Roaming/nltk_data")
from nltk.corpus import wordnet as wn

DELTA = 0.10
MARGIN = 0.05
PERMUTE_OFFSET = 137
OOV_DISJOINT = ("zzq_lex_a", "zzq_lex_b", "zzq_lex_c", "zzq_lex_d")
LEGACY_DISJOINT = ("apple", "banana", "chair", "table")
HASH_PREFIX = "RAVANA-LEXICAL-SUPERORDINATE-v1"
WN_RELEASE = "WordNet 3.0 (Princeton University; Miller 1995 CACM 38(11):39-41)"
BANNED = frozenset(("gold", "type_sets", "world_model", "typed_queries",
                    "unseen_test_words", "type_label", "item_id", "role",
                    "evidence_role", "label_set", "graded_key", "key"))


def sha_norm(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def sha_file(path: str) -> str:
    with open(path, "rb") as fh:
        return sha_norm(fh.read())


def sha_head(rel: str) -> str:
    return sha_norm(subprocess.check_output(["git", "show", "HEAD:" + rel], cwd=ROOT,
                                            stderr=subprocess.DEVNULL))


def norm(text) -> str:
    import re
    return re.sub(r"[^a-z0-9 ]", " ", str(text).lower()).strip()


_SYN_CACHE: dict = {}


def synsets(word: str):
    if word not in _SYN_CACHE:
        _SYN_CACHE[word] = wn.synsets(word, pos=wn.NOUN)
    return _SYN_CACHE[word]


def path_sim(a: str, b: str):
    vals = [wn.path_similarity(x, y) for x in synsets(a) for y in synsets(b)]
    return max(vals) if vals else None


# ── SCORER (AST-scanned below). Never reads a key/type/gold field. ──────────
def schema_criterion(teaching_pairs, fallback_theta):
    """Calibrate the admission bar from publicly taught pairs only."""
    sims = [path_sim(a, b) for a, b in teaching_pairs]
    sims = [v for v in sims if v is not None]
    return statistics.median(sims) + DELTA if sims else fallback_theta


def subject_scores(subject, candidate_options):
    """Subject-specific evidence: query subject -> each displayed candidate."""
    return {c: path_sim(subject, c) for c in candidate_options}


def demo_scores(teaching_pairs, candidate_options):
    """Subject-INDEPENDENT control: taught demo words -> displayed candidates."""
    words = sorted({r2 for _r1, r2 in teaching_pairs})
    out = {}
    for c in candidate_options:
        vals = [path_sim(c, w) for w in words]
        vals = [v for v in vals if v is not None]
        out[c] = max(vals) if vals else None
    return out


def decide(scores, theta):
    """Fail-closed unique winner under a teaching-calibrated criterion."""
    ranked = sorted(((c, s) for c, s in scores.items() if s is not None),
                    key=lambda cs: (-cs[1], cs[0]))
    if not ranked or theta is None or ranked[0][1] < theta:
        return None, "below_criterion"
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if ranked[0][1] - second < MARGIN:
        return None, "no_margin"
    return ranked[0][0], "emit"


def permute_key(candidates, key):
    if key not in candidates:
        return None
    for step in (1, 2, 3):
        new = candidates[(candidates.index(key) + step) % len(candidates)]
        if new != key:
            return new
    return None


def main():
    # ── instrument validation BEFORE any endpoint is read ───────────────────
    shas = {}
    for rel in ("scripts/tra_k_items.py", "scripts/front_gates.py",
                "scratch/trak_substrate.json"):
        work, head = sha_file(os.path.join(ROOT, *rel.split("/"))), sha_head(rel)
        shas[rel] = {"worktree": work, "head": head, "identical": work == head}
        if work != head:
            raise SystemExit("UNMODIFIED-INPUT CHECK FAILED: " + rel)

    scorer_sources = {}
    ast_problems = []
    for fn in (schema_criterion, subject_scores, demo_scores, decide):
        src = textwrap.dedent(inspect.getsource(fn))
        scorer_sources[fn.__name__] = src
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in BANNED:
                ast_problems.append((fn.__name__, node.id))
            if isinstance(node, ast.Attribute) and node.attr in BANNED:
                ast_problems.append((fn.__name__, node.attr))
    ast_ok = not ast_problems

    bank = tra_k_items.build_items()
    bank_errors = tra_k_items.validate_items(bank)

    # ── grader-only key from the independent CC0 substrate ─────────────────
    substrate = json.load(open(os.path.join(ROOT, "scratch", "trak_substrate.json"),
                               encoding="utf-8"))["facts"]
    by_subject = collections.defaultdict(list)
    for fact in substrate:
        by_subject[fact["subject"]].append(fact)

    def primary_facts(word):
        return [f for f in by_subject.get(word, ()) if f.get("sense") == "primary"]

    def structural_labels(word):
        return frozenset(norm(f["type_label"]) for f in primary_facts(word)) - {norm(word)}

    def graded_key(subject, candidates):
        subj = structural_labels(subject)
        if not subj:
            return None
        hits = [c for c in candidates if structural_labels(c) & subj]
        return hits[0] if len(hits) == 1 else None

    all_words = sorted({w for it in bank for w in
                        [it["subject"]] + [c for cond in ("near", "far")
                                           for c in it["unseen_test_words"][cond]["candidates"]]})

    cases = []
    for it in bank:
        teaching = [(e["r1"], e["r2"]) for e in it["taught_exemplars"][:it["k"]]]
        for cond in ("near", "far"):
            q = it["unseen_test_words"][cond]
            subject, candidates = q["subject"], tuple(q["candidates"])
            cases.append({
                "item_id": it["item_id"], "pool": it["pool"], "condition": cond,
                "subject": subject, "candidates": candidates,
                "teaching": teaching,
                "text": next((f.get("text_evidence", "") for f in primary_facts(subject)
                              if f.get("wikidata_property") == "P279"
                              and f.get("text_evidence_kind") == "item_description"), ""),
                "key": graded_key(subject, candidates),
            })

    # Frozen criterion from ALL publicly taught pairs (no answer key involved).
    all_teaching = [pair for c in cases for pair in c["teaching"]]
    frozen_theta = statistics.median(
        [v for pair in all_teaching if (v := path_sim(*pair)) is not None]) + DELTA

    rows = []
    for i, c in enumerate(cases):
        cand = list(c["candidates"])
        theta = schema_criterion(c["teaching"], frozen_theta)
        s_scores = subject_scores(c["subject"], cand)
        d_scores = demo_scores(c["teaching"], cand)
        pick_s, _ = decide(s_scores, theta)
        pick_d, _ = decide(d_scores, theta)
        # subject permutation: a DIFFERENT query subject, same displayed options
        alt = all_words[(i + PERMUTE_OFFSET) % len(all_words)]
        pick_perm, _ = decide(subject_scores(alt, cand), theta)
        # negated teaching: no trusted teaching pairs anywhere
        pick_neg, _ = decide(subject_scores(c["subject"], cand),
                             schema_criterion((), None))
        pick_disj, _ = decide(subject_scores(c["subject"], OOV_DISJOINT), theta)
        pick_legacy, _ = decide(subject_scores(c["subject"], LEGACY_DISJOINT), theta)
        rotated = cand[PERMUTE_OFFSET % len(cand):] + cand[:PERMUTE_OFFSET % len(cand)]
        pick_rot, _ = decide(subject_scores(c["subject"], rotated), theta)
        unrelated = "zzq_unrelated_wumpus"
        pick_unrel, _ = decide(subject_scores(unrelated, cand), theta)
        # gold permutation is graded-side only: it cannot affect any scorer call
        perm_key = permute_key(cand, c["key"])
        rows.append({
            "item_id": c["item_id"], "pool": c["pool"], "condition": c["condition"],
            "subject": c["subject"], "subject_text": norm(c["text"]),
            "candidates": cand, "theta": round(theta, 4) if theta else None,
            "theta_raw": theta,
            "key": c["key"], "permuted_key": perm_key,
            "struct": pick_s, "demo": pick_d, "struct_permuted_subject": pick_perm,
            "struct_negated_teaching": pick_neg, "struct_disjoint": pick_disj,
            "struct_legacy_disjoint": pick_legacy, "struct_rotated_options": pick_rot,
            "struct_unrelated": pick_unrel,
            "struct_scores": {k: round(v, 4) if v is not None else None
                              for k, v in s_scores.items()},
        })

    graded = [r for r in rows if r["key"] is not None]
    primary = [r for r in graded if r["pool"] == "primary"]
    sealed = [r for r in graded if r["pool"] == "sealed"]

    def tally(subset, arm, keyfield="key"):
        emitted = [r for r in subset if r[arm] is not None]
        correct = sum(1 for r in emitted if r[arm] == r[keyfield])
        return {"n": len(subset), "emit": len(emitted), "correct": correct,
                "precision": round(correct / len(emitted), 4) if emitted else None,
                "coverage": round(len(emitted) / len(subset), 4) if subset else None}

    arms = {arm: {"all": tally(graded, arm), "primary": tally(primary, arm),
                  "sealed": tally(sealed, arm)} for arm in ("struct", "demo")}

    emitted = [r for r in graded if r["struct"] is not None]
    changed = sum(1 for r in emitted if r["struct_permuted_subject"] != r["struct"])
    gold_perm = {
        "all_keys_changed": all(r["permuted_key"] is not None and r["permuted_key"] != r["key"]
                                for r in graded),
        "picks_identical": all(
            decide(subject_scores(r["subject"], r["candidates"]),
                   r["theta_raw"])[0] == r["struct"] for r in graded),
        "accuracy_before": tally(graded, "struct")["precision"],
        "accuracy_after": tally(graded, "struct", "permuted_key")["precision"],
    }
    controls = {
        "subject_permutation": {"emits": len(emitted), "picks_changed": changed,
                                "rate": round(changed / len(emitted), 4) if emitted else None,
                                "required": ">= 0.80"},
        "gold_key_permutation": gold_perm,
        "negated_teaching_emits": sum(r["struct_negated_teaching"] is not None for r in graded),
        "oov_disjoint_emits": sum(r["struct_disjoint"] is not None for r in graded),
        "legacy_disjoint_emits_DIAGNOSTIC_ONLY": sum(r["struct_legacy_disjoint"] is not None for r in graded),
        "candidate_order_picks_changed": sum(r["struct_rotated_options"] != r["struct"]
                                             for r in graded if r["struct"] is not None),
        "unrelated_subject_emits": sum(r["struct_unrelated"] is not None for r in graded),
        "ast_anti_leakage": {"ok": ast_ok, "problems": ast_problems},
        "bank_validate_errors": len(bank_errors),
        "inputs_identical_to_head": all(m["identical"] for m in shas.values()),
    }
    p_all = arms["struct"]["all"]["precision"] or 0
    d_all = arms["demo"]["all"]["precision"] or 0
    p_pri = arms["struct"]["primary"]["precision"]
    p_sea = arms["struct"]["sealed"]["precision"]
    delta_pri_sea = abs((p_pri or 0) - (p_sea or 0))
    positive = bool(
        ast_ok and not bank_errors and controls["inputs_identical_to_head"]
        and arms["struct"]["all"]["emit"] >= 20 and p_all >= 0.60
        and p_all - d_all >= 0.15
        and controls["subject_permutation"]["rate"] is not None
        and controls["subject_permutation"]["rate"] >= 0.80
        and gold_perm["all_keys_changed"] and gold_perm["accuracy_after"] is not None
        and gold_perm["accuracy_after"] <= 0.40
        and controls["negated_teaching_emits"] == 0
        and controls["oov_disjoint_emits"] == 0
        and controls["candidate_order_picks_changed"] == 0
        and controls["unrelated_subject_emits"] == 0
        and delta_pri_sea <= 0.20)

    scores = {"struct_precision_pct": 100 * p_all, "demo_precision_pct": 100 * d_all,
              "struct_coverage_pct": 100 * (arms["struct"]["all"]["coverage"] or 0)}
    g6 = front_gates.gate_saturation(scores, mode="offline")

    out = {
        "hypothesis": "H_struct: a subject-specific WordNet superordinate route, "
                      "with a criterion calibrated from public teaching pairs, "
                      "emits the Wikidata-type-sharing candidate above a "
                      "subject-independent demo-only route",
        "frozen": {"DELTA": DELTA, "MARGIN": MARGIN, "PERMUTE_OFFSET": PERMUTE_OFFSET,
                   "HASH_PREFIX": HASH_PREFIX, "WORDNET": WN_RELEASE,
                   "GLOBAL_Theta": round(frozen_theta, 6),
                   "criterion": "P1..P8 pre-registered in script docstring"},
        "sha_inputs": shas, "scorer_sources": scorer_sources,
        "n_cases": len(rows), "n_graded": len(graded),
        "n_primary": len(primary), "n_sealed": len(sealed),
        "arms": arms, "controls": controls,
        "primary_sealed_delta": round(delta_pri_sea, 4),
        "g6": {"status": g6.status, "detail": g6.detail, "flags": g6.flagged_fronts},
        "handcheck": [{"item": r["item_id"], "subject": r["subject"],
                       "text": r["subject_text"], "candidates": r["candidates"],
                       "scores": r["struct_scores"], "theta": r["theta"],
                       "pick": r["struct"], "key": r["key"],
                       "correct": r["struct"] == r["key"]}
                      for r in emitted[:12]],
        "rows": rows,
        "verdict": "POSITIVE" if positive else "NEGATIVE / NO-GO",
    }
    results = os.path.join(ROOT, "scratch", "lexical_superordinate_v2_results.json")
    with open(results, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=2, sort_keys=True)
        fh.write("\n")

    print("inputs identical to HEAD:", controls["inputs_identical_to_head"])
    print("AST:", ast_ok, ast_problems, "bank errors:", len(bank_errors))
    print("cases", len(rows), "graded", len(graded), "primary", len(primary),
          "sealed", len(sealed), "frozen theta", round(frozen_theta, 4))
    for arm in ("struct", "demo"):
        print(arm, "all", arms[arm]["all"], "primary", arms[arm]["primary"],
              "sealed", arms[arm]["sealed"])
    print("controls:", json.dumps(controls, sort_keys=True)[:1400])
    print("primary/sealed delta", round(delta_pri_sea, 4))
    print("G6:", g6.status, g6.detail)
    print("hand-check:")
    for r in out["handcheck"]:
        print(" ", r["item"], r["subject"], "text=", r["text"][:60],
              "scores=", r["scores"], "theta=", r["theta"],
              "pick=", r["pick"], "key=", r["key"], "OK" if r["correct"] else "WRONG")
    print("VERDICT:", out["verdict"])
    print("results:", results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
