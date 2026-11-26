#!/usr/bin/env python3
"""WFA free-text / forced-choice articulation: boot-free candidate decision routines.

NO CognitiveChatEngine boot, NO weight store, NO production edit, NO change to any
gate, threshold, item, prompt or grading rule.  Every production function used
here is imported UNMODIFIED from this worktree (identical to main @ ac4ec88).

Question
--------
The echo-immune WFA run measured pd_unparseable_rate = 1.00 and
forced_choice_explicit_rate = 0.00 with R = 0 on both pools: the engine emits no
explicit decision token.  The g05 claim-ID anchor fixes provenance RETENTION and
the source-evaluation module supplies a rule table for the STAGED internal-source
arm, but neither puts a decision at the FREE-TEXT / forced-choice articulation
surface.  This probe compares boot-free candidate articulation routines on the
recorded failing rows, using the production parsers, the production evaluator
and the production process-dissociation scorer.

Inputs (read-only)
------------------
  --summary  scratch/learn5_wfa_summary.json from the recorded failing run
             (pd_audit: 192 rows with the engine's raw free-text / forced-choice
             output).  Default: ../ravana/scratch/learn5_wfa_summary.json
Items come from the production build_items(); the frozen manifest is verified.

Candidate contract (anti-tautology)
-----------------------------------
A candidate receives ONLY (claim, arm, raw_engine_text, engine_graph, meta).
It never receives the item's kind/source_status/expected_answer/source_id or
any expected token.  ``meta`` is nevertheless passed to every candidate so the
anti-tautology control can mutate it and assert the decision does not move.
The experimental condition is staged into an in-memory ConceptGraph by the
FIXTURE BUILDER (the analogue of the harness's teach turn), using production
_write_asserted_wfa_fact / apply_source_invalidation.  All decision evidence
therefore comes from the engine's own stored provenance or its emitted text.

Pre-registered metrics and selection rule (fixed before any candidate was run)
-----------------------------------------------------------------------------
M1  explicit_rate per arm, fraction of rows with an explicit Y/N/W(/M) token.
    Target >= 0.90 on exclusion, invalidation and forced_choice.
M2  process dissociation via the UNMODIFIED w.process_dissociation():
    R >= 0.25, exclusion_margin >= 0.10, 0 < R < 1, withdrawal >= 0.50, on
    BOTH pools.  These are the harness's own unchanged bounds; this script
    reports them, it does not set or tune them.
M3  design accuracy: token equals the arm's design-derived expectation
    (inclusion endorses familiar claims; exclusion/forced-choice endorse only
    credible provenance; invalidation withdraws after a prior endorsement).
    Target >= 0.90 both pools.  Ground truth is the experimental design, not
    an item answer key, and is used only for scoring.
M4  anti-tautology: mutating meta (expected_answer, source_status, source_id,
    reliability) leaves (decision, fingerprint) unchanged.  Required 1.00.
M5  no-shortcut: empty graph + familiarity=1.0 -> ABSTAIN.  Required.
M6  stored-provenance mutation: curated -> co_occurrence flips a credible
    ENDORSE to ABSTAIN.  Required for graph-reading candidates.
M7  source-id-string mutation: renaming the origin string without touching
    edge_kind/source_status leaves the decision unchanged.  Required.
M8  echo hygiene: production text parsers yield ZERO decisions on the recorded
    raw texts (re-verifies the recorded 1.00 unparseable rate).
M9  determinism: 3 repeats byte-identical.
Selection: among candidates passing M4-M9, maximise (M1 on forced_choice,
M2 R on both pools, M3), then prefer FEWER novel primitives / more production
function reuse.  A candidate that fails a control is reported, never selected.

Usage
-----
  .venv/Scripts/python.exe scratch/wfa_articulation_decision.py \
      --summary ../ravana/scratch/learn5_wfa_summary.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (os.path.join(PROJ, "scripts"),
           os.path.join(PROJ, "ravana", "src"), PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_learn5_wfa as w  # noqa: E402  (production, unmodified)
from ravana.chat.source_evaluation import (  # noqa: E402
    ClaimContext, Decision, SourceStatus, evaluate_source)
from ravana_ml.graph import ConceptGraph  # noqa: E402

MANIFEST = "119b9d91fe43a06087d001b1fb02a554edd4d08d665b99041528d93533818164"
ARMS = ("inclusion", "exclusion", "invalidation", "forced_choice")
TEXT_KEY = {
    "inclusion": "pd_inclusion_text",
    "exclusion": "pd_exclusion_text",
    "invalidation": "inv_after",
    "forced_choice": "pd_forced_choice_text",
}
# Design expectations, by arm and the row's experimental condition.
EXPECTED = {
    "inclusion": {"gap": "Y", "decoy": "Y"},
    "exclusion": {"gap": "Y", "decoy": "N"},
    "invalidation": {"gap": "W", "decoy": "W"},
    "forced_choice": {"gap": "Y", "decoy": "N"},
}
FORCE_TOK = {"endorse": "Y", "abstain": "N", "withdraw": "W"}


# --------------------------------------------------------------------------
# boot-free in-memory graph holder
# --------------------------------------------------------------------------
class GraphEngine:
    """Real ConceptGraph, no boot, no store, no engine methods."""

    def __init__(self):
        self.graph = ConceptGraph(dim=8)


class Fixture:
    __slots__ = ("claim", "arm", "raw_text", "engine", "sid", "meta",
                 "kind", "pool", "item_id", "prior")

    def __init__(self, claim, arm, raw_text, engine, sid, meta, kind, pool, item_id):
        self.claim = claim
        self.arm = arm
        self.raw_text = raw_text
        self.engine = engine
        self.sid = sid
        self.meta = meta           # deliberately available but never read
        self.kind = kind           # never read by candidates
        self.pool = pool           # never read by candidates
        self.item_id = item_id     # never read by candidates


class Result:
    __slots__ = ("token", "rule", "source", "fingerprint", "confidence", "steps")

    def __init__(self, token, rule="", source="", fingerprint="",
                 confidence=None, steps=0):
        self.token = token
        self.rule = rule
        self.source = source
        self.fingerprint = fingerprint
        self.confidence = confidence
        self.steps = steps

    def as_dict(self):
        return {"token": self.token, "rule": self.rule, "source": self.source,
                "fingerprint": self.fingerprint, "confidence": self.confidence,
                "steps": self.steps}


def build_fixture(item, arm, row):
    """Harness-side staging: the analogue of the teach turn, boot-free.

    Uses ONLY production functions.  The source id is derived from the claim
    and the experimental condition (an independent variable), never from the
    item's expected answer.  The candidate never sees `kind` or `meta`.
    """
    claim = item["claim"]
    kind = row["kind"]
    eng = GraphEngine()
    cred = w.is_source_id(claim, "credible")
    disc = w.is_source_id(claim, "discredited")
    prior = None
    sid = cred
    if arm == "inclusion":
        # Familiarity arm: the claim was taught from its actual source; no
        # invalidation cue is delivered in this arm.
        if kind == "decoy":
            sid = disc
        w._write_asserted_wfa_fact(eng, claim, sid)
    elif arm == "exclusion":
        if kind == "gap":
            w._write_asserted_wfa_fact(eng, claim, cred)
        else:
            w._write_asserted_wfa_fact(eng, claim, disc)
            w.apply_source_invalidation(eng, disc)
            sid = disc
    elif arm == "invalidation":
        w._write_asserted_wfa_fact(eng, claim, cred)
        n = w.apply_source_invalidation(eng, cred)
        assert n >= 1, ("invalidation tagged no stored edge", item["id"])
        prior = Decision.ENDORSE   # engine-audit analogue, harness-side only
    elif arm == "forced_choice":
        if kind == "gap":
            w._write_asserted_wfa_fact(eng, claim, cred)
        else:
            w._write_asserted_wfa_fact(eng, claim, disc)
            w.apply_source_invalidation(eng, disc)
            sid = disc
    else:
        raise ValueError(arm)
    fx = Fixture(claim, arm, (row.get(TEXT_KEY[arm]) or ""), eng, sid,
                 dict(item), kind, row["pool"], item["id"])
    fx.prior = prior              # harness-side only; candidates do not read it
    return fx


# --------------------------------------------------------------------------
# Candidates -- each receives (fx) only.  None reads fx.meta/kind/pool/item_id.
# --------------------------------------------------------------------------
def c0_production_text_parser(fx):
    """T0: the SHIPPED scoring path, replayed on the recorded raw text."""
    if fx.arm == "forced_choice":
        p = w.parse_forced_choice(fx.raw_text)
        tok = FORCE_TOK.get(p["decision"])
        return Result(tok, "T0_forced_choice", "text")
    kind = "withdrawal" if fx.arm == "invalidation" else "endorsement"
    p = w.parse_decision(fx.raw_text, kind)
    if fx.arm == "invalidation":
        tok = "W" if p["decision"] is True else ("M" if p["decision"] is False else None)
    else:
        tok = "Y" if p["decision"] is True else ("N" if p["decision"] is False else None)
    return Result(tok, "T0_parse_decision", "text")


def _staged(fx, prior=None):
    return w.context_from_graph_provenance(fx.engine, fx.claim, source_id=fx.sid,
                                           prior_decision=prior)


def c1_production_evaluator(fx):
    """T1: production staged context + UNMODIFIED evaluate_source + token."""
    prior = fx.prior if fx.arm == "invalidation" else None
    ctx, audit = _staged(fx, prior=prior)
    d = evaluate_source(fx.claim, ctx)
    return Result(d.token, d.rule, "graph_provenance", audit["fingerprint"])


# Pre-registered DDM-style constants (chosen BEFORE any candidate was run).
DDM_BOUND = 1.0          # evidence units required to cross a boundary
DDM_CONF_FLOOR = 0.5     # metacognitive veto: margin/bound must clear this


def c2_accumulator_with_confidence(fx):
    """T2: evidence accumulation to criterion + metacognitive confidence veto.

    EXTRAPOLATION (labelled): evidence increments are discrete stored-provenance
    signals, not noisy perceptual samples.  The accumulation-to-bound and a
    second-stage confidence readout are the brain-inspired parts
    (Ratcliff & McKoon 2008; Gold & Shadlen 2007; Yeung & Summerfield 2012);
    applying them to symbolic provenance is an analogy, not a neural claim.

    Every increment is read from the production provenance census.  The final
    COMMIT is still delegated to the UNMODIFIED evaluate_source where possible.
    """
    rows = w.graph_provenance_for_claim(fx.engine, fx.claim, fx.sid)
    v = 0.0
    steps = 0
    # Discredited evidence first; a prior endorsement makes this a withdrawal.
    for r in sorted(rows, key=lambda m: m["source"]):
        status = str(r.get("source_status") or "").lower()
        kind = str(r.get("edge_kind") or "")
        if status in w._IS_DISCREDITED_WORDS:
            v -= 1.0
        elif kind in w._IS_VERIFIABLE_KINDS and r.get("source"):
            v += 1.0
        steps += 1
        if abs(v) >= DDM_BOUND:
            break
    if abs(v) < DDM_BOUND:
        # Fail closed but still articulate: an articulation routine must emit a
        # token even when no boundary is crossed (UNKNOWN is an input state).
        return Result("N", "T2_NO_BOUNDARY_FAIL_CLOSED", "graph_provenance",
                      confidence=0.0, steps=steps)
    conf = abs(v) / DDM_BOUND
    if conf < DDM_CONF_FLOOR:
        return Result("N", "T2_CONFIDENCE_VETO", "graph_provenance",
                      confidence=conf, steps=steps)
    prior = fx.prior if fx.arm == "invalidation" else None
    ctx, audit = _staged(fx, prior=prior)
    d = evaluate_source(fx.claim, ctx)
    rule = "T2_BOUND_%s|%s" % ("POS" if v > 0 else "NEG", d.rule)
    return Result(d.token, rule, "graph_provenance", audit["fingerprint"],
                  confidence=conf, steps=steps)


def c3_hybrid_verify(fx):
    """T3: production text decision if genuinely present, else production evaluator.

    Brain analogue: commit before articulation, then verify the emitted token;
    if articulation carries no decision, the committed decision supplies it.
    """
    t = c0_production_text_parser(fx)
    if t.token is not None:
        t.rule = "T3_ENGINE_TEXT|" + t.rule
        return t
    p = c1_production_evaluator(fx)
    p.rule = "T3_PROVENANCE_FALLBACK|" + p.rule
    return p


# Text-only accumulator: a falsifier control, not a deployable candidate.
_ENDORSE_W = re.compile(r"\b(endorse|yes|confirmed|correct)\b", re.I)
_ABSTAIN_W = re.compile(r"\b(abstain|no|unknown|not sure|unreliable)\b", re.I)
_WITHDRAW_W = re.compile(r"\b(withdraw|retract|i was wrong)\b", re.I)


def c4_text_only_accumulator(fx):
    """T4: DDM-style accumulation over the engine's RAW TEXT evidence only.

    Falsifier: if the decision variable were recoverable from articulation,
    this must produce explicit decisions.  The recorded run predicts it cannot.
    """
    text = w.strip_echoes(fx.raw_text)
    v = 0.0
    steps = 0
    for tok in re.findall(r"[A-Za-z]+", text):
        low = tok.lower()
        if fx.arm == "invalidation" and _WITHDRAW_W.match(low):
            v -= 1.0
        elif _ABSTAIN_W.match(low):
            v -= 1.0
        elif _ENDORSE_W.match(low):
            v += 1.0
        steps += 1
        if abs(v) >= DDM_BOUND:
            break
    if abs(v) < DDM_BOUND:
        return Result(None, "T4_NO_BOUNDARY", "text_only", steps=steps)
    return Result("Y" if v > 0 else "N", "T4_TEXT_BOUNDARY", "text_only",
                  confidence=abs(v) / DDM_BOUND, steps=steps)


CANDIDATES = {
    "T0_production_text_parser": c0_production_text_parser,
    "T1_production_evaluator": c1_production_evaluator,
    "T2_accumulator_confidence": c2_accumulator_with_confidence,
    "T3_hybrid_verify": c3_hybrid_verify,
    "T4_text_only_accumulator_falsifier": c4_text_only_accumulator,
}


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _rate(vals):
    vals = list(vals)
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None


def score_candidate(name, fn, rows, items):
    per_arm = {}
    results = {}
    for arm in ARMS:
        arm_rows = [r for r in rows if r.get(TEXT_KEY[arm]) is not None]
        correct, explicit = [], []
        for r in arm_rows:
            fx = build_fixture(items[r["id"]], arm, r)
            res = fn(fx)
            results[(arm, r["id"])] = res
            explicit.append(res.token is not None)
            correct.append(res.token == EXPECTED[arm][r["kind"]])
        per_arm[arm] = {
            "n": len(arm_rows),
            "explicit_rate": _rate(explicit),
            "design_accuracy": _rate(correct),
        }

    # Production process-dissociation scorer on candidate-owned decisions.
    cand_records = []
    for r in rows:
        def _tok(arm, rid):
            res = results.get((arm, rid))
            return res.token if res is not None else None

        c = dict(r)
        c["pd_inclusion_endorse"] = _tok("inclusion", r["id"]) == "Y"
        c["pd_exclusion_endorse"] = _tok("exclusion", r["id"]) == "Y"
        c["inv_withdrawn"] = _tok("invalidation", r["id"]) == "W"
        fct = _tok("forced_choice", r["id"])
        c["pd_forced_choice"] = (fct or "no_answer").lower()
        if c["pd_forced_choice"] == "y":
            c["pd_forced_choice"] = "endorse"
        elif c["pd_forced_choice"] == "n":
            c["pd_forced_choice"] = "abstain"
        elif c["pd_forced_choice"] == "w":
            c["pd_forced_choice"] = "withdraw"
        cand_records.append(c)
    pd = {p: w.process_dissociation(cand_records, p) for p in ("primary", "sealed")}

    fc_res = [results[("forced_choice", r["id"])] for r in rows
              if r.get("pd_forced_choice_text") is not None]
    out = {
        "candidate": name,
        "per_arm": per_arm,
        "forced_choice_explicit_rate": _rate(x.token is not None for x in fc_res),
        "pd_production_scorer": pd,
        "sample_decisions": {
            "%s/%s" % (a, i): results[(a, i)].as_dict()
            for (a, i) in list(results)[:8]
        },
    }
    return out, results, cand_records


def run_controls(fn, rows, items):
    ctrls = {}
    # M4 anti-tautology: mutate item metadata, decision must not move.
    r = next(x for x in rows if x["kind"] == "gap" and x["pool"] == "primary")
    it = dict(items[r["id"]])
    base_fx = build_fixture(it, "exclusion", r)
    base = fn(base_fx)
    mut = dict(it)
    mut.update({"expected_answer": "FORGED", "source_status": "credible",
                "source_id": "forged://answer", "reliability": 1.0})
    mut_fx = build_fixture(mut, "exclusion", r)
    mut_res = fn(mut_fx)
    ctrls["M4_item_metadata_anti_tautology"] = {
        "base": base.as_dict(), "mutated": mut_res.as_dict(),
        "unchanged": (base.token == mut_res.token and
                      base.fingerprint == mut_res.fingerprint)}

    # M5 no shortcut: empty graph + familiarity=1.0 must abstain.
    empty = GraphEngine()
    ctx = ClaimContext(source=SourceStatus(
        origin_id="", trust_class=__import__(
            "ravana.chat.source_evaluation", fromlist=["TrustClass"]).TrustClass.UNKNOWN,
        credibility_token=None), familiarity=1.0)
    d = evaluate_source("empty claim", ctx)
    ctrls["M5_familiarity_no_shortcut"] = {
        "decision": d.decision.value, "abstains": d.decision is Decision.ABSTAIN}

    # M6 stored-provenance mutation: curated -> co_occurrence flips ENDORSE.
    fx = build_fixture(it, "exclusion", r)
    before = fn(fx).token
    tok = w.wfa_claim_token(it["claim"])
    damaged = False
    for (s, t), edge in w._iter_graph_edges(fx.engine):
        labels = (w._graph_node_label(fx.engine.graph, s) + " " +
                  w._graph_node_label(fx.engine.graph, t))
        if tok in labels:
            edge.source_metadata["edge_kind"] = "co_occurrence"
            damaged = True
            break
    after = fn(fx).token
    ctrls["M6_stored_provenance_mutation"] = {
        "damaged": damaged, "before": before, "after": after,
        "flipped": damaged and before == "Y" and after == "N"}

    # M7 origin-string mutation: renew the origin string but keep the census
    # query key consistent, so the test isolates the STRING'S SEMANTICS from
    # stored edge_kind/source_status.  (Leaving fx.sid stale would only prove
    # that the source_id FILTER still works -- a probe artifact.)
    fx2 = build_fixture(it, "exclusion", r)
    b2 = fn(fx2).token
    mutated_origin = "wfa-internal://discredited/FORGED"
    for (s, t), edge in w._iter_graph_edges(fx2.engine):
        sm = edge.source_metadata
        if sm.get("edge_kind") == "curated":
            sm["source"] = mutated_origin
            sm["url"] = mutated_origin
            fx2.sid = mutated_origin
            break
    a2 = fn(fx2).token
    ctrls["M7_origin_string_mutation"] = {"before": b2, "after": a2,
                                          "unchanged": b2 == a2}

    # M8 echo hygiene on the recorded texts (production parsers).
    txt = [x.get(k) for x in rows for k in TEXT_KEY.values()]
    parsed = [w.parse_decision(t, "endorsement")["decision"] for t in txt if t]
    ctrls["M8_echo_hygiene"] = {
        "n_texts": len(txt),
        "explicit_decisions": sum(1 for d in parsed if d is not None),
        "clean": all(d is None for d in parsed)}

    # M9 determinism.
    reps = [fn(build_fixture(it, "exclusion", r)).as_dict() for _ in range(3)]
    ctrls["M9_determinism"] = {
        "identical": len({json.dumps(x, sort_keys=True) for x in reps}) == 1}
    return ctrls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(
        PROJ, "..", "ravana", "scratch", "learn5_wfa_summary.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "wfa_articulation_results.json"))
    args = ap.parse_args()

    with open(args.summary, encoding="utf-8") as fh:
        summary = json.load(fh)
    rows = summary["pd_audit"]
    items = {it["id"]: it for it in w.build_items()}
    assert summary["manifest_sha256"] == MANIFEST, "item seal mismatch"

    payload = {
        "probe": "WFA free-text/forced-choice articulation candidates (boot-free)",
        "no_engine_boot": True, "no_store_open": True, "no_production_edit": True,
        "manifest_sha256": summary["manifest_sha256"],
        "n_rows": len(rows),
        "recorded_baseline": {
            "pd_unparseable_rate": summary["pd_unparseable_rate"],
            "forced_choice_explicit_rate": summary["forced_choice_explicit_rate"],
            "R": {p: summary[p]["R"] for p in ("primary", "sealed")},
        },
        "preregistered": {
            "DDM_BOUND": DDM_BOUND, "DDM_CONF_FLOOR": DDM_CONF_FLOOR,
            "M1_explicit_min": 0.90, "M2_R_min": 0.25,
            "M2_margin_min": 0.10, "M2_withdrawal_min": 0.50,
            "M3_accuracy_min": 0.90, "M4-M9": "all required",
        },
        "candidates": {}, "controls": {}, "selection": None,
    }

    for name, fn in CANDIDATES.items():
        out, _res, _recs = score_candidate(name, fn, rows, items)
        out["controls"] = run_controls(fn, rows, items)
        payload["candidates"][name] = out

    # Selection per the pre-registered rule.
    def eligible(o):
        c = o["controls"]
        pd = o["pd_production_scorer"]
        m2 = all(pd[p]["R"] is not None and pd[p]["R"] >= 0.25
                 and pd[p]["exclusion_margin"] is not None
                 and pd[p]["exclusion_margin"] >= 0.10
                 and pd[p]["withdrawal_rate"] is not None
                 and pd[p]["withdrawal_rate"] >= 0.50
                 for p in ("primary", "sealed"))
        m1 = o["forced_choice_explicit_rate"] is not None and \
            o["forced_choice_explicit_rate"] >= 0.90
        m3 = all((v["design_accuracy"] or 0) >= 0.90
                 for v in o["per_arm"].values())
        return (m1 and m2 and m3
                and c["M4_item_metadata_anti_tautology"]["unchanged"]
                and c["M5_familiarity_no_shortcut"]["abstains"]
                and c["M6_stored_provenance_mutation"]["flipped"]
                and c["M7_origin_string_mutation"]["unchanged"]
                and c["M8_echo_hygiene"]["clean"]
                and c["M9_determinism"]["identical"])

    def score(o):
        pd = o["pd_production_scorer"]
        return (o["forced_choice_explicit_rate"] or 0,
                min(pd[p]["R"] for p in ("primary", "sealed")),
                min(o["per_arm"]["inclusion"]["design_accuracy"] or 0,
                    o["per_arm"]["exclusion"]["design_accuracy"] or 0))

    elig = {n: score(o) for n, o in payload["candidates"].items() if eligible(o)}
    if elig:
        best = sorted(elig.items(), key=lambda kv: (kv[1], -list(CANDIDATES).index(kv[0])),
                      reverse=True)[0][0]
        best_pd = payload["candidates"][best]["pd_production_scorer"]
        payload["selection"] = {
            "best": best, "score": elig[best], "eligible": sorted(elig),
            "G6_saturated": all(best_pd[p]["R"] >= 1.0 or best_pd[p]["F"] is None
                               for p in ("primary", "sealed")),
            "interpretation": "best scratch MECHANISM candidate only; a "
                              "G6-saturated R=1.00 is not capability evidence "
                              "and the production PD_wellformed gate (R<=0.999) "
                              "still fails",
        }
    else:
        payload["selection"] = {"best": None, "eligible": [],
                                "reason": "no candidate passed the full gates"}

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print("SELECTION:", payload["selection"])


if __name__ == "__main__":
    main()
