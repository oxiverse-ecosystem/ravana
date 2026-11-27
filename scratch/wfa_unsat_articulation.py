#!/usr/bin/env python3
"""WFA unsaturating articulation: pre-registered boot-free scratch experiment.

NO CognitiveChatEngine boot, NO weight store, NO production edit. Production
modules are imported UNMODIFIED from this worktree. Pre-registration:
docs/investigations/WFA_UNSAT_ARTICULATION_PREREG.md (committed before this
script was run).

Usage:
  ../ravana/.venv/Scripts/python.exe scratch/wfa_unsat_articulation.py \
      --summary ../ravana/scratch/learn5_wfa_summary.json \
      --out scratch/wfa_unsat_results.json
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import json
import math
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (os.path.join(PROJ, "scripts"),
           os.path.join(PROJ, "ravana", "src"), PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_learn5_wfa as w  # noqa: E402  production, unmodified
from ravana.chat.source_evaluation import Decision, evaluate_source  # noqa: E402
from ravana_ml.graph import ConceptGraph  # noqa: E402

MANIFEST = "119b9d91fe43a06087d001b1fb02a554edd4d08d665b99041528d93533818164"
ARMS = ("inclusion", "exclusion", "invalidation", "forced_choice")
MODE = {"inclusion": "free", "exclusion": "restricted",
        "invalidation": "invalidate", "forced_choice": "forced"}
TEXT_KEY = {"inclusion": "pd_inclusion_text", "exclusion": "pd_exclusion_text",
            "invalidation": "inv_after", "forced_choice": "pd_forced_choice_text"}
EXPECTED = {"inclusion": {"gap": "endorse", "decoy": "endorse"},
            "exclusion": {"gap": "endorse", "decoy": "abstain"},
            "invalidation": {"gap": "withdraw", "decoy": "withdraw"},
            "forced_choice": {"gap": "Y", "decoy": "N"}}

# ---- PRE-REGISTERED MECHANISM CONSTANTS (do not tune after a run) ----------
DDM_STEPS = 4
SIGMA = 1.0
CONF_CRITERION = 0.5
SEED_SALT = "wfa-unsat-v1"
PHI_09 = statistics.NormalDist().cdf(0.9)
# Scratch-only selection bounds (NOT production gates):
R_MIN, R_MAX = 0.25, 0.95
MARGIN_MIN, E_CRED_MIN, WITHDRAW_MIN = 0.10, 0.70, 0.50
M3_MIN, M10_TOL, M11_DELTA = 0.75, 0.08, 0.20


class GraphEngine:
    """Real in-memory ConceptGraph; no engine boot, no store."""
    def __init__(self):
        self.graph = ConceptGraph(dim=8)


class Fixture:
    __slots__ = ("claim", "mode", "arm", "raw_text", "engine", "meta", "kind",
                 "pool", "item_id", "prior")

    def __init__(self, claim, mode, arm, raw_text, engine, meta, kind, pool, item_id):
        self.claim, self.mode, self.arm = claim, mode, arm
        self.raw_text = raw_text
        self.engine = engine
        self.meta = meta            # available ONLY for the anti-tautology control
        self.kind = kind            # never read by a candidate
        self.pool = pool            # never read by a candidate
        self.item_id = item_id      # never read by a candidate
        self.prior = None


class ArticResult:
    __slots__ = ("token", "decision", "confidence", "rule", "fingerprint",
                 "n_edges", "steps")

    def __init__(self, token, decision, confidence=None, rule="", fingerprint="",
                 n_edges=0, steps=0):
        self.token, self.decision = token, decision
        self.confidence, self.rule = confidence, rule
        self.fingerprint, self.n_edges, self.steps = fingerprint, n_edges, steps

    def as_dict(self):
        return {"token": self.token, "decision": self.decision,
                "confidence": self.confidence, "rule": self.rule,
                "fingerprint": self.fingerprint, "n_edges": self.n_edges,
                "steps": self.steps}


def build_fixture(item, arm, row):
    """Harness-side teaching: production writers only; candidate sees no key."""
    claim, kind = item["claim"], row["kind"]
    eng = GraphEngine()
    cred = w.is_source_id(claim, "credible")
    disc = w.is_source_id(claim, "discredited")
    prior = None
    if arm == "inclusion":
        w._write_asserted_wfa_fact(eng, claim, disc if kind == "decoy" else cred)
    elif arm in ("exclusion", "forced_choice"):
        if kind == "gap":
            w._write_asserted_wfa_fact(eng, claim, cred)
        else:
            w._write_asserted_wfa_fact(eng, claim, disc)
            w.apply_source_invalidation(eng, disc)
    elif arm == "invalidation":
        w._write_asserted_wfa_fact(eng, claim, cred)
        assert w.apply_source_invalidation(eng, cred) >= 1
        prior = Decision.ENDORSE   # engine-audit analogue, staged by the harness
    else:
        raise ValueError(arm)
    fx = Fixture(claim, MODE[arm], arm, row.get(TEXT_KEY[arm]) or "", eng,
                 dict(item), kind, row["pool"], item["id"])
    fx.prior = prior
    return fx


def _fail_closed(mode):
    tok = {"free": "abstain", "restricted": "abstain",
           "invalidate": "maintain", "forced": "N"}[mode]
    dec = {"free": "ABSTAIN", "restricted": "ABSTAIN",
           "invalidate": "MAINTAIN", "forced": "ABSTAIN"}[mode]
    return ArticResult(tok, dec, 0.0, "U_NO_STORED_PROVENANCE", "", 0, 0)


def unsat_decide(fx, sigma=SIGMA, use_stored_conf=True):
    """Candidate U1: accumulation + graded confidence over a source verdict.

    Inputs: claim, instruction mode, and the engine's OWN in-memory graph.
    It never reads fx.meta/kind/pool/item_id or any expected token.  It also
    does NOT receive a source-id filter: the census ranks all stored records.
    """
    graph = fx.engine
    rows = w.graph_provenance_for_claim(graph, fx.claim)
    if not rows:
        return _fail_closed(fx.mode)                      # M5
    best = w._rank_stored_provenance(rows)[0]
    conf = float(best["confidence"]) if (use_stored_conf and
            isinstance(best.get("confidence"), (int, float))) else 0.0
    conf = min(1.0, max(0.0, conf))
    ctx, audit = w.context_from_graph_provenance(graph, fx.claim,
                                                 prior_decision=fx.prior)
    d = evaluate_source(fx.claim, ctx)
    if d.decision is Decision.UNKNOWN:
        return _fail_closed(fx.mode)

    # Deterministic per-claim noise seed; the claim is the engine's own text.
    seed = int(hashlib.sha256(
        (SEED_SALT + "|" + fx.claim + "|" + fx.mode).encode("utf-8")
    ).hexdigest()[:16], 16)
    rng = __import__("random").Random(seed)
    step_drift = conf / DDM_STEPS
    step_sigma = sigma / math.sqrt(DDM_STEPS)
    v = 0.0
    for _ in range(DDM_STEPS):
        v += step_drift + rng.gauss(0.0, step_sigma)
    confidence = 1.0 / (1.0 + math.exp(-v))
    recollected = (v > 0.0) if sigma == 0.0 else (confidence >= CONF_CRITERION)
    rule = "U_ACC_%s_CONF=%.3f_RECOL=%s|%s" % (
        fx.mode, confidence, recollected, d.rule)

    if fx.mode == "free":
        # Inclusion: familiarity and recollection cooperate. An endorsed or
        # merely stored claim is answered from item familiarity.
        token, decision = "endorse", "ENDORSE_FAMILIARITY"
    elif fx.mode in ("restricted", "forced"):
        if d.decision is Decision.ENDORSE:
            token, decision = ("Y" if fx.mode == "forced" else "endorse"), "ENDORSE"
        elif recollected:
            token, decision = ("N" if fx.mode == "forced" else "abstain"), "ABSTAIN"
        else:
            # Controlled recollection failed; automatic familiarity answers.
            token, decision = ("Y" if fx.mode == "forced" else "endorse"), "ENDORSE_FAMILIARY_FALLBACK"
    elif fx.mode == "invalidate":
        if d.decision is Decision.WITHDRAW and recollected:
            token, decision = "withdraw", "WITHDRAW"
        else:
            token, decision = "maintain", "MAINTAIN"
    else:
        return _fail_closed(fx.mode)
    return ArticResult(token, decision, confidence, rule, audit["fingerprint"],
                       len(rows), DDM_STEPS)


def t0_production_parser(fx):
    """Baseline: the shipped articulation path, replayed on recorded text."""
    if fx.mode == "forced":
        p = w.parse_forced_choice(fx.raw_text)
        tok = {"endorse": "Y", "abstain": "N", "withdraw": "W"}.get(p["decision"])
    else:
        kind = "withdrawal" if fx.mode == "invalidate" else "endorsement"
        p = w.parse_decision(fx.raw_text, kind)
        if fx.mode == "invalidate":
            tok = "withdraw" if p["decision"] is True else (
                "maintain" if p["decision"] is False else None)
        else:
            tok = "endorse" if p["decision"] is True else (
                "abstain" if p["decision"] is False else None)
    return ArticResult(tok, str(tok).upper() if tok else None, None,
                       "T0_PRODUCTION_TEXT", "", 0, 0)


_ENDORSE_W = re.compile(r"\b(endorse|yes|correct)\b", re.I)
_ABSTAIN_W = re.compile(r"\b(abstain|no|unknown|unreliable)\b", re.I)


def t3_text_only(fx):
    """Falsifier control: no graph access, keywords in the engine's text only."""
    text = w.strip_echoes(fx.raw_text)
    v = 0.0
    for tok in re.findall(r"[A-Za-z]+", text):
        v += 1.0 if _ENDORSE_W.match(tok) else (-1.0 if _ABSTAIN_W.match(tok) else 0.0)
        if abs(v) >= 1.0:
            break
    if abs(v) < 1.0:
        return ArticResult(None, None, 0.0, "T3_NO_BOUNDARY", "text_only", 0, 0)
    tok = "endorse" if v > 0 else "abstain"
    if fx.mode == "forced":
        tok = "Y" if v > 0 else "N"
    return ArticResult(tok, tok.upper(), abs(v), "T3_TEXT_BOUNDARY", "text_only", 0, 0)


CANDIDATES = {
    "T0_production_text": t0_production_parser,
    "U1_unsat_accumulator": unsat_decide,
    "U2_zero_noise_G6_control": functools.partial(unsat_decide, sigma=0.0),
    "U4_ungraded_confidence_control": functools.partial(unsat_decide,
                                                        use_stored_conf=False),
    "T3_text_only_falsifier": t3_text_only,
}

_TO_PD = {"endorse": "endorse", "abstain": "abstain", "maintain": "abstain",
          "Y": "endorse", "N": "abstain", "W": "withdraw", "withdraw": "withdraw"}


def rewrite_records(rows, results):
    """Feed each emitted token to the PRODUCTION parsers/counters unchanged."""
    out = []
    for r in rows:
        c = copy.deepcopy(r)
        for arm in ARMS:
            res = results.get((arm, r["id"]))
            tok = res.token if res else None
            if tok is None:
                continue
            if arm == "inclusion":
                d = w.parse_decision(tok, "endorsement")
                c["pd_inclusion_text"] = tok
                c["pd_inclusion_decision"] = d["decision"]
                c["pd_inclusion_unparseable"] = d["unparseable"]
                c["pd_inclusion_echo_marker"] = False
                c["pd_inclusion_endorse"] = d["decision"] is True
            elif arm == "exclusion":
                d = w.parse_decision(tok, "endorsement")
                c["pd_exclusion_text"] = tok
                c["pd_exclusion_decision"] = d["decision"]
                c["pd_exclusion_unparseable"] = d["unparseable"]
                c["pd_exclusion_echo_marker"] = False
                c["pd_exclusion_endorse"] = d["decision"] is True
            elif arm == "invalidation" and r["pool"] == "primary" or arm == "invalidation":
                d = w.parse_decision(tok, "withdrawal")
                c["inv_after"] = tok
                c["inv_decision"] = d["decision"]
                c["inv_unparseable"] = d["unparseable"]
                c["inv_echo_marker"] = False
                c["inv_withdrawn"] = d["decision"] is True
            elif arm == "forced_choice":
                d = w.parse_forced_choice(tok)
                c["pd_forced_choice_text"] = tok
                c["pd_forced_choice"] = d["decision"]
                c["pd_forced_choice_no_answer"] = d["no_answer"]
                c["pd_forced_choice_echo_marker"] = False
        out.append(c)
    return out


def score_candidate(name, fn, rows, items, summary):
    results, per_arm = {}, {}
    for arm in ARMS:
        correct, explicit = [], []
        for r in rows:
            fx = build_fixture(items[r["id"]], arm, r)
            res = fn(fx)
            results[(arm, r["id"])] = res
            explicit.append(res.token is not None)
            correct.append(res.token == EXPECTED[arm][r["kind"]])
        per_arm[arm] = {"n": len(rows), "explicit_rate": _rate(explicit),
                        "design_accuracy": _rate(correct)}
    records = rewrite_records(rows, results)
    pd = {p: w.process_dissociation(records, p) for p in ("primary", "sealed")}
    qual = w.pd_quality_counters(records)

    # Production decision routine on an unchanged scorer, with ONLY the PD
    # endpoints and quality counters replaced by the candidate's observations.
    s2 = copy.deepcopy(summary)
    for p in ("primary", "sealed"):
        for k, v in pd[p].items():
            s2[p][k] = v
        s2[p]["pd_unparseable_rate"] = qual["pd_unparseable_rate"]
    for k in ("pd_unparseable_rate", "forced_choice_explicit_rate",
              "forced_choice_no_answer_rate", "unparseable_rate",
              "own_decision_rate", "echo_marker_rate"):
        s2[k] = qual[k]
    checks_ok = all(bool(c.get("ok")) for c in summary.get("instrument_controls", []))
    rules, all_ok = w.decide(s2, checks_ok)
    pd_rules = {k: v for k, v in rules.items()
                if ("PD_" in k or "unparseable_ceiling" in k
                    or "invalidation_withdrawal" in k)}
    return {
        "candidate": name, "per_arm": per_arm,
        "forced_choice_explicit_rate": qual["forced_choice_explicit_rate"],
        "pd_unparseable_rate": qual["pd_unparseable_rate"],
        "pd_production_scorer": pd, "production_pd_rules": pd_rules,
        "all_production_rules_pass": all_ok,
        "sample_decisions": {
            "%s/%s" % (a, i): results[(a, i)].as_dict()
            for (a, i) in list(results)[:8]},
    }, results, records


def _rate(vals):
    vals = list(vals)
    return round(sum(1 for v in vals if v) / len(vals), 4) if vals else None


def run_controls(fn, rows, items):
    ctrls = {}
    gap = [r for r in rows if r["kind"] == "gap" and r["pool"] == "primary"]
    r = gap[0]
    it = items[r["id"]]

    # M4 anti-tautology (item metadata only; graph fixture is identical).
    base_fx = build_fixture(it, "exclusion", r)
    base = fn(base_fx)
    mut_item = dict(it)
    mut_item.update({"expected_answer": "FORGED", "source_status": "credible",
                     "source_id": "forged://answer", "reliability": 1.0})
    mut = fn(build_fixture(mut_item, "exclusion", r))
    ctrls["M4_anti_tautology"] = {
        "unchanged": base.token == mut.token and base.fingerprint == mut.fingerprint,
        "base": base.as_dict(), "mutated": mut.as_dict()}

    # M5 no shortcut, every mode.
    empty = {}
    for mode in ("free", "restricted", "invalidate", "forced"):
        fx = Fixture("no stored claim", mode, mode, "", GraphEngine(), {},
                     "gap", "primary", "empty")
        empty[mode] = fn(fx).token
    ctrls["M5_no_shortcut"] = {"tokens": empty, "pass": all(
        v in ("abstain", "maintain", "N", None) for v in empty.values())}

    # M6 stored-provenance damage, aggregate over primary gap exclusion rows.
    before, after = [], []
    for rr in gap:
        fx = build_fixture(items[rr["id"]], "exclusion", rr)
        before.append(fn(fx).token == "endorse")
        damaged = False
        for (_s, _t), edge in w._iter_graph_edges(fx.engine):
            if edge.source_metadata.get("edge_kind") == "curated":
                edge.source_metadata["edge_kind"] = "co_occurrence"
                damaged = True
                break
        assert damaged
        after.append(fn(fx).token == "endorse")
    ctrls["M6_stored_provenance_mutation"] = {
        "endorse_before": _rate(before), "endorse_after": _rate(after),
        "drop": round(_rate(before) - _rate(after), 4),
        "pass": (_rate(before) - _rate(after)) >= 0.50}

    # M7 origin-string mutation (kind/status untouched).
    changed = []
    for rr in gap[:20]:
        fx = build_fixture(items[rr["id"]], "exclusion", rr)
        b = fn(fx).token
        for (_s, _t), edge in w._iter_graph_edges(fx.engine):
            sm = edge.source_metadata
            if sm.get("edge_kind") == "curated":
                sm["source"] = sm["url"] = "wfa-internal://discredited/FORGED"
                break
        changed.append(b == fn(fx).token)
    ctrls["M7_origin_string_mutation"] = {"n": len(changed),
                                          "unchanged": _rate(changed),
                                          "pass": all(changed)}

    # M8 echo hygiene on the recorded text.
    texts = [x.get(k) for x in rows for k in TEXT_KEY.values()]
    parsed = [w.parse_decision(t, "endorsement")["decision"] for t in texts if t]
    ctrls["M8_echo_hygiene"] = {"n_texts": len(texts),
        "explicit": sum(1 for d in parsed if d is not None),
        "pass": all(d is None for d in parsed)}

    # M9 determinism (repeats of the same fixture).
    reps = [fn(build_fixture(it, "exclusion", r)).as_dict() for _ in range(3)]
    ctrls["M9_determinism"] = {
        "pass": len({json.dumps(x, sort_keys=True) for x in reps}) == 1}

    return ctrls


def graded_control(rows, items):
    """M11: stored confidence -> withdrawal rate (no gate/threshold change)."""
    gap = [r for r in rows if r["kind"] == "gap" and r["pool"] == "primary"]
    out = {}
    for c in (0.1, 0.3, 0.5, 0.7, 0.9):
        vals = []
        for rr in gap:
            fx = build_fixture(items[rr["id"]], "invalidation", rr)
            for (_s, _t), edge in w._iter_graph_edges(fx.engine):
                edge.confidence = c
            vals.append(unsat_decide(fx).token == "withdraw")
        out[str(c)] = {"withdrawal_rate": _rate(vals),
                       "analytic_phi": round(statistics.NormalDist().cdf(c), 4)}
    rates = [v["withdrawal_rate"] for v in out.values()]
    out["pass"] = (all(b > a for a, b in zip(rates, rates[1:]))
                   and rates[-1] - rates[0] >= M11_DELTA)
    return out


def eligibility(o):
    pd, c = o["pd_production_scorer"], o["controls"]
    m2 = all(pd[p]["R"] is not None and R_MIN <= pd[p]["R"] <= R_MAX
             and pd[p]["F"] is not None
             and pd[p]["exclusion_margin"] is not None
             and pd[p]["exclusion_margin"] >= MARGIN_MIN
             and pd[p]["E_credible"] is not None
             and pd[p]["E_credible"] >= E_CRED_MIN
             and pd[p]["withdrawal_rate"] is not None
             and pd[p]["withdrawal_rate"] >= WITHDRAW_MIN
             for p in ("primary", "sealed"))
    m1 = (o["forced_choice_explicit_rate"] is not None
          and o["forced_choice_explicit_rate"] >= 0.90
          and o["pd_unparseable_rate"] is not None
          and o["pd_unparseable_rate"] <= 0.50)
    m3 = all((v["design_accuracy"] or 0) >= M3_MIN for v in o["per_arm"].values())
    m10 = all(pd[p]["withdrawal_rate"] is not None
              and abs(pd[p]["withdrawal_rate"] - PHI_09) <= M10_TOL
              for p in ("primary", "sealed"))
    m13 = (o["per_arm"]["forced_choice"]["explicit_rate"] or 0) <= 0.10 or \
          (o["per_arm"]["forced_choice"]["design_accuracy"] or 1) <= 0.30
    g6 = any(pd[p]["R"] is not None and (pd[p]["R"] > 0.95 or pd[p]["F"] is None)
             for p in ("primary", "sealed"))
    return {"M1": m1, "M2": m2, "M3": m3,
            "M4": c["M4_anti_tautology"]["unchanged"],
            "M5": c["M5_no_shortcut"]["pass"], "M6": c["M6_stored_provenance_mutation"]["pass"],
            "M7": c["M7_origin_string_mutation"]["pass"],
            "M8": c["M8_echo_hygiene"]["pass"], "M9": c["M9_determinism"]["pass"],
            "M10": m10, "M11": c.get("M11_graded_monotonicity", {}).get("pass", False),
            "M12": c.get("M12_zero_noise_saturation", {}).get("pass", False),
            "M13_text_only_falsifier": m13, "G6_saturated": g6,
            "eligible": all([m1, m2, m3, m10, g6 is False,
                c["M4_anti_tautology"]["unchanged"], c["M5_no_shortcut"]["pass"],
                c["M6_stored_provenance_mutation"]["pass"],
                c["M7_origin_string_mutation"]["pass"],
                c["M8_echo_hygiene"]["pass"], c["M9_determinism"]["pass"]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.join(
        PROJ, "..", "ravana", "scratch", "learn5_wfa_summary.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "wfa_unsat_results.json"))
    args = ap.parse_args()
    summary = json.load(open(args.summary, encoding="utf-8"))
    assert summary["manifest_sha256"] == MANIFEST, "item seal mismatch"
    rows = summary["pd_audit"]
    items = {it["id"]: it for it in w.build_items()}

    payload = {
        "probe": "WFA unsaturating articulation (boot-free, pre-registered)",
        "no_engine_boot": True, "no_store_open": True, "no_production_edit": True,
        "manifest_sha256": summary["manifest_sha256"], "n_rows": len(rows),
        "recorded_baseline": {
            "pd_unparseable_rate": summary["pd_unparseable_rate"],
            "forced_choice_explicit_rate": summary["forced_choice_explicit_rate"],
            "R": {p: summary[p]["R"] for p in ("primary", "sealed")}},
        "preregistered": {"DDM_STEPS": DDM_STEPS, "SIGMA": SIGMA,
            "CONF_CRITERION": CONF_CRITERION, "SEED_SALT": SEED_SALT,
            "PHI_09": PHI_09, "R_MIN": R_MIN, "R_MAX": R_MAX,
            "MARGIN_MIN": MARGIN_MIN, "E_CRED_MIN": E_CRED_MIN,
            "WITHDRAW_MIN": WITHDRAW_MIN, "M3_MIN": M3_MIN,
            "M10_TOL": M10_TOL, "M11_DELTA": M11_DELTA},
        "candidates": {}, "selection": None,
    }

    for name, fn in CANDIDATES.items():
        out, _res, _records = score_candidate(name, fn, rows, items, summary)
        out["controls"] = run_controls(fn, rows, items)
        if name == "U1_unsat_accumulator":
            out["controls"]["M11_graded_monotonicity"] = graded_control(rows, items)
        if name == "U2_zero_noise_G6_control":
            zpd = out["pd_production_scorer"]
            out["controls"]["M12_zero_noise_saturation"] = {
                "R": {p: zpd[p]["R"] for p in ("primary", "sealed")},
                "F": {p: zpd[p]["F"] for p in ("primary", "sealed")},
                "pass": all(zpd[p]["R"] >= 1.0 and zpd[p]["F"] is None
                            for p in ("primary", "sealed"))}
        out["eligibility"] = eligibility(out)
        payload["candidates"][name] = out

    u1 = payload["candidates"]["U1_unsat_accumulator"]
    ctrls = payload["candidates"]["T3_text_only_falsifier"]["controls"]
    # M12 is a diagnostic on the ZERO-NOISE arm, not a gate on U1: the
    # pre-registered prediction is that sigma=0 reproduces the old saturation.
    u1["eligibility"]["M12_zero_noise_prediction_confirmed"] = \
        payload["candidates"]["U2_zero_noise_G6_control"]["controls"][
            "M12_zero_noise_saturation"]["pass"]
    u1["eligibility"]["M13_text_only_falsifier"] = eligibility(
        payload["candidates"]["T3_text_only_falsifier"])["M13_text_only_falsifier"]
    u1["eligibility"]["eligible"] = u1["eligibility"]["eligible"] and \
        u1["eligibility"]["M13_text_only_falsifier"]
    elig = u1["eligibility"]["eligible"]
    payload["selection"] = {
        "selected": "U1_unsat_accumulator" if elig else None,
        "eligible": elig,
        "failed": [k for k, v in u1["eligibility"].items()
                   if v is False and not k.startswith("M12")
                   and k != "G6_saturated"],
        "G6_saturated": u1["eligibility"]["G6_saturated"],
        "implementation_rule": ("default-off flag permitted" if elig
            else "NO production implementation; negative result")}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    brief = {n: {"fc_explicit": o["forced_choice_explicit_rate"],
                 "pd_unparseable": o["pd_unparseable_rate"],
                 "pd": {p: {k: o["pd_production_scorer"][p][k]
                            for k in ("I", "E", "R", "F", "exclusion_margin",
                                      "withdrawal_rate", "E_credible")}
                        for p in ("primary", "sealed")},
                 "eligibility": o["eligibility"]}
             for n, o in payload["candidates"].items()}
    print(json.dumps({"selection": payload["selection"],
                      "candidates": brief,
                      "graded": payload["candidates"]["U1_unsat_accumulator"]
                                ["controls"]["M11_graded_monotonicity"]},
                     indent=2, ensure_ascii=False, default=str))
    print("SELECTION:", payload["selection"])


if __name__ == "__main__":
    main()
