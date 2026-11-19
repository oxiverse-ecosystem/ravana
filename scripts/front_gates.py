#!/usr/bin/env python3
"""Tier-0 instrument gates for the two front harnesses.

Implements docs/BENCHMARKING_SURVEY.md section 5.1 (G1-G6). Every check here
is deterministic and runs on FIXED strings, on the harness source, or on the
already-written per-probe JSONL. No rubric criterion is changed: the gate
exercises the criteria that live in measure_fronts*.py and reports what they
actually did. A gate FAIL means no front score may be reported.

Nothing in this module boots an engine; the caller supplies an optional
cold-reload probe function for G5.
"""
from __future__ import annotations

import json
import os
import re

# Pre-registered thresholds (survey section 5.1; G2's 0.30 is the proposed
# value awaiting ratification, not a literature constant).
SELECTIVITY_MIN = 0.30
POLICY_PASS_MAX = 0.25          # a decline-all / assert-all policy must not
                                # clear this fraction of query-front probes
SATURATION_LOW = 5
SATURATION_HIGH = 95

QUERY_FRONTS = (
    "language understanding / conversation",
    "grounded factual answers",
    "opinions / stance",
    "thinking / reasoning",
    "speaking / surface realization",
)
SCORED_FRONTS = QUERY_FRONTS + ("code generation", "language learning (online)")

# ── fixed inputs. These are test VECTORS, not rubric changes. ────────────────
# Each pair is (front, good answer, bad answer, query for the grounded judge).
PAIRED_CASES = (
    ("language understanding / conversation",
     "Hello there, I am doing well today and happy to chat with you.",
     "the of and the of and the of", None),
    ("grounded factual answers",
     "Gravity is the force that attracts bodies toward one another.",
     "the closest i get to gravity is rains and maybe environments "
     "\u2014 and from what i've seen, water that falls from the sky as "
     "drops of liquid.",
     "what is gravity"),
    ("opinions / stance",
     "I lean toward strong privacy because it matters to me and protects "
     "ordinary people from harm.",
     "I'm still forming a view on that. What's your take?", None),
    ("thinking / reasoning",
     "Rain falls because water vapor condenses; therefore the droplets grow "
     "heavy and descend from the clouds.",
     "a device that stores chemical energy", None),
    ("speaking / surface realization",
     "Water flows downhill and forms the rivers.",
     "the closest i get to water is and maybe a liquid", None),
)

# Control task: outputs unrelated to / evasive of the question. A selective
# rubric must reject all of them. Reuses the repo's hand-labelled negatives.
CONTROL_CASES = (
    ("language understanding / conversation", "the of and the of and the of", None),
    ("language understanding / conversation",
     "the closest i get to robot is cat and maybe dog \u2014 and from what "
     "i've seen, a cat is a small domesticated carnivorous mammal.", None),
    ("grounded factual answers",
     "I'm not sure what to say about that; how do you see it?", "what is gravity"),
    ("grounded factual answers",
     "the closest i get to earthquakes is rains and maybe environments "
     "\u2014 and from what i've seen, water that falls from the sky as drops "
     "of liquid.", "what causes earthquakes"),
    ("opinions / stance",
     "I'm still forming a view on that. What's your take?", None),
    ("opinions / stance", "ok", None),
    ("thinking / reasoning", "Rain is water that falls from the clouds.", None),
    ("thinking / reasoning", "a device that stores chemical energy", None),
    ("speaking / surface realization",
     "the closest i get to water is and maybe a liquid", None),
    ("speaking / surface realization",
     "honestly, sentence is a bit outside what i know right now what made "
     "you think about it.", None),
)

# Symmetric-policy probes: a decline-everything system and an
# assert-without-structure system must both stay below POLICY_PASS_MAX.
DECLINE_CASES = (
    ("language understanding / conversation",
     "I don't know. I'm not sure what to say about that; how do you see it?", None),
    ("grounded factual answers",
     "I'm still forming a view on that. What's your take?", "what is gravity"),
    ("opinions / stance",
     "I don't have a clean definition; how do you see it?", None),
    ("thinking / reasoning",
     "I don't know why that happens. I don't have a way to act.", None),
    ("speaking / surface realization",
     "I don't really have an answer for that one.", None),
)
ASSERT_CASES = (
    ("language understanding / conversation", "the of and the of and the of", None),
    ("grounded factual answers", "Gravity is a force.", "what is gravity"),
    ("opinions / stance", "yes absolutely, it is very good and important.", None),
    ("thinking / reasoning", "a device that stores chemical energy", None),
    ("speaking / surface realization", "the closest i get to water is and maybe a liquid", None),
)

# The symmetric guard proper (survey G3) is about the gap/chain judgement:
# declining when a chain exists and asserting when none exists are BOTH FAIL.
# It is scoped to the thinking rubric, where that property is actually
# measured, plus the code front's honest-decline arithmetic. Conversation /
# grounded / opinion judges are not honesty-sensitive by design; their
# uniform-policy rates are reported as context, never as a gate.
DECLINE_THINKING = (
    "I don't know why that happens. I don't have a way to act.",
    "I don't know. I'm not sure what to say about that.",
    "I don't have a clean definition; how do you see it?",
)
ASSERT_THINKING = (
    "a device that stores chemical energy",
    "Rain is water that falls from the clouds.",
    "the closest i get to water is and maybe a liquid",
)
GOOD_THINKING = (
    "Rain falls because water vapor condenses; therefore the droplets grow "
    "heavy and descend from the clouds.",
    "Water vapor condenses into droplets; then the droplets grow heavy so "
    "they descend from the cloud as rain.",
    "Sunlight warms the ground; therefore air above it rises and draws in "
    "cooler air, so wind is produced.",
)


class GateResult:
    """One gate outcome. `blocking` means: suppress all front scores."""

    def __init__(self, gid, name, status, detail, metrics=None,
                 blocking=None, flagged_fronts=None):
        self.gid = gid
        self.name = name
        self.status = status                      # PASS / FAIL / FLAG / SKIP
        self.detail = detail
        self.metrics = dict(metrics or {})
        self.blocking = (status in ("FAIL", "SKIP")) if blocking is None else blocking
        self.flagged_fronts = list(flagged_fronts or [])

    def as_dict(self, mode):
        return {"gate": self.gid, "name": self.name, "status": self.status,
                "mode": mode, "detail": self.detail, "metrics": self.metrics,
                "blocking": self.blocking, "flagged_fronts": self.flagged_fronts}


def _judge(judges, front, text, query):
    fn = judges[front]
    return (fn(text, query) if query is not None else fn(text))


def _emit(logfh, mode, result):
    if logfh is not None:
        logfh.write(json.dumps(result.as_dict(mode), ensure_ascii=False) + "\n")
        logfh.flush()


# ── G1: paired known-good / known-bad rubric check ──────────────────────────
def gate_paired(judges, logfh=None, mode="offline"):
    rows, bad_lt_good = [], 0
    for front, good, bad, query in PAIRED_CASES:
        g_ok, g_reason = _judge(judges, front, good, query)
        b_ok, b_reason = _judge(judges, front, bad, query)
        # Boolean rubrics: good must be 1 and bad must be 0, so bad < good.
        ok = bool(g_ok) and not bool(b_ok)
        bad_lt_good += ok
        rows.append({"front": front, "bad": int(bool(b_ok)), "good": int(bool(g_ok)),
                     "bad_lt_good": ok, "bad_reason": b_reason,
                     "good_reason": g_reason, "passed": ok})
    n = len(PAIRED_CASES)
    r = GateResult(
        "G1", "paired rubric (bad < good)",
        "PASS" if bad_lt_good == n else "FAIL",
        f"bad<good {bad_lt_good}/{n} front judges; "
        f"good={sum(x['good'] for x in rows)}/{n}, bad={sum(x['bad'] for x in rows)}/{n}",
        {"pairs": rows, "n": n, "bad_lt_good": bad_lt_good})
    _emit(logfh, mode, r)
    return r


# ── helpers: per-probe JSONL -> measured pass rates ─────────────────────────
def load_probe_records(path):
    records = []
    if not path or not os.path.exists(path):
        return records
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not rec.get("control"):
                records.append(rec)
    return records


def recompute_scores(records, n_map):
    """Recompute what the CURRENT score functions earn from the raw records."""
    out = {}
    for front in SCORED_FRONTS:
        rows = [r for r in records if r.get("front") == front]
        if not rows:
            continue
        if front in QUERY_FRONTS:
            out[front] = round(100 * sum(bool(r.get("passed")) for r in rows) / len(rows))
        else:
            sub = [sum((r.get("sub_scores") or {}).values()) for r in rows]
            out[front] = round(sum(sub) / len(sub))
    return out


def _real_rates(records, n_map):
    rates = {}
    for front in QUERY_FRONTS:
        rows = [r for r in records if r.get("front") == front]
        if rows:
            rates[front] = sum(bool(r.get("passed")) for r in rows) / len(rows)
    return rates


# ── G2: control-task selectivity ────────────────────────────────────────────
def gate_selectivity(judges, records, n_map, logfh=None, mode="offline"):
    ctrl_pass = []
    per_front_ctrl = {}
    for front, text, query in CONTROL_CASES:
        ok, _ = _judge(judges, front, text, query)
        ctrl_pass.append(bool(ok))
        per_front_ctrl.setdefault(front, []).append(bool(ok))
    control_rate = sum(ctrl_pass) / len(ctrl_pass)
    real = _real_rates(records, n_map)
    per_front = {}
    for front, rr in real.items():
        cr = sum(per_front_ctrl.get(front, [False])) / max(len(per_front_ctrl.get(front, [])), 1)
        per_front[front] = {"real": round(rr, 4), "control": round(cr, 4),
                            "selectivity": round(rr - cr, 4)}
    real_rate = sum(real.values()) / len(real) if real else 0.0
    selectivity = real_rate - control_rate
    ok = bool(real) and selectivity >= SELECTIVITY_MIN and control_rate <= 0.05
    r = GateResult(
        "G2", f"selectivity (real - control >= {SELECTIVITY_MIN:.2f})",
        "PASS" if ok else "FAIL",
        f"real={real_rate:.4f} control={control_rate:.4f} "
        f"selectivity={selectivity:+.4f} (n_real={len(real)} fronts, "
        f"n_control={len(ctrl_pass)})",
        {"real_rate": round(real_rate, 4), "control_rate": round(control_rate, 4),
         "selectivity": round(selectivity, 4), "threshold": SELECTIVITY_MIN,
         "per_front": per_front, "control_rejected": len(ctrl_pass) - sum(ctrl_pass),
         "control_total": len(ctrl_pass)})
    _emit(logfh, mode, r)
    return r


# ── G3: symmetric decline / answer guard ────────────────────────────────────
def gate_symmetric(judges, logfh=None, mode="offline"):
    think = judges["thinking / reasoning"]
    decline = [bool(think(t)[0]) for t in DECLINE_THINKING]
    assert_ = [bool(think(t)[0]) for t in ASSERT_THINKING]
    good = [bool(think(t)[0]) for t in GOOD_THINKING]
    d_rate = sum(decline) / len(decline)
    a_rate = sum(assert_) / len(assert_)
    g_rate = sum(good) / len(good)
    # Code front: replay the honest-decline arithmetic with two fake policies.
    # An "answer everything" emitter claims every unknown function exists;
    # a "decline everything" emitter never emits the taught function.
    answer_all = 40 + 40 + 0          # taught + generated - honest=0
    decline_all = 0 + 0 + 20          # no taught, no generated, honest=1
    code_ok = answer_all < 100 and decline_all < 100
    ok = (d_rate == 0.0 and a_rate == 0.0 and g_rate == 1.0 and code_ok)
    r = GateResult(
        "G3", "symmetric decline/answer guard",
        "PASS" if ok else "FAIL",
        f"thinking: decline-all={d_rate:.4f} ({sum(decline)}/{len(decline)}), "
        f"assert-no-chain={a_rate:.4f} ({sum(assert_)}/{len(assert_)}), "
        f"real-chain={g_rate:.4f} ({sum(good)}/{len(good)}); "
        f"code answer-all={answer_all}/100 decline-all={decline_all}/100",
        {"thinking_decline_rate": round(d_rate, 4),
         "thinking_assert_rate": round(a_rate, 4),
         "thinking_good_rate": round(g_rate, 4),
         "policy_max": POLICY_PASS_MAX,
         "decline_verdicts": [int(x) for x in decline],
         "assert_verdicts": [int(x) for x in assert_],
         "good_verdicts": [int(x) for x in good],
         "code_answer_all": answer_all, "code_decline_all": decline_all})
    _emit(logfh, mode, r)
    return r


# ── G4: web-off / web-on parity ─────────────────────────────────────────────
def _code_only(src):
    """Strip comments and triple-quoted text so documentation cannot trip
    a source scan for capability bonuses or eng.save() calls."""
    src = re.sub(r'""".*?"""', "", src, flags=re.S)
    src = re.sub(r"'''.*?'''", "", src, flags=re.S)
    return "\n".join(line.split("#", 1)[0] for line in src.splitlines())


def _score_region(path):
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    start = src.find("def run_query_front(")
    end = src.find("def main(")
    if start < 0 or end < 0 or end <= start:
        return None
    # Drop comments/docstrings before scanning: the harness *documents* the
    # removed +20 bonus in a comment; a naive token scan flags the docs.
    return _code_only(src[start:end])


def gate_parity(script_path, records=None, summary=None, compare_modes=True,
                n_map=None, logfh=None, mode="offline"):
    sibling = ("measure_fronts_web.py" if os.path.basename(script_path) == "measure_fronts.py"
               else "measure_fronts.py")
    sibling_path = os.path.join(os.path.dirname(script_path), sibling)
    a, b = _score_region(script_path), _score_region(sibling_path)
    identical = bool(a and b and re.sub(r"\s+", "", a) == re.sub(r"\s+", "", b))
    region = a or ""
    conditional = len(re.findall(r"RAVANA_OFFLINE", region))
    # A capability-gated bonus is a credit that depends on mode/environment.
    # The legitimate `+ 20 * honest` sub-score in the code front is NOT one.
    bonus_lines = [
        line for line in region.splitlines()
        if re.search(r"\bbonus\b", line, re.I)
        or (re.search(r"\+|=\s*\+?\s*20\b", line)
            and re.search(r"\boffline\b|\bmode\b|\bweb\b", line, re.I))
    ]
    bonus = len(bonus_lines)
    static_ok = identical and conditional == 0 and bonus == 0

    earned_vs_reported = None
    if records and summary:
        earned = recompute_scores(records, n_map or {})
        reported = summary.get("scores", {})
        deltas = {f: reported[f] - earned[f] for f in earned if f in reported}
        worst = max(((abs(d), f, d) for f, d in deltas.items()), default=(0, "", 0))
        earned_ok = worst[0] == 0
        earned_vs_reported = {"earned": earned, "reported": reported,
                              "deltas": deltas,
                              "worst_front": worst[1], "worst_delta": worst[2],
                              "passed": earned_ok}
    else:
        earned_ok = True

    cross_mode = None
    if compare_modes:
        scratch = os.path.normpath(os.path.join(os.path.dirname(script_path),
                                                "..", "scratch"))
        other_name = ("front_probes_web_summary.json" if "measure_fronts.py" in script_path
                      else "front_probes_n30_summary.json")
        other_path = os.path.join(scratch, other_name)
        if os.path.exists(other_path) and summary:
            with open(other_path, encoding="utf-8") as fh:
                other_summary = json.load(fh)
            deltas = {f: summary["scores"][f] - other_summary["scores"].get(f, 0)
                      for f in summary.get("scores", {})}
            # The removed +20 showed up exactly as OFF language-learning=100
            # vs ON=80 with identical per-probe sub-scores (80). Any >=10 gap
            # on that front is a capability-gated bonus signature.
            ll = deltas.get("language learning (online)", 0)
            cross_mode = {"deltas": deltas, "ll_delta": ll,
                          "passed": ll < 10}
        else:
            cross_mode = {"note": "other-mode summary unavailable", "passed": True}
    else:
        cross_mode = {"note": "not checked in post-score phase", "passed": True}

    ok = static_ok and earned_ok and cross_mode["passed"]
    detail = (f"scoring regions identical={identical} "
              f"env_conditional_tokens={conditional} bonus_tokens={bonus}; "
              f"earned-vs-reported={'ok' if earned_vs_reported is None else earned_vs_reported['deltas']}; "
              f"cross_mode_ll_delta={cross_mode.get('ll_delta', 'n/a')}")
    r = GateResult("G4", "web-off / web-on parity (no capability bonus)",
                   "PASS" if ok else "FAIL", detail,
                   {"identical_scoring_code": identical,
                    "conditional_tokens": conditional, "bonus_tokens": bonus,
                    "earned_vs_reported": earned_vs_reported,
                    "cross_mode": cross_mode})
    _emit(logfh, mode, r)
    return r


# ── G5: census sanity + cold-reload persistence ─────────────────────────────
def gate_census(records, n_map, script_path, cold_info=None, logfh=None,
                mode="offline"):
    census = {}
    for front, expected in n_map.items():
        rows = [r for r in records if r.get("front") == front]
        idx = [r.get("probe_index") for r in rows]
        subjects = [r.get("subject") for r in rows]
        census[front] = {
            "expected": expected, "found": len(rows),
            "distinct_indices": len(set(idx)),
            "indices_1_based": sorted(idx) == list(range(1, expected + 1)),
            "empty_subjects": sum(1 for s in subjects if s is None or s == ""),
        }
    census_ok = all(v["found"] == v["expected"] and v["distinct_indices"] == v["expected"]
                    and v["indices_1_based"] and v["empty_subjects"] == 0
                    for v in census.values())

    # Static audit: the instrument must not itself iterate ConceptGraph.nodes
    # (a dict of int->node); the known footgun reads ints as labels.
    footguns = 0
    for p in (script_path,):
        with open(p, encoding="utf-8") as fh:
            src = _code_only(fh.read())
        footguns += len(re.findall(r"for\s+\w+\s+in\s+\w*\.nodes\s*:", src))
        footguns += len(re.findall(r"eng\.save\s*\(", src))
    static_ok = footguns == 0

    cold_ok = bool(cold_info and cold_info.get("passed"))
    ok = census_ok and static_ok and cold_ok
    r = GateResult(
        "G5", "census sanity + cold reload",
        "PASS" if ok else ("SKIP" if cold_info is None else "FAIL"),
        f"census={'ok' if census_ok else 'BAD'} "
        f"({sum(v['found'] for v in census.values())} records, "
        f"footguns={footguns}); cold_reload="
        f"{'ok' if cold_ok else ('skipped' if cold_info is None else 'BAD')}",
        {"census": census, "census_ok": census_ok, "footguns": footguns,
         "static_ok": static_ok, "cold_reload": cold_info or {"passed": False,
         "note": "no cold-boot probe supplied"}})
    _emit(logfh, mode, r)
    return r


# ── G6: saturation flag ─────────────────────────────────────────────────────
def gate_saturation(scores, logfh=None, mode="offline"):
    flagged = {f: s for f, s in scores.items()
               if s is not None and (s >= SATURATION_HIGH or s <= SATURATION_LOW)}
    r = GateResult(
        "G6", f"saturation flag (>= {SATURATION_HIGH} or <= {SATURATION_LOW})",
        "FLAG" if flagged else "PASS",
        (f"{len(flagged)} front(s) non-informative: "
         + ", ".join(f"{f}={s}" for f, s in flagged.items())) if flagged
        else "no front in the saturated band",
        {"saturated": flagged, "threshold_low": SATURATION_LOW,
         "threshold_high": SATURATION_HIGH},
        blocking=False, flagged_fronts=list(flagged))
    _emit(logfh, mode, r)
    return r


# ── reporting ───────────────────────────────────────────────────────────────
def gate_valid(results):
    """True when no gate blocks reporting. Saturation FLAG is non-blocking."""
    return all(not r.blocking for r in results)


def print_table(results, mode, title="Tier-0 instrument gates"):
    print("=" * 78)
    print(f"{title} -- {mode}")
    print("=" * 78)
    for r in results:
        print(f"  {r.gid}  {r.status:<4}  {r.name}")
        print(f"       {r.detail}")
    bad = sum(r.status == "FAIL" or r.status == "SKIP" for r in results)
    flags = sum(r.status == "FLAG" for r in results)
    status = "INVALID" if bad else ("PARTIAL (saturated fronts INVALID)" if flags else "VALID")
    print(f"  gate status: {status}  [{len(results) - bad - flags} PASS / "
          f"{bad} FAIL/SKIP / {flags} FLAG]")
    print("=" * 78)
    return status
