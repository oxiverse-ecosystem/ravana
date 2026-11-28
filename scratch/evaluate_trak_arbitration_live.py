#!/usr/bin/env python3
"""Ordered falsifier evaluation for the TRA-K source-arbitration live A/B.

Reads the five arm reports written by m123_combined_runner.py and the boot-free
self-test, evaluates the pre-registered falsifiers IN ORDER, and writes a
verdict JSON. No engine, no store write, no gold/key input.
"""
from __future__ import annotations
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "m123_ab_out")
TAGS = {"OFF": "off1", "OFF2": "off2", "M123": "m123",
        "ARBITRATION": "arb", "SHAM": "sham"}


def load():
    reports = {}
    for name, tag in TAGS.items():
        p = os.path.join(OUT, "trak_%s.json" % tag)
        if not os.path.isfile(p):
            return None, "missing report: %s" % p
        reports[name] = json.load(open(p, encoding="utf-8"))
    selftest = os.path.join(HERE, "trak_arbitration_live_selftest.json")
    if not os.path.isfile(selftest):
        return None, "missing self-test: %s" % selftest
    return (reports, json.load(open(selftest, encoding="utf-8"))), None


def endpoints(rep, pool, key):
    return (rep.get("harness", {}).get("summary", {}).get(pool, {}) or {}).get(key)


def num(v):
    return float(v) if isinstance(v, (int, float)) else 0.0


def evaluate(reports, selftest):
    checks, details = [], {}

    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        details[name] = detail

    # L1 — instrument validity on every arm.
    bad = [(n, r.get("status"), r.get("falsifiers_local"),
            r.get("guard", {}).get("failed_controls"))
           for n, r in reports.items()
           if r.get("status") != "OK" or not r.get("scored")
           or r.get("falsifiers_local")
           or not r.get("guard", {}).get("preflight_ok")
           or not r.get("guard", {}).get("sha_pins_match")
           or not r.get("guard", {}).get("delayed_option_audit", {}).get("ok")
           or not r.get("guard", {}).get("controls_ok")
           or not r.get("guard", {}).get("c11_anti_fake_ok")
           or r.get("guard", {}).get("backend_error") is not None
           or not r.get("guard", {}).get("store_prefix_identical")]
    add("L1_instrument_valid", not bad, bad or "all five arms OK/scored/controls OK")

    # L2 — AST anti-leak + boot-free wiring self-test.
    ast_bad = [n for n, r in reports.items()
               if not r.get("guard", {}).get("arbitration", {})
               .get("ast_anti_leakage", {}).get("ok")]
    add("L2_ast_anti_leak_clean", not ast_bad and selftest.get("ok", False),
        {"bad_arms": ast_bad, "selftest_ok": selftest.get("ok")})

    # L3 — default-off flag isolation (ARBITRATION only).
    flags = {n: r.get("config", {}).get("env_flags", {}).get("RAVANA_SOURCE_ARBITRATION")
             for n, r in reports.items()}
    add("L3_flag_only_in_arbitration",
        flags.get("ARBITRATION") == "1"
        and all(flags[n] is None for n in flags if n != "ARBITRATION"), flags)

    # L4 — repeated baseline (A2): OFF == OFF2 on every reported endpoint.
    diffs = []
    for pool in ("primary", "sealed"):
        for key in ("near", "far", "foil", "negative", "delayed", "growth",
                    "delayed_retention", "baseline", "transfer"):
            a = endpoints(reports["OFF"], pool, key)
            b = endpoints(reports["OFF2"], pool, key)
            if a != b:
                diffs.append((pool, key, a, b))
    add("L4_off_equals_off2", not diffs, diffs or "identical on all endpoints")

    # L5 — matched wrong-tag/permuted SHAM, flat far.
    arb, sham = reports["ARBITRATION"], reports["SHAM"]
    dec_keys = ("self_loop_decisions", "slot_decisions", "slot_writes",
                "arbitration_calls", "supersedes", "separations", "identity_slots")
    da = arb["guard"]["arbitration"]["module_decisions"]
    ds = sham["guard"]["arbitration"]["module_decisions"]
    mismatch = {k: (da.get(k), ds.get(k)) for k in dec_keys if da.get(k) != ds.get(k)}
    sham_picks = sham["counters"].get("emitted_picks", 0)
    flat = all(num(endpoints(sham, p, "far")) <=
               num(endpoints(reports["OFF"], p, "far")) + 0.05
               for p in ("primary", "sealed"))
    add("L5_sham_matched_and_flat", not mismatch and sham_picks == 0 and flat,
        {"decision_mismatch": mismatch, "sham_emitted_picks": sham_picks,
         "sham_far": {p: endpoints(sham, p, "far") for p in ("primary", "sealed")},
         "off_far": {p: endpoints(reports["OFF"], p, "far")
                     for p in ("primary", "sealed")}})

    # L6 — mechanism actually fired (A5/A9).
    mdec = arb["guard"]["arbitration"]["module_decisions"]
    mechanism = (mdec.get("slot_writes", 0) > 0 and mdec.get("separations", 0) > 0
                 and mdec.get("identity_slots", 0) > 0)
    substrate = all(num(endpoints(reports["M123"], p, "far")) < 0.5
                    for p in ("primary", "sealed"))
    add("L6_mechanism_fired", mechanism and substrate,
        {"arbitration_decisions": mdec,
         "m123_far": {p: endpoints(reports["M123"], p, "far")
                      for p in ("primary", "sealed")}})

    # L7 — pre-registered primary endpoint: far >=0.80 on both pools and gain.
    far = {p: num(endpoints(arb, p, "far")) for p in ("primary", "sealed")}
    gain = {p: round(far[p] - num(endpoints(reports["M123"], p, "far")), 4)
            for p in ("primary", "sealed")}
    add("L7_far_reach_and_gain",
        all(v >= 0.80 for v in far.values())
        and all(g >= 0.50 for g in gain.values()),
        {"arbitration_far": far, "gain_over_m123": gain})

    # L8 — near retention / growth held; reject endpoints not worsened (A7).
    reg = {}
    for pool in ("primary", "sealed"):
        for key in ("near", "growth", "delayed_retention"):
            a = num(endpoints(arb, pool, key))
            b = num(endpoints(reports["M123"], pool, key))
            reg["%s.%s" % (pool, key)] = {"arbitration": a, "m123": b,
                                          "delta": round(a - b, 4)}
    hold = all(v["delta"] >= -0.05 for v in reg.values())
    reject_keys = ("foil", "negative", "unrelated")
    rej = {}
    for pool in ("primary", "sealed"):
        for key in reject_keys:
            a, b = endpoints(arb, pool, key), endpoints(reports["M123"], pool, key)
            if a is None or b is None:
                continue
            rej["%s.%s" % (pool, key)] = {"arbitration": a, "m123": b}
    no_worse = all(num(v["arbitration"]) <= num(v["m123"]) + 0.05
                   for v in rej.values())
    add("L8_near_growth_held_no_regression", hold and no_worse,
        {"retention": reg, "rejects": rej})

    # L9 — k=0 cold null unchanged and no answer-side change (A8).
    null = all(num(endpoints(arb, p, "baseline")) <= 0.05
               for p in ("primary", "sealed"))
    pins = all(bool(v) for v in (selftest["checks"]["sha_pins"] or {}).values())
    add("L9_cold_null_and_pins", null and pins,
        {"arb_baseline": {p: endpoints(arb, p, "baseline")
                          for p in ("primary", "sealed")}, "pins_ok": pins})

    return checks, details


def main():
    data, err = load()
    if err:
        print("VERDICT NO-GO", err)
        return 1
    reports, selftest = data
    checks, details = evaluate(reports, selftest)
    fired = [c["name"] for c in checks if not c["ok"]]
    passed = not fired
    ep = {}
    for name, rep in reports.items():
        ep[name] = {p: {k: endpoints(rep, p, k) for k in
                        ("near", "far", "foil", "negative", "growth",
                         "delayed_retention", "baseline")}
                    for p in ("primary", "sealed")}
    verdict = {
        "schema": "trak-arbitration-live-verdict-v1",
        "verdict": "PASS_MECHANISM_ONLY" if passed else "NO-GO",
        "full_bank_claim": False,
        "n_per_pool": 2,
        "falsifiers_in_order": checks,
        "first_falsifier_fired": fired[0] if fired else None,
        "endpoints": ep,
        "details": details,
        "note": ("Focused n=2/pool mechanism probe on isolated stores. "
                 "No full-bank verdict and no production default-path change."),
    }
    p = os.path.join(OUT, "verdict.json")
    json.dump(verdict, open(p, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("VERDICT", verdict["verdict"], "first_fired=", verdict["first_falsifier_fired"])
    for c in checks:
        print("  %-36s %s" % (c["name"], c["ok"]))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
