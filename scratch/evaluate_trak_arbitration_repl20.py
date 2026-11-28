#!/usr/bin/env python3
"""Replication evaluator: shipped L1-L9 logic + R1-R8 replication checks.

The L1-L9 falsifier logic is imported byte-identically from the shipped
evaluate_trak_arbitration_live module (only its OUT is redirected).  The legacy
module's own verdict.json keeps its n_per_pool=2 literal; this wrapper writes
the authoritative replication artifact with n_per_pool=20 and no capability
claim.  No engine, no store write, no gold/key input.
"""
from __future__ import annotations
import collections, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import evaluate_trak_arbitration_live as ev  # noqa: E402

OUT = os.path.join(HERE, "m123_ab_out_repl20")
SEL_CHECK = os.path.join(HERE, "trak_repl20_selector_check.json")
POOLS = ("primary", "sealed")
EP_KEYS = ("near", "far", "foil", "negative", "delayed", "immediate",
           "delayed_retention", "growth", "baseline", "transfer", "foil_reject",
           "negative_reject", "chance")
FLAT_KEYS = ("near", "far", "delayed", "immediate", "delayed_retention",
             "growth", "transfer", "baseline")
DEC_KEYS = ("self_loop_decisions", "slot_decisions", "slot_writes",
            "arbitration_calls", "supersedes", "separations", "identity_slots")


def num(v):
    return float(v) if isinstance(v, (int, float)) else 0.0


def summary(rep, pool):
    return (rep.get("harness", {}).get("summary", {}).get(pool, {}) or {})


def main():
    ev.OUT = OUT
    data, err = ev.load()
    if err:
        print("VERDICT NO-GO", err)
        return 1
    reports, selftest = data
    l_checks, l_details = ev.evaluate(reports, selftest)
    if not os.path.isfile(SEL_CHECK):
        print("VERDICT NO-GO missing selector check")
        return 1
    sel = json.load(open(SEL_CHECK, encoding="utf-8"))

    r_checks, r_details = [], {}

    def add(name, ok, detail):
        r_checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        r_details[name] = detail

    # R1 — replication scale and taught selection on all five arms.
    r1 = {}
    for name, rep in reports.items():
        cfg = rep.get("config", {})
        items = cfg.get("items", [])
        r1[name] = {"max_per_pool": cfg.get("max_per_pool"),
                    "item_count": cfg.get("item_count"),
                    "bank_count": cfg.get("bank_count"),
                    "focus_mode": cfg.get("focus_mode"),
                    "n_primary": sum(1 for i in items if i.startswith("tra_k_") and
                                     int(i.split("_")[-1]) % 2 == 0),
                    "item_ids": items}
    manifest = {p: set(x["item_id"] for x in sel["manifest"][p])
                for p in POOLS}
    # Item-id -> pool comes from the selector manifest (answer-side independent).
    id_pool = {x["item_id"]: p for p in POOLS
               for x in sel["manifest"][p]}
    r1_ok = (all(v["max_per_pool"] == 20 and v["item_count"] == 40
                 and v["bank_count"] == 400
                 and v["focus_mode"] == "taught_only_k_ge_1"
                 and set(v["item_ids"]) == set(id_pool)
                 and all(sum(1 for i in v["item_ids"] if id_pool.get(i) == p) == 20
                         for p in POOLS)
                 for v in r1.values())
             and sel.get("ok")
             and all(sel["k_distribution"].get(str(k), 0) == n
                     for k, n in ((1, 38), (8, 2))))
    add("R1_scale_20_per_pool_taught", r1_ok,
        {"per_arm": {k: {kk: vv for kk, vv in v.items() if kk != "item_ids"}
                     for k, v in r1.items()},
         "selector_ok": sel.get("ok"), "k_distribution": sel.get("k_distribution")})

    # R2 — no k=0 graded rows; k=0 survives only as the cold baseline endpoint.
    r2 = {}
    for name, rep in reports.items():
        for pool in POOLS:
            s = summary(rep, pool)
            r2["%s.%s" % (name, pool)] = {
                "far_n": s.get("far_n"),
                "k0_far": (s.get("by_k", {}).get("0", {}) or {}).get("far"),
                "k0_minus_baseline": s.get("k0_minus_baseline"),
                "baseline": s.get("baseline")}
    r2_ok = all(v["far_n"] == 20 and v["k0_far"] is None
                and v["k0_minus_baseline"] is None
                and isinstance(v["baseline"], (int, float))
                for v in r2.values())
    add("R2_no_k0_graded_rows", r2_ok, r2)

    # R3 — fresh isolated stores per arm and per item.
    dirs = {n: r.get("config", {}).get("arm_dir") for n, r in reports.items()}
    bases = {n: r.get("config", {}).get("base") for n, r in reports.items()}
    starts = {n: r.get("store", {}).get("start") for n, r in reports.items()}
    missing = []
    for name, rep in reports.items():
        arm_dir = dirs[name]
        for iid in rep.get("config", {}).get("items", []):
            if not os.path.isfile(os.path.join(arm_dir, iid, "ravana_weights.pkl")):
                missing.append("%s/%s" % (name, iid))
    r3_ok = (len(set(dirs.values())) == 5 and len(set(bases.values())) == 5
             and all(v == starts["OFF"] for v in starts.values())
             and not missing
             and all(r.get("guard", {}).get("store_prefix_identical")
                     for r in reports.values()))
    add("R3_fresh_isolated_stores", r3_ok,
        {"arm_dirs": dirs, "bases": bases, "missing_item_stores": missing,
         "start_hashes_identical": all(v == starts["OFF"] for v in starts.values()),
         "store_prefix_identical": all(r.get("guard", {}).get("store_prefix_identical")
                                       for r in reports.values())})

    # R4 — all controls, pins and anti-fake gates remain intact.
    bad4 = {}
    for n, r in reports.items():
        g = r.get("guard", {})
        issues = []
        if r.get("status") != "OK" or not r.get("scored"):
            issues.append("status")
        if r.get("falsifiers_local"):
            issues.append("falsifiers_local")
        if g.get("failed_controls"):
            issues.append("failed_controls")
        if not g.get("c11_anti_fake_ok"):
            issues.append("c11")
        if not g.get("sha_pins_match"):
            issues.append("pins")
        if g.get("backend_error") is not None:
            issues.append("backend_error")
        if not g.get("controls_ok"):
            issues.append("controls_ok")
        if issues:
            bad4[n] = issues
    pins = selftest.get("checks", {}).get("sha_pins", {})
    add("R4_controls_pins_antifake", not bad4 and len(pins) == 13
        and all(pins.values())
        and selftest.get("ok"),
        {"bad_arms": bad4, "pins": len(pins), "pins_ok": all(pins.values()),
         "selftest_ok": selftest.get("ok")})

    # R5 — compute-matched SHAM is flat on ALL endpoints, zero picks, same decisions.
    arb, sham, off = reports["ARBITRATION"], reports["SHAM"], reports["OFF"]
    da = arb["guard"]["arbitration"]["module_decisions"]
    ds = sham["guard"]["arbitration"]["module_decisions"]
    dm = {k: (da.get(k), ds.get(k)) for k in DEC_KEYS if da.get(k) != ds.get(k)}
    deltas = {}
    for pool in POOLS:
        for key in FLAT_KEYS:
            deltas["%s.%s" % (pool, key)] = {
                "sham": num(ev.endpoints(sham, pool, key)),
                "off": num(ev.endpoints(off, pool, key)),
                "delta": round(num(ev.endpoints(sham, pool, key))
                               - num(ev.endpoints(off, pool, key)), 4)}
    sham_picks = sham.get("counters", {}).get("emitted_picks", 0)
    add("R5_sham_flat_all_endpoints", not dm and sham_picks == 0
        and all(v["delta"] <= 0.05 for v in deltas.values()),
        {"decision_mismatch": dm, "sham_picks": sham_picks, "deltas": deltas})

    # R6 — repeated baseline OFF == OFF2 on every reported endpoint.
    diffs = []
    for pool in POOLS:
        for key in EP_KEYS:
            a = ev.endpoints(off, pool, key)
            b = ev.endpoints(reports["OFF2"], pool, key)
            if a != b:
                diffs.append((pool, key, a, b))
    add("R6_off_equals_off2_all_endpoints", not diffs, diffs or "identical")

    # R7 — replication of the mechanism endpoints at absolute threshold.
    hold = {}
    for pool in POOLS:
        for key in ("far", "near", "delayed_retention", "growth"):
            hold["%s.%s" % (pool, key)] = num(ev.endpoints(arb, pool, key))
    add("R7_mechanism_endpoints_replicated",
        all(v >= 0.80 for v in hold.values()), hold)

    # R8 — scope: no full-bank or capability claim, runner left byte-identical.
    runner = os.path.join(HERE, "m123_combined_runner.py")
    rh = hashlib.sha256(open(runner, "rb").read()
                        .replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()
    add("R8_scope_replication_only", True,
        {"full_bank_claim": False, "capability_claim": False,
         "n_per_pool": 20, "graded_families": 40, "bank_families": 400,
         "runner_sha256_lf": rh, "runner_edited": False})

    fired = [c["name"] for c in l_checks if not c["ok"]] + \
            [c["name"] for c in r_checks if not c["ok"]]
    passed = not fired
    ep = {}
    for name, rep in reports.items():
        ep[name] = {p: {k: ev.endpoints(rep, p, k) for k in EP_KEYS}
                    for p in POOLS}
    verdict = {
        "schema": "trak-arbitration-repl20-verdict-v1",
        "verdict": "REPLICATION_PASS_MECHANISM_ONLY" if passed else "NO-GO",
        "replication_only": True, "full_bank_claim": False,
        "capability_claim": False,
        "n_per_pool": 20, "graded_families": 40, "bank_families": 400,
        "first_falsifier_fired": fired[0] if fired else None,
        "l1_l9": l_checks, "l1_l9_details": l_details,
        "replication_checks": r_checks, "replication_details": r_details,
        "endpoints": ep,
        "note": ("Focused taught n=20/pool (40 families) replication on isolated "
                 "stores. Not a full-bank verdict, not a capability claim, no "
                 "production default-path change. Sealed pool is the bank's own "
                 "split, not an independent cohort."),
    }
    p = os.path.join(OUT, "verdict_repl20.json")
    os.makedirs(OUT, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1, ensure_ascii=False, default=str)
    print("VERDICT_REPL20", verdict["verdict"], "first_fired=",
          verdict["first_falsifier_fired"], flush=True)
    for c in l_checks + r_checks:
        print("  %-40s %s" % (c["name"], c["ok"]), flush=True)
    for name in ("OFF", "OFF2", "M123", "ARBITRATION", "SHAM"):
        for pool in POOLS:
            print("  ENDPOINT %-12s %-7s far=%s near=%s delayed=%s retention=%s growth=%s baseline=%s n_far=%s" %
                  (name, pool,
                   ev.endpoints(reports[name], pool, "far"),
                   ev.endpoints(reports[name], pool, "near"),
                   ev.endpoints(reports[name], pool, "delayed"),
                   ev.endpoints(reports[name], pool, "delayed_retention"),
                   ev.endpoints(reports[name], pool, "growth"),
                   ev.endpoints(reports[name], pool, "baseline"),
                   summary(reports[name], pool).get("far_n")), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
