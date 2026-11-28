#!/usr/bin/env python3
"""Boot-free replication selector check for --max-per-pool 20 --taught-focus.

No engine boot, no store write, no network. Imports the SHIPPED runner's
focused_items_taught() and the SHIPPED harness summarize() on the fake backend.
"""
from __future__ import annotations
import collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
WT = os.path.dirname(HERE)
for _p in (HERE, WT, os.path.join(WT, "scripts"),
           os.path.join(WT, "ravana", "src"),
           os.path.join(WT, "ravana_ml", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("RAVANA_OFFLINE", "1")

import m123_preflight as pf            # noqa: E402
import measure_tra_k as mtk            # noqa: E402
import m123_combined_runner as r        # noqa: E402 (shipped, unmodified)


def main():
    checks = []

    def add(name, ok, detail=""):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    bank, _channel = pf.load_bank()
    sel = r.focused_items_taught(bank, 20)
    per_pool = collections.Counter(it["pool"] for it in sel)
    k_dist = collections.Counter(int(it["k"]) for it in sel)
    manifest = {p: [{"item_id": it["item_id"], "k": int(it["k"])}
                    for it in sel if it["pool"] == p]
                for p in ("primary", "sealed")}

    add("repl20_item_count", len(sel) == 40, {"n": len(sel)})
    add("repl20_per_pool_20", all(per_pool[p] == 20 for p in ("primary", "sealed")),
        dict(per_pool))
    add("repl20_all_taught", all(int(it["k"]) >= 1 for it in sel), dict(k_dist))
    add("repl20_exactly_one_k8_per_pool",
        all(sum(1 for it in sel if it["pool"] == p and int(it["k"]) == 8) == 1
            for p in ("primary", "sealed")), dict(k_dist))
    add("repl20_no_k0_graded", all(int(it["k"]) != 0 for it in sel),
        {"k0": k_dist.get(0, 0)})

    minimal = [{"pool": it["pool"], "k": it["k"]} for it in bank]
    minimal_pairs = [(it["pool"], it["k"])
                     for it in r.focused_items_taught(minimal, 20)]
    full_pairs = [(it["pool"], it["k"]) for it in sel]
    add("repl20_ignores_answer_side_fields", minimal_pairs == full_pairs,
        {"minimal_n": len(minimal_pairs), "full_n": len(full_pairs)})

    ast = r.ast_anti_leakage_taught_focus()
    add("repl20_ast_anti_leakage", ast["ok"], ast["problems"])

    # Unmodified summarize() on the focused subset, shipped fake backend.
    rows = mtk.run_sweep(sel, backend=mtk.FakeBackend())
    summary = mtk.summarize(sel, rows)
    baseline = {p: summary[p].get("baseline") for p in ("primary", "sealed")}
    far_n = {p: summary[p].get("far_n") for p in ("primary", "sealed")}
    k0_far = {p: summary[p]["by_k"]["0"]["far"]
              for p in ("primary", "sealed")}
    k0_delta = {p: summary[p].get("k0_minus_baseline")
                for p in ("primary", "sealed")}
    add("repl20_k0_baseline_only_no_graded_rows",
        all(isinstance(baseline[p], (int, float)) for p in baseline)
        and all(far_n[p] == 20 for p in far_n)
        and all(k0_far[p] is None for p in k0_far)
        and all(k0_delta[p] is None for p in k0_delta),
        {"baseline": baseline, "far_n": far_n, "k0_far": k0_far,
         "k0_minus_baseline": k0_delta})

    ok = all(c["ok"] for c in checks)
    report = {
        "schema": "trak-arbitration-repl20-selector-v1",
        "ok": ok, "passed": sum(c["ok"] for c in checks), "total": len(checks),
        "engine_booted": False, "stores_written": False,
        "runner": r.__file__, "max_per_pool": 20,
        "k_distribution": dict(k_dist), "manifest": manifest,
        "checks": checks,
    }
    out = os.path.join(HERE, "trak_repl20_selector_check.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False, default=str)
    print("REPL20_SELECTOR", "OK" if ok else "INVALID",
          json.dumps({"passed": report["passed"], "total": report["total"],
                      "failed": [c["name"] for c in checks if not c["ok"]],
                      "k_distribution": dict(k_dist)}, default=str), flush=True)
    print("DONE", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
