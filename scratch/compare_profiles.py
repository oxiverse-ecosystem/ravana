#!/usr/bin/env python3
"""Compare PR #549 profile_summary.json (main repo) with the current fixed-main one."""
import json
import os

OLD = r"C:\Users\Likhith\Documents\projects\ravana\scratch\profile_summary.json"
NEW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile_summary.json")


def short(path):
    parts = path.replace("\\", "/").split("/")
    return parts[-1]


def fmt(row):
    return "%9.4f cum %9.4f tot %10d nc  %s:%s %s" % (
        row["cumtime"], row["tottime"], row["ncalls"],
        short(row["file"]), row["line"], row["func"])


old = json.load(open(OLD, encoding="utf-8"))
new = json.load(open(NEW, encoding="utf-8"))
print("OLD boot_s=%s nodes=%s glove=%s" % (old["boot_s"], old["nodes"], old["glove_words"]))
print("NEW boot_s=%s nodes=%s glove=%s" % (new["boot_s"], new["nodes"], new["glove_words"]))
for ot, nt in zip(old["turns"], new["turns"]):
    print("\n" + "=" * 100)
    print("TURN %s  prompt_len old=%d new=%d  wall old=%.3fs new=%.3fs  ratio=%.2fx" % (
        nt["name"], ot["prompt_len"], nt["prompt_len"], ot["wall_s"],
        nt["wall_s"], ot["wall_s"] / max(nt["wall_s"], 1e-9)))
    print("  OLD activate=%s ground=%s" % (
        [(r["ncalls"], r["cumtime"]) for r in ot["activate_rows"]],
        [(r["ncalls"], r["cumtime"]) for r in ot["ground_query_rows"]]))
    print("  NEW activate=%s ground=%s" % (
        [(r["ncalls"], r["cumtime"]) for r in nt["activate_rows"]],
        [(r["ncalls"], r["cumtime"]) for r in nt["ground_query_rows"]]))
    old_top = {(r["func"], r["line"]): r for r in ot["top_cumulative"]}
    print("\n  NEW TOP CUMULATIVE (20)  [old cumtime / old ncalls where available]")
    for r in nt["top_cumulative"][:20]:
        key = (r["func"], r["line"])
        prior = old_top.get(key)
        suffix = ""
        if prior:
            suffix = " | OLD cum=%.4f nc=%d" % (prior["cumtime"], prior["ncalls"])
        print("   " + fmt(r) + suffix)
    print("\n  OLD TOP CUMULATIVE (12, for reference)")
    for r in ot["top_cumulative"][:12]:
        print("   " + fmt(r))
