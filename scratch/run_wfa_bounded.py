#!/usr/bin/env python3
"""SCRATCH ONLY: run the UNMODIFIED bounded WFA harness under A or B re cache.

A = stdlib default re cache (512/256). B = larger re cache (8192/8192),
installed from this scratch driver before the harness main() runs.
Outputs are copied to side-specific files for byte comparison.
"""
import json
import os
import re
import shutil
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, os.path.join(WT, "ravana", "src"))
os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["WFA_BOUNDED_DIAGNOSTIC"] = "1"
os.environ["WFA_IS_MAX_PER_POOL"] = os.environ.get("WFA_MAX", "2")
os.environ["WFA_IS_FRESH_ENGINE"] = "1"

SIDE = os.environ.get("SIDE", "A")
if SIDE == "B":
    re._MAXCACHE = 8192
    re._MAXCACHE2 = 8192

import measure_learn5_wfa as wfa  # noqa: E402

_real = re._compiler.compile
_n = [0]


def _counted(pattern, flags):
    _n[0] += 1
    return _real(pattern, flags)


re._compiler.compile = _counted
t0 = time.time()
rc = wfa.main([])
elapsed = time.time() - t0
re._compiler.compile = _real
print("SIDE=%s WFA_EXIT=%s ELAPSED=%.1fs REAL_COMPILES=%d" % (
    SIDE, rc, elapsed, _n[0]), flush=True)
for src, suffix in (("learn5_wfa_summary.json", "summary.json"),
                    ("learn5_wfa_probes.jsonl", "probes.jsonl")):
    dst = os.path.join(WT, "scratch", "bounded_%s_%s" % (SIDE, suffix))
    shutil.copyfile(os.path.join(WT, "scratch", src), dst)
with open(os.path.join(WT, "scratch", "bounded_%s_meta.json" % SIDE), "w",
          encoding="utf-8") as fh:
    json.dump({"side": SIDE, "rc": rc, "elapsed_s": round(elapsed, 3),
               "real_compiles": _n[0],
               "max_per_pool": os.environ["WFA_IS_MAX_PER_POOL"]}, fh, indent=2)
