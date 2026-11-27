#!/usr/bin/env python3
"""SCRATCH ONLY: cProfile the bounded WFA harness with the larger re cache.

Same bounded env as scratch/run_wfa_bounded.py; candidate cache installed before
main(). Dumps cumulative/tottime stats for after-fix hotspot triage.
"""
import cProfile
import os
import pstats
import re
import sys

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, os.path.join(WT, "ravana", "src"))
os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["WFA_BOUNDED_DIAGNOSTIC"] = "1"
os.environ["WFA_IS_MAX_PER_POOL"] = "2"
os.environ["WFA_IS_FRESH_ENGINE"] = "1"
re._MAXCACHE = 8192
re._MAXCACHE2 = 8192

import measure_learn5_wfa as wfa  # noqa: E402

prof = cProfile.Profile()
prof.enable()
rc = wfa.main([])
prof.disable()
out = os.path.join(WT, "scratch", "profile_bounded_after_re_cache.txt")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("rc=%s\n" % rc)
    ps = pstats.Stats(prof, stream=fh)
    ps.sort_stats("cumulative").print_stats(40)
    fh.write("\n===== tottime =====\n")
    ps.sort_stats("tottime").print_stats(25)
print("PROFILE_EXIT=%s -> %s" % (rc, out), flush=True)
