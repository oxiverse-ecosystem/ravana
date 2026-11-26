#!/usr/bin/env python3
"""SCRATCH ONLY: cProfile the bounded (2/pool) WFA run with a hard 180 s watchdog.

Runs the UNMODIFIED production harness (scripts/measure_learn5_wfa.py at 90623cc)
with the exact bounded env from the parent task, dumps cProfile stats sorted by
cumulative time when the watchdog fires. No production file is edited.
"""
import cProfile
import os
import pstats
import sys
import threading
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, os.path.join(WT, "ravana", "src"))

os.environ["RAVANA_OFFLINE"] = "1"
os.environ["WFA_IS_FRESH_ENGINE"] = "1"
os.environ["WFA_IS_MAX_PER_POOL"] = "2"

import measure_learn5_wfa as wfa  # noqa: E402

OUT = os.path.join(WT, "scratch", "profile_bounded_timeout.stats")
DURATION = float(os.environ.get("PROFILE_SECONDS", "180"))
_counts = {"inc_exc": 0, "inv": 0, "fc": 0, "boot": 0, "turn": 0}


def _tagged(cls_name, key):
    cls = getattr(wfa, cls_name)
    orig = cls.__call__

    def wrapped(self, *a, **kw):
        _counts[key] += 1
        if _counts[key] % 10 == 0:
            print("[COUNT] %s=%d t=%.1f" % (key, _counts[key], time.time() - T0),
                  flush=True)
        return orig(self, *a, **kw)
    cls.__call__ = wrapped


T0 = time.time()
for _cls, _key in (("EngineEndorsementResponder", "inc_exc"),
                   ("EngineInvalidationResponder", "inv"),
                   ("ForcedChoiceResponder", "fc")):
    _tagged(_cls, _key)

prof = cProfile.Profile()
_dumped = {"done": False}


def _dump():
    if _dumped["done"]:
        return
    _dumped["done"] = True
    try:
        prof.disable()
    except Exception:
        pass
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("WATCHDOG dumped at %.1fs\n" % (time.time() - T0))
        fh.write("counts=%r\n\n" % (_counts,))
        ps = pstats.Stats(prof, stream=fh)
        ps.sort_stats("cumulative").print_stats(70)
        fh.write("\n===== by tottime =====\n")
        ps.sort_stats("tottime").print_stats(30)
    print("[WATCHDOG] dumped stats -> %s counts=%r" % (OUT, _counts), flush=True)
    os._exit(0)


timer = threading.Timer(DURATION, _dump)
timer.daemon = True
timer.start()

print("[PROFILE] wt=%s py=%s" % (WT, sys.executable), flush=True)
print("[PROFILE] harness=%s" % wfa.__file__, flush=True)
print("[PROFILE] env IS_FRESH=%s IS_MAX=%s OFFLINE=%s duration=%.0fs" % (
    os.environ.get("WFA_IS_FRESH_ENGINE"), os.environ.get("WFA_IS_MAX_PER_POOL"),
    os.environ.get("RAVANA_OFFLINE"), DURATION), flush=True)

prof.enable()
try:
    wfa.main([])
finally:
    prof.disable()
    timer.cancel()
    if not _dumped["done"]:
        _dumped["done"] = True
        with open(OUT, "w", encoding="utf-8") as fh:
            fh.write("COMPLETED at %.1fs\n" % (time.time() - T0))
            fh.write("counts=%r\n\n" % (_counts,))
            ps = pstats.Stats(prof, stream=fh)
            ps.sort_stats("cumulative").print_stats(70)
            fh.write("\n===== by tottime =====\n")
            ps.sort_stats("tottime").print_stats(30)
    print("[PROFILE] completed in %.1fs counts=%r" % (time.time() - T0, _counts),
          flush=True)
