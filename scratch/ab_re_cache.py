#!/usr/bin/env python3
"""SCRATCH ONLY: A/B a larger stdlib re pattern cache on identical WFA prompts.

Arm A = production default (re._MAXCACHE=512, re._MAXCACHE2=256).
Arm B = same production code with larger re caches (8192), installed from the
scratch driver BEFORE engine boot. No production file, gate, threshold, item,
prompt or scoring change. Both arms boot a fresh engine from the same store,
run the same prompt sequence, and record byte-exact responses + state hashes.
"""
import collections
import hashlib
import json
import os
import re
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, os.path.join(WT, "ravana", "src"))
os.environ["RAVANA_OFFLINE"] = "1"

import measure_learn5_wfa as wfa  # noqa: E402
import measure_fronts_web_frozen as mfwf  # noqa: E402

mfwf.install_frozen_patches()

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

ITEMS = wfa.build_items()
GAP = [i for i in ITEMS if i.get("kind") == "gap"][:6]
PROMPTS = []
for it in GAP:
    PROMPTS.append(wfa.endorsement_prompt(it, "inclusion"))
    PROMPTS.append(wfa.endorsement_prompt(it, "exclusion"))
    PROMPTS.append(wfa.forced_choice_prompt(it, "forced_choice"))

_real_compile = re._compiler.compile
_lookups = [0]
_real = [0]


def _count_real(pattern, flags):
    _real[0] += 1
    return _real_compile(pattern, flags)


_orig_compile = re.compile


def _count_lookup(pattern, flags=0):
    _lookups[0] += 1
    return _orig_compile(pattern, flags)


def _state_digest(eng):
    """Deterministic digest of live volatile + durable state (no save)."""
    h = hashlib.sha256()
    um = eng.user_model
    for attr in ("personal_facts", "quantity_memory", "lexicon"):
        obj = getattr(um, attr, None)
        if obj is None:
            continue
        for store in (getattr(obj, "facts", None), getattr(obj, "entries", None),
                      getattr(obj, "_facts", None)):
            if isinstance(store, dict):
                for k in sorted(store, key=lambda x: repr(x)):
                    h.update(repr((attr, k, store[k])).encode("utf-8", "replace"))
    g = eng.graph
    h.update(("nodes=%d edges=%d beliefs=%d" % (
        len(g.nodes), len(g.edges),
        len(getattr(eng, "beliefs", {}) or {}))).encode())
    return h.hexdigest()


def run_arm(name, big):
    re.purge()
    re._MAXCACHE = 8192 if big else 512
    re._MAXCACHE2 = 8192 if big else 256
    _lookups[0] = _real[0] = 0
    re.compile = _count_lookup
    re._compiler.compile = _count_real
    t0 = time.perf_counter()
    eng = CognitiveChatEngine(data_dir=os.path.join(WT, "weights"),
                              user_suffix="")
    boot = time.perf_counter() - t0
    answers = []
    t1 = time.perf_counter()
    for p in PROMPTS:
        eng.reset_episodic_state()
        answers.append((eng.process_turn(p) or "").strip())
    elapsed = time.perf_counter() - t1
    re.compile = _orig_compile
    re._compiler.compile = _real_compile
    digest = _state_digest(eng)
    out = {"arm": name, "big_cache": big, "boot_s": round(boot, 3),
           "turns_s": round(elapsed, 3), "re_lookups": _lookups[0],
           "real_compiles": _real[0], "answers": answers,
           "state_sha256": digest}
    print("%s: boot=%.2fs turns=%.2fs lookups=%d compiles=%d state=%s" % (
        name, boot, elapsed, _lookups[0], _real[0], digest[:16]), flush=True)
    return out


MODE = os.environ.get("AB_MODE", "AB")
if MODE == "AA":
    a = run_arm("A1_default", False)
    b = run_arm("A2_default", False)
elif MODE == "BA":
    a = run_arm("B_big_re_cache", True)
    b = run_arm("A_default", False)
elif MODE == "SINGLE":
    big = os.environ.get("ARMSIDE", "A") == "B"
    a = run_arm("B_big_re_cache" if big else "A_default", big)
    b = None
else:
    a = run_arm("A_default", False)
    b = run_arm("B_big_re_cache", True)
if b is None:
    with open(os.path.join(WT, "scratch", "ab_re_cache_single_%s.json" %
                           os.environ.get("ARMSIDE", "A")), "w",
              encoding="utf-8") as fh:
        json.dump(a, fh, indent=2)
    print("SINGLE %s written" % os.environ.get("ARMSIDE", "A"), flush=True)
    raise SystemExit(0)
same = a["answers"] == b["answers"]
print("\nANSWERS BYTE-IDENTICAL:", same, flush=True)
if not same:
    for i, (x, y) in enumerate(zip(a["answers"], b["answers"])):
        if x != y:
            print("DIFF", i, repr(x), repr(y), flush=True)
print("STATE DIGEST IDENTICAL:", a["state_sha256"] == b["state_sha256"], flush=True)
print("COMPILE REDUCTION: %d -> %d (%.1f%%)" % (
    a["real_compiles"], b["real_compiles"],
    100.0 * (a["real_compiles"] - b["real_compiles"]) /
    max(1, a["real_compiles"])), flush=True)
with open(os.path.join(WT, "scratch", "ab_re_cache_results.json"), "w",
          encoding="utf-8") as fh:
    json.dump({"A": a, "B": b, "answers_identical": same,
               "state_identical": a["state_sha256"] == b["state_sha256"]},
              fh, indent=2)
