#!/usr/bin/env python3
"""SCRATCH ONLY: identify dynamic regex patterns in a short WFA turn sequence.

Imports the UNMODIFIED production harness/engine, installs the harness's frozen
boundary, and wraps re.compile to count pattern/flag pairs and their callers.
No production file is edited.
"""
import collections
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

_orig_compile = re.compile
_counts = collections.Counter()          # (pattern, flags) -> count
_sites = collections.Counter()           # caller site -> count
_tot = [0]
_uncached = collections.Counter()


def _wrapped(pattern, flags=0):
    _tot[0] += 1
    key = (pattern if isinstance(pattern, str) else "<pattern>", int(flags or 0))
    _counts[key] += 1
    frame = sys._getframe(1)
    # re.search/sub/match/finditer call re.compile; the real caller is 2 up.
    outer = sys._getframe(2)
    loc = "%s:%s" % (os.path.basename(outer.f_code.co_filename), outer.f_lineno)
    _sites[loc] += 1
    if isinstance(pattern, str) and (pattern, int(flags or 0)) not in re._cache:
        _uncached[key] += 1
    return _orig_compile(pattern, flags)


re.compile = _wrapped

from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

data_dir = os.path.join(WT, "weights")
t0 = time.perf_counter()
eng = CognitiveChatEngine(data_dir=data_dir, user_suffix="")
print("boot %.2fs" % (time.perf_counter() - t0), flush=True)

items = wfa.build_items()
gap = [i for i in items if i.get("kind") == "gap"][:4]
prompts = []
for it in gap:
    prompts.append(wfa.endorsement_prompt(it, "inclusion"))
    prompts.append(wfa.endorsement_prompt(it, "exclusion"))
    prompts.append(wfa.forced_choice_prompt(it, "forced_choice"))

# re.compile wrapper installed only for the measured turn section.
_mine_calls = [0]
import ravana.chat.user_model as um  # noqa: E402
_mine_orig = um.UserModel.mine_personal_facts
_mine_t = [0.0]


def _mine(self, *a, **kw):
    _mine_calls[0] += 1
    t = time.perf_counter()
    try:
        return _mine_orig(self, *a, **kw)
    finally:
        _mine_t[0] += time.perf_counter() - t


um.UserModel.mine_personal_facts = _mine

t0 = time.perf_counter()
answers = []
for p in prompts:
    eng.reset_episodic_state()
    answers.append((eng.process_turn(p) or "").strip())
elapsed = time.perf_counter() - t0
print("turns=%d elapsed=%.2fs mine_calls=%d mine_time=%.2fs compile_calls=%d"
      % (len(prompts), elapsed, _mine_calls[0], _mine_t[0], _tot[0]), flush=True)
print("answers:", answers, flush=True)

print("\n=== unique dynamic patterns by compile calls (top 25) ===", flush=True)
for (pat, flags), n in _counts.most_common(25):
    print("%6d uncached=%6d flags=%d %s" %
          (n, _uncached[(pat, flags)], flags, repr(pat)[:220]), flush=True)
print("\n=== call sites (top 20) ===", flush=True)
for site, n in _sites.most_common(20):
    print("%6d %s" % (n, site), flush=True)
print("\nunique_patterns=%d unique_uncached=%d" %
      (len(_counts), len(_uncached)), flush=True)
