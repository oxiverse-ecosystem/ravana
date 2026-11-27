#!/usr/bin/env python3
"""SCRATCH: cProfile a focused WFA PD-turn slice on the GloVe-present store.

Run only after the capped full run has stopped (no concurrent engine boots).
Uses the harness's own items, responders and run_pd_arms; no gate/threshold/
item/scoring change, no full-bank verdict.
"""
import cProfile
import os
import pstats
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scripts"))
sys.path.insert(0, os.path.join(WT, "ravana", "src"))
os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["RAVANA_PRIOR_ACT_TRACE"] = "1"
os.environ["RAVANA_GLOVE_BOOT_MEMO"] = "1"

import measure_learn5_wfa as wfa  # noqa: E402
import measure_fronts_web_frozen as mfwf  # noqa: E402
from ravana.chat.engine import CognitiveChatEngine  # noqa: E402

mfwf.install_frozen_patches()
items = wfa.build_items()
subset = ([it for it in items if it["kind"] == "gap"][:2] +
          [it for it in items if it["kind"] == "decoy"][:1])
records = [{"id": it["id"], "pool": it["pool"], "kind": it["kind"]} for it in subset]
print("PROFILE_ITEMS=%d" % len(subset), flush=True)

eng = CognitiveChatEngine(data_dir=os.path.join(WT, "weights"), user_suffix="")
arms = wfa.WFAArms(
    inclusion=wfa.EngineEndorsementResponder(eng, eng.reset_episodic_state),
    exclusion=wfa.EngineEndorsementResponder(eng, eng.reset_episodic_state),
    invalidation=wfa.EngineInvalidationResponder(eng, eng.reset_episodic_state),
    forced_choice=wfa.ForcedChoiceResponder(eng, eng.reset_episodic_state))

prof = cProfile.Profile()
t0 = time.time()
prof.enable()
wfa.run_pd_arms(subset, records, arms)
prof.disable()
elapsed = time.time() - t0
prof.dump_stats(os.path.join(WT, "scratch", "wfa_glove_turns_after_memo.prof"))
with open(os.path.join(WT, "scratch", "wfa_glove_turns_after_memo.stats"), "w") as fh:
    pstats.Stats(prof, stream=fh).sort_stats("cumulative").print_stats(60)
print("PROFILE_ELAPSED=%.1fs PROFILE_WRITTEN" % elapsed, flush=True)
