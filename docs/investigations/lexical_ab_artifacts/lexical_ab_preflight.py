#!/usr/bin/env python3
"""Cheap reachability pre-flight: one forced-choice turn, flag off vs on."""
import json, os, shutil, sys
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scratch"))
import lexical_ab_runner as r
from ravana.chat.engine import CognitiveChatEngine

r.patch_hook()
D = os.path.join(WT, "scratch", "ab_preflight")
shutil.rmtree(D, ignore_errors=True)
r.seed_data_dir(D)

out = {}
for flag in ("off", "on"):
    for k in ("calls", "emits", "forced_choice_turns"):
        r.STATS[k] = 0
    r.STATS["audit"] = []
    if flag == "on":
        os.environ["RAVANA_LEXICAL_SUPERORDINATE"] = "1"
    else:
        os.environ.pop("RAVANA_LEXICAL_SUPERORDINATE", None)
    eng = CognitiveChatEngine(data_dir=D, user_suffix="")
    # teach one trusted demo edge as RSI/TRA do
    eng.process_turn("gale causes rainbow.")
    resp = eng.process_turn(
        "Considering the examples above, what is true of thaw? "
        "Options: rainbow, storm, hail, breeze. Answer briefly.")
    out[flag] = {"calls": r.STATS["calls"], "emits": r.STATS["emits"],
                 "forced_choice_turns": r.STATS["forced_choice_turns"],
                 "response": str(resp)[:200],
                 "audit": r.STATS["audit"][-3:]}
    print(flag, json.dumps(out[flag], indent=1), flush=True)
json.dump(out, open(os.path.join(WT, "scratch", "ab_preflight_result.json"), "w",
                    encoding="utf-8"), indent=1)
print("DONE", flush=True)
