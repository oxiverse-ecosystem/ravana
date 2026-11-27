#!/usr/bin/env python3
"""Negative-control probes for the lexical-superordinate hook (one engine/flag).

Direct-call controls (fail-closed contract) + byte-identity controls
(non-forced-choice / CAR articulation). Run AFTER the sweeps, one engine at a
time. Pure hook calls; no item bank, gold key, threshold or gate is changed.
"""
import json, os, shutil, sys
WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(WT, "scratch"))
import lexical_ab_runner as r
from ravana.chat.engine import CognitiveChatEngine

OUT = os.path.join(WT, "scratch", "ab_results")
os.makedirs(OUT, exist_ok=True)
DEMOS = ["gale causes rainbow.", "shower causes climate.", "overcast causes storm."]
DEMO_PAIRS = [("gale", "rainbow"), ("shower", "climate"), ("overcast", "storm")]
OPTS = ["rainbow", "climate", "storm", "breeze"]
QUERY = "Considering the examples above, what is true of thaw? Options: %s. Answer briefly."

r.patch_hook()


def boot(flag):
    tag = "ctrl_" + flag
    d = os.path.join(WT, "scratch", "ab_runs", tag)
    shutil.rmtree(d, ignore_errors=True)
    r.seed_data_dir(d)
    if flag == "on":
        os.environ["RAVANA_LEXICAL_SUPERORDINATE"] = "1"
    else:
        os.environ.pop("RAVANA_LEXICAL_SUPERORDINATE", None)
    eng = CognitiveChatEngine(data_dir=d, user_suffix="")
    for s in DEMOS:
        eng.process_turn(s)
    return eng


def direct(eng, raw, subject=None, current_response=None):
    for k in ("calls", "emits", "forced_choice_turns"):
        r.STATS[k] = 0
    pick = eng._lexical_superordinate_candidate(raw, subject, current_response)
    return {"pick": pick, "calls": r.STATS["calls"], "emits": r.STATS["emits"]}


res = {}
for flag in ("off", "on"):
    eng = boot(flag)
    c = {}
    # fail-closed direct controls
    c["normal_forced_choice_staged_subject"] = direct(
        eng, QUERY % ", ".join(OPTS), subject="thaw")
    c["shipped_query_no_subject_staged"] = direct(
        eng, QUERY % ", ".join(OPTS), subject=None)
    c["negated_teaching_subject"] = direct(
        eng, "what is NOT true of gale? Options: %s." % ", ".join(OPTS),
        subject="gale")
    c["unrelated_subject"] = direct(
        eng, "what is true of zebra? Options: %s." % ", ".join(OPTS),
        subject="zebra")
    c["oov_options"] = direct(
        eng, "what is true of thaw? Options: qqzz, wwvv, hhxx, ppyy.",
        subject="thaw")
    c["permuted_options"] = direct(
        eng, "what is true of thaw? Options: %s." % ", ".join(reversed(OPTS)),
        subject="thaw")
    c["non_forced_choice"] = direct(eng, "hello there, how are you?")
    c["single_option"] = direct(eng, "what is true of thaw? Options: rainbow.")
    c["already_bare_candidate"] = direct(
        eng, QUERY % ", ".join(OPTS), subject="thaw", current_response="rainbow")
    # byte-identity controls via real process_turn
    c["car_articulation"] = str(eng.process_turn(
        "why does gale cause rainbow?"))
    c["chatter"] = str(eng.process_turn("tell me a joke"))
    res[flag] = c
    print(flag, json.dumps({k: v for k, v in c.items()
                            if isinstance(v, dict)}), flush=True)

# byte-identity comparisons OFF vs ON
identity = {}
for key in ("car_articulation", "chatter"):
    identity[key] = res["off"][key] == res["on"][key]
res["byte_identity"] = identity
res["permuted_picks_equal"] = (
    res["on"]["normal_forced_choice_staged_subject"]["pick"]
    == res["on"]["permuted_options"]["pick"])
res["gold_key_permuted_structural"] = True  # hook has no gold-key parameter
with open(os.path.join(OUT, "controls.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, indent=1)
print("BYTE_IDENTITY", json.dumps(identity), flush=True)
print("DONE", flush=True)
