"""Verify item D: fail-closed retrieval on the LIVE engine.

1. _needs_web_search override: an informational/why query about a KNOWN subject
   with hollow graph edges (e.g. "dream", "sky") now returns True (was False ->
   answered from graph, never web-searched). This is the Q10/Q14 root cause fix.
2. web_unverified degradation: when the web is forced (informational) but the
   search pipeline yields nothing, _web_direct_answer returns an honest
   "couldn't verify" abstention rather than None (which would fall to hollow
   graph). Simulated by monkeypatching search() to always raise SearchError.

Mirrors scripts/ravana_chat.py import order.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src"),
         os.path.join(_PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.models import CognitiveResponseContext

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                          data_dir="/tmp/ravana_d_test")

# ---- (1) informational override on a subject that IS a graph node ----
# Find a real noun node with >=3 strong outgoing edges so the edge-count gate
# would otherwise return False (proving the informational override fires).
known_subject = None
for cand in eng._concept_keywords:
    nids = eng._concept_keywords[cand]
    if not nids:
        continue
    nid = nids[0]
    se = 0
    for tid, e in eng.graph.get_outgoing(nid):
        if e.weight > 0.3:
            se += 1
    for src, e in eng.graph.get_incoming(nid):
        if e.weight > 0.3:
            se += 1
    if se >= 3:
        known_subject = cand
        break
assert known_subject, "no graph node with >=3 strong edges found for baseline test"
print(f"[D] baseline subject (>=3 strong edges) = {known_subject!r}")
q_why = f"why is {known_subject} important"
q_what = f"what is {known_subject}"
needs_why = eng._needs_web_search(known_subject, query=q_why)
needs_what = eng._needs_web_search(known_subject, query=q_what)
# baseline: same subject, NON-informational utterance must NOT force web
needs_chitchat = eng._needs_web_search(known_subject, query=f"i was thinking about {known_subject} today")
print(f"[D] _needs_web_search({known_subject!r}, why)     = {needs_why}  (expect True)")
print(f"[D] _needs_web_search({known_subject!r}, what)    = {needs_what}  (expect True)")
print(f"[D] _needs_web_search({known_subject!r}, chitchat)= {needs_chitchat}  (expect False)")

# ---- (2) web_unverified graceful degradation ----
orig_search = eng.search_engine.search
def _empty_search(*a, **k):
    raise type('SearchError', (Exception,), {})(  # emulate SearchError
        "All search APIs failed (simulated)")
# Use the real SearchError class from the engine's search module.
import ravana.web.learner as _lw
def _empty_search2(*a, **k):
    raise _lw.SearchError("All search APIs failed (simulated)")
eng.search_engine.search = _empty_search2

ctx = CognitiveResponseContext(subject=known_subject, raw_input=q_why,
                               associated_concepts=[("honesty", 0.8)])
ans, strat = eng._web_direct_answer(ctx)
print(f"[D] web_direct_answer (all-empty) -> {ans!r} [{strat}]")
print(f"    abstains_honestly = {strat == 'web_unverified' and ans is not None}  (expect True)")

eng.search_engine.search = orig_search
eng.stop_background_learning()

ok = (needs_why and needs_what and not needs_chitchat
      and strat == "web_unverified" and ans is not None)
print("\nVERDICT:", "CONFIRMED — informational queries force web; empty web aborts honestly."
      if ok else "CHECK — a D assertion failed.")
raise SystemExit(0 if ok else 1)
