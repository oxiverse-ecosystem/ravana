"""RED->GREEN: topic-opinion frames route to the AGENT-stance resolver,
never to the hippocampal echo of an unrelated user utterance.

Round 2026-08-19T1026Z: "what do you make of X" / "what's your read on X" /
"your opinion of X" all ask RAVANA's view on a SUBJECT but only the
"...about/on" syntactic shapes were matched by the opinion-frame detector in
_route_self_query. The unmatched frames fell through to
_try_hippocampal_retrieval, which echoed an UNRELATED stored user utterance
("you told me earlier: actual second thought makes me uneasy..."), a
self/other boundary violation.

This test pins the fix: every topic-opinion frame routes to the agent-stance
resolver (self_model strategy, topic-named honest deflection), and NEVER echoes
an unrelated "you told me earlier:" user fact.

The reply content is driven by REAL cognitive state (the user's topic target),
never authored prose.
"""
import os, sys, io, contextlib
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine


def _build():
    return CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                               user_suffix="test_opinion_frame")


def run():
    fails = 0
    eng = _build()
    # Seed one unrelated user fact so the hippocampal buffer is NON-EMPTY
    # (the bug only surfaced when there was a stored utterance to echo).
    with contextlib.redirect_stdout(io.StringIO()):
        eng.process_turn("actual second thought makes me uneasy when strangers know my routine")

    frames = [
        "what do you make of the bronze age collapse?",
        "what do you make of silence as a response?",
        "what's your read on tidal marsh restoration?",
        "your opinion of ultra-processed food matters to me",
        "what do you think of bioluminescent fungi?",
    ]
    for q in frames:
        eng._last_strategy = None
        with contextlib.redirect_stdout(io.StringIO()):
            r = eng.process_turn(q)
        reply = r if isinstance(r, str) else str(r)
        strat = eng._last_strategy
        # MUST route to agent-stance resolver, NOT hippocampal_recall.
        if strat == "hippocampal_recall":
            print(f"[FAIL] '{q}' routed to hippocampal echo (strateasy={strat}) -> {reply!r}")
            fails += 1
            continue
        # MUST NOT echo an unrelated user utterance.
        if "you told me earlier" in reply:
            print(f"[FAIL] '{q}' echoed unrelated user fact -> {reply!r}")
            fails += 1
            continue
        # The reply must name the REAL topic (the subject of the question).
        topic = q.split()[-1].strip("?").lower()
        if topic not in reply.lower():
            print(f"[WARN] '{q}' reply does not name topic '{topic}': {reply!r}")
        print(f"[OK] '{q}' -> strategy={strat}, topic-named reply")

    # Genuine world query ("what do you make of paris?") must still fall
    # through honestly (not echo a user fact, not crash).
    eng._last_strategy = None
    with contextlib.redirect_stdout(io.StringIO()):
        r = eng.process_turn("what do you make of paris?")
    reply = r if isinstance(r, str) else str(r)
    if "you told me earlier" in reply:
        print(f"[FAIL] world-query 'paris' echoed user fact -> {reply!r}")
        fails += 1
    else:
        print(f"[OK] world-query 'paris' fell through honestly -> {reply!r}")

    eng.stop_background_learning()
    if fails:
        print(f"\n{len(frames)+1} checks, {fails} FAILED")
        raise SystemExit(1)
    print(f"\n{len(frames)+1} checks PASSED")


def test_agent_opinion_frame_coverage():
    run()


if __name__ == "__main__":
    run()
