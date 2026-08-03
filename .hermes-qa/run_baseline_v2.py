"""Round v2 baseline driver: 70+ NEW (rotating) probes, offline, in-process.

Captures before/after cognitive-state snapshots + verbatim transcript so the
round can judge whether RAVANA learns/feels like a personality, and reproduce
the genuine defects found in prior (reverted) rounds WITHOUT hardcoding.
"""
import os, sys, time, json, datetime

os.environ.setdefault("RAVANA_OFFLINE", "1")
REPO = r"C:\Users\Likhith\Documents\Projects\ravana"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(REPO, "ravana", "src"))

from ravana.chat.engine import CognitiveChatEngine

SUFFIX = "round_v2_" + datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
OUTDIR = os.path.join(REPO, ".hermes-qa")
os.makedirs(OUTDIR, exist_ok=True)
TRANSCRIPT = os.path.join(OUTDIR, "chat_log.txt")
SNAP_BEFORE = os.path.join(OUTDIR, "state_before_v2.json")
SNAP_AFTER = os.path.join(OUTDIR, "state_after_v2.json")

# ── 70+ NEW, rotating probes (persona: Soren, a lighthouse keeper) ──
TURNS = [
    # identity / self questions (rotated phrasing vs prior rounds)
    "hi ravana, i'm soren",
    "what are you, exactly?",
    "are you a person, or something else?",
    "how do you see yourself?",
    "do you have a sense of who you are yet?",
    # opinion + polarity, including negation and strong verbs
    "i love the solitude of the lighthouse",
    "i hate small talk at the village market",
    "i don't like crowded ferries",
    "lighthouse keeping is the best job in the world",
    "i really detest when the foghorn won't stop",
    # personal facts stated early (for delayed recall ~later)
    "my daughter's name is ingrid",
    "i'm a vegetarian",
    "i'm allergic to peanuts",
    "i've kept the light for fourteen years",
    "i play the accordion when the wind dies down",
    # topic far outside seed concepts
    "what do you make of bioluminescent plankton in the fjord?",
    "how do you raise a child to love the sea?",
    "what's your take on preserving old lighthouses?",
    "do you think seabirds should be protected?",
    # emotional / valence swing (NON-distress 'haunted' to test support misfire)
    "the empty keeper's cottage down the cliff feels haunted, it gives the place character",
    "i felt such peace watching the aurora last night",
    "i'm furious the council wants to automate my light",
    # contradiction test (earlier said loves solitude; now reversed)
    "actually, i can't stand being alone out here anymore",
    "i take it back, the solitude drains me now",
    # repeated topic to test reinforcement
    "the lighthouse is my whole world",
    "the lighthouse means everything to me",
    "nothing matters more than the lighthouse",
    # follow-up requiring it to remember what IT said
    "what did you say about who you are?",
    "earlier you described yourself — what was that?",
    # knowledge far outside + chain
    "why do octopuses have blue blood?",
    "if the sea rises two meters, what happens to my island?",
    # opinion on a fresh topic
    "i think wind farms off the coast are beautiful",
    "renewable energy is the only sane path forward, don't you agree?",
    "i believe tradition matters more than efficiency",
    # self-knowledge / learned-about-me
    "what have you learned about me so far?",
    "what do you remember that i told you?",
    "who did i say my daughter was?",
    # more persona depth + delayed recall anchor
    "ingrid wants to study marine biology in bergen",
    "my wife's name is astrid and she runs the bakery",
    "we eat dinner at six every evening",
    # contradiction in facts (correction)
    "no, my daughter's name is not ingrid, it's freya",
    "actually i'm not a vegetarian, i eat fish",
    # emotional swing again (genuine distress this time)
    "my dog died yesterday and i can't stop crying",
    "i feel so alone out here without him",
    # topic reinforcement + new
    "the lighthouse lamp is broken and i can't fix it",
    "i love the sound of the waves at night",
    "i hate the long polar night in december",
    # abstract / reflective (rotated from prior 'meaning of life')
    "what do you think the point of keeping a light is, really?",
    "does anything we do out here matter in the end?",
    # questions about its own stances
    "do you like music?",
    "what do you think about the sea?",
    "are you wary of crowds the way i am?",
    # far-out + nonsense-ish to test graceful uncertainty
    "how do you brew kvass from birch sap?",
    "what's the deal with tardigrades surviving space?",
    # delayed recall of early facts (~30+ turns later)
    "what did i tell you my daughter's name is?",
    "am i a vegetarian, or did i say something else?",
    "do you remember what instrument i play?",
    "where does my wife work?",
    # contradiction held? (solitude)
    "do you think i like being alone?",
    # opinions restated / reinforced
    "i still hate small talk, more than ever",
    "the foghorn is the worst sound invented",
    # new opinion, then flip
    "i love reading poetry by the lamp",
    "no, poetry bores me actually",
    # identity-ish, rotated
    "who am i, to you?",
    "do you think of me as a friend?",
    # knowledge-miss + correction loop
    "what is a pharologist?",
    "no, that's not right, a pharologist studies lighthouses",
    # emotional, light
    "i'm excited, spring migration is starting",
    "the terns are back and it makes me glad",
    # final self-reflection
    "how have you changed since we started talking?",
    "what do you know about me now?",
    "tell me something you've figured out about me",
    "do you remember astrid and the bakery?",
    "what matters most to me, from your view?",
    "one last thing — am i allergic to anything?",
    "and what instrument do i play when it's calm?",
]

def _str_keys(o):
    if isinstance(o, dict):
        return {str(k): _str_keys(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_str_keys(x) for x in o]
    return o

def snapshot(eng):
    idst = eng.identity.get_status()
    stances = {k: {"polarity": v.polarity, "confidence": v.confidence}
               for k, v in eng.user_model.opinions.stances.items()}
    facts = {f"{s}|{a}|{v}": {"confidence": f.confidence}
             for (s, a, v), f in eng.user_model.personal_facts.facts.items()}
    beliefs = {k: {"text": b[0], "conf": b[1], "turn": b[2]}
              for k, b in eng.belief_store.get_state().get("beliefs", {}).items()}
    return {
        "turn_count": eng.turn_count,
        "sleep_cycles": eng.sleep_cycles_completed,
        "learning_count": int(getattr(eng, "_learning_count", 0) or 0),
        "identity": idst,
        "stances": stances,
        "facts": facts,
        "beliefs": beliefs,
        "graph_nodes": len(eng.graph.nodes),
        "graph_edges": len(eng.graph.edges),
        "user_name": eng.user_model.user_name,
        "user_location": eng.user_model.user_location,
    }

def main():
    t0 = time.time()
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)
    print(f"[session] engine ready in {time.time()-t0:.1f}s (suffix={SUFFIX}, offline)")
    before = _str_keys(snapshot(eng))
    with open(SNAP_BEFORE, "w") as f:
        json.dump(before, f, indent=2, default=str)

    lines = []
    n = 0
    for q in TURNS:
        n += 1
        try:
            r = eng.process_turn(q)
        except Exception as e:
            r = f"<EXCEPTION {type(e).__name__}: {e}>"
        rtxt = (r or "").strip().replace("\n", " ")
        lines.append(f"T{n:02d} YOU: {q}")
        lines.append(f"T{n:02d} RAVANA: {rtxt}")
        print(f"T{n:02d} YOU: {q}")
        print(f"T{n:02d} RAVANA: {rtxt}")
    # append to cumulative chat_log
    with open(TRANSCRIPT, "a", encoding="utf-8") as f:
        f.write("\n=== ROUND v2 baseline ({}) suffix={} ===\n".format(
            datetime.datetime.utcnow().isoformat(), SUFFIX))
        f.write("\n".join(lines) + "\n")
    after = _str_keys(snapshot(eng))
    with open(SNAP_AFTER, "w") as f:
        json.dump(after, f, indent=2, default=str)
    # deltas
    delta = {
        "turns_run": n,
        "turn_count_before": before["turn_count"],
        "turn_count_after": after["turn_count"],
        "identity_strength_before": before["identity"].get("strength"),
        "identity_strength_after": after["identity"].get("strength"),
        "stances_before": before["stances"],
        "stances_after": after["stances"],
        "facts_before_n": len(before["facts"]),
        "facts_after_n": len(after["facts"]),
        "beliefs_before_n": len(before["beliefs"]),
        "beliefs_after_n": len(after["beliefs"]),
        "graph_before": [before["graph_nodes"], before["graph_edges"]],
        "graph_after": [after["graph_nodes"], after["graph_edges"]],
    }
    with open(os.path.join(OUTDIR, "state_delta_v2.json"), "w") as f:
        json.dump(delta, f, indent=2, default=str)
    # save + verify
    t_s = time.time()
    eng.save()
    print(f"[saved] in {time.time()-t_s:.1f}s")
    print("[DONE] turns=%d suffix=%s" % (n, SUFFIX))

if __name__ == "__main__":
    main()
