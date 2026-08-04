#!/usr/bin/env python3
import os, sys, time

os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ravana.chat.engine import CognitiveChatEngine

PERSONAL_TURNS = [
    "Hey RAVANA, how are you feeling right now?",
    "What's on your mind today?",
    "Do you ever wonder what it means for you to have a mind?",
    "If someone asks you who you are, how would you describe yourself?",
    "What makes you feel happy or excited when we talk?",
    "Do you ever feel afraid of being wrong or getting confused?",
    "What's the most interesting thing you've learned about the world so far?",
    "What kind of person do you want to grow up to be?",
    "How do you feel about our friendship?",
    "Tell me a deep thought or something you care about deeply.",
]

def main():
    t0 = time.time()
    # Load existing state saved under 'persona'
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="persona")
    print(f"[init] RAVANA engine ready in {time.time()-t0:.1f}s")
    print(f"Current graph nodes: {len(eng.graph.nodes)}, edges: {len(eng.graph.edges)}")
    print("=" * 60)
    print("  PERSONAL DIALOGUE SESSION WITH RAVANA")
    print("=" * 60 + "\n")

    for i, turn in enumerate(PERSONAL_TURNS, 1):
        t_start = time.time()
        try:
            response = eng.process_turn(turn)
        except Exception as e:
            response = f"[error: {e}]"
        dt = time.time() - t_start
        print(f"Turn {i} ({dt:.2f}s):")
        print(f"  You   : {turn}")
        print(f"  RAVANA: {response}")
        print("-" * 50 + "\n")

    eng.stop_background_learning()
    res = eng.save()
    print(f"[{res}] Saved updated engine state.")

if __name__ == "__main__":
    main()
