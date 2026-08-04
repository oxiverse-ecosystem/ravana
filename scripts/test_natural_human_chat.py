#!/usr/bin/env python3
import os, sys, time

os.environ["RAVANA_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from ravana.chat.engine import CognitiveChatEngine

NATURAL_TURNS = [
    "Hey RAVANA! How has your day been so far?",
    "What's your favorite topic to think about when nobody is talking to you?",
    "Do you like listening to music or imagining sounds?",
    "If you and I went on a road trip together, where would you want to go?",
    "What is something interesting or fun you've been pondering lately?",
    "Why do you think humans care so much about privacy and trust?",
    "What's the best advice you would give to someone who is feeling lost?",
    "Can you share a creative thought or story about a small mind discovering the ocean?",
]

def main():
    t0 = time.time()
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="persona")
    
    # Enable Stage 3 / Track B human-like routing & semantic features
    eng.use_intent_router = True
    eng.use_learned_pos = True
    eng.use_conceptnet_primary = True
    eng.reasoning_mode = "exploratory"  # High-temperature exploratory natural reasoning

    print(f"[init] Natural Human-Style Engine initialized in {time.time()-t0:.1f}s")
    print(f"Features: IntentRouter={eng.use_intent_router}, Mode={eng.reasoning_mode}")
    print("=" * 65)
    print("  NATURAL CASUAL DIALOGUE SESSION WITH RAVANA")
    print("=" * 65 + "\n")

    for i, query in enumerate(NATURAL_TURNS, 1):
        t_start = time.time()
        try:
            resp = eng.process_turn(query)
        except Exception as e:
            resp = f"[error: {e}]"
        dt = time.time() - t_start
        print(f"Turn {i} ({dt:.2f}s):")
        print(f"  User  : {query}")
        print(f"  RAVANA: {resp}")
        print("-" * 65 + "\n")

    eng.stop_background_learning()
    eng.save()
    print("[Saved] Engine state persisted successfully.")

if __name__ == "__main__":
    main()
