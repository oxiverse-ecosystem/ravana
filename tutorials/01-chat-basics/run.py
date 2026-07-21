"""Tutorial 01: Chat Engine Basics — create engine, send queries, save state.

This is the first tutorial in a progression (01→02→03 build a mini-system).
Run this first, then Tutorial 02 loads the saved state.

Usage:
    python tutorials/01-chat-basics/run.py
"""
import os
import sys

# Make sure local source trees are importable when running from repo root.
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ravana", "src"))
sys.path.insert(0, os.path.join(ROOT, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(ROOT, "ravana-v2", "src"))

from ravana.chat.engine import CognitiveChatEngine


def main() -> None:
    # 1. Create the engine — starts as a "baby" with a small concept graph
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)

    # 2. Send queries — each turn may trigger web learning or sleep
    queries = [
        "what is trust",
        "tell me about love",
        "what is memory",
    ]

    for q in queries:
        answer = engine.process_turn(q)
        print("Q:", q)
        print("A:", answer)
        print(
            "  [nodes=" + str(len(engine.graph.nodes))
            + ", edges=" + str(len(engine.graph.edges))
            + ", sleep=" + str(engine.sleep_cycles_completed)
            + "]"
        )

    # 3. Save — Tutorial 02 will load this state
    engine.save()
    print("\n[OK] State saved to data/ravana_weights.pkl")
    print("  Ready for Tutorial 02: python tutorials/02-decoder-training/run.py")


if __name__ == "__main__":
    main()
