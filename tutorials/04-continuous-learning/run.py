"""Tutorial 04: Continuous Background Learning.

Layer 1 of 3 added on top of the mini-system from tutorials 01-03.
Demonstrates background web learning with curiosity-driven topic selection.

Usage:
    python tutorials/04-continuous-learning/run.py --cycles 5 --delay 2
"""
import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from scripts.ravana_chat import CognitiveChatEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Background learning demo")
    parser.add_argument("--cycles", type=int, default=3, help="Number of learning cycles")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between cycles")
    parser.add_argument("--no-curiosity", action="store_true",
                        help="Disable autonomous topic selection (use queue only)")
    args = parser.parse_args()

    # 1. Create engine with background learning
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    print("=== Continuous Background Learning Demo ===\n")

    # 2. Start the background learning thread
    engine.start_background_learning()
    if args.no_curiosity:
        engine._curiosity_drive_enabled = False
        print("  [curiosity disabled — using queue only]\n")

    # 3. Push research topics to the learning queue
    # The curiosity drive also auto-selects topics if enabled
    topics = ["consciousness neuroscience", "human memory psychology", "sleep dreaming"]
    for t in topics:
        engine._bg_learning_queue.append(t)
    print(f"  Queued {len(topics)} topics: {', '.join(topics)}\n")

    # 4. Poll graph growth over N cycles
    from time import sleep
    initial_nodes = len(engine.graph.nodes)
    initial_edges = len(engine.graph.edges)

    for i in range(args.cycles):
        sleep(args.delay)
        nodes = len(engine.graph.nodes)
        edges = len(engine.graph.edges)
        print(
            f"  cycle={i+1}/{args.cycles}"
            f"  nodes={nodes} (+{nodes - initial_nodes})"
            f"  edges={edges} (+{edges - initial_edges})"
            f"  searches={engine._bg_search_count}"
        )

    # 5. Stop and save
    engine.stop_background_learning()
    engine.save()
    print(f"\n  [OK] State saved. Graph grew by {len(engine.graph.nodes) - initial_nodes} nodes"
          f" and {len(engine.graph.edges) - initial_edges} edges.")


if __name__ == "__main__":
    main()
