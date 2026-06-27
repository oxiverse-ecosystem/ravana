#!/usr/bin/env python3
"""
RAVANA Continuous Learner — autonomous background learning with no chat needed
==============================================================================
Loads the same CognitiveChatEngine used by ravana_chat.py. Instead of using
hardcoded topic lists, the curiosity drive (Phase 18) autonomously selects
what to learn based on prediction error, information gaps, contradiction pairs,
and novelty — mirroring how the brain's dopaminergic system drives exploration
toward the "Goldilocks zone" of optimal learning progress.

The curiosity drive uses these signals (from neuroscience research):
- Prediction Error (Friston Active Inference): concepts with high surprise
- Information Gap (Loewenstein): unresolved queries
- Cognitive Dissonance: contradiction pairs in the graph
- Novelty: least-visited or dormant-edge concepts
- Learning Progress (Oudeyer): PE delta tracking

Usage:
    python scripts/ravana_learn.py [--cycles N] [--delay SECONDS]
    python scripts/ravana_learn.py --cycles 10 --delay 5
    python scripts/ravana_learn.py              # runs indefinitely until Ctrl+C

Press Ctrl+C at any time to save weights and exit gracefully.
"""

import sys
import os
import time
import argparse

# Ensure project root is on sys.path
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2", "src"))

# Try package-qualified import first, fall back to direct sibling import
try:
    from scripts.ravana_chat import CognitiveChatEngine
except ImportError:
    from ravana_chat import CognitiveChatEngine


def _seed_from_graph_curiosity(engine, max_topics: int = 5):
    """Bootstrap the learning queue from the graph's own curiosity signals.

    Picks concepts with the highest autonomous curiosity scores:
    high prediction error, contradiction involvement, low visit count,
    or dormant edges. This replaces hardcoded seed topics.
    
    Now uses engine._get_curiosity_scores() which also includes edge-level
    prediction free energy for curiosity drive.
    """
    if not engine._curiosity_drive_enabled:
        return 0

    # Use the engine's unified curiosity scoring (includes edge-level PE)
    curiosity_scores = engine._get_curiosity_scores(max_topics=max_topics * 2)
    
    candidates = [label for label, score in curiosity_scores if score > 0.1]
    
    # Add high-degree hubs for serendipity (not covered by curiosity scores)
    if len(engine.graph.nodes) > 0:
        degrees = {}
        for nid in engine.graph.nodes:
            out = len(list(engine.graph.get_outgoing(nid)))
            degrees[nid] = out
        top_hubs = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        for nid, _ in top_hubs:
            node = engine.graph.get_node(nid)
            if node and node.label:
                label = node.label.lower()
                if len(label) >= 3 and label not in candidates:
                    candidates.append(label)

    # Queue the candidates
    queued = 0
    with engine._bg_lock:
        for topic in candidates[:max_topics]:
            if topic not in engine._bg_learning_queue:
                engine._bg_learning_queue.append(topic)
                queued += 1
    if queued > 0:
        engine._bg_idle_event.set()
    return queued


def main():
    parser = argparse.ArgumentParser(
        description="RAVANA Continuous Learner — autonomous learning without chat"
    )
    parser.add_argument("--cycles", type=int, default=0,
                        help="Exit after this many learning cycles (0 = infinite)")
    parser.add_argument("--delay", type=float, default=15.0,
                        help="Seconds to wait between learning cycles")
    parser.add_argument("--dim", type=int, default=64,
                        help="Graph dimension (must match existing weights)")
    parser.add_argument("--seed-rng", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--trace", action="store_true",
                        help="Enable trace output")
    parser.add_argument("--no-curiosity", action="store_true",
                        help="Disable autonomous curiosity-driven topic selection")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory for weights and GloVe cache")
    args = parser.parse_args()

    print(f"  [Learn] RAVANA Continuous Learner")
    print(f"  [Learn] Curiosity drive: {'ENABLED' if not args.no_curiosity else 'DISABLED'}")
    print(f"  [Learn] Press Ctrl+C to save weights and exit")

    # ── Load / create engine ──
    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed_rng,
                                  baby_mode=True, data_dir=args.data_dir)
    if args.trace:
        engine._trace_enabled = True
    if args.no_curiosity:
        engine._curiosity_drive_enabled = False

    # Start background learning thread
    engine.start_background_learning()

    # ── Bootstrap learning queue from graph's own curiosity signals ──
    # Instead of hardcoded topics, the curiosity drive selects what to learn
    # based on prediction error, contradictions, dormant edges, and novelty
    initial_queued = _seed_from_graph_curiosity(engine, max_topics=8)
    print(f"  [Learn] Curiosity-bootstrapped {initial_queued} initial topics from graph")
    print(f"  [Learn] Graph: {len(engine.graph.nodes)} concepts, {len(engine.graph.edges)} edges")
    print()

    # ── Main loop: keep alive, periodically print status ──
    cycles_completed = 0
    last_word_count = len(engine.graph.nodes)
    last_edge_count = len(engine.graph.edges)
    try:
        while True:
            time.sleep(args.delay)

            # ── Auto-select curiosity topics when queue runs low ──
            # The bg_learn_loop already calls _auto_select_curiosity_topics when
            # its queue is empty. Here we also check periodically to ensure the
            # curiosity drive stays active even if the bg thread is busy.
            queued_this_cycle = 0
            if engine._curiosity_drive_enabled:
                with engine._bg_lock:
                    queue_size = len(engine._bg_learning_queue)
                if queue_size <= 2:
                    queued_this_cycle = _seed_from_graph_curiosity(engine, max_topics=4)

            # ── Print periodic status ──
            cycles_completed += 1
            current_words = len(engine.graph.nodes)
            current_edges = len(engine.graph.edges)
            new_words = current_words - last_word_count
            new_edges = current_edges - last_edge_count

            if new_words > 0 or new_edges > 0 or queued_this_cycle > 0 or cycles_completed % 5 == 0:
                print(f"  [Learn] Cycle {cycles_completed}: "
                      f"{current_words} concepts (+{new_words}), "
                      f"{current_edges} edges (+{new_edges}), "
                      f"{engine._bg_search_count} web searches, "
                      f"decoder={engine._decoder_training_count}, "
                      f"urgency={engine._curiosity_urgency:.2f}"
                      f"{f', curiosity-queued {queued_this_cycle}' if queued_this_cycle else ''}")
                last_word_count = current_words
                last_edge_count = current_edges

            # ── If curiosity is disabled, re-seed from graph every 10 cycles ──
            if not engine._curiosity_drive_enabled and cycles_completed % 10 == 0:
                queued = _seed_from_graph_curiosity(engine, max_topics=3)
                if queued > 0:
                    print(f"  [Learn] Re-seeded {queued} topics (curiosity disabled)")

            # ── Check for cycles limit ──
            if args.cycles > 0 and cycles_completed >= args.cycles:
                print(f"  [Learn] Reached {args.cycles} cycles, saving and exiting...")
                break

    except KeyboardInterrupt:
        pass
    finally:
        engine.stop_background_learning()
        result = engine.save()
        print(f"  [Learn] {result}")
        print(f"  [Learn] Final graph: {len(engine.graph.nodes)} concepts, "
              f"{len(engine.graph.edges)} edges, "
              f"{engine._bg_search_count} web searches performed, "
              f"decoder trained on {engine._decoder_training_count} sentences "
              f"({engine._decoder_web_training_count} from web)")
        print("  [Learn] Goodbye!")


if __name__ == "__main__":
    main()
