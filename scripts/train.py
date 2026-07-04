#!/usr/bin/env python3
"""
RAVANA Training — full human-like training pipeline
====================================================
Trains the decoder on real English + web learning + reasoning cycles.

Modes:
  phase2    — Heavy decoder training on teen_seeds.txt (fast, ~1hr)
  full      — Full pipeline: seed → web → consolidate → evaluate (~3-5hrs)
  test      — Quick diagnostic that decoder trains on real text

Usage:
    python scripts/train.py --mode phase2
    python scripts/train.py --mode full [--web-topics 10] [--cycles 3]
    python scripts/train.py --mode test
"""
import sys, os, time, json, re
import numpy as np

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))

os.environ["RAVANA_SILENT"] = "1"
from scripts.ravana_chat import CognitiveChatEngine

# ─── Topics for web learning ───
WEB_TOPICS = [
    "consciousness neuroscience",
    "time travel physics",
    "quantum mechanics explained",
    "artificial intelligence ethics",
    "human memory psychology",
    "philosophy of mind",
    "evolutionary biology",
    "cybersecurity basics",
    "climate change science",
    "meditation mindfulness",
]

SEED_QUERIES = [
    "what is trust",
    "tell me about love",
    "what happens if time travel possible",
    "explain consciousness",
    "how does memory work",
    "what is justice",
    "tell me about freedom",
    "explain evolution",
    "what is artificial intelligence",
    "how does the brain work",
]

EVAL_QUESTIONS = [
    "what is trust",
    "tell me about love",
    "what happens if time travel possible",
    "explain consciousness",
    "how does memory work",
    "what is the meaning of life",
    "who are you",
    "what is justice",
    "explain freedom",
]

# ─── Shared ───

def load_engine(args, reset=False):
    save_path = os.path.join(_proj_root, "data", "ravana_weights.pkl")
    if reset and os.path.exists(save_path):
        os.remove(save_path)
        print(f"  [Reset] Deleted saved weights, starting fresh!")
    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed, baby_mode=True)
    nd = engine.neural_decoder
    if nd._total_training_examples == 0:
        nd.reset_plasticity(stability=0.5)
        print("  [reset] Decoder plasticity reset")
    return engine, nd


def load_corpus(engine):
    corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    engine._freeze_decoder_vocab = False
    words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", text.lower()))
    new_for_vocab = [w for w in words_in_corpus if w not in engine._decoder_word_to_idx]
    if new_for_vocab:
        engine._expand_decoder_vocab(new_for_vocab)
    engine._freeze_decoder_vocab = True
    nd = engine.neural_decoder
    all_sentences = nd.prepare_sentences(
        text, engine._decoder_word_to_embed, engine._decoder_word_to_idx,
        min_sentence_len=3,
    )
    return text, all_sentences, nd


def train_seed_corpus(engine, nd, all_sentences, n_passes=200, pp=2000, si=10, pe=20):
    """Train decoder on seed corpus with sampled softmax.
    
    Early stopping uses training CE (now honest after removing
    self-conditioning cheat). Final evaluation is done separately
    by the evaluate() function after training completes.
    Returns total sentences trained.
    """
    n_avail = len(all_sentences)
    pp = min(pp, n_avail)
    rng = np.random.RandomState(42)
    total = 0
    best_ce = float('inf')
    stall = 0
    t0 = time.time()

    for i in range(n_passes):
        idx = rng.choice(n_avail, size=pp, replace=False)
        for j in idx:
            s = all_sentences[j]
            nd.train_on_sentence(
                s['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
                word_indices=s['word_indices'], conditioning_embs=s['conditioning_embs'],
            )
        total += pp
        if (i+1) % si == 0:
            nd.sleep_cycle()
            ce = nd._avg_cross_entropy
            if ce < best_ce - 1e-3:
                best_ce = ce; stall = 0
            else:
                stall += 1
                if stall >= pe:
                    print(f"  Early stop at pass {i+1} (best CE={best_ce:.3f})")
                    break
        if (i+1) % 5 == 0:
            elapsed = time.time()-t0
            rate = elapsed/(i+1)
            print(f"  Pass {i+1}/{n_passes}: CE={nd._avg_cross_entropy:.3f} "
                  f"t1={nd._avg_top1_acc:.3f} t5={nd._avg_top5_acc:.3f} "
                  f"({rate:.1f}s/pass, ETA={(n_passes-i-1)*rate:.0f}s)", flush=True)

    if n_passes % si != 0:
        nd.sleep_cycle()
    engine._decoder_seed_training_count += total
    engine._decoder_training_count += total
    return total


def evaluate(engine, questions=EVAL_QUESTIONS):
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    for q in questions:
        t0 = time.time()
        resp = engine.process_turn(q)
        t = time.time()-t0
        print(f"  Q: {q}")
        print(f"  A: {resp[:150] if resp else '<None>'}")
        print(f"    [{engine._last_strategy}] ({t:.1f}s)")
        print()


# ─── Mode: phase2 ───

def _mode_phase2(args):
    """Heavy decoder training on teen_seeds.txt with web learning."""
    print("="*60)
    print("RAVANA Decoder Training — Phase 2 (human-like speech)")
    print("="*60)
    t0 = time.time()
    engine, nd = load_engine(args, reset=args.reset)

    print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
    print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
    print(f"  Pre-training: CE={nd._avg_cross_entropy:.3f} t1={nd._avg_top1_acc:.3f} "
          f"trained={nd._total_training_examples}")
    print()

    # Phase 1: Heavy seed corpus training
    print("[Phase 1] Seed corpus training...")
    text, all_sentences, nd = load_corpus(engine)
    n_seed = train_seed_corpus(engine, nd, all_sentences, n_passes=200, pp=2000, si=10, pe=25)
    print(f"  {n_seed} seed sentences trained in {time.time()-t0:.0f}s")
    print(f"  CE={nd._avg_cross_entropy:.3f} t1={nd._avg_top1_acc:.3f} t5={nd._avg_top5_acc:.3f}")
    print()

    # Phase 2: Curiosity-driven web learning (autonomous topic selection)
    if not args.no_web:
        print("[Phase 2] Seeding curiosity signals (running seed queries)...")
        engine._bg_learning_active = True
        engine._curiosity_drive_enabled = True
        engine._network_available = True  # force fresh web attempts
        t1 = time.time()
        for q in SEED_QUERIES:
            try:
                engine.process_turn(q)
            except Exception:
                pass
        print(f"  {len(SEED_QUERIES)} queries in {time.time()-t1:.0f}s")

        print(f"[Phase 2] Curiosity-driven web learning ({args.web_topics} topics)...")
        t1 = time.time()
        engine._bg_learning_queue.clear()
        for i in range(args.web_topics):
            topics = engine._auto_select_curiosity_topics(max_topics=3)
            if not topics:
                engine._bg_learning_queue = list(WEB_TOPICS[:5])
                engine._impossible_queries.clear()
                topics = engine._auto_select_curiosity_topics(max_topics=3)
                if not topics:
                    for topic in WEB_TOPICS[:args.web_topics]:
                        engine._network_available = True
                        try:
                            result, _ = engine.learn_from_web(
                                topic + " explained with examples", max_results=2)
                            print(f"  [{i+1}/{args.web_topics}] {topic} -> {result}")
                        except Exception:
                            print(f"  [{i+1}/{args.web_topics}] {topic} -> offline")
                    break
            for topic in topics[:2]:
                engine._network_available = True
                query = engine._generate_curiosity_query(topic, source_type="prediction_error")
                try:
                    result, _ = engine.learn_from_web(query, max_results=2)
                    print(f"  [{i+1}/{args.web_topics}] {query} -> {result}")
                except Exception:
                    print(f"  [{i+1}/{args.web_topics}] {query} -> offline")
        print(f"  Web learning done in {time.time()-t1:.0f}s")
        print(f"  Web training: {engine._decoder_web_training_count} sentences")
        print()

    # Phase 3: Consolidation — more seed corpus passes with web-expanded vocab
    print("[Phase 3] Consolidation training...")
    t2 = time.time()
    n_consolidate = train_seed_corpus(engine, nd, all_sentences, n_passes=50, pp=2000, si=10, pe=10)
    print(f"  {n_consolidate} consolidation sentences in {time.time()-t2:.0f}s")
    print()

    # Phase 4: Evaluate + save
    print("[Phase 4] Evaluating...")
    evaluate(engine)

    print("Saving...")
    engine._needs_seed_training = False
    result = engine.save()
    print(f"  {result}")

    print()
    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
    print(f"  Decoder: {engine._decoder_training_count} total, "
          f"{engine._decoder_seed_training_count} seed, {engine._decoder_web_training_count} web")
    print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
    print(f"  Final: CE={nd._avg_cross_entropy:.3f} t1={nd._avg_top1_acc:.3f} "
          f"t5={nd._avg_top5_acc:.3f}")
    print()


# ─── Mode: full ───

def _mode_full(args):
    """Full training pipeline — single-phase seed corpus + web knowledge.
    
    FIX: Eliminated multi-cycle approach which caused catastrophic forgetting.
    Now delegates to phase2 (single long seed training + web + consolidation).
    No alternating cycles that overwrite previous learning.
    """
    # Phase 2 is already the correct single-phase approach (seed → web → consolidate)
    _mode_phase2(args)


# ─── Mode: test ───

def _mode_test(args):
    """Quick diagnostic: verify decoder trains and generates."""
    engine, nd = load_engine(args)
    text, all_sentences, nd = load_corpus(engine)

    print("Training on 50 sentences...")
    t0 = time.time()
    for s in all_sentences[:50]:
        nd.train_on_sentence(s['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
            word_indices=s['word_indices'], conditioning_embs=s['conditioning_embs'])
    nd.sleep_cycle()
    print(f"  CE={nd._avg_cross_entropy:.4f} t1={nd._avg_top1_acc:.4f} ({time.time()-t0:.1f}s)")

    print("\nGenerating responses...")
    for q in ["what is trust", "tell me about love", "hello"]:
        t0 = time.time()
        r = engine.process_turn(q)
        print(f"  Q: {q}")
        print(f"  A: {r[:120] if r else '<None>'} [{engine._last_strategy}] ({time.time()-t0:.1f}s)")

    engine.save()
    print("\nTest complete.")


# ─── CLI ───

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA unified trainer")
    parser.add_argument("--mode", choices=["phase2", "full", "test"], default="phase2",
                        help="Training mode (default: phase2)")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reset", action="store_true", help="Delete saved weights and start fresh")

    # Phase 2 / Full options
    parser.add_argument("--no-web", action="store_true", help="Skip web learning (offline mode)")
    parser.add_argument("--web-topics", type=int, default=5, help="Number of web topics to learn (default: 5)")
    parser.add_argument("--cycles", type=int, default=3, help="Training cycles for full mode (default: 3)")

    args = parser.parse_args()

    modes = {"phase2": _mode_phase2, "full": _mode_full, "test": _mode_test}
    modes[args.mode](args)


if __name__ == "__main__":
    main()
