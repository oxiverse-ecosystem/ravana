#!/usr/bin/env python3
"""
RAVANA Training — unified trainer
===================================
Merges functionality from: train_decoder_phase2.py, iterative_train.py,
train_chat_loop.py, simple_train_loop.py, test_decoder_training.py.

Modes:
  phase2    — Decoder-only training on teen_seeds.txt (Phase 2+3)
  iterative — Full iterative training with web learning + checkpoints
  test      — Quick diagnostic test that decoder trains on real text

Usage:
    python scripts/train.py --mode phase2
    python scripts/train.py --mode iterative [--cycles 5]
    python scripts/train.py --mode test
    python scripts/train.py --mode phase2 --dim 128 --seed 0
"""
import sys, os, time, json, re
import numpy as np

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))

os.environ["RAVANA_SILENT"] = "1"
from scripts.ravana_chat import CognitiveChatEngine


# ─── Shared utilities ───

TEMPLATE_MARKERS = [
    "relates to", "connects with", "links to", "is associated with",
    "leads to", "results in", "causes", "influences", "contributes to",
    "stands against", "contrasts with", "on the other hand"
]
NATURAL_MARKERS = [
    "furthermore", "moreover", "however", "although", "consequently",
    "therefore", "nevertheless", "nonetheless", "in addition", "similarly",
    "accordingly", "subsequently", "meanwhile", "conversely"
]


def _ensure_corpus(root: str) -> bool:
    """Fetch teen_seeds.txt if missing. Returns True if available after attempt."""
    corpus_path = os.path.join(root, "data", "corpora", "teen_seeds.txt")
    if os.path.exists(corpus_path):
        return True
    print("[0/3] teen_seeds.txt not found — gathering from internet...", flush=True)
    gather_script = os.path.join(root, "scripts", "gather_teen_seeds.py")
    if os.path.exists(gather_script):
        import subprocess
        ret = subprocess.call([sys.executable, gather_script, "--force"])
        if ret != 0:
            print(f"  [warn] Gather script exited with code {ret}", flush=True)
    else:
        print(f"  [warn] gather_teen_seeds.py not found", flush=True)
    return os.path.exists(corpus_path)


def evaluate_response(response: str) -> dict:
    """Evaluate response quality: template vs natural language markers."""
    r_low = response.lower()
    template_count = sum(1 for m in TEMPLATE_MARKERS if m in r_low)
    natural_count = sum(1 for m in NATURAL_MARKERS if m in r_low)
    words = len(response.split())
    sentences = len([s for s in response.split('.') if s.strip()])
    unique_words = len(set(w.lower() for w in response.split()))
    diversity = unique_words / max(1, words)
    template_ratio = template_count / max(1, template_count + natural_count)
    natural_ratio = natural_count / max(1, template_count + natural_count)
    return {
        "template_count": template_count, "natural_count": natural_count,
        "template_ratio": template_ratio, "natural_ratio": natural_ratio,
        "words": words, "sentences": sentences, "diversity": diversity,
        "quality_score": natural_ratio * diversity,
    }


# ─── Mode: phase2 ───

def _mode_phase2(args):
    """Decoder-only training on teen_seeds.txt (Phase 2+3)."""
    print("=" * 60)
    print("RAVANA Decoder Training — Phase 2+3")
    print("=" * 60)
    print()
    _ensure_corpus(_proj_root)

    print("[1/3] Loading CognitiveChatEngine...", flush=True)
    t0 = time.time()
    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed, baby_mode=True)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
    print(f"  Decoder: {engine._decoder_training_count} total, {engine._decoder_web_training_count} web, {engine._decoder_seed_training_count} seed")
    print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
    print()

    engine.neural_decoder.reset_plasticity(stability=0.5)
    print("  [reset] Decoder plasticity reset to stability=0.5")
    print()

    corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
    if not os.path.exists(corpus_path):
        print(f"  WARNING: Corpus not found at {corpus_path}")
        return

    print("[2/3] Training decoder on teen_seeds.txt corpus (real English only)...", flush=True)
    t1 = time.time()
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", corpus_text.lower()))
    new_for_vocab = [w for w in words_in_corpus if w not in engine._decoder_word_to_idx]
    if new_for_vocab:
        engine._expand_decoder_vocab(new_for_vocab)

    nd = engine.neural_decoder
    all_sentences = nd.prepare_sentences(
        corpus_text, engine._decoder_word_to_embed, engine._decoder_word_to_idx,
        min_sentence_len=3,
    )
    n_available = len(all_sentences)
    sentences_per_pass = min(300, n_available)
    n_passes = 1000
    sleep_interval = 5
    patience = 15
    target_ce = 3.0

    passes = 0
    rng = np.random.RandomState(42)
    best_ce = float('inf')
    stall_count = 0

    try:
        for i in range(n_passes):
            idx = rng.choice(n_available, size=sentences_per_pass, replace=False)
            batch = [all_sentences[i] for i in idx]
            for sent in batch:
                nd.train_on_sentence(
                    sent['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
                    word_indices=sent['word_indices'],
                    conditioning_embs=sent['conditioning_embs'],
                )
            passes += sentences_per_pass

            if (i + 1) % sleep_interval == 0:
                nd.sleep_cycle()

            if (i + 1) % 5 == 0:
                print(f"  Pass {i+1}/{n_passes}: CE={nd._avg_cross_entropy:.3f} "
                      f"top1={nd._avg_top1_acc:.3f} top5={nd._avg_top5_acc:.3f} "
                      f"(sentences={passes})", flush=True)

            if nd._avg_cross_entropy <= target_ce and nd._avg_top1_acc > 0.30:
                print(f"  Target CE={target_ce} reached at pass {i+1} — stopping early")
                break

            if (i + 1) >= 10 and (i + 1) % sleep_interval == 0:
                if nd._avg_cross_entropy < best_ce - 1e-3:
                    best_ce = nd._avg_cross_entropy
                    stall_count = 0
                else:
                    stall_count += 1
                    if stall_count >= patience:
                        print(f"  Early stopping at pass {i+1} "
                              f"(no CE improvement for {patience} checks, best={best_ce:.3f})")
                        break
    except KeyboardInterrupt:
        print(f"  [Interrupted] Consolidating partial training ({passes} sentences)...", flush=True)

    if (passes // sentences_per_pass % sleep_interval) != 0:
        nd.sleep_cycle()

    engine._decoder_seed_training_count = passes
    engine._decoder_training_count += passes
    corpus_sentences = passes

    print(f"  Trained {corpus_sentences} sentences in {time.time() - t1:.1f}s")
    print()

    # Step 3: Skip synthetic graph training (was causing template artifacts)
    print("[3/3] Skipping synthetic graph training (was causing template artifacts)...")
    print(f"  Graph edges still available for conditioning embeddings ({len(engine.graph.edges)} edges)")
    print()

    print("Saving engine state...", flush=True)
    result = engine.save()
    print(f"  Save: {result}")
    print()

    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time: {time.time() - t0:.1f}s")
    print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
    print(f"  Decoder total: {engine._decoder_training_count}")
    print(f"  Decoder seed:  {engine._decoder_seed_training_count}")
    print(f"  Decoder web:   {engine._decoder_web_training_count}")
    print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
    print(f"  Final metrics: CE={nd._avg_cross_entropy:.3f} "
          f"top1={nd._avg_top1_acc:.3f} top5={nd._avg_top5_acc:.3f}")
    print()

    # Quick smoke test
    print("Smoke test: generating a response...", flush=True)
    t4 = time.time()
    test_response = engine.process_turn("what is trust")
    test_time = time.time() - t4
    print(f"  Generated in {test_time:.1f}s")
    print(f"  Strategy: {engine._last_strategy}")
    print(f"  Response: {test_response[:200] if len(test_response) > 200 else test_response}")
    print()
    print("Phase 2+3 complete!")


# ─── Mode: iterative ───

ITERATIVE_TEST_QUESTIONS = [
    "what happens if time travel possible",
    "why not possible til now",
    "make time machine",
    "switch to cybersecurity",
    "tell me about consciousness",
    "how does memory work",
    "what is the meaning of life",
    "explain quantum mechanics",
]

ITERATIVE_WEB_TOPICS = [
    "time travel physics",
    "consciousness neuroscience",
    "quantum mechanics explained",
]


def _evaluate_model(engine, questions):
    """Evaluate model on test questions, return results list."""
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    results = []
    for q in questions:
        response = engine.process_turn(q)
        eval_result = evaluate_response(response)
        eval_result["question"] = q
        eval_result["response"] = response
        results.append(eval_result)
        print(f"  Q: {q}")
        print(f"  A: {response}")
        print(f"  Quality: {eval_result['quality_score']:.3f} "
              f"(nat={eval_result['natural_ratio']:.2f}, tmpl={eval_result['template_ratio']:.2f})")
        print()
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    avg_natural = sum(r['natural_ratio'] for r in results) / len(results)
    avg_template = sum(r['template_ratio'] for r in results) / len(results)
    print(f"  AVG Quality: {avg_quality:.3f} | Natural: {avg_natural:.2f} | Template: {avg_template:.2f}")
    return results


def _mode_iterative(args):
    """Full iterative training with cycles, web learning, checkpoints."""
    cycles = args.cycles
    eval_every = 1

    print("Starting Iterative Decoder Training")
    print(f"Cycles: {cycles}, Eval every: {eval_every}")
    _ensure_corpus(_proj_root)

    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed, baby_mode=False)
    engine._bg_learning_active = False
    engine._needs_seed_training = False
    engine._needs_synthetic_training = False
    engine.baby_mode = False

    print("\n*** INITIAL STATE ***")
    _evaluate_model(engine, ITERATIVE_TEST_QUESTIONS)

    corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
    corpus_text = None
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()

    for cycle in range(1, cycles + 1):
        print(f"\n{'='*60}")
        print(f"TRAINING CYCLE {cycle}")
        print(f"{'='*60}")

        # Phase 1: Seed corpus training
        print("  [Phase 1] Seed corpus training: 50 passes...")
        if corpus_text:
            total_passes = 0
            for _ in range(50):
                err, n = engine.neural_decoder.train_on_text(
                    corpus_text, engine._decoder_word_to_embed,
                    engine._decoder_word_to_idx, min_sentence_len=3, max_sentences=200)
                total_passes += n
            engine.neural_decoder.sleep_cycle()
            engine._decoder_seed_training_count += total_passes
            engine._decoder_training_count += total_passes
            print(f"  [Phase 1] Done: {total_passes} sentences")

        # Phase 2: Web article training
        print("  [Phase 2] Web article training...")
        for topic in ITERATIVE_WEB_TOPICS:
            try:
                engine.learn_from_web(topic, max_results=2)
            except Exception as e:
                print(f"    Web training error for {topic}: {e}")
        print(f"  [Phase 2] Done: web_training={engine._decoder_web_training_count}")

        print(f"  [Summary] Total training: {engine._decoder_training_count} sentences")

        if cycle % eval_every == 0:
            results = _evaluate_model(engine, ITERATIVE_TEST_QUESTIONS)
            avg_quality = sum(r['quality_score'] for r in results) / len(results)

            checkpoint = {
                "cycle": cycle,
                "training_count": engine._decoder_training_count,
                "seed_training": engine._decoder_seed_training_count,
                "web_training": engine._decoder_web_training_count,
                "avg_quality": avg_quality,
                "results": results,
            }
            os.makedirs("checkpoints", exist_ok=True)
            with open(f"checkpoints/iterative_checkpoint_c{cycle}.json", "w") as f:
                json.dump(checkpoint, f, indent=2)
            print(f"\n  Checkpoint saved: checkpoints/iterative_checkpoint_c{cycle}.json")

            if avg_quality > 0.5:
                print(f"  *** QUALITY THRESHOLD REACHED ({avg_quality:.3f} > 0.5) ***")
                break

    print("\n*** TRAINING COMPLETE ***")
    final_results = _evaluate_model(engine, ITERATIVE_TEST_QUESTIONS)
    os.makedirs("checkpoints", exist_ok=True)
    with open("checkpoints/iterative_final.json", "w") as f:
        json.dump({"final_training_count": engine._decoder_training_count, "final_results": final_results}, f, indent=2)


# ─── Mode: test ───

def _mode_test(args):
    """Quick diagnostic: verify decoder trains on real text."""
    print("Loading engine...")
    start = time.time()
    engine = CognitiveChatEngine(dim=args.dim, baby_mode=True)
    print(f"Engine loaded in {time.time() - start:.1f}s")

    print("\n=== INITIAL STATE ===")
    print(f"Decoder training count (total): {engine._decoder_training_count}")
    print(f"Decoder training count (web):   {engine._decoder_web_training_count}")
    print(f"Decoder vocab size: {len(engine._decoder_word_to_idx)}")
    print(f"Decoder exists: {engine.neural_decoder is not None}")

    print("\n=== TEST 1: _learn_from_text with real article text ===")
    sample_text = (
        "The truth is often more complex than people realize. "
        "Scientists study truth through careful observation and experimentation. "
        "There are many different theories about what truth means in philosophy. "
        "The search for truth drives human knowledge forward. "
        "Understanding truth requires both logic and evidence. "
        "Truth can be found through honest inquiry and open minded debate."
    )
    old_total = engine._decoder_training_count
    old_web = engine._decoder_web_training_count
    old_decoder_examples = engine.neural_decoder._total_training_examples if engine.neural_decoder else 0

    new_concepts = engine._learn_from_text(sample_text, "truth", source_url="test")

    delta_total = engine._decoder_training_count - old_total
    delta_web = engine._decoder_web_training_count - old_web
    delta_decoder = engine.neural_decoder._total_training_examples - old_decoder_examples if engine.neural_decoder else 0

    print(f"New concepts added: {new_concepts}")
    print(f"Decoder total count delta: {delta_total}")
    print(f"Decoder web count delta:   {delta_web}")
    print(f"NeuralDecoder internal examples delta: {delta_decoder}")
    print("DECODER IS TRAINING!" if delta_decoder > 0 else "Decoder did NOT train!")

    print("\n=== TEST 2: Call _learn_from_text again with different text ===")
    sample_text2 = (
        "Freedom is the power to act and speak without hindrance. "
        "People value freedom as a fundamental human right. "
        "The concept of freedom has evolved throughout history. "
        "Philosophers debate the limits of freedom in society. "
        "Freedom requires responsibility and respect for others. "
        "True freedom means being able to make choices about ones own life."
    )
    old_total = engine._decoder_training_count
    new_concepts2 = engine._learn_from_text(sample_text2, "freedom", source_url="test")
    delta_total2 = engine._decoder_training_count - old_total
    print(f"New concepts added: {new_concepts2}")
    print(f"Decoder total count delta: {delta_total2}")
    print("TRAINING PERSISTS!" if delta_total2 > 0 else "Decoder did NOT train on second call")

    print("\n=== DONE ===")
    engine.save()
    print("State saved.")


# ─── CLI ───

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA unified trainer")
    parser.add_argument("--mode", choices=["phase2", "iterative", "test"], default="phase2",
                        help="Training mode (default: phase2)")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension (default: 64)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cycles", type=int, default=5, help="Iterations for iterative mode (default: 5)")
    args = parser.parse_args()

    modes = {"phase2": _mode_phase2, "iterative": _mode_iterative, "test": _mode_test}
    modes[args.mode](args)


if __name__ == "__main__":
    main()
