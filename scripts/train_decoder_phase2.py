#!/usr/bin/env python3
"""
RAVANA Decoder Training — Phase 2+3
=====================================
Loads the existing CognitiveChatEngine, trains the neural decoder on:
1. The teen_seeds.txt corpus (natural English sentences — primary training)
2. Synthetic sentences from graph edges are SKIPPED — they produce
   "X connects with Y" template artifacts that poison the decoder.

Then saves the updated state so the chat script picks up the improvements.

Usage:
    python scripts/train_decoder_phase2.py
"""
import sys, os, time
import numpy as np
_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))

os.environ["RAVANA_SILENT"] = "1"

print("=" * 60)
print("RAVANA Decoder Training — Phase 2+3")
print("=" * 60)
print()

# ── Auto-fetch teen_seeds.txt if missing ──
corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
if not os.path.exists(corpus_path):
    print("[0/3] teen_seeds.txt not found — gathering from internet...", flush=True)
    gather_script = os.path.join(_proj_root, "scripts", "gather_teen_seeds.py")
    if os.path.exists(gather_script):
        import subprocess
        ret = subprocess.call([sys.executable, gather_script, "--force"])
        if ret != 0:
            print(f"  [warn] Gather script exited with code {ret}", flush=True)
    else:
        print(f"  [warn] gather_teen_seeds.py not found at {gather_script}", flush=True)
    print()

# ── Load Engine ──
print("[1/3] Loading CognitiveChatEngine...", flush=True)
t0 = time.time()
from scripts.ravana_chat import CognitiveChatEngine

engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
load_time = time.time() - t0

print(f"  Loaded in {load_time:.1f}s")
print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
print(f"  Decoder: {engine._decoder_training_count} total, {engine._decoder_web_training_count} web, {engine._decoder_seed_training_count} seed")
print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
print()

# ── Reset decoder plasticity so this training burst lands at full strength ──
engine.neural_decoder.reset_plasticity(stability=0.5)
print(f"  [reset] Decoder plasticity reset to stability=0.5 (was freezing after prior runs)")
print()

# ── Step 1: Seed corpus training (the ONLY training — no synthetic templates) ──
print("[2/3] Training decoder on teen_seeds.txt corpus (real English only)...", flush=True)
t1 = time.time()
corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
if not os.path.exists(corpus_path):
    print(f"  WARNING: Corpus still not found at {corpus_path} after gather attempt")
    corpus_sentences = 0
else:
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus_text = f.read()

    # Expand decoder vocab with corpus words first (one-time)
    import re
    words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", corpus_text.lower()))
    new_for_vocab = [w for w in words_in_corpus if w not in engine._decoder_word_to_idx]
    if new_for_vocab:
        engine._expand_decoder_vocab(new_for_vocab)

    # Pre-process corpus once: cache word_indices + conditioning_embs per sentence
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
    target_ce = 3.0  # Lower target — we want better next-word prediction

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

seed_time = time.time() - t1
print(f"  Trained {corpus_sentences} sentences in {seed_time:.1f}s")
print(f"  Decoder seed count: {engine._decoder_seed_training_count}")
print()

# ── Step 2: Synthetic graph training SKIPPED ──
# The synthetic "{s} connects with {o}" templates were poisoning the decoder.
# The decoder now trains ONLY on real English from teen_seeds.txt.
# Graph knowledge is used for conditioning embeddings, not for template text.
print("[3/3] Skipping synthetic graph training (was causing template artifacts)...", flush=True)
print(f"  Graph edges still available for conditioning embeddings ({len(engine.graph.edges)} edges)")
print()

# ── Save ──
print("Saving engine state...", flush=True)
t3 = time.time()
result = engine.save()
save_time = time.time() - t3
print(f"  Save: {result}")
print(f"  Saved in {save_time:.1f}s")
print()

# ── Summary ──
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
print(f"  Total time: {time.time() - t0:.1f}s")
print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
print(f"  Decoder total: {engine._decoder_training_count}")
print(f"  Decoder seed:  {engine._decoder_seed_training_count}")
print(f"  Decoder web:   {engine._decoder_web_training_count}")
print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
print(f"  Final metrics: CE={engine.neural_decoder._avg_cross_entropy:.3f} "
      f"top1={engine.neural_decoder._avg_top1_acc:.3f} "
      f"top5={engine.neural_decoder._avg_top5_acc:.3f}")
print()

# Quick smoke test
print("Smoke test: generating a simple response...", flush=True)
t4 = time.time()
test_response = engine.process_turn("what is trust")
test_time = time.time() - t4
print(f"  Generated in {test_time:.1f}s")
print(f"  Strategy: {engine._last_strategy}")
print(f"  Response: {test_response[:300] if len(test_response) > 300 else test_response}")
print()
print("✅ Phase 2+3 complete! Ready for interactive chat.")
