import sys, os, time, re, json
import numpy as np

_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
os.environ["RAVANA_SILENT"] = "1"

from scripts.ravana_chat import CognitiveChatEngine

print("=" * 60)
print("RAVANA Decoder Training — Phase 2+3 (FIXED)")
print("=" * 60)
print()
t0 = time.time()

engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
print()

engine.neural_decoder.reset_plasticity(stability=0.5)
print("  [reset] Decoder plasticity reset to stability=0.5")
print()

corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
with open(corpus_path, "r", encoding="utf-8") as f:
    corpus_text = f.read()

engine._freeze_decoder_vocab = False
words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", corpus_text.lower()))
new_for_vocab = [w for w in words_in_corpus if w not in engine._decoder_word_to_idx]
if new_for_vocab:
    engine._expand_decoder_vocab(new_for_vocab)
engine._freeze_decoder_vocab = True

nd = engine.neural_decoder
all_sentences = nd.prepare_sentences(
    corpus_text, engine._decoder_word_to_embed, engine._decoder_word_to_idx,
    min_sentence_len=3,
)
n_available = len(all_sentences)
sentences_per_pass = min(1000, n_available)
n_passes = 120
sleep_interval = 5
patience = 30
target_ce = 2.5

print(f"[2/3] Training on teen_seeds.txt ({n_available} sents, {sentences_per_pass}/pass, {n_passes} passes)...")
print(f"  sleep_interval={sleep_interval}, patience={patience}")
print()

rng = np.random.RandomState(42)
passes = 0
best_ce = float('inf')
stall_count = 0
print_interval = max(1, n_passes // 50)

try:
    for i in range(n_passes):
        idx = rng.choice(n_available, size=sentences_per_pass, replace=False)
        batch = [all_sentences[j] for j in idx]
        for sent in batch:
            nd.train_on_sentence(
                sent['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
                word_indices=sent['word_indices'],
                conditioning_embs=sent['conditioning_embs'],
            )
        passes += sentences_per_pass

        if (i + 1) % sleep_interval == 0:
            nd.sleep_cycle()

        if (i + 1) % print_interval == 0:
            eta = (time.time() - t0) / (i + 1) * (n_passes - i - 1)
            print(f"  Pass {i+1}/{n_passes}: CE={nd._avg_cross_entropy:.3f} "
                  f"top1={nd._avg_top1_acc:.3f} top5={nd._avg_top5_acc:.3f} "
                  f"(sents={passes}, ETA={eta:.0f}s)", flush=True)

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
    print(f"  [Interrupted] Consolidating partial training ({passes} sentences)...")

if n_passes % sleep_interval != 0:
    nd.sleep_cycle()

engine._decoder_seed_training_count = passes
engine._decoder_training_count += passes

print(f"  Trained {passes} sentences in {time.time() - t0:.1f}s")
print()

print("[3/3] Skipping synthetic graph training...")
print(f"  Graph edges still available for conditioning ({len(engine.graph.edges)} edges)")
print()

print("Saving engine state...")
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
print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
print(f"  Final metrics: CE={nd._avg_cross_entropy:.3f} "
      f"top1={nd._avg_top1_acc:.3f} top5={nd._avg_top5_acc:.3f}")
print()

# Quick smoke test
print("Smoke test: generating a response...")
t4 = time.time()
test_response = engine.process_turn("what is trust")
test_time = time.time() - t4
print(f"  Generated in {test_time:.1f}s")
print(f"  Strategy: {engine._last_strategy}")
print(f"  Response: {test_response[:200] if len(test_response) > 200 else test_response}")
print()
print("Done!")
