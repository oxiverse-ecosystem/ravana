import sys
import os
import time
import numpy as np

_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

from scripts.ravana_chat import CognitiveChatEngine, CognitiveResponseContext

print("Initializing engine...")
engine = CognitiveChatEngine(dim=64, baby_mode=True)
print("Engine loaded.")

nd = engine.neural_decoder

# Let's run a custom intensive training loop on the seed corpus
corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
with open(corpus_path, "r", encoding="utf-8") as f:
    text = f.read()

# Expand vocabulary first
import re
words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", text.lower()))
new_for_vocab = [w for w in words_in_corpus if w not in engine._decoder_word_to_idx]
if new_for_vocab:
    engine._expand_decoder_vocab(new_for_vocab)

all_sentences = nd.prepare_sentences(
    text, engine._decoder_word_to_embed, engine._decoder_word_to_idx,
    min_sentence_len=3,
)

n_available = len(all_sentences)
print(f"Total available sentences: {n_available}")

# Let's train for 150 passes!
n_passes = 150
sentences_per_pass = n_available
rng = np.random.RandomState(42)

print("\n=== Starting Intensive Training (150 passes) ===")
t_start = time.time()
for pass_idx in range(n_passes):
    idx = rng.permutation(n_available)
    err_sum = 0.0
    for j in range(n_available):
        sent = all_sentences[idx[j]]
        err = nd.train_on_sentence(
            sent['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
            word_indices=sent['word_indices'],
            conditioning_embs=sent['conditioning_embs'],
        )
        err_sum += err
    avg_err = err_sum / n_available
    
    if (pass_idx + 1) % 10 == 0:
        print(f"Pass {pass_idx+1}/{n_passes} - Loss: {avg_err:.4f} - CE EMA: {nd._avg_cross_entropy:.4f} - Top1 EMA: {nd._avg_top1_acc:.4f}")
        nd.sleep_cycle()

print(f"Training completed in {time.time()-t_start:.1f}s")
print(f"Final CE EMA: {nd._avg_cross_entropy:.4f} - Top1 EMA: {nd._avg_top1_acc:.4f}")

# Let's test generation now!
print("\n=== Testing Generation after Intensive Training ===")
topics = ["ravana", "oxiverse", "hello", "trust"]
for topic in topics:
    ctx = CognitiveResponseContext(
        raw_input=f"what is {topic}",
        subject=topic,
        associated_concepts=[(topic, 1.0)],
        past_topics=[]
    )
    # Generate
    try:
        raw_resp = engine._generate_with_decoder(ctx)
        print(f"Topic: {topic} -> Response: {raw_resp}")
    except Exception as e:
        print(f"Error generating for {topic}: {e}")
