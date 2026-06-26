import sys
import os
import time

_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

# Remove weights to start fresh
weights_path = os.path.join(_proj_root, "data", "ravana_weights.pkl")
if os.path.exists(weights_path):
    os.remove(weights_path)

from scripts.ravana_chat import CognitiveChatEngine

print("Initializing engine...")
engine = CognitiveChatEngine(dim=64, baby_mode=True)
print("Engine initialized.")

# Pre-cache sentences from teen_seeds.txt
corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
corpus_text = open(corpus_path, "r", encoding="utf-8").read()

nd = engine.neural_decoder
all_sentences = nd.prepare_sentences(
    corpus_text, engine._decoder_word_to_embed, engine._decoder_word_to_idx,
    min_sentence_len=3
)
n_available = len(all_sentences)
sentences_per_pass = min(200, n_available)

print(f"Prepared {n_available} sentences. Training in batches of {sentences_per_pass}...")

import numpy as np
rng = np.random.RandomState(42)

for epoch in range(1, 11): # 10 epochs of 10 passes = 100 passes total
    err_sum = 0.0
    passes = 0
    for _ in range(10):
        idx = rng.permutation(n_available)
        batch = [all_sentences[idx[j]] for j in range(sentences_per_pass)]
        for sent in batch:
            err = nd.train_on_sentence(
                sent['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
                word_indices=sent['word_indices'],
                conditioning_embs=sent['conditioning_embs']
            )
            err_sum += err
        passes += sentences_per_pass
        
    nd.sleep_cycle()
    print(f"Epoch {epoch*10} passes: CE={nd._avg_cross_entropy:.4f}, Top1={nd._avg_top1_acc:.4f}, Top5={nd._avg_top5_acc:.4f}")
