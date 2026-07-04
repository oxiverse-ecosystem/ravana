import sys, os, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ravana', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ravana_ml', 'src'))
os.environ['RAVANA_SILENT'] = '1'

from scripts.ravana_chat import CognitiveChatEngine
import numpy as np

engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
nd = engine.neural_decoder
nd.reset_plasticity(stability=0.5)

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'corpora', 'teen_seeds.txt'), 'r', encoding='utf-8') as f:
    corpus_text = f.read()
all_sentences = nd.prepare_sentences(
    corpus_text, engine._decoder_word_to_embed, engine._decoder_word_to_idx,
    min_sentence_len=3)
n_avail = len(all_sentences)
print(f'Available sentences: {n_avail}')

t0 = time.time()
for i in range(10):
    s = all_sentences[i]
    nd.train_on_sentence(
        s['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
        word_indices=s['word_indices'],
        conditioning_embs=s['conditioning_embs'],
    )
elapsed = time.time() - t0
print(f'10 sentences in {elapsed:.2f}s ({elapsed/10*1000:.1f}ms per sentence)')
