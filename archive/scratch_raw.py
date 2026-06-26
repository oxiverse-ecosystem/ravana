import sys
import os
import numpy as np

_proj_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

from scripts.ravana_chat import CognitiveChatEngine, CognitiveResponseContext

print("Loading engine...")
engine = CognitiveChatEngine(dim=64, baby_mode=True)
print("Engine loaded.")

nd = engine.neural_decoder

# Let's get the conditioning embeddings for "ravana"
topic = "ravana"
ctx = CognitiveResponseContext(
    raw_input=f"what is {topic}",
    subject=topic,
    associated_concepts=[(topic, 1.0)],
    past_topics=[]
)

# Let's reproduce the conditioning embeddings build
concept_embs = []
subj_lower = topic.lower()
if subj_lower in engine._concept_keywords:
    nids = engine._concept_keywords[subj_lower]
    node = engine.graph.get_node(nids[0])
    if node and node.vector is not None:
        concept_embs.append(node.vector.copy())

conditioning_embs = np.stack(concept_embs, axis=0).astype(np.float32)

bos_idx = engine._decoder_word_to_idx.get("<bos>", 0)
eos_idx = engine._decoder_word_to_idx.get("<eos>", 2)

print("\n--- TEST 1: Pure Generation (No BG, No cerebellar, No biases, temp=0.0) ---")
try:
    gen1 = nd.generate(
        conditioning_embs=conditioning_embs,
        max_steps=28,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        temperature=0.0,
        cerebellar_ngram=None,
        idx_to_word=None,
        basal_ganglia=None,
        content_word_ids=None,
        token_boost=None
    )
    words1 = [engine._decoder_idx_to_word[idx] for idx in gen1 if idx in engine._decoder_idx_to_word]
    print("Tokens:", gen1)
    print("Text:", " ".join(words1))
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n--- TEST 2: Pure Generation (temp=0.5, top-p) ---")
try:
    gen2 = nd.generate(
        conditioning_embs=conditioning_embs,
        max_steps=28,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        temperature=0.5,
        cerebellar_ngram=None,
        idx_to_word=None,
        basal_ganglia=None,
        content_word_ids=None,
        token_boost=None
    )
    words2 = [engine._decoder_idx_to_word[idx] for idx in gen2 if idx in engine._decoder_idx_to_word]
    print("Tokens:", gen2)
    print("Text:", " ".join(words2))
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n--- TEST 3: Generation with token boost of topic (temp=0.5, top-p) ---")
try:
    subject_idx = engine._decoder_word_to_idx.get(topic.lower())
    token_boost = {subject_idx: 3.0} if subject_idx is not None else None
    gen3 = nd.generate(
        conditioning_embs=conditioning_embs,
        max_steps=28,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        temperature=0.5,
        cerebellar_ngram=None,
        idx_to_word=None,
        basal_ganglia=None,
        content_word_ids=None,
        token_boost=token_boost
    )
    words3 = [engine._decoder_idx_to_word[idx] for idx in gen3 if idx in engine._decoder_idx_to_word]
    print("Tokens:", gen3)
    print("Text:", " ".join(words3))
except Exception as e:
    import traceback
    traceback.print_exc()
