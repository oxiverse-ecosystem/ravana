"""Debug: inspect concept graph edges for fire/heat/expansion"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer

tok = WordTokenizer()
train = [
    ("heat causes expansion", "expansion"),
    ("heat melts ice", "ice"),
    ("fire produces smoke", "smoke"),
    ("fire creates heat", "heat"),
    ("cold freezes water", "water"),
    ("kindness causes trust", "trust"),
    ("kindness creates friendship", "friendship"),
    ("anger produces conflict", "conflict"),
    ("rain causes flooding", "flooding"),
    ("sun produces heat", "heat"),
    ("sun causes growth", "growth"),
    ("exercise strengthens muscles", "muscles"),
    ("exercise causes sweating", "sweating"),
    ("fire is hot", "hot"),
    ("sun is hot", "hot"),
    ("exercise produces heat", "heat"),
    ("love causes trust", "trust"),
]
all_texts = set()
for t, _ in train: all_texts.add(t)
for t in ["fire causes expansion", "love causes trust", "anger causes expansion"]:
    all_texts.add(t)
for t in sorted(all_texts): tok.encode(t)

model = RLMv2(vocab_size=tok.vocab_size+5, embed_dim=32, concept_dim=32, n_concepts=tok.vocab_size, sleep_interval=100)
model._tokenizer = tok

# Train
for epoch in range(500):
    for text, target_word in train:
        ids = tok.encode(text)
        target_id = tok.encode(target_word)[0]
        ctx = np.array(ids[:-1], dtype=np.int64)
        tgt = np.array([target_id], dtype=np.int64)
        model.learn(ctx, tgt)

# Build reverse map: concept_id → word
cid_to_word = {}
for tok_id in range(tok.vocab_size):
    bindings = model.binding_map.get_concepts(tok_id, min_confidence=0.1)
    for b in bindings:
        cid_to_word[b.concept_id] = tok.decode([tok_id])

def get_word(cid):
    return cid_to_word.get(cid, f"c{cid}")

# Print ALL edges
print(f"--- All {len(model.graph.edges)} edges ---")
for (src, tgt), edge in model.graph.edges.items():
    print(f"  {get_word(src)} → {get_word(tgt)}  type={edge.relation_type}  w={edge.weight:.3f}")

# Fire's outgoing
fire_id = tok.encode("fire")[0]
fire_bindings = model.binding_map.get_concepts(fire_id, min_confidence=0.1)
if fire_bindings:
    fire_cid = fire_bindings[0].concept_id
    print(f"\n--- Fire (cid={fire_cid}) outgoing ---")
    for tgt_id, edge in model.graph.get_outgoing(fire_cid):
        print(f"  fire → {get_word(tgt_id)}  type={edge.relation_type}  w={edge.weight:.3f}")
        for tgt2_id, edge2 in model.graph.get_outgoing(tgt_id):
            hop = edge.weight * edge.confidence * edge2.weight * edge2.confidence * 0.6
            print(f"    2-hop → {get_word(tgt2_id)}  type={edge2.relation_type}  hop={hop:.4f}")

# Test forward
print(f"\n--- Forward: fire causes ---")
ctx = tok.encode("fire causes")
ctx = np.array(ctx, dtype=np.int64)
logits = model.forward(ctx)
flat = logits.data.flatten()
top10 = np.argsort(flat)[::-1][:10]
for i, idx in enumerate(top10):
    word = tok.decode([int(idx)])
    print(f"  #{i+1}: {word} (score={flat[idx]:.4f})")

expansion_id = tok.encode("expansion")[0]
print(f"\n  expansion score: {flat[expansion_id]:.6f}")
print(f"  expansion rank: {int(np.argsort(flat)[::-1].tolist().index(expansion_id)) + 1}")
