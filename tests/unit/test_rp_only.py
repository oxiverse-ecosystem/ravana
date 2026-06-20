from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

# Minimal setup
tokenizer = WordTokenizer()
# Encode all texts to build vocab
texts = [
    "heat causes expansion",
    "fire produces warmth",
    "kindness leads to trust",
    "anger causes conflict",
    "ice causes expansion",
    "light produces warmth",
    "honesty leads to trust",
    "sadness causes conflict",
]
for t in texts:
    tokenizer.encode(t)

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Vocab: {tokenizer.word_to_id}")

model = RLMv2(vocab_size=tokenizer.vocab_size, embed_dim=64, concept_dim=64, n_concepts=100)
model.disable_spreading_activation = True  # Disable spreading, rely on RP
model.use_rp_for_analogy = True
model.use_rp_hidden = True
model.use_rp_contrastive = True  # Enable contrastive

# Train a few facts
facts = [
    ("heat causes ", "expansion"),
    ("fire produces ", "warmth"),
    ("kindness leads to ", "trust"),
    ("anger causes ", "conflict"),
]

for epoch in range(20):
    for inp, tgt in facts:
        input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
        target_ids = np.array(tokenizer.encode(tgt), dtype=np.int64)
        model.learn(input_ids, target_ids)

# Test on held-out subjects
test_facts = [
    ("ice causes ", "expansion"),  # held-out subject for causal
    ("light produces ", "warmth"),
    ("honesty leads to ", "trust"),
    ("sadness causes ", "conflict"),
]

print("\nTest with spreading DISABLED (RP only):")
correct = 0
for inp, tgt in test_facts:
    input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
    target_ids = np.array(tokenizer.encode(tgt), dtype=np.int64)
    logits = model.forward(input_ids).data.flatten()
    pred_id = int(np.argmax(logits))
    pred_tok = tokenizer.decode([pred_id])
    is_correct = pred_tok == tgt
    if is_correct:
        correct += 1
    print(f"  {inp.strip()} -> {tgt} (pred: {pred_tok}, correct: {is_correct})")
print(f"Accuracy: {correct}/{len(test_facts)} = {correct/len(test_facts)*100:.1f}%")