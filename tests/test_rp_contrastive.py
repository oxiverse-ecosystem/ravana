import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

# Pre-encode ALL text (training + test) to fix vocabulary before model creation
all_texts = [
    "heat causes expansion",
    "fire produces warmth",
    "kindness leads to trust",
    "anger causes conflict",
    "ice causes expansion",
    "light produces warmth",
    "honesty leads to trust",
    "sadness causes conflict",
    "friction produces heat",
    "gravity pulls objects",
    "rain causes growth",
    "cold freezes water",
    "patience creates understanding",
    "honesty builds respect",
    "generosity creates gratitude",
    # Test facts:
    "ice causes expansion",
    "light produces warmth",
    "honesty leads to trust",
    "sadness causes conflict",
    "cold melts ice",  # novel relation word "melts"
]
tokenizer = WordTokenizer()
for t in all_texts:
    tokenizer.encode(t)

print(f"Vocab size: {tokenizer.vocab_size}")

model = RLMv2(vocab_size=tokenizer.vocab_size, embed_dim=64, concept_dim=64, n_concepts=100)
model.disable_spreading_activation = True
model.use_rp_for_analogy = True
model.use_rp_hidden = True
model.use_rp_contrastive = True

# Train facts with multiple relations per domain
facts = [
    ("heat causes ", "expansion", 0),
    ("fire produces ", "warmth", 0),
    ("friction produces ", "heat", 0),
    ("gravity pulls ", "objects", 0),
    ("rain causes ", "growth", 0),
    ("cold freezes ", "water", 0),
    ("kindness leads to ", "trust", 1),
    ("honesty builds ", "respect", 1),
    ("generosity creates ", "gratitude", 1),
    ("patience creates ", "understanding", 1),
    ("anger causes ", "conflict", 1),
    ("sadness causes ", "conflict", 1),
]

# Train with contrastive loss
print("Training with contrastive loss...")
for epoch in range(50):
    total_loss = 0
    for inp, tgt, domain in facts:
        input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
        target_ids = np.array(tokenizer.encode(tgt), dtype=np.int64)
        model.current_domain_id = domain
        model.set_domain(domain)
        res = model.learn(input_ids, target_ids)
        total_loss += res["loss"]
    # Train RP with contrastive loss
    model._rp_contrastive_lambda = 1.0
    # Manually call RP backward with contrastive loss for each fact
    for inp, tgt, domain in facts:
        input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
        target_ids = np.array(tokenizer.encode(tgt), dtype=np.int64)
        model.set_domain(domain)
        model.forward(input_ids)  # to populate cache
        model._rp_backward(int(target_ids[0]), )
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}: loss={total_loss/len(facts):.6f}")

# Test on held-out subjects
test_facts = [
    ("ice causes ", "expansion", 0),
    ("light produces ", "warmth", 0),
    ("honesty leads to ", "trust", 1),
    ("sadness causes ", "conflict", 1),
    ("cold melts ", "ice", 0),  # novel relation word
]

print("\nTest with spreading DISABLED (RP only):")
correct = 0
for inp, tgt, domain in test_facts:
    input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
    target_ids = np.array(tokenizer.encode(tgt), dtype=np.int64)
    model.set_domain(domain)
    logits = model.forward(input_ids).data.flatten()
    pred_id = int(np.argmax(logits))
    pred_tok = tokenizer.decode([pred_id])
    is_correct = pred_tok == tgt
    if is_correct:
        correct += 1
    print(f"  Domain {domain}: {inp.strip()} -> {tgt} (pred: {pred_tok}, correct: {is_correct})")
print(f"Accuracy: {correct}/{len(test_facts)} = {correct/len(test_facts)*100:.1f}%")