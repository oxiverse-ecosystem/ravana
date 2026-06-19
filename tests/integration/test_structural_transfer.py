"""
Structural Transfer Diagnostic for RLMv2.

Tests what the model CAN actually do:
1. In-distribution recall: predicts trained facts (should be 100%)
2. Subject generalization: same predicate+target, similar subject (RP generalization)
3. Predicate discrimination: same subject, OOD predicate with trained target
4. Cross-domain: Domain A verb + Domain B subject (if target trained)
"""

import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_phase4_integrated import inject_minilm_embeddings
from experiments.experiment_cross_domain import (
    build_domain_a_science, build_domain_b_social,
    train_rlm_on_domain
)

def strip_trailing_spaces(facts):
    return [(i.rstrip(), t.rstrip(), r) for i, t, r in facts]

def run():
    config = type('Config', (), {
        'n_train_repeats': 35,
        'embed_dim': 64, 'concept_dim': 64, 'n_hidden': 128,
        'n_layers': 3, 'sleep_interval': 300, 'seed': 42
    })()

    np.random.seed(config.seed)
    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()

    train_a = strip_trailing_spaces(domain_a["train"])
    test_a = strip_trailing_spaces(domain_a["test"])
    train_b = strip_trailing_spaces(domain_b["train"])
    test_b = strip_trailing_spaces(domain_b["test"])

    tokenizer = WordTokenizer()
    for inp, tgt, _ in train_a + test_a + train_b + test_b:
        tokenizer.encode(inp)
        tokenizer.encode(tgt)

    vocab_size = len(tokenizer.word_to_id)
    model = RLMv2(vocab_size=vocab_size+5, embed_dim=64, concept_dim=64,
                  n_concepts=vocab_size, sleep_interval=300, gate_concept_creation=False)
    model._tokenizer = tokenizer
    inject_minilm_embeddings(model, tokenizer)
    model._pretrain_encoder_autoencoder(epochs=300, lr=0.01)

    # Phase 1: Train on Domain A
    print("Phase 1: Training on Domain A (Science)...")
    model.set_domain(0)
    acc_a, err_a = train_rlm_on_domain(model, domain_a["train"], tokenizer,
                                        n_repeats=config.n_train_repeats, domain_tag="science")
    model.set_domain(None)

    n_train = len(train_a)
    n_correct = sum(1 for a in acc_a[-n_train:] if a > 0.5) if len(acc_a) >= n_train else 0
    final_train_acc = float(np.mean(acc_a[-min(10, len(acc_a)):]) if acc_a else 0)
    print(f"  Training accuracy (final repeats): {final_train_acc:.1%}")
    print(f"  Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")

    # Build train targets set
    train_targets = set(t for _, t, _ in train_a)

    # ── Test 1: In-distribution recall ──
    print("\n── Test 1: In-Distribution Recall ──")
    correct = 0
    for inp, tgt, _ in train_a[:20]:
        input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
        logits = model.forward(input_ids)
        pred = tokenizer.decode([int(np.argmax(logits.data))])
        if pred == tgt: correct += 1
    print(f"  Recall on 20 trained facts: {correct}/20 = {correct/20:.0%}")

    # ── Test 2: Subject generalization ──
    # Find test facts where target IS a train target
    print("\n── Test 2: Subject Generalization (same target, similar subject) ──")
    testable_facts = [(i, t, r) for i, t, r in test_a if t in train_targets]
    if testable_facts:
        correct, top10 = 0, 0
        for inp, tgt, _ in testable_facts:
            input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
            logits = model.forward(input_ids)
            probs = logits.data.flatten()
            target_id = tokenizer.encode(tgt)[0]
            pred = tokenizer.decode([int(np.argmax(probs))])
            in_top10 = target_id in set(np.argsort(probs)[-10:])
            if pred == tgt: correct += 1
            if in_top10: top10 += 1
            flag = "OK" if pred == tgt else ("~" if in_top10 else "X")
            print(f"  [{flag}] '{inp}' -> '{tgt}' (pred: '{pred}')")
        print(f"  Top-1: {correct}/{len(testable_facts)} = {correct/max(1,len(testable_facts)):.0%}")
        print(f"  Top-10: {top10}/{len(testable_facts)} = {top10/max(1,len(testable_facts)):.0%}")
    else:
        print("  No testable facts found (0 targets in training)")

    # ── Test 3: Predicate OOD ──
    # Take trained facts and swap to a novel predicate
    print("\n── Test 3: Predicate OOD (novel predicate, trained target) ──")
    ood_pred = "accelerates"
    predicate_seen = model._seen_predicates
    if ood_pred not in predicate_seen:
        print(f"  Novel predicate: '{ood_pred}' (not in training predicates)")
        test_cases = [
            ("heat accelerates", "expansion", "causal"),
            ("fire accelerates", "warmth", "causal"),
        ]
        correct, top10 = 0, 0
        for inp, tgt, _ in test_cases:
            input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
            logits = model.forward(input_ids)
            probs = logits.data.flatten()
            target_id = tokenizer.encode(tgt)[0]
            pred = tokenizer.decode([int(np.argmax(probs))])
            in_top10 = target_id in set(np.argsort(probs)[-10:])
            if pred == tgt: correct += 1
            if in_top10: top10 += 1
            flag = "OK" if pred == tgt else ("~" if in_top10 else "X")
            print(f"  [{flag}] '{inp}' -> '{tgt}' (pred: '{pred}', seen preds: {len(predicate_seen)})")
        print(f"  Top-1: {correct}/{len(test_cases)} = {correct/max(1,len(test_cases)):.0%}")
    else:
        print(f"  '{ood_pred}' already seen, skipping")

    # ── Test 4: Cross-domain (A verb + B subject, B target) ──
    print("\n── Test 4: Cross-domain (Domain A verb + Domain B subject) ──")
    b_causal = [(i, t, r) for i, t, r in train_b if r == "causal"]
    b_targets = set(t for _, t, _ in b_causal)
    correct, top10 = 0, 0
    tested = 0
    for orig_inp, orig_tgt, _ in b_causal[:8]:
        subject = orig_inp.split()[0]
        for a_verb in ["causes ", "produces ", "enables "]:
            new_inp = f"{subject} {a_verb}"
            input_ids = np.array(tokenizer.encode(new_inp), dtype=np.int64)
            logits = model.forward(input_ids)
            probs = logits.data.flatten()
            target_id = tokenizer.encode(orig_tgt)[0]
            pred = tokenizer.decode([int(np.argmax(probs))])
            in_top10 = target_id in set(np.argsort(probs)[-10:])
            if pred == orig_tgt: correct += 1
            if in_top10: top10 += 1
            tested += 1
            flag = "OK" if pred == orig_tgt else ("~" if in_top10 else "X")
            print(f"  [{flag}] '{new_inp}' -> '{orig_tgt}' (pred: '{pred}')")
    print(f"  Top-1: {correct}/{tested} = {correct/max(1,tested):.0%}")
    print(f"  Top-10: {top10}/{tested} = {top10/max(1,tested):.0%}")

    print("\nDiagnostic complete.")

if __name__ == "__main__":
    run()
