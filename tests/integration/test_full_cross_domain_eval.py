"""Quick full cross-domain probe evaluation to check current state."""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest
from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_cross_domain import (
    build_domain_a_science, build_domain_b_social,
    train_rlm_on_domain, evaluate_rlm,
)


@pytest.mark.slow
@pytest.mark.skip(reason="Full cross-domain eval takes >2min even with reduced params; run manually")
def test_cross_domain_structural_transfer():
    """Full cross-domain probe evaluation: train A, train B, retrain both, test probes."""
    tokenizer = WordTokenizer()

    # Pre-build vocab
    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()
    all_facts = domain_a['train'] + domain_a['test'] + domain_b['train'] + domain_b['test']
    for input_text, target_text, _ in all_facts:
        tokenizer.encode(input_text)
        tokenizer.encode(target_text)
    # Pre-tokenize cross-domain probes
    for t in ["kindness causes ", "trust", "anger produces ", "conflict",
              "sharing enables ", "friendship", "heat causes ", "expansion",
              "trust is ", "fragile", "friction produces ", "heat",
              "patience creates ", "understanding", "gossip spreads ", "mistrust",
              "collaboration produces ", "innovation", "gravity pulls ", "objects",
              "inclusion builds ", "belonging", "compassion reduces ", "suffering",
              "fire produces ", "warmth", "leadership inspires ", "action",
              "apology restores ", "harmony", "oxygen enables ", "combustion",
              "rivalry spurs ", "growth", "grief deepens ", "empathy",
              "curiosity sparks ", "discovery", "trust enables ", "vulnerability"]:
        tokenizer.encode(t)

    vocab_size = tokenizer.vocab_size
    np.random.seed(42)

    model = RLMv2(
        vocab_size=vocab_size + 5, embed_dim=64, concept_dim=64,
        n_concepts=vocab_size, sleep_interval=100,
    )
    model._tokenizer = tokenizer

    # Inject MiniLM and pretrain autoencoder
    from experiments.experiment_phase4_integrated import inject_minilm_embeddings
    inject_minilm_embeddings(model, tokenizer)
    print("Pre-training encoder autoencoder on MiniLM embeddings...")
    model._pretrain_encoder_autoencoder(epochs=50, lr=0.01)  # Reduced from 300 for test speed

    # Train Domain A
    train_rlm_on_domain(model, domain_a['train'], tokenizer, n_repeats=5,  # Reduced from 35
                         domain_tag="science", buffer_for_replay=True)
    model.snapshot_replay_buffer("science")
    model.activate_domain_memories("science")
    model.sleep_cycle()

    # Train Domain B
    train_rlm_on_domain(model, domain_b['train'], tokenizer, n_repeats=5,  # Reduced from 35
                         domain_tag="social", buffer_for_replay=True)
    model.snapshot_replay_buffer("social")
    model.activate_domain_memories("social")
    model.sleep_cycle()

    # Retrain Domain A
    train_rlm_on_domain(model, domain_a['train'], tokenizer, n_repeats=3)  # Reduced from 20
    model.sleep_cycle()

    # Retrain Domain B
    train_rlm_on_domain(model, domain_b['train'], tokenizer, n_repeats=3)  # Reduced from 20
    model.sleep_cycle()

    # 1. Structural transfer test on trained facts (retrieve trained facts with swapped relation verbs)
    print("\n============================================================")
    print("STRUCTURAL TRANSFER ON TRAINED FACTS (Verb Swapping)")
    print("============================================================")
    transfer_train = test_structural_transfer(model, tokenizer, domain_a["train"], domain_b["train"])
    for r in transfer_train['probes']:
        status = "PASS" if r['correct'] else ("T10" if r['in_top10'] else "FAIL")
        print(f"  {status:>4}  {r['input']:<28} expected={r['expected']:<14} got={r['predicted']:<14}")
    print(f"\nTrain-fact Transfer Top-1: {transfer_train['top1_accuracy']:.1%}  Top-10: {transfer_train['top10_accuracy']:.1%}")

    # 2. Structural transfer test on held-out test facts
    print("\n============================================================")
    print("STRUCTURAL TRANSFER ON HELD-OUT TEST FACTS")
    print("============================================================")
    transfer_test = test_structural_transfer(model, tokenizer, domain_a["test"], domain_b["test"])
    for r in transfer_test['probes']:
        status = "PASS" if r['correct'] else ("T10" if r['in_top10'] else "FAIL")
        print(f"  {status:>4}  {r['input']:<28} expected={r['expected']:<14} got={r['predicted']:<14}")
    print(f"\nTest-fact Transfer Top-1: {transfer_test['top1_accuracy']:.1%}  Top-10: {transfer_test['top10_accuracy']:.1%}")

    assert len(transfer_train['probes']) > 0, "No train probes generated"
    assert len(transfer_test['probes']) > 0, "No test probes generated"
    # Verify we get positive transfer on the trained concepts
    assert transfer_train['top1_accuracy'] > 0.1, f"Trained transfer too low: {transfer_train['top1_accuracy']:.1%}"
