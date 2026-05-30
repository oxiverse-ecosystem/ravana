"""Quick full cross-domain probe evaluation to check current state."""
import sys, os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pytest
from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_cross_domain import (
    build_domain_a_science, build_domain_b_social,
    train_rlm_on_domain, evaluate_rlm,
)
from experiments.experiment_cross_domain import test_structural_transfer as _run_structural_transfer


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

    model = RLM(
        vocab_size=vocab_size, embed_dim=64, concept_dim=64,
        n_concepts=vocab_size, n_hidden=128, n_layers=3,
        sleep_interval=100, tokenizer=tokenizer,
    )

    # Train Domain A
    train_rlm_on_domain(model, domain_a['train'], tokenizer, n_repeats=3,
                         domain_tag="science", buffer_for_replay=True)
    model.snapshot_replay_buffer("science")
    model.activate_domain_memories("science")
    model.sleep_cycle()

    # Train Domain B
    train_rlm_on_domain(model, domain_b['train'], tokenizer, n_repeats=3,
                         domain_tag="social", buffer_for_replay=True)
    model.snapshot_replay_buffer("social")
    model.activate_domain_memories("social")
    model.sleep_cycle()

    # Retrain Domain A
    train_rlm_on_domain(model, domain_a['train'], tokenizer, n_repeats=2)
    model.sleep_cycle()

    # Retrain Domain B
    train_rlm_on_domain(model, domain_b['train'], tokenizer, n_repeats=2)
    model.sleep_cycle()

    # Full structural transfer test
    transfer = _run_structural_transfer(model, tokenizer)

    failures = [r for r in transfer['probes'] if not r['correct']]
    for r in transfer['probes']:
        status = "PASS" if r['correct'] else ("T10" if r['in_top10'] else "FAIL")
        print(f"  {status:>4}  {r['input']:<28} expected={r['expected']:<14} got={r['predicted']:<14}")

    print(f"\nTop-1: {transfer['top1_accuracy']:.1%}  Top-10: {transfer['top10_accuracy']:.1%}")

    # At minimum, top-10 should be above 50% for a working architecture
    assert transfer['top10_accuracy'] >= 0.5, (
        f"Cross-domain top-10 too low: {transfer['top10_accuracy']:.1%}"
    )
