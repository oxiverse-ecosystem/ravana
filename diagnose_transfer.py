#!/usr/bin/env python3
"""
Cross-Domain Semantic Transfer & Held-Out Diagnostic
=====================================================
Tests whether RAVANA learns transferable causal structure or just memorizes patterns.
"""

import sys, os, numpy as np
sys.path.insert(0, "/c/Users/Likhith/Documents/projects/ravana")
sys.path.insert(0, "/c/Users/Likhith/Documents/projects/ravana/ravana-v2")

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_cross_domain import (
    build_domain_a_science, build_domain_b_social,
    train_rlm_on_domain, evaluate_rlm
)
from tests.test_structural_transfer import strip_trailing_spaces
from experiments.experiment_phase4_integrated import inject_minilm_embeddings


def build_model_and_tokenizer():
    """Build trained RLMv2 model with tokenizer."""
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
    vocab_size = tokenizer.vocab_size + 5
    
    model = RLMv2(
        vocab_size=vocab_size, embed_dim=64, concept_dim=64,
        n_concepts=vocab_size, sleep_interval=300,
        freeze_token_embeds_in_rp=True,
        latent_dim=64,
    )
    model._tokenizer = tokenizer
    model.use_cross_domain_alignment = True
    model.use_shared_relation_embeds = False
    model.alignment_lr = 0.05
    model.use_rp_for_analogy = True
    import ravana_ml.nn.rlm_v2 as rlm_v2_module
    rlm_v2_module._KEYWORD_MAP['causal'].extend(['enables', 'enable', 'shapes', 'shape'])
    inject_minilm_embeddings(model, tokenizer)
    model._pretrain_encoder_autoencoder(epochs=5, lr=0.01)
    
    return model, tokenizer, train_a, test_a, train_b, test_b


def analyze_held_out_failures(model, tokenizer, test_facts, domain_name):
    """Analyze why held-out facts fail."""
    print(f"\n{'='*60}")
    print(f"HELD-OUT FAILURE ANALYSIS: {domain_name}")
    print(f"{'='*60}")
    
    failures = []
    for inp, tgt, rel_type in test_facts:
        input_ids = np.array(tokenizer.encode(inp), dtype=np.int64)
        logits = model.forward(input_ids)
        probs_data = logits.data.flatten()
        target_id = tokenizer.encode(tgt)[0]
        pred_id = int(np.argmax(probs_data))
        pred = tokenizer.decode([pred_id])
        
        if pred != tgt:
            # Analyze the failure
            subject = inp.split()[0]
            relation = inp.split()[1] if ' ' in inp else inp
            
            # Check if entities are OOV
            subject_id = tokenizer.encode(subject)[0]
            subject_oov = subject_id >= len(model.token_embed.weight.data) - 5
            
            # Get top-5 predictions
            top5 = np.argsort(probs_data)[-5:][::-1]
            top5_tokens = [tokenizer.decode([t]) for t in top5]
            
            failures.append({
                'input': inp,
                'target': tgt,
                'pred': pred,
                'subject': subject,
                'relation': relation,
                'top5': top5_tokens,
                'prob_target': probs_data[target_id],
                'prob_pred': probs_data[pred_id],
                'subject_oov': subject_oov,
            })
            
    print(f"Total held-out: {len(test_facts)}")
    print(f"Failures: {len(failures)}")
    print(f"Success rate: {(len(test_facts) - len(failures)) / len(test_facts) * 100:.1f}%")
    
    # Analyze failure patterns
    if failures:
        print("\nFailure patterns:")
        oov_count = sum(1 for f in failures if f['subject_oov'])
        print(f"  OOV subjects: {oov_count}/{len(failures)}")
        
        # Check if target appears in top5
        top5_count = sum(1 for f in failures if f['target'] in f['top5'])
        print(f"  Target in top-5: {top5_count}/{len(failures)}")
        
        # Show top 10 failures
        for f in failures[:10]:
            print(f"  {f['input'][:40]:40s} target={f['target']:12s} pred={f['pred']:12s} "
                  f"target_prob={f['prob_target']:.3f} pred_prob={f['prob_pred']:.3f}")
    
    return failures


def test_science_to_social_semantic_transfer(model, tokenizer):
    """Test if Science causal structure transfers to novel Social queries."""
    print(f"\n{'='*60}")
    print("SCIENCE → SOCIAL SEMANTIC TRANSFER TEST (simplified)")
    print(f"(Using full model with both domains trained)")
    print(f"{'='*60}")
    
    # Use the full model's tokenizer and trained model
    # Science causal verbs + Social subjects
    social_subjects = ['kindness', 'anger', 'sharing', 'patience', 'honesty', 
                       'empathy', 'greed', 'jealousy', 'generosity', 'rudeness',
                       'listening', 'teaching', 'neglect', 'celebration', 'criticism',
                       'forgiveness', 'praise', 'isolation', 'teamwork', 'gossip',
                       'mentorship', 'bullying', 'collaboration', 'rejection',
                       'inclusion', 'betrayal', 'gratitude', 'boredom', 'competition',
                       'compassion', 'sarcasm', 'trust', 'leadership', 'apology',
                       'neglect', 'humor', 'rivalry', 'grief', 'curiosity']
    
    a_causal_verbs = ['causes ', 'produces ', 'enables ', 'creates ', 'drives ', 'shapes ']
    
    # Ground truth from Social domain
    ground_truth = {
        'kindness': 'trust',
        'anger': 'conflict',
        'sharing': 'friendship',
        'patience': 'understanding',
        'honesty': 'respect',
        'empathy': 'connection',
        'greed': 'loneliness',
        'jealousy': 'resentment',
        'generosity': 'gratitude',
        'rudeness': 'offense',
        'listening': 'rapport',
        'teaching': 'knowledge',
        'neglect': 'distance',
        'celebration': 'bonds',
        'criticism': 'defensiveness',
        'forgiveness': 'wounds',
        'praise': 'confidence',
        'isolation': 'sadness',
        'teamwork': 'success',
        'gossip': 'mistrust',
        'mentorship': 'skills',
        'bullying': 'trauma',
        'collaboration': 'innovation',
        'rejection': 'withdrawal',
        'inclusion': 'belonging',
        'betrayal': 'loyalty',
        'gratitude': 'relationships',
        'boredom': 'exploration',
        'competition': 'excellence',
        'compassion': 'suffering',
        'sarcasm': 'tension',
        'trust': 'vulnerability',
        'leadership': 'action',
        'apology': 'harmony',
        'humor': 'conflict',
        'rivalry': 'growth',
        'grief': 'empathy',
        'curiosity': 'discovery',
    }
    
    correct, top10, tested = 0, 0, 0
    
    for subject, expected in ground_truth.items():
        for a_verb in ['causes ', 'produces ', 'enables ']:
            new_inp = f"{subject} {a_verb}"
            input_ids = np.array(tokenizer.encode(new_inp), dtype=np.int64)
            logits = model.forward(input_ids)
            probs_data = logits.data.flatten()
            target_id = tokenizer.encode(expected)[0]
            pred_id = int(np.argmax(probs_data))
            pred = tokenizer.decode([pred_id])
            in_top10 = target_id in set(np.argsort(probs_data)[-10:])
            
            if pred == expected:
                correct += 1
            if in_top10:
                top10 += 1
            tested += 1
            
            status = '✓' if pred == expected else ('~' if in_top10 else '✗')
            print(f"  {subject} {a_verb} → expected={expected:12s} pred={pred:12s} {status}")
    
    print(f"\nScience→Social semantic transfer: Top-1={correct}/{tested}={correct/tested*100:.1f}%, "
          f"Top-10={top10}/{tested}={top10/tested*100:.1f}%")
    
    return correct, top10, tested


def test_full_model_transfer(model, tokenizer, train_a, train_b):
    """Test cross-domain transfer on full trained model."""
    print(f"\n{'='*60}")
    print("FULL MODEL CROSS-DOMAIN TRANSFER (Science→Social)")
    print(f"{'='*60}")
    
    domain_b = build_domain_b_social()
    train_b = strip_trailing_spaces(domain_b["train"])
    
    b_causal = [(i, t, r) for i, t, r in train_b if r == 'causal']
    a_causal_verbs = ['causes ', 'produces ', 'enables ', 'creates ', 'drives ', 'shapes ']
    
    correct, top10, tested = 0, 0, 0
    for orig_inp, orig_tgt, _ in b_causal[:10]:
        subject = orig_inp.split()[0]
        for a_verb in a_causal_verbs:
            new_inp = f"{subject} {a_verb}"
            input_ids = np.array(tokenizer.encode(new_inp), dtype=np.int64)
            logits = model.forward(input_ids)
            probs_data = logits.data.flatten()
            target_id = tokenizer.encode(orig_tgt)[0]
            pred_id = int(np.argmax(probs_data))
            pred = tokenizer.decode([pred_id])
            in_top10 = target_id in set(np.argsort(probs_data)[-10:])
            
            if pred == orig_tgt:
                correct += 1
            if in_top10:
                top10 += 1
            tested += 1
    
    print(f"Full model transfer: Top-1={correct}/{tested}={correct/tested*100:.1f}%, "
          f"Top-10={top10}/{tested}={top10/tested*100:.1f}%")
    return correct, top10, tested


def main():
    np.random.seed(42)
    print("="*60)
    print("SEMANTIC TRANSFER & HELD-OUT DIAGNOSTIC")
    print("="*60)
    
    # Test 1: Full model with both domains
    model, tokenizer, train_a, test_a, train_b, test_b = build_model_and_tokenizer()
    
    from experiments.experiment_cross_domain import train_rlm_on_domain
    train_rlm_on_domain(model, train_a, tokenizer, n_repeats=8, domain_tag='science')
    model.sleep_cycle()
    train_rlm_on_domain(model, train_b, tokenizer, n_repeats=8, domain_tag='social')
    model.sleep_cycle()
    
    for _ in range(200):
        model._cross_domain_relation_alignment()
    
    print(f"\nAlignment quality: {model.measure_cross_domain_alignment()}")
    
    # Analyze held-out failures
    analyze_held_out_failures(model, tokenizer, test_a, "Science")
    analyze_held_out_failures(model, tokenizer, test_b, "Social")
    
    # Test Science→Social semantic transfer on science-only model
    test_science_to_social_semantic_transfer(model, tokenizer)
    
    # Test full model cross-domain transfer
    test_full_model_transfer(model, tokenizer, train_a, train_b)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("If Science-only model transfers to Social causal structure:")
    print("  → Semantic alignment works, held-out failure is entity/relation rarity")
    print("If Science-only model FAILS to transfer:")
    print("  → Semantic alignment is broken, W_rel collapsing distinct relations")
    print("If full model PCX=75% but held-out=8%:")
    print("  → Overfitting to training entities, not learning transferable structure")


if __name__ == "__main__":
    import numpy as np
    main()