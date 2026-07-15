#!/usr/bin/env python3
"""Test expanded domains with W_rel alignment."""
import sys, os, numpy as np
sys.path.insert(0, "/c/Users/Likhith/Documents/projects/ravana")
sys.path.insert(0, "/c/Users/Likhith/Documents/projects/ravana/ravana-v2")

import pytest

from ravana_ml.nn.rlm_v2 import RLMv2
from ravana_ml.tokenizer import WordTokenizer
from experiments.experiment_phase4_integrated import inject_minilm_embeddings
import numpy as np

# ============================================================
# EXPANDED DOMAIN DEFINITIONS (from experiment_cross_domain.py)
# ============================================================

def strip_trailing_spaces(facts):
    return [(i.rstrip(), t.rstrip(), r) for i, t, r in facts]

def _subject_holdout_split(facts, seed=42, holdout_ratio=0.2):
    from collections import defaultdict as _dd
    rng = np.random.RandomState(seed)
    subject_facts = _dd(list)
    for fact in facts:
        subject = fact[0].split()[0].lower()
        subject_facts[subject].append(fact)
    all_subjects = list(subject_facts.keys())
    rng.shuffle(all_subjects)
    n_holdout = max(1, int(len(all_subjects) * holdout_ratio))
    holdout_subjects = set(all_subjects[:n_holdout])
    train, test = [], []
    for subject, entries in subject_facts.items():
        if subject in holdout_subjects:
            test.extend(entries)
        else:
            train.extend(entries)
    return {"train": train, "test": test}


# ---- Domain A: Science (already expanded in experiment_cross_domain.py) ----
def build_domain_a_science():
    from experiments.experiment_cross_domain import build_domain_a_science as orig
    return orig()


# ---- Domain B: Social (EXPANDED VERSION) ----
def build_domain_b_social():
    facts = [
        # Causal facts (existing ~40)
        ("kindness leads to ", "trust", "causal"),
        ("anger causes ", "conflict", "causal"),
        ("sharing builds ", "friendship", "causal"),
        ("lying destroys ", "trust", "causal"),
        ("patience creates ", "understanding", "causal"),
        ("honesty builds ", "respect", "causal"),
        ("empathy creates ", "connection", "causal"),
        ("greed causes ", "loneliness", "causal"),
        ("jealousy causes ", "resentment", "causal"),
        ("generosity creates ", "gratitude", "causal"),
        ("rudeness causes ", "offense", "causal"),
        ("listening builds ", "rapport", "causal"),
        ("teaching builds ", "knowledge", "causal"),
        ("neglect causes ", "distance", "causal"),
        ("celebration builds ", "bonds", "causal"),
        ("criticism causes ", "defensiveness", "causal"),
        ("forgiveness heals ", "wounds", "causal"),
        ("praise boosts ", "confidence", "causal"),
        ("isolation causes ", "sadness", "causal"),
        ("teamwork creates ", "success", "causal"),
        ("gossip spreads ", "mistrust", "causal"),
        ("mentorship builds ", "skills", "causal"),
        ("bullying causes ", "trauma", "causal"),
        ("collaboration produces ", "innovation", "causal"),
        ("rejection causes ", "withdrawal", "causal"),
        ("inclusion builds ", "belonging", "causal"),
        ("betrayal destroys ", "loyalty", "causal"),
        ("gratitude strengthens ", "relationships", "causal"),
        ("boredom triggers ", "exploration", "causal"),
        ("competition drives ", "excellence", "causal"),
        ("compassion reduces ", "suffering", "causal"),
        ("sarcasm creates ", "tension", "causal"),
        ("trust enables ", "vulnerability", "causal"),
        ("leadership inspires ", "action", "causal"),
        ("apology restores ", "harmony", "causal"),
        ("neglect weakens ", "bonds", "causal"),
        ("humor defuses ", "conflict", "causal"),
        ("rivalry spurs ", "growth", "causal"),
        ("grief deepens ", "empathy", "causal"),
        ("curiosity sparks ", "discovery", "causal"),
        # NEW: Additional causal facts (same concepts, new relations)
        ("kindness fosters ", "trust", "causal"),
        ("kindness nurtures ", "trust", "causal"),
        ("anger provokes ", "conflict", "causal"),
        ("anger triggers ", "conflict", "causal"),
        ("sharing cultivates ", "friendship", "causal"),
        ("sharing encourages ", "friendship", "causal"),
        ("lying erodes ", "trust", "causal"),
        ("lying undermines ", "trust", "causal"),
        ("patience fosters ", "understanding", "causal"),
        ("patience breeds ", "understanding", "causal"),
        ("honesty earns ", "respect", "causal"),
        ("honesty commands ", "respect", "causal"),
        ("empathy fosters ", "connection", "causal"),
        ("empathy builds ", "connection", "causal"),
        ("greed breeds ", "loneliness", "causal"),
        ("greed invites ", "loneliness", "causal"),
        ("jealousy breeds ", "resentment", "causal"),
        ("jealousy fuels ", "resentment", "causal"),
        ("generosity inspires ", "gratitude", "causal"),
        ("generosity evokes ", "gratitude", "causal"),
        ("rudeness provokes ", "offense", "causal"),
        ("rudeness invites ", "offense", "causal"),
        ("listening cultivates ", "rapport", "causal"),
        ("listening establishes ", "rapport", "causal"),
        ("teaching imparts ", "knowledge", "causal"),
        ("teaching transmits ", "knowledge", "causal"),
        ("neglect creates ", "distance", "causal"),
        ("neglect breeds ", "distance", "causal"),
        ("celebration strengthens ", "bonds", "causal"),
        ("celebration deepens ", "bonds", "causal"),
        ("criticism triggers ", "defensiveness", "causal"),
        ("criticism provokes ", "defensiveness", "causal"),
        ("forgiveness mends ", "wounds", "causal"),
        ("forgiveness closes ", "wounds", "causal"),
        ("praise builds ", "confidence", "causal"),
        ("praise nurtures ", "confidence", "causal"),
        ("isolation breeds ", "sadness", "causal"),
        ("isolation deepens ", "sadness", "causal"),
        ("teamwork fosters ", "success", "causal"),
        ("teamwork yields ", "success", "causal"),
        ("gossip breeds ", "mistrust", "causal"),
        ("gossip sows ", "mistrust", "causal"),
        ("mentorship develops ", "skills", "causal"),
        ("mentorship nurtures ", "skills", "causal"),
        ("bullying inflicts ", "trauma", "causal"),
        ("bullying causes ", "trauma", "causal"),
        ("collaboration generates ", "innovation", "causal"),
        ("collaboration yields ", "innovation", "causal"),
        ("rejection triggers ", "withdrawal", "causal"),
        ("rejection causes ", "withdrawal", "causal"),
        ("inclusion fosters ", "belonging", "causal"),
        ("inclusion nurtures ", "belonging", "causal"),
        ("betrayal shatters ", "loyalty", "causal"),
        ("betrayal breaks ", "loyalty", "causal"),
        ("gratitude deepens ", "relationships", "causal"),
        ("gratitude cements ", "relationships", "causal"),
        ("boredom sparks ", "exploration", "causal"),
        ("boredom drives ", "exploration", "causal"),
        ("competition fuels ", "excellence", "causal"),
        ("competition sharpens ", "excellence", "causal"),
        ("compassion alleviates ", "suffering", "causal"),
        ("compassion eases ", "suffering", "causal"),
        ("sarcasm breeds ", "tension", "causal"),
        ("sarcasm creates ", "tension", "causal"),
        ("trust allows ", "vulnerability", "causal"),
        ("trust invites ", "vulnerability", "causal"),
        ("leadership motivates ", "action", "causal"),
        ("leadership galvanizes ", "action", "causal"),
        ("apology repairs ", "harmony", "causal"),
        ("apology mends ", "harmony", "causal"),
        ("neglect erodes ", "bonds", "causal"),
        ("neglect frays ", "bonds", "causal"),
        ("humor diffuses ", "conflict", "causal"),
        ("humor eases ", "conflict", "causal"),
        ("rivalry fuels ", "growth", "causal"),
        ("rivalry drives ", "growth", "causal"),
        ("grief fosters ", "empathy", "causal"),
        ("grief awakens ", "empathy", "causal"),
        ("curiosity ignites ", "discovery", "causal"),
        ("curiosity drives ", "discovery", "causal"),
        # Semantic facts
        ("friendship is ", "valuable", "semantic"),
        ("family is ", "important", "semantic"),
        ("trust is ", "fragile", "semantic"),
        ("wisdom is ", "rare", "semantic"),
        ("courage is ", "admirable", "semantic"),
        ("patience is ", "virtuous", "semantic"),
        ("humor is ", "helpful", "semantic"),
        ("loyalty is ", "noble", "semantic"),
        ("rudeness is ", "harmful", "semantic"),
        ("kindness is ", "powerful", "semantic"),
        ("honesty is ", "essential", "semantic"),
        ("grief is ", "natural", "semantic"),
        ("ambition is ", "driving", "semantic"),
        ("solitude is ", "peaceful", "semantic"),
        ("chaos is ", "destabilizing", "semantic"),
        ("harmony is ", "restorative", "semantic"),
        ("resentment is ", "corrosive", "semantic"),
        ("hope is ", "resilient", "semantic"),
        ("pride is ", "dangerous", "semantic"),
        ("grace is ", "inspiring", "semantic"),
        # NEW: Additional semantic facts (new concepts + existing)
        ("compassion is ", "healing", "semantic"),
        ("integrity is ", "steadfast", "semantic"),
        ("forgiveness is ", "liberating", "semantic"),
        ("gratitude is ", "transformative", "semantic"),
        ("humility is ", "grounding", "semantic"),
        ("curiosity is ", "expansive", "semantic"),
        ("resilience is ", "enduring", "semantic"),
        ("presence is ", "powerful", "semantic"),
        ("authenticity is ", "magnetic", "semantic"),
        ("vulnerability is ", "courageous", "semantic"),
        ("empathy is ", "bridging", "semantic"),
        ("patience is ", "timeless", "semantic"),
        ("kindness is ", "contagious", "semantic"),
        ("honesty is ", "clarifying", "semantic"),
        ("trust is ", "foundational", "semantic"),
        ("love is ", "transformative", "semantic"),
        ("friendship is ", "sustaining", "semantic"),
        ("loyalty is ", "anchoring", "semantic"),
        ("respect is ", "earned", "semantic"),
        ("understanding is ", "liberating", "semantic"),
        ("acceptance is ", "peaceful", "semantic"),
        ("boundaries are ", "healthy", "semantic"),
        ("communication is ", "connecting", "semantic"),
        ("listening is ", "validating", "semantic"),
        ("apology is ", "accountable", "semantic"),
        ("gratitude is ", "abundant", "semantic"),
        ("growth is ", "nonlinear", "semantic"),
        ("healing is ", "layered", "semantic"),
        ("intimacy is ", "vulnerable", "semantic"),
        ("joy is ", "contagious", "semantic"),
        ("kindness is ", "revolutionary", "semantic"),
        ("leadership is ", "service", "semantic"),
        ("mentorship is ", "investing", "semantic"),
        ("neuroplasticity is ", "hopeful", "semantic"),
        ("optimism is ", "strategic", "semantic"),
        ("presence is ", "medicinal", "semantic"),
        ("questions are ", "expansive", "semantic"),
        ("rest is ", "productive", "semantic"),
        ("safety is ", "prerequisite", "semantic"),
        ("trust is ", "earned", "semantic"),
        ("understanding is ", "bridging", "semantic"),
        ("vulnerability is ", "human", "semantic"),
        ("worthiness is ", "inherent", "semantic"),
        ("xenial is ", "welcoming", "semantic"),
        ("yearning is ", "guiding", "semantic"),
        ("zeal is ", "fuel", "semantic"),
    ]

    from experiments.experiment_cross_domain import _subject_holdout_split
    return _subject_holdout_split(facts, seed=42)


# ============================================================
# MAIN TEST
# ============================================================

@pytest.mark.slow
def test_expanded_domains():
    """Heavy RLMv2 training integration test (marked slow).

    Runs real encoder pretraining + two-domain training + cross-domain
    alignment + full evaluation. Takes ~10-15 min of CPU; excluded from a
    bare `pytest` run. Run explicitly with `-m slow`.
    """
    np.random.seed(42)
    
    domain_a = build_domain_a_science()
    domain_b = build_domain_b_social()
    
    train_a = strip_trailing_spaces(domain_a["train"])
    test_a = strip_trailing_spaces(domain_a["test"])
    train_b = strip_trailing_spaces(domain_b["train"])
    test_b = strip_trailing_spaces(domain_b["test"])
    
    print(f"Domain A: {len(train_a)} train, {len(test_a)} test")
    print(f"Domain B: {len(train_b)} train, {len(test_b)} test")
    
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
    model.use_verb_offset = True  # Enable verb-stem offset for held-out generalization
    inject_minilm_embeddings(model, tokenizer)
    model._pretrain_encoder_autoencoder(epochs=5, lr=0.01)
    
    # Patch relation keyword map
    import ravana_ml.nn.rlm_v2 as rlm_v2_module
    rlm_v2_module._KEYWORD_MAP['causal'].extend(['enables', 'enable', 'shapes', 'shape'])
    
    from experiments.experiment_cross_domain import train_rlm_on_domain
    train_rlm_on_domain(model, train_a, tokenizer, n_repeats=10, domain_tag='science')
    model.sleep_cycle()
    train_rlm_on_domain(model, train_b, tokenizer, n_repeats=10, domain_tag='social')
    model.sleep_cycle()
    
    # Compute verb offsets after training
    print("\n[Verb Offset] Computing verb offsets from training data...")
    model._compute_verb_offsets()
    
    # Cross-domain alignment
    for _ in range(200):
        model._cross_domain_relation_alignment()
    
    print(f"Alignment quality: {model.measure_cross_domain_alignment()}")
    
    # Evaluate
    from experiments.experiment_cross_domain import evaluate_rlm
    
    print("\n=== FULL EVALUATION ===")
    for name, test in [('Science held-out', test_a), ('Social held-out', test_b),
                       ('Science train', train_a), ('Social train', train_b)]:
        r = evaluate_rlm(model, test, tokenizer)
        print(f'{name}: Top-1={r["top1_accuracy"]:.1%}, Top-10={r["top10_accuracy"]:.1%} (n={r["n_tested"]})')
    
    # Cross-domain transfer test
    b_causal = [(i, t, r) for i, t, r in train_b if r == 'causal']
    a_causal_verbs = ['causes ', 'produces ', 'enables ', 'creates ', 'drives ', 'shapes ']
    
    correct, top10, tested = 0, 0, 0
    for orig_inp, orig_tgt, _ in b_causal[:10]:
        subject = orig_inp.split()[0]
        for a_verb in a_causal_verbs:
            new_inp = f'{subject} {a_verb}'
            input_ids = np.array(tokenizer.encode(new_inp), dtype=np.int64)
            logits = model.forward(input_ids)
            probs_data = logits.data.flatten()
            target_id = tokenizer.encode(orig_tgt)[0]
            pred_id = int(np.argmax(probs_data))
            in_top10 = target_id in set(np.argsort(probs_data)[-10:])
            
            if int(np.argmax(probs_data)) == target_id:
                correct += 1
            if in_top10:
                top10 += 1
            tested += 1
    
    print(f'\nCross-domain transfer: Top-1={correct}/{tested}={correct/tested:.1%}, Top-10={top10}/{tested}={top10/tested:.1%}')


if __name__ == "__main__":
    test_expanded_domains()