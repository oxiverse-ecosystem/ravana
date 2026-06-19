#!/usr/bin/env python3
"""
Iterative Decoder Training Loop for RAVANA

Runs training cycles + evaluation until decoder produces natural language.
"""
import os
import sys
import json
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from scripts.ravana_chat import CognitiveChatEngine

# Test questions to evaluate quality
TEST_QUESTIONS = [
    "what happens if time travel possible",
    "why not possible til now",
    "make time machine",
    "switch to cybersecurity",
    "tell me about consciousness",
    "how does memory work",
    "what is the meaning of life",
    "explain quantum mechanics",
]

TEMPLATE_MARKERS = [
    "relates to", "connects with", "links to", "is associated with",
    "leads to", "results in", "causes", "influences", "contributes to",
    "stands against", "contrasts with", "on the other hand"
]

NATURAL_MARKERS = [
    "furthermore", "moreover", "however", "although", "consequently",
    "therefore", "nevertheless", "nonetheless", "in addition", "similarly",
    "accordingly", "subsequently", "meanwhile", "conversely"
]

def evaluate_response(response: str) -> dict:
    """Evaluate response quality: template vs natural language."""
    r_low = response.lower()
    
    template_count = sum(1 for m in TEMPLATE_MARKERS if m in r_low)
    natural_count = sum(1 for m in NATURAL_MARKERS if m in r_low)
    
    # Word count and sentence count
    words = len(response.split())
    sentences = len([s for s in response.split('.') if s.strip()])
    
    # Diversity: unique words ratio
    unique_words = len(set(w.lower() for w in response.split()))
    diversity = unique_words / max(1, words)
    
    template_ratio = template_count / max(1, template_count + natural_count)
    natural_ratio = natural_count / max(1, template_count + natural_count)
    
    return {
        "template_count": template_count,
        "natural_count": natural_count,
        "template_ratio": template_ratio,
        "natural_ratio": natural_ratio,
        "words": words,
        "sentences": sentences,
        "diversity": diversity,
        "quality_score": natural_ratio * diversity  # Combined metric
    }

def run_training_cycle(engine: CognitiveChatEngine, cycle: int):
    """Run one training cycle: seed corpus + web articles."""
    print(f"\n{'='*60}")
    print(f"TRAINING CYCLE {cycle}")
    print(f"{'='*60}")
    
    # Phase 1: Heavy seed corpus training (50 passes Hebbian)
    print(f"  [Phase 1] Seed corpus training: 50 passes...")
    corpus_path = os.path.join(PROJ_ROOT, "data", "corpora", "teen_seeds.txt")
    if os.path.exists(corpus_path):
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_text = f.read()
        
        total_err = 0.0
        total_passes = 0
        for _ in range(50):
            err, n = engine.neural_decoder.train_on_text(
                corpus_text,
                engine._decoder_word_to_embed,
                engine._decoder_word_to_idx,
                min_sentence_len=3, max_sentences=200
            )
            total_err += err
            total_passes += n
        
        engine.neural_decoder.sleep_cycle()
        engine._decoder_seed_training_count += total_passes
        engine._decoder_training_count += total_passes
        print(f"  [Phase 1] Done: {total_passes} sentences, avg_err={total_err/50:.4f}")
    
    # Phase 2: Web article training (if network available)
    print(f"  [Phase 2] Web article training...")
    # This happens automatically during process_turn via web_learner
    # We'll just trigger a few searches
    for topic in ["time travel physics", "consciousness neuroscience", "quantum mechanics explained"]:
        try:
            engine.learn_from_web(topic, max_results=2)
        except Exception as e:
            print(f"    Web training error for {topic}: {e}")
    
    print(f"  [Phase 2] Done: web_training={engine._decoder_web_training_count}")
    
    print(f"  [Summary] Total training: {engine._decoder_training_count} sentences")

def evaluate_model(engine: CognitiveChatEngine) -> list:
    """Evaluate model on test questions."""
    print(f"\n{'='*60}")
    print(f"EVALUATION")
    print(f"{'='*60}")
    
    results = []
    for q in TEST_QUESTIONS:
        response = engine.process_turn(q)
        eval_result = evaluate_response(response)
        eval_result["question"] = q
        eval_result["response"] = response
        results.append(eval_result)
        
        print(f"  Q: {q}")
        print(f"  A: {response}")
        print(f"  Quality: {eval_result['quality_score']:.3f} (nat={eval_result['natural_ratio']:.2f}, tmpl={eval_result['template_ratio']:.2f}, div={eval_result['diversity']:.2f})")
        print()
    
    avg_quality = sum(r['quality_score'] for r in results) / len(results)
    avg_natural = sum(r['natural_ratio'] for r in results) / len(results)
    avg_template = sum(r['template_ratio'] for r in results) / len(results)
    
    print(f"  AVG Quality: {avg_quality:.3f} | Natural: {avg_natural:.2f} | Template: {avg_template:.2f}")
    
    return results

def main():
    cycles = 5
    eval_every = 1
    
    print("Starting Iterative Decoder Training")
    print(f"Cycles: {cycles}, Eval every: {eval_every}")
    
    # Initialize engine
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=False)
    engine._bg_learning_active = False
    engine._needs_seed_training = False
    engine._needs_synthetic_training = False
    engine.baby_mode = False
    
    # Initial evaluation (cycle 0)
    print("\n*** INITIAL STATE ***")
    evaluate_model(engine)
    
    for cycle in range(1, cycles + 1):
        run_training_cycle(engine, cycle)
        
        if cycle % eval_every == 0:
            results = evaluate_model(engine)
            avg_quality = sum(r['quality_score'] for r in results) / len(results)
            
            # Save checkpoint
            checkpoint = {
                "cycle": cycle,
                "training_count": engine._decoder_training_count,
                "seed_training": engine._decoder_seed_training_count,
                "web_training": engine._decoder_web_training_count,
                "avg_quality": avg_quality,
                "results": results
            }
            
            with open(f".iterative_checkpoint_c{cycle}.json", "w") as f:
                json.dump(checkpoint, f, indent=2)
            
            print(f"\n  Checkpoint saved: .iterative_checkpoint_c{cycle}.json")
            
            if avg_quality > 0.5:
                print(f"  *** QUALITY THRESHOLD REACHED ({avg_quality:.3f} > 0.5) ***")
                break
    
    print("\n*** TRAINING COMPLETE ***")
    final_results = evaluate_model(engine)
    
    # Save final checkpoint
    with open(".iterative_final.json", "w") as f:
        json.dump({
            "final_training_count": engine._decoder_training_count,
            "final_results": final_results
        }, f, indent=2)

if __name__ == "__main__":
    main()