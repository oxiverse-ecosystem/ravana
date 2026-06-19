#!/usr/bin/env python3
"""
Interactive Chat + Training Loop for RAVANA

Chats with the model, evaluates responses, trains more, repeats.
"""
import os
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from scripts.ravana_chat import CognitiveChatEngine

TEST_QUERIES = [
    "what happens if time travel possible",
    "why is it not possible til now", 
    "make time machine",
    "switch to cybersecurity",
    "tell me about consciousness",
    "how does memory work",
]

TEMPLATE_MARKERS = ["relates to", "connects with", "links to", "associated with", 
                    "causes", "leads to", "results in", "influences",
                    "stands against", "opposite of", "differs from"]

def evaluate_response(response: str) -> dict:
    r_low = response.lower()
    template_count = sum(1 for m in TEMPLATE_MARKERS if m in r_low)
    words = len(response.split())
    sentences = len([s for s in response.split('.') if s.strip()])
    unique = len(set(w.lower() for w in response.split()))
    diversity = unique / max(1, words)
    template_score = template_count / max(1, sentences)
    return {
        "template_count": template_count,
        "template_score": template_score,
        "sentences": sentences,
        "words": words,
        "diversity": diversity,
        "quality": diversity * (1 - min(1, template_score/3))
    }

def chat_and_evaluate(engine, queries):
    results = []
    for q in queries:
        resp = engine.process_turn(q)
        eval_r = evaluate_response(resp)
        eval_r["q"] = q
        eval_r["a"] = resp
        results.append(eval_r)
        print(f"Q: {q}")
        print(f"A: {resp}")
        print(f"  Quality: {eval_r['quality']:.3f} (tmpl={eval_r['template_score']:.1f}, div={eval_r['diversity']:.2f}, sent={eval_r['sentences']})")
        print()
    avg_q = sum(r['quality'] for r in results) / len(results)
    print(f"AVG QUALITY: {avg_q:.3f}")
    return results

def train_seed_corpus(engine, passes=10):
    corpus_path = os.path.join(PROJ_ROOT, "data", "corpora", "teen_seeds.txt")
    with open(corpus_path, "r") as f:
        text = f.read()
    for i in range(passes):
        err, n = engine.neural_decoder.train_on_text(
            text, engine._decoder_word_to_embed, engine._decoder_word_to_idx, min_sentence_len=3)
    engine.neural_decoder.sleep_cycle()
    print(f"  Trained {passes} passes, total: {engine._decoder_training_count}")

def main():
    print("="*60)
    print("RAVANA INTERACTIVE TRAINING LOOP")
    print("="*60)
    
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=False)
    engine._bg_learning_active = False
    engine._needs_seed_training = False
    engine._needs_synthetic_training = False
    engine.baby_mode = False
    
    cycle = 0
    while cycle < 5:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle}")
        print(f"{'='*60}")
        
        # Chat and evaluate
        print("** EVALUATION **")
        results = chat_and_evaluate(engine, TEST_QUERIES)
        
        avg_quality = sum(r['quality'] for r in results) / len(results)
        if avg_quality > 0.4:
            print(f"\n*** GOOD QUALITY REACHED ({avg_quality:.3f}) ***")
            break
        
        # Train more
        print("** TRAINING **")
        train_seed_corpus(engine, passes=5)
        
        # Trigger web learning on interesting topics
        if cycle % 2 == 1:
            print("** WEB LEARNING TRIGGER **")
            try:
                engine.learn_from_web("time travel physics paradoxes", max_results=1)
                engine.learn_from_web("consciousness neuroscience theories", max_results=1)
            except Exception as e:
                print(f"  Web learning error: {e}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("FINAL EVALUATION")
    print(f"{'='*60}")
    chat_and_evaluate(engine, TEST_QUERIES)

if __name__ == "__main__":
    main()