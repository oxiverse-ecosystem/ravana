#!/usr/bin/env python3
"""Simple seed corpus training + evaluation loop"""
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from scripts.ravana_chat import CognitiveChatEngine

TEST_QUERIES = [
    "what happens if time travel possible",
    "why is it not possible til now", 
    "make time machine",
    "switch to cybersecurity",
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

def evaluate(engine):
    for q in TEST_QUERIES:
        resp = engine.process_turn(q)
        eval_r = evaluate_response(resp)
        print(f"Q: {q}")
        print(f"A: {resp}")
        print(f"  tmpl={eval_r['template_score']:.1f} div={eval_r['diversity']:.2f} sent={eval_r['sentences']}")
    avg_q = sum(evaluate_response(engine.process_turn(q))['quality'] for q in TEST_QUERIES) / len(TEST_QUERIES)
    print(f"AVG QUALITY: {avg_q:.3f}\n")
    return avg_q

def train_passes(engine, n):
    corpus_path = os.path.join(PROJ_ROOT, "data", "corpora", "teen_seeds.txt")
    with open(corpus_path, "r") as f:
        text = f.read()
    for i in range(n):
        err, _ = engine.neural_decoder.train_on_text(text, engine._decoder_word_to_embed, engine._decoder_word_to_idx, min_sentence_len=3)
    engine.neural_decoder.sleep_cycle()
    print(f"  Trained {n} passes, total: {engine._decoder_training_count}")

def main():
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=False)
    engine._bg_learning_active = False
    engine._needs_seed_training = False
    engine._needs_synthetic_training = False
    engine.baby_mode = False
    
    print("INITIAL")
    evaluate(engine)
    
    for cycle in range(5):
        print(f"CYCLE {cycle+1}: Training 5 passes...")
        train_passes(engine, 5)
        avg_q = evaluate(engine)
        if avg_q > 0.5:
            print("GOOD QUALITY!")
            break

if __name__ == "__main__":
    main()