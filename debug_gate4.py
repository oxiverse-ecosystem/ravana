import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_gate_score")

# Test the engine.py version (score-only)
print("=== engine.py _relevance_gate (score-only) ===")
r1 = eng._relevance_gate("what is sleep", "car engine repair requires mechanical tools and oil changes")
print(f"sleep vs car engine repair: {r1!r}")

r2 = eng._relevance_gate("what is sleep", "sleep is when the mind rests and dreams happen")
print(f"sleep vs sleep answer: {r2!r}")

r3 = eng._relevance_gate("what do you dream about", "when i let my mind wander, i find myself drawn to people who hoard every book ever read")
print(f"dream vs books answer: {r3!r}")

r4 = eng._relevance_gate("", "some answer")
print(f"empty query: {r4!r}")

r5 = eng._relevance_gate("some query", "")
print(f"empty answer: {r5!r}")

# Check the score directly
print("\n=== Relevance scorer scores ===")
s1 = eng._relevance_scorer.score("what is sleep", "car engine repair requires mechanical tools and oil changes", query_turn=0, threshold=eng._relevance_threshold)
print(f"sleep vs car engine repair score: {s1:.4f} (threshold: {eng._relevance_threshold.base_threshold:.4f})")

s2 = eng._relevance_scorer.score("what is sleep", "sleep is when the mind rests and dreams happen", query_turn=0, threshold=eng._relevance_threshold)
print(f"sleep vs sleep answer score: {s2:.4f}")

s3 = eng._relevance_scorer.score("what do you dream about", "when i let my mind wander, i find myself drawn to people who hoard every book ever read", query_turn=0, threshold=eng._relevance_threshold)
print(f"dream vs books answer score: {s3:.4f}")
