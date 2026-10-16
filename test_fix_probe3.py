import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src'):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_rel_gate")

# Test the relevance gate directly
answer = "car engine repair requires mechanical tools and oil changes"
result = eng._relevance_gate("what is sleep", answer)
print(f"_relevance_gate('what is sleep', '{answer[:30]}...'): {result!r}")

# Check the score and overlap
from ravana.chat.relevance_store import _content_words, _jaccard_overlap
q_words = set(_content_words("what is sleep"))
a_words = set(_content_words(answer))
overlap = _jaccard_overlap(q_words, a_words)
print(f"q_words: {q_words}")
print(f"a_words: {a_words}")
print(f"jaccard overlap: {overlap}")

score = eng._relevance_scorer.score("what is sleep", answer, query_turn=eng.turn_count, threshold=eng._relevance_threshold)
print(f"relevance score: {score}")
print(f"threshold: {eng._relevance_threshold.base_threshold}")
print(f"score >= threshold: {score >= eng._relevance_threshold.base_threshold}")
print(f"overlap >= 0.15: {overlap >= 0.15}")
