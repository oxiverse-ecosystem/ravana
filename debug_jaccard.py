import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.relevance_store import _content_words, _jaccard_overlap
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_jaccard")

# Test case 1: "what is sleep" vs "xyzyx qwvut plmno"
q1 = set(_content_words("what is sleep"))
a1 = set(_content_words("xyzyx qwvut plmno"))
print(f"q1: {q1}")
print(f"a1: {a1}")
print(f"jaccard(q1, a1): {_jaccard_overlap(q1, a1):.4f}")
print(f"score: {eng._relevance_scorer.score('what is sleep', 'xyzyx qwvut plmno', query_turn=0, threshold=eng._relevance_threshold):.4f}")
print()

# Test case 2: "what do you dream about" vs "xyzyx qwvut plmno"
q2 = set(_content_words("what do you dream about"))
a2 = set(_content_words("xyzyx qwvut plmno"))
print(f"q2: {q2}")
print(f"a2: {a2}")
print(f"jaccard(q2, a2): {_jaccard_overlap(q2, a2):.4f}")
print(f"score: {eng._relevance_scorer.score('what do you dream about', 'xyzyx qwvut plmno', query_turn=0, threshold=eng._relevance_threshold):.4f}")
print()

# Test case 3: "remind me what you said about board games" vs "i said: hmm, i'm not totally sure about friends yet, and that's okay. what's your take on it?"
q3 = set(_content_words("remind me what you said about board games"))
a3 = set(_content_words("i said: hmm, i'm not totally sure about friends yet, and that's okay. what's your take on it?"))
print(f"q3: {q3}")
print(f"a3: {a3}")
print(f"jaccard(q3, a3): {_jaccard_overlap(q3, a3):.4f}")
print(f"score: {eng._relevance_scorer.score('remind me what you said about board games', 'i said: hmm, i\\'m not totally sure about friends yet, and that\\'s okay. what\\'s your take on it?', query_turn=0, threshold=eng._relevance_threshold):.4f}")
print()

# Test case 4: "what is sleep" vs "sleep is when the mind rests and dreams happen"
q4 = set(_content_words("what is sleep"))
a4 = set(_content_words("sleep is when the mind rests and dreams happen"))
print(f"q4: {q4}")
print(f"a4: {a4}")
print(f"jaccard(q4, a4): {_jaccard_overlap(q4, a4):.4f}")
print(f"score: {eng._relevance_scorer.score('what is sleep', 'sleep is when the mind rests and dreams happen', query_turn=0, threshold=eng._relevance_threshold):.4f}")
