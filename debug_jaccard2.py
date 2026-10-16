import os, sys
os.environ["RAVANA_OFFLINE"] = "1"
PROJ = r"C:\Users\Likhith\Documents\Projects\ravana"
for p in (PROJ, f"{PROJ}\\ravana_ml\\src", f"{PROJ}\\ravana\\src", f"{PROJ}\\ravana-v2\\src"):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.relevance_store import _content_words, _jaccard_overlap
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="test_jaccard3")

# Test case 1: "what is sleep" vs "xyzyx qwvut plmno"
q1 = set(_content_words("what is sleep"))
a1 = set(_content_words("xyzyx qwvut plmno"))
print("q1:", q1)
print("a1:", a1)
print("jaccard(q1, a1):", round(_jaccard_overlap(q1, a1), 4))
s1 = eng._relevance_scorer.score("what is sleep", "xyzyx qwvut plmno", query_turn=0, threshold=eng._relevance_threshold)
print("score:", round(s1, 4))
print()

# Test case 2: "what do you dream about" vs "xyzyx qwvut plmno"
q2 = set(_content_words("what do you dream about"))
a2 = set(_content_words("xyzyx qwvut plmno"))
print("q2:", q2)
print("a2:", a2)
print("jaccard(q2, a2):", round(_jaccard_overlap(q2, a2), 4))
s2 = eng._relevance_scorer.score("what do you dream about", "xyzyx qwvut plmno", query_turn=0, threshold=eng._relevance_threshold)
print("score:", round(s2, 4))
print()

# Test case 3: recall answer
q3 = set(_content_words("remind me what you said about board games"))
ans3 = "i said: hmm, i'm not totally sure about friends yet, and that's okay. what's your take on it?"
a3 = set(_content_words(ans3))
print("q3:", q3)
print("a3:", a3)
print("jaccard(q3, a3):", round(_jaccard_overlap(q3, a3), 4))
s3 = eng._relevance_scorer.score("remind me what you said about board games", ans3, query_turn=0, threshold=eng._relevance_threshold)
print("score:", round(s3, 4))
print()

# Test case 4: relevant answer
q4 = set(_content_words("what is sleep"))
a4 = set(_content_words("sleep is when the mind rests and dreams happen"))
print("q4:", q4)
print("a4:", a4)
print("jaccard(q4, a4):", round(_jaccard_overlap(q4, a4), 4))
s4 = eng._relevance_scorer.score("what is sleep", "sleep is when the mind rests and dreams happen", query_turn=0, threshold=eng._relevance_threshold)
print("score:", round(s4, 4))
