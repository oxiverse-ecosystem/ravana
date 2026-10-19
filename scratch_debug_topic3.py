import os, sys, re
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_topic_gr')
eng.process_turn('i love open source software')
eng.process_turn('what do you think about open source?')

q = "do you ever feel curious about open source?"
t = q.lower().strip()

# Check what _topic_grounded_self_experience returns
tg = eng._topic_grounded_self_experience(q, t)
print("TOPIC_GROUNDED:", repr(tg))

# Check the topic extraction in the gate
_topic_word = None
_tw_m = re.search(r"\b(?:about|of|on)\s+([a-z][a-z\s]{2,40})\s*$", t)
if _tw_m and _tw_m.group(1).strip():
    _topic_word = _tw_m.group(1).strip().split()[0]
if not _topic_word:
    _tw_m = re.search(r"\b(dream|think|feel|wonder|curious|afraid|scared|hope|wish|regret|love|like|enjoy)\s+about\s*$", t)
    if _tw_m:
        _topic_word = _tw_m.group(1).strip()
print("TOPIC_WORD:", repr(_topic_word))
print("TOPIC_WORD_IN_REPLY:", _topic_word in tg.lower() if _topic_word and tg else "N/A")

# Check what the relevance gate returns
gate = getattr(eng, "_relevance_gate", None)
if gate is not None and tg:
    gated = gate(q, tg)
    print("GATE_RESULT:", repr(gated))
else:
    print("GATE: None or no topic_grounded")

# Now check what _route_self_query actually returns
result = eng._route_self_query(q)
print("ROUTE_SELF_QUERY:", repr(result))

eng.stop_background_learning()
