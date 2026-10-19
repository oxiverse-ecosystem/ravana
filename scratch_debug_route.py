import os, sys, re
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_topic_gr')
eng.process_turn('i love open source software')
eng.process_turn('what do you think about open source?')

# Check what the PFC classifies this as
q = "do you ever feel curious about open source?"
print("PFC route:", eng._route_intent(q))
print()

# Check _is_self_experience_query
print("is_self_experience:", eng._is_self_experience_query(q))

# Check what _route_self_experience returns
t = q.lower().strip()
result = eng._route_self_experience(q, t)
print("route_self_experience:", repr(result))
print()

# Check the strategy
print("last_strategy:", eng._last_strategy)
print("last_responses:", eng._last_responses[-1] if eng._last_responses else None)

eng.stop_background_learning()
