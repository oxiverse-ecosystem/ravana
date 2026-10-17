import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro')

# Test topic grounded
t = "what do you dream about"
result = eng._topic_grounded_self_experience("what do you dream about", t)
print('TOPIC_GROUNDED result:', repr(result))

# Test what the full process_turn returns
r = eng.process_turn("what do you dream about")
print('FULL REPLY:', repr(r))
print('_last_responses:', eng._last_responses)
