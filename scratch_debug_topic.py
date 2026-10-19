import os, sys, re
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro')

# Check what _topic_grounded_self_experience returns
t = "what do you dream about"
result = eng._topic_grounded_self_experience("what do you dream about", t)
print('TOPIC_GROUNDED result:', repr(result))

# Check the topic word extraction
_topic_word = None
_tw_m = re.search(
    r"\b(?:about|of|on)\s+([a-z][a-z\s]{2,40})\s*$", t)
if _tw_m:
    _topic_word = _tw_m.group(1).strip().split()[0] if _tw_m.group(1).strip() else None
print('Topic word:', repr(_topic_word))
print('Topic word in result:', _topic_word in result.lower() if _topic_word else 'N/A')
