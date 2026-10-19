import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_topic_gr')
eng.process_turn('i love open source software')
eng.process_turn('what do you think about open source?')

# Check agent stances
agent_stances = getattr(eng, '_agent_stances', {})
print("AGENT STANCES:", agent_stances)

# Check what _find_topic_connection returns
result = eng._find_topic_connection("open source")
print("FIND_TOPIC_CONNECTION:", repr(result))

# Check the topic extraction
import re
t = "do you ever feel curious about open source"
_topic_m = re.search(r"\b(?:about|of|on)\s+([a-z][a-z\s]{2,40})\s*$", t)
if _topic_m:
    print("TOPIC:", repr(_topic_m.group(1).strip()))
else:
    print("NO TOPIC MATCHED")

# Check what _topic_grounded_self_experience returns
tg = eng._topic_grounded_self_experience("do you ever feel curious about open source?", t)
print("TOPIC_GROUNDED:", repr(tg))

eng.stop_background_learning()
