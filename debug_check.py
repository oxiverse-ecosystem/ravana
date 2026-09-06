import os, sys
os.environ['RAVANA_OFFLINE']='1'
for p in ['.', 'ravana/src', 'ravana_ml/src', 'ravana-v2/src']:
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='_debug')
print('has _agent_self_stance_reply:', hasattr(eng, '_agent_self_stance_reply'))
print('has _agent_stance_on:', hasattr(eng, '_agent_stance_on'))
print('has _agent_stance_key:', hasattr(eng, '_agent_stance_key'))
print('has _agent_own_stances:', hasattr(eng, '_agent_own_stances'))
print('has _agent_values:', hasattr(eng, '_agent_values'))
# Check what the _agent_self_stance_reply call actually resolves to
# by tracing through the code path in engine.py around line 4273
import inspect
src = inspect.getsource(eng._route_self_query)
# find the contrastive block
lines = src.split('\n')
for i, line in enumerate(lines):
    if '_agent_self_stance_reply' in line or '_agent_stance_on' in line:
        print(f'  line {i}: {line.strip()}')
