import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro')
eng.process_turn('i love silence')
r = eng.process_turn('changed my mind — silence is actually wonderful')
print('STANCE REPLY:', repr(r))
print()

# Check what the reply should contain
r_l = r.lower() if r else ''
print('Contains silence:', 'silence' in r_l)
print('Contains wonderful:', 'wonderful' in r_l)
print('Contains good:', 'good' in r_l)
print('Contains positive:', 'positive' in r_l)
