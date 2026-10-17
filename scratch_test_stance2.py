import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro')
eng.process_turn('i love silence')

# Check what _route_own_stance_inversion returns
r = eng._route_own_stance_inversion('changed my mind — silence is actually wonderful')
print('INVERSION result:', repr(r))

# Check what _route_own_stance_revisit returns
r2 = eng._route_own_stance_revisit('changed my mind — silence is actually wonderful')
print('REVISIT result:', repr(r2))

# Check the full process_turn
r3 = eng.process_turn('changed my mind — silence is actually wonderful')
print('FULL REPLY:', repr(r3))
print('LAST STRATEGY:', eng._last_strategy)
