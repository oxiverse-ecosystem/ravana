import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro')
eng.process_turn('my dog biscuit sleeps on the wedging table')

# Check what's stored
facts = eng.user_model.personal_facts.facts
for k, v in sorted(facts.items()):
    print(f'  {k} -> {v.value!r} (conf={v.confidence:.2f})')
