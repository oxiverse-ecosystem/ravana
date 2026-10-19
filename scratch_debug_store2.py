import os, sys, re
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_rfix01')
eng.process_turn('i think board games are a great way to bond with friends')

store = getattr(eng, '_own_replies', {})
for k, entries in store.items():
    for e in entries:
        print(f"KEY={k!r}")
        print(f"  src_tokens={e.get('src_tokens', [])}")
        print(f"  text={e.get('text', '')!r}")
        print(f"  ALL KEYS: {list(e.keys())}")
        # Print all fields
        for field, val in e.items():
            if field != 'text':
                print(f"  {field}={val!r}")

eng.stop_background_learning()
