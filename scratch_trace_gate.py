"""Check gate behavior with trace."""
import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for p in (PROJ, os.path.join(PROJ, 'ravana', 'src'), os.path.join(PROJ, 'ravana_ml', 'src'), os.path.join(PROJ, 'ravana-v2', 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

suffix = 'd4_trace'
for f in (f'weights/ravana_weights{suffix}.pkl', f'weights/ravana_usermodel{suffix}.pkl'):
    full = os.path.join(PROJ, f)
    if os.path.exists(full):
        os.remove(f)

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
eng._trace_enabled = True

reply = eng.process_turn("what do you dream about")
print(f"\nReply: {reply!r}")

eng.stop_background_learning()
print("DONE")
