"""Debug: run gate with trace enabled."""
import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for p in (PROJ, os.path.join(PROJ, 'ravana', 'src'), os.path.join(PROJ, 'ravana_ml', 'src'), os.path.join(PROJ, 'ravana-v2', 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

suffix = 'd4_debug3'
for f in (f'weights/ravana_weights{suffix}.pkl', f'weights/ravana_usermodel{suffix}.pkl'):
    full = os.path.join(PROJ, f)
    if os.path.exists(full):
        os.remove(f)

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
eng._trace_enabled = True

answer = "dream is a bit outside what i've formed a view on. i'm still figuring out what i think about things like that. what about you — what's your take?"
result = eng._relevance_gate("what do you dream about", answer)
print(f"\nResult: {result!r}")

eng.stop_background_learning()
print("DONE")
