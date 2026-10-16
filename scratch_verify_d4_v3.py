"""Check replies for remaining failing tests."""
import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for p in (PROJ, os.path.join(PROJ, 'ravana', 'src'), os.path.join(PROJ, 'ravana_ml', 'src'), os.path.join(PROJ, 'ravana-v2', 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

suffix = 'd4_verify3'
for f in (f'weights/ravana_weights{suffix}.pkl', f'weights/ravana_usermodel{suffix}.pkl'):
    full = os.path.join(PROJ, f)
    if os.path.exists(full):
        os.remove(f)

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
eng._trace_enabled = True

# Test 2: "what makes you feel calm"
print("=== Test 2: what makes you feel calm ===")
reply = eng.process_turn("what makes you feel calm")
print(f"Reply: {reply!r}")
print(f"Has 'calm': {'calm' in reply.lower()}")
print(f"Has 'haven't': {'havent' in reply.lower()}")
print(f"Has 'outside': {'outside' in reply.lower()}")
print()

# Test 4: "do you ever feel curious about open source?"
print("=== Test 4: open source context ===")
eng.process_turn("i love open source software")
eng.process_turn("what do you think about open source?")
reply = eng.process_turn("do you ever feel curious about open source?")
print(f"Reply: {reply!r}")
print(f"Has 'open source': {'open source' in reply.lower()}")
print(f"Has 'havent': {'havent' in reply.lower()}")

eng.stop_background_learning()
print("\nDONE")
