"""Check actual replies after the adaptive Jaccard fix."""
import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for p in (PROJ, os.path.join(PROJ, 'ravana', 'src'), os.path.join(PROJ, 'ravana_ml', 'src'), os.path.join(PROJ, 'ravana-v2', 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

suffix = 'd4_verify2'
for f in (f'weights/ravana_weights{suffix}.pkl', f'weights/ravana_usermodel{suffix}.pkl'):
    full = os.path.join(PROJ, f)
    if os.path.exists(full):
        os.remove(f)

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)

probes = [
    ("what do you dream about", "dreams"),
    ("what makes you feel calm", "calm"),
    ("do you think people can change", "people"),
]

for query, label in probes:
    reply = eng.process_turn(query)
    print(f"\n{label}: {query}")
    print(f"  -> {reply}")

# Test with open source context
eng.process_turn("i love open source software")
eng.process_turn("what do you think about open source?")
reply = eng.process_turn("do you ever feel curious about open source?")
print(f"\nopen source: do you ever feel curious about open source?")
print(f"  -> {reply}")

eng.stop_background_learning()
print("\nDONE")
