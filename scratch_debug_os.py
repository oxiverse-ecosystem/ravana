"""Debug the open source test case."""
import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for p in (PROJ, os.path.join(PROJ, 'ravana', 'src'), os.path.join(PROJ, 'ravana_ml', 'src'), os.path.join(PROJ, 'ravana-v2', 'src')):
    if p not in sys.path:
        sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

suffix = 'd4_debug_os'
for f in (f'weights/ravana_weights{suffix}.pkl', f'weights/ravana_usermodel{suffix}.pkl'):
    full = os.path.join(PROJ, f)
    if os.path.exists(full):
        os.remove(f)

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=suffix)
eng._trace_enabled = True

# Set up context
eng.process_turn("i love open source software")
eng.process_turn("what do you think about open source?")

# Check agent stances
print(f"agent_stances: {getattr(eng, '_agent_stances', {})}")

# Now test the query
t = "do you ever feel curious about open source?"
print(f"\nQuery: {t}")

# Check what topic_grounded returns
topic_result = eng._topic_grounded_self_experience(t, t.lower())
print(f"topic_grounded result: {topic_result!r}")

if topic_result:
    gate_result = eng._relevance_gate(t, topic_result)
    print(f"gate result: {gate_result!r}")

# Full process_turn
print("\n--- process_turn ---")
reply = eng.process_turn(t)
print(f"Reply: {reply!r}")

eng.stop_background_learning()
print("DONE")
