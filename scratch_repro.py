import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

# Test 1: pet activity leak
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro')
eng.process_turn('my dog biscuit sleeps on the wedging table')
r = eng.process_turn('what is my dog name')
print('TEST1 REPLY:', repr(r))
print()

# Test 2: stance inversion
eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro2')
eng2.process_turn('i love silence')
r2 = eng2.process_turn('changed my mind — silence is actually wonderful')
print('TEST2 REPLY:', repr(r2))
print()

# Test 3: topic grounded
eng3 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro3')
r3 = eng3.process_turn('what do you dream about')
print('TEST3 REPLY:', repr(r3))
print()

# Test 4: d3 first person
eng4 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro4')
eng4.process_turn('my dog just learned a new trick and he is so proud of himself now')
r4 = eng4._route_agent_own_recall('what did i just tell you about my dog')
print('TEST4 REPLY:', repr(r4))
print()

# Test 5: retirement marker
from ravana.chat.retirement import RetirementLedger, looks_like_retraction
led = RetirementLedger()
r5 = looks_like_retraction('patna, franks he lives in delhi', led.all_markers())
print('TEST5 looks_like_retraction:', r5)
print('TEST5 all_markers:', led.all_markers())
