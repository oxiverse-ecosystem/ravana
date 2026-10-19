import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

results = []

# Test 1: pet activity leak
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro')
eng.process_turn('my dog biscuit sleeps on the wedging table')
r = eng.process_turn('what is my dog name')
ok = 'biscuit' in r.lower() and 'sleeps' not in r.lower() and 'wedging' not in r.lower()
results.append(('pet_activity_name', ok, r))

# Test 2: stance inversion
eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro2')
eng2.process_turn('i love silence')
r2 = eng2.process_turn('changed my mind — silence is actually wonderful')
ok2 = 'silence' in r2.lower()
results.append(('stance_inversion', ok2, r2))

# Test 3: topic grounded
eng3 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro3')
r3 = eng3.process_turn('what do you dream about')
ok3 = 'dream' in r3.lower() or "haven't" in r3.lower() or 'outside' in r3.lower()
results.append(('topic_grounded_dream', ok3, r3))

# Test 4: d3 first person
eng4 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_repro4')
eng4.process_turn('my dog just learned a new trick and he is so proud of himself now')
r4 = eng4._route_agent_own_recall('what did i just tell you about my dog')
ok4 = r4 is None
results.append(('d3_first_person', ok4, r4))

# Test 5: retirement marker
from ravana.chat.retirement import RetirementLedger, looks_like_retraction
led = RetirementLedger()
r5 = looks_like_retraction('patna, franks he lives in delhi', led.all_markers())
ok5 = not r5
results.append(('retirement_marker', ok5, r5))

# Print results
for name, ok, val in results:
    status = 'PASS' if ok else 'FAIL'
    print(f'{status}: {name} -> {val!r}')

# Summary
passed = sum(1 for _, ok, _ in results if ok)
print(f'\n{passed}/{len(results)} passed')
