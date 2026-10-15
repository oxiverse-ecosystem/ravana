import os, sys, io, contextlib
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

# Test 1: d6 opinion pronoun leak
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='debug1')
with contextlib.redirect_stdout(io.StringIO()):
    eng.process_turn('i live on a houseboat moored in a quiet lagoon.')
    eng.process_turn('i have a pet crow named mica.')
    r = eng.process_turn('how do you feel about us moving away?')
print('TEST 1 REPLY:', repr(r))
print('TEST 1 STRATEGY:', eng._last_strategy)
print()

# Test 2: own stance persistence
eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='debug2')
with contextlib.redirect_stdout(io.StringIO()):
    r2 = eng2.process_turn('what do you think about open source')
print('TEST 2 REPLY:', repr(r2))
print('TEST 2 STRATEGY:', eng2._last_strategy)
print('TEST 2 AGENT STANCES:', eng2._agent_stances)
print()

# Test 3: freeform contradiction
eng3 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='debug3')
with contextlib.redirect_stdout(io.StringIO()):
    eng3.process_turn('i really love street art, especially big murals on warehouse walls')
    held = None
    s = eng3.user_model.opinions.stances.get('street art')
    if s: held = s.polarity
    print('TEST 3 HELD:', held)
    eng3.process_turn('i saw a mural downtown that was just tagged over a local business\'s sign — changed my mind, not all street art is good')
    after = None
    s = eng3.user_model.opinions.stances.get('street art')
    if s: after = s.polarity
    print('TEST 3 AFTER:', after)
print()

# Test 4: defect A
eng4 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='debug4')
with contextlib.redirect_stdout(io.StringIO()):
    eng4.process_turn("i love knitting, it's the most calming thing in the world.")
    eng4.process_turn("actually, you know what, i've come to hate knitting — it just makes my hands ache.")
    _key = eng4.user_model.opinions.resolve_topic("knitting")
    _s = eng4.user_model.opinions.stances.get(_key)
    print('TEST 4 STANCE:', _s.polarity if _s else None)
    ans = eng4._user_stance_reply("do you still think i love knitting?")
    print('TEST 4 REPLY:', repr(ans))
print()
