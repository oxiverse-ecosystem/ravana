import os, sys
os.environ['RAVANA_OFFLINE'] = '1'
PROJ = r'C:\Users\Likhith\Documents\Projects\ravana'
for p in (PROJ, f'{PROJ}\\ravana_ml\\src', f'{PROJ}\\ravana\\src', f'{PROJ}\\ravana-v2\\src'):
    sys.path.insert(0, p)
from ravana.chat.engine import CognitiveChatEngine

# Test 1: remind me what you said
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_rfix01')
eng.process_turn('i think board games are a great way to bond with friends')
out = eng.process_turn('remind me what you said about board games')
print('TEST1 REPLY:', repr(out))
print('TEST1 STRATEGY:', eng._last_strategy)
eng.stop_background_learning()
print('---')

# Test 2: topic grounded
eng2 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_topic_gr')
eng2.process_turn('i love open source software')
eng2.process_turn('what do you think about open source?')
out2 = eng2.process_turn('do you ever feel curious about open source?')
print('TEST2 REPLY:', repr(out2))
print('TEST2 STRATEGY:', eng2._last_strategy)
eng2.stop_background_learning()
print('---')

# Test 3: fix03 infinitive
eng3 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='fix03test')
r = eng3.process_turn("my neighbors dog barks every morning at 6am and ive started to hate it")
print('TEST3 REPLY:', repr(r))
print('TEST3 STRATEGY:', eng3._last_strategy)
eng3.stop_background_learning()
print('---')

# Test 4: stance inversion
eng4 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='test_stance_inv')
eng4.process_turn('i hate silence')
s_before = eng4.user_model.opinions.stances.get('silence')
print('TEST4 BEFORE:', s_before.polarity if s_before else None)
r4 = eng4.process_turn('changed my mind — silence is actually wonderful')
s_after = eng4.user_model.opinions.stances.get('silence')
print('TEST4 AFTER:', s_after.polarity if s_after else None)
print('TEST4 REPLY:', repr(r4))
print('TEST4 STRATEGY:', eng4._last_strategy)
eng4.stop_background_learning()
print('---')

# Test 5: pet name
eng5 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='attrgate2040')
eng5.process_turn('my dog biscuit sleeps on the wedging table')
out5 = eng5.process_turn('what is my dog name')
print('TEST5 REPLY:', repr(out5))
print('TEST5 STRATEGY:', eng5._last_strategy)
eng5.stop_background_learning()
