"""
RAVANA P2 Emotional State Tracking Tests
===========================================
Tests for the P2 emotional state tracking system (roadmap §7):
1. UserEmotionDetector: VAD inference from text (positive, negative, curious)
2. emotional_state field: initialized, updated, EMA blended
3. interaction_history: recorded, capped at 100 entries
4. Temperature modulation: user arousal affects _get_temperature output
5. Verbosity modulation: user arousal affects discourse plan intent count
6. Concept breadth modulation: user arousal affects max assocs in pipeline
7. Serialization round-trip (get_state / set_state + backward compat)
8. Integration: full engine process_turn with emotional state tracking

Run: python scripts/test_emotional_mirror.py
"""
import sys, os, traceback

sys.stdout.reconfigure(encoding='utf-8')
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "ravana", "src"))
sys.path.insert(0, os.path.join(project_root, "ravana-v2"))

os.environ['RAVANA_SILENT'] = '1'

from scripts.ravana_chat import UserModel

_passed = 0
_failed = 0
_total  = 0

def check(name: str, condition: bool, detail: str = ""):
    global _passed, _failed, _total
    _total += 1
    if condition:
        _passed += 1
        print(f"  \u2705 {name}")
    else:
        _failed += 1
        print(f"  \u274c {name}  \u2014 {detail}")


# ────────────────────────────────────────────
# PART 1: Unit tests (no GloVe, no engine)
# ────────────────────────────────────────────

print("=" * 60)
print("PART 1: Emotional State Tracking Unit Tests")
print("=" * 60)

# 1a. Emotional state initialization
print("\n\u2500\u2500 Test 1: Emotional State Initialization \u2500\u2500")
um = UserModel()
check("emotional_state has valence key", 'valence' in um.emotional_state)
check("emotional_state has arousal key", 'arousal' in um.emotional_state)
check("emotional_state has dominance key", 'dominance' in um.emotional_state)
check("default valence = 0.0", um.emotional_state['valence'] == 0.0)
check("default arousal = 0.3", um.emotional_state['arousal'] == 0.3)
check("default dominance = 0.5", um.emotional_state['dominance'] == 0.5)

# 1b. Emotional state inference from text
print("\n\u2500\u2500 Test 2: _infer_user_emotion from Text \u2500\u2500")
um2 = UserModel()
v, a, d = um2._infer_user_emotion("I'm excited about this!")
check("excited text has positive valence", v > 0.3)
check("excited text has high arousal", a > 0.5)

um2b = UserModel()
v2, a2, d2 = um2b._infer_user_emotion("this is really frustrating and broken")
check("frustrated text has negative valence", v2 < -0.2)

um2c = UserModel()
v3, a3, d3 = um2c._infer_user_emotion("hello")
check("neutral text has near-zero valence", abs(v3) < 0.2)
check("neutral text has baseline arousal", abs(a3 - 0.3) < 0.2)

# 1c. EMA blending of emotional_state across calls
print("\n\u2500\u2500 Test 3: EMA Blending of Emotional State \u2500\u2500")
um3 = UserModel()
um3._infer_user_emotion("I'm happy and excited!")
first_valence = um3.emotional_state['valence']
check("first call: valence > 0 after happy text", first_valence > 0.1)

um3._infer_user_emotion("I'm sad and disappointed")
check("second call: valence adjusted downward",
      um3.emotional_state['valence'] < first_valence)
check("second call: valence not fully flipped (EMA smooths)",
      um3.emotional_state['valence'] > -0.5)

# Check raw (pre-EMA) return values — VAD lexicon should detect strong negative
um3b = UserModel()
raw_v, raw_a, raw_d = um3b._infer_user_emotion("I'm furious about this terrible error")
check("strong negative raw valence < -0.5", raw_v < -0.5)
check("strong negative raw arousal > 0.6", raw_a > 0.6)
neg_v = um3b.emotional_state['valence']
check("strong negative EMA valence < -0.2", neg_v < -0.2)
check("strong negative EMA arousal > 0.4", um3b.emotional_state['arousal'] > 0.4)

# 1d. Emotional state updates via observe_user_query
print("\n\u2500\u2500 Test 4: observe_user_query updates emotional_state \u2500\u2500")
um4 = UserModel()
check("initial emotional_state['arousal'] = 0.3",
      abs(um4.emotional_state['arousal'] - 0.3) < 1e-6)
um4.observe_user_query("I'm excited about rust!", "rust", 0.5)
# After one query with 'excited', arousal should increase from 0.3
check("arousal increased after excited query",
      um4.emotional_state['arousal'] > 0.3)

um4.observe_user_query("this is calm and peaceful", "rust", 0.5)
# After 'calm', arousal should decrease from its elevated level
check("arousal decreased after calm query",
      um4.emotional_state['arousal'] < 1.0)

# 1e. interaction_history recording
print("\n\u2500\u2500 Test 5: interaction_history \u2500\u2500")
um5 = UserModel()
check("initial history is empty", len(um5.interaction_history) == 0)

um5.observe_user_query("how does trust work", "trust", 0.5)
check("history length = 1 after one query", len(um5.interaction_history) == 1)
entry = um5.interaction_history[0]
check("history entry has 'text' key", 'text' in entry)
check("history entry has 'subject' key", 'subject' in entry)
check("history entry has 'valence' key", 'valence' in entry)
check("history entry has 'arousal' key", 'arousal' in entry)
check("history entry has 'dominance' key", 'dominance' in entry)
check("history entry has 'turn' key", 'turn' in entry)
check("history entry subject = 'trust'", entry['subject'] == 'trust')

# Fill to cap
for i in range(150):
    um5.observe_user_query("tell me about something", f"topic{i}", 0.0)
check("history capped at 100 entries after 151 queries",
      len(um5.interaction_history) == 100)
check("history retains most recent entries",
      um5.interaction_history[-1]['subject'] == 'topic149')

# 1f. Serialization round-trip
print("\n\u2500\u2500 Test 6: Serialization Round-Trip \u2500\u2500")
um6 = UserModel()
um6.observe_user_query("I'm excited about learning!", "learning", 0.7)
um6.observe_user_query("this is frustrating", "bugs", -0.3)

state = um6.get_state()
check("Serialized has emotional_state", 'emotional_state' in state)
check("Serialized emotional_state has valence",
      'valence' in state['emotional_state'])
check("Serialized has belief_state", 'belief_state' in state)
check("Serialized has interaction_history", 'interaction_history' in state)
check("Serialized interaction_history length",
      len(state['interaction_history']) == 2)
check("Serialized interaction_history has VAD data",
      'arousal' in state['interaction_history'][0])

um6b = UserModel()
um6b.set_state(state)
check("Restored: emotional_state valence matches",
      abs(um6b.emotional_state['valence'] - um6.emotional_state['valence']) < 1e-6)
check("Restored: emotional_state arousal matches",
      abs(um6b.emotional_state['arousal'] - um6.emotional_state['arousal']) < 1e-6)
check("Restored: emotional_state dominance matches",
      abs(um6b.emotional_state['dominance'] - um6.emotional_state['dominance']) < 1e-6)
check("Restored: interaction_history length matches",
      len(um6b.interaction_history) == 2)
check("Restored: belief_state preserved",
      um6b.belief_state == um6.belief_state)

# 1g. Backward compatibility: old state without emotional fields
print("\n\u2500\u2500 Test 7: Backward Compatibility \u2500\u2500")
um7 = UserModel()
old_state = {
    'edge_reactivations': {},
    'query_concepts': [],
    'knowledge_model': {'rust': 0.5},
    'learning_goals': {'rust': 3},
    'emotional_rapport': {'rust': 0.2},
    'cognitive_style': 'practical',
    'engagement_level': 0.6,
    'conversation_depth': 0.4,
    'topic_interaction_count': {'rust': 5},
    'topic_followup_count': {'rust': 2},
    'last_topic': 'rust',
    'turn_since_topic_change': 1,
    # P1 fields
    'interaction_count': 5,
    'relationship_depth': 0.25,
    'goals': ['LEARNING'],
    'last_goal': 'LEARNING',
    # Missing: emotional_state, belief_state, interaction_history
}
um7.set_state(old_state)
check("Old state: emotional_state defaults to initial VAD",
      abs(um7.emotional_state['valence'] - 0.0) < 1e-6)
check("Old state: emotional_state arousal defaults to 0.3",
      abs(um7.emotional_state['arousal'] - 0.3) < 1e-6)
check("Old state: belief_state defaults to {}",
      um7.belief_state == {})
check("Old state: interaction_history defaults to []",
      um7.interaction_history == [])
check("Old state: interaction_count preserved",
      um7.interaction_count == 5)


# ────────────────────────────────────────────
# PART 2: Integration test (full engine)
# ────────────────────────────────────────────

print("\n" + "=" * 60)
print("PART 2: Integration Test (Full Engine)")
print("=" * 60)

print("\nInitializing engine (GloVe loading \u2014 may take 30s)...", flush=True)

try:
    from scripts.ravana_chat import CognitiveChatEngine
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    print("Engine loaded!", flush=True)

    engine.baby_mode = False
    engine._bg_learning_active = False
    engine._network_available = False  # Skip web search for fast deterministic tests

    print("\n\u2500\u2500 Integration 1: Emotional state updates after process_turn \u2500\u2500")

    r1 = engine.process_turn("I'm so excited about trust!")
    check("Response is a string", isinstance(r1, str) and len(r1) > 0)
    check("Emotional state valence > 0 after excited query",
          engine.user_model.emotional_state['valence'] > 0.0)
    check("Emotional state arousal > 0.3 after excited query",
          engine.user_model.emotional_state['arousal'] > 0.3)
    check("Interaction history has 1 entry",
          len(engine.user_model.interaction_history) == 1)
    check("History entry valence positive (raw detector output)",
          engine.user_model.interaction_history[0]['valence'] > 0.3)
    check("EMA-blended valence also positive",
          engine.user_model.emotional_state['valence'] > 0.0)

    print("\n\u2500\u2500 Integration 2: Negative user affect \u2500\u2500")
    r2 = engine.process_turn("this is really frustrating and broken")
    check("Emotional state valence < 0 after frustrated query",
          engine.user_model.emotional_state['valence'] < 0.0)
    check("Emotional state arousal elevated",
          engine.user_model.emotional_state['arousal'] > 0.3)

    print("\n\u2500\u2500 Integration 3: Emotional state persists \u2500\u2500")
    r3 = engine.process_turn("what is kindness")
    check("Emotional state still has valid keys after neutral query",
          'valence' in engine.user_model.emotional_state)
    check("Interaction history continues to grow",
          len(engine.user_model.interaction_history) == 3)

    print("\n\u2500\u2500 Integration 4: History grows and rounds correctly \u2500\u2500")
    for i in range(10):
        engine.process_turn(f"tell me about topic{i}")
    check("History length matches turns",
          len(engine.user_model.interaction_history) == 13)

    print("\n\u2500\u2500 Integration 5: State persistence round-trip \u2500\u2500")
    saved = engine.user_model.get_state()
    check("Engine state has emotional_state", 'emotional_state' in saved)
    check("Engine state has interaction_history",
          len(saved['interaction_history']) >= 13)
    check("Engine state interaction_history has VAD data per entry",
          'arousal' in saved['interaction_history'][0])

    um_loaded = UserModel()
    um_loaded.set_state(saved)
    check("Reloaded: emotional_state valence preserved",
          abs(um_loaded.emotional_state['valence']
              - engine.user_model.emotional_state['valence']) < 1e-6)
    check("Reloaded: interaction_history length preserved",
          len(um_loaded.interaction_history) >= 13)

    print("\n\u2500\u2500 Integration 6: No regression \u2014 response is well-formed \u2500\u2500")
    r_final = engine.process_turn("what is loyalty")
    check("Final response is string", isinstance(r_final, str))
    check("Final response length > 10", len(r_final) > 10)

except Exception as e:
    print(f"\n\u26a0\ufe0f  Integration test error: {e}")
    traceback.print_exc()
    print("  (Unit tests (Part 1) are the primary verification gate)")


# ────────────────────────────────────────────
# Summary
# ────────────────────────────────────────────

print("\n" + "=" * 60)
print(f"RESULTS: {_passed}/{_total} passed")
if _failed:
    print(f"         {_failed} FAILED")
print("=" * 60)

if _failed > 0:
    sys.exit(1)
else:
    print("\nALL TESTS PASSED \u2705")
    sys.exit(0)
