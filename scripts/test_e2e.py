"""
End-to-end test of the RAVANA chat system with the E+F+G pipeline.
Sends 'what is trust' and verifies the response is well-formed.
"""
import sys, os, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')
project_root = 'C:/Users/Likhith/Documents/projects/ravana'

# Write a temp test script that handles import paths correctly
test_code = r'''
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'PROJECT_ROOT')
sys.path.insert(0, os.path.join(r'PROJECT_ROOT', 'ravana-v2'))
os.chdir(r'PROJECT_ROOT')
os.environ['RAVANA_SILENT'] = '1'

print("=== IMPORTING ===", flush=True)
# Direct import using sys.path
import scripts.ravana_chat as rc
CognitiveChatEngine = rc.CognitiveChatEngine

print("=== INITIALIZING ===", flush=True)
engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
print("=== INITIALIZED ===", flush=True)

# Phase C test
print("=== PHASE C ===", flush=True)
if hasattr(engine, 'cerebellar_ngram'):
    cn = engine.cerebellar_ngram
    cn.learn_chain(['trust', 'connect', 'knowledge'], successful=True,
                   chain_hops=[('trust', 'connect'), ('connect', 'knowledge')])
    s = cn.get_transition_strength('trust', 'knowledge')
    print("Cerebellar: trust->knowledge strength = %.3f" % s, flush=True)

# Phase D test
print("=== PHASE D ===", flush=True)
if hasattr(engine, 'pfc_workspace'):
    pos = getattr(engine, '_concept_pos', {})
    plan = engine.pfc_workspace.plan_discourse(
        'what is trust', 'trust', pos,
        [('knowledge', 0.9), ('people', 0.7), ('loyalty', 0.6)])
    print("PFC plan: %d intents (%s)" % (len(plan.intents), plan.question_type), flush=True)
    for intent in plan.intents:
        print("  Intent: %s subject=%s marker=%s" % (intent.type, intent.subject, intent.discourse_marker), flush=True)

# Phase E test
print("=== PHASE E ===", flush=True)
if hasattr(engine, 'syntactic_assembly'):
    pos = getattr(engine, '_concept_pos', {})
    frame = engine.syntactic_assembly.bind_to_sentence('trust', 'semantic', 'knowledge', pos)
    print("Frame: %s %s %s" % (frame.subject_concept, frame.verb_phrase, frame.object_concept), flush=True)

# Phase F+G test
print("=== PHASE F+G ===", flush=True)
if hasattr(engine, 'surface_realizer'):
    from ravana.language.surface_realizer import DiscourseState
    ctx = DiscourseState(sentence_index=0, discourse_type='explain')
    realized = engine.surface_realizer.realize(frame, ctx, dopamine_tone=0.5, cerebellar_ngram=getattr(engine, 'cerebellar_ngram', None))
    print("Realized: %s" % realized, flush=True)
    ctx2 = DiscourseState(sentence_index=1, discourse_type='elaborate')
    realized2 = engine.surface_realizer.realize(frame, ctx2, dopamine_tone=0.7, cerebellar_ngram=getattr(engine, 'cerebellar_ngram', None), discourse_marker='furthermore')
    print("With marker: %s" % realized2, flush=True)

# Full pipeline test
print("=== FULL PIPELINE ===", flush=True)
try:
    response = engine.process_turn('what is trust')
    print("RESPONSE: %s" % response, flush=True)
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 10, "Response too short: " + response
    assert response[0].isupper(), "Response should start with capital"
    print("PASS: Well-formed response!", flush=True)
except Exception as e:
    print("FAIL: %s" % str(e), flush=True)
    import traceback
    traceback.print_exc()

print("=== TEST COMPLETE ===", flush=True)
'''.replace('PROJECT_ROOT', project_root)

test_path = os.path.join(project_root, 'scripts', '_e2e_test.py')
with open(test_path, 'w', encoding='utf-8') as f:
    f.write(test_code)

print("Running end-to-end test (30-60s for GloVe loading)...")
start = time.time()
result = subprocess.run([sys.executable, test_path], capture_output=True, text=True, timeout=120, cwd=project_root)
elapsed = time.time() - start

print("Elapsed: %.1fs" % elapsed)
print()
print("STDOUT:")
out = result.stdout
print(out[-8000:] if len(out) > 8000 else out)

if result.stderr:
    print()
    print("STDERR (last 2000 chars):")
    print(result.stderr[-2000:])

try:
    os.remove(test_path)
except:
    pass

# Analyze results
checks = [
    ("Import", "=== IMPORTING ===" in out and "=== INITIALIZED ===" in out),
    ("Phase C Cerebellar", "Cerebellar:" in out),
    ("Phase D PFC", "PFC plan:" in out),
    ("Phase E Assembly", "Frame:" in out),
    ("Phase F+G Realizer", "Realized:" in out),
    ("Discourse marker", "furthermore" in out),
    ("Full pipeline", "RESPONSE:" in out),
    ("Well-formed", "PASS: Well-formed response!" in out),
]

print("\n" + "=" * 60)
print("TEST RESULTS")
print("=" * 60)
for name, status in checks:
    print("  %s %s" % ("✅" if status else "❌", name))

passed = all(v for _, v in checks)
print("\n%s" % ("ALL TESTS PASSED ✅" if passed else "SOME TESTS FAILED ❌"))
