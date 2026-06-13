"""
End-to-end test of the RAVANA chat system with the E+F+G pipeline.
Sends 'what is trust' and verifies the response is well-formed.
"""
import sys, os, subprocess, time, json

sys.stdout.reconfigure(encoding='utf-8')
project_root = 'C:/Users/Likhith/Documents/projects/ravana'
os.chdir(project_root)
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'ravana-v2'))

# Method 1: Run the chat script in a subprocess with input piped in
print("=" * 60)
print("RAVANA Chat End-to-End Test")
print("=" * 60)
print()

# Create a temporary input script that sends messages and captures output
test_script = '''
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'{project_root}/ravana-v2')
sys.path.insert(0, r'{project_root}')

# Suppress verbose output
os.environ['RAVANA_SILENT'] = '1'

from scripts.ravana_chat import CognitiveChatEngine

print("=== INITIALIZING ===", flush=True)
engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
print("=== INITIALIZED ===", flush=True)

# Test Phase C: CerebellarNgram
print("=== TEST: PHASE C ===", flush=True)
if hasattr(engine, 'cerebellar_ngram'):
    cn = engine.cerebellar_ngram
    cn.learn_chain(['trust', 'connect', 'knowledge'], successful=True, 
                   chain_hops=[('trust', 'connect'), ('connect', 'knowledge')])
    strength = cn.get_transition_strength('trust', 'knowledge')
    print(f"Cerebellar: trust->knowledge strength = {{strength:.3f}}", flush=True)

# Test Phase D: PrefrontalWorkspace
print("=== TEST: PHASE D ===", flush=True)
if hasattr(engine, 'pfc_workspace'):
    plan = engine.pfc_workspace.plan_discourse(
        'what is trust', 'trust', 
        getattr(engine, '_concept_pos', {{}}),
        [('knowledge', 0.9), ('people', 0.7), ('loyalty', 0.6)]
    )
    print(f"PFC plan: {{len(plan.intents)}} intents ({{plan.question_type}})", flush=True)
    for intent in plan.intents:
        print(f"  Intent: {{intent.type}} subject={{intent.subject}} marker={{intent.discourse_marker}}", flush=True)

# Test Phase E: SyntacticCellAssembly
print("=== TEST: PHASE E ===", flush=True)
if hasattr(engine, 'syntactic_assembly'):
    pos = getattr(engine, '_concept_pos', {{}})
    frame = engine.syntactic_assembly.bind_to_sentence(
        'trust', 'semantic', 'knowledge', pos
    )
    print(f"Frame: {{frame.subject_concept}} {{frame.verb_phrase}} {{frame.object_concept}}", flush=True)

# Test Phase F+G: SurfaceRealizer
print("=== TEST: PHASE F+G ===", flush=True)
if hasattr(engine, 'surface_realizer'):
    from ravana.language.surface_realizer import DiscourseState
    ctx = DiscourseState(sentence_index=0, discourse_type='explain')
    realized = engine.surface_realizer.realize(
        frame, ctx, dopamine_tone=0.5,
        cerebellar_ngram=getattr(engine, 'cerebellar_ngram', None)
    )
    print(f"Realized: {{realized}}", flush=True)
    
    # Test with discourse marker
    ctx2 = DiscourseState(sentence_index=1, discourse_type='elaborate')
    realized2 = engine.surface_realizer.realize(
        frame, ctx2, dopamine_tone=0.7,
        cerebellar_ngram=getattr(engine, 'cerebellar_ngram', None),
        discourse_marker='furthermore'
    )
    print(f"With marker: {{realized2}}", flush=True)

# Test full process_turn
print("=== TEST: FULL PIPELINE ===", flush=True)
try:
    response = engine.process_turn('what is trust')
    print(f"RESPONSE: {{response}}", flush=True)
    # Check for well-formedness
    assert isinstance(response, str), "Response should be a string"
    assert len(response) > 10, f"Response too short: {{response}}"
    assert response[0].isupper(), "Response should start with capital"
    assert response.endswith('.'), "Response should end with period"
    print("PASS: Well-formed response!", flush=True)
except Exception as e:
    print(f"FAIL: Error in process_turn: {{e}}", flush=True)
    import traceback
    traceback.print_exc()

print("=== TEST COMPLETE ===", flush=True)
'''.format(project_root=project_root)

# Save and run the test script
test_path = os.path.join(project_root, 'scripts', '_e2e_test.py')
with open(test_path, 'w', encoding='utf-8') as f:
    f.write(test_script)

print("Running end-to-end test (this may take 30-60 seconds for GloVe loading)...")
print()

start = time.time()
result = subprocess.run(
    [sys.executable, test_path],
    capture_output=True, text=True, timeout=120,
    cwd=project_root
)
elapsed = time.time() - start

print(f"Elapsed: {elapsed:.1f}s")
print()
print("STDOUT:")
print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)

if result.stderr:
    print()
    print("STDERR (last 1000 chars):")
    print(result.stderr[-1000:])

# Clean up
try:
    os.remove(test_path)
except:
    pass

# Determine pass/fail
stdout = result.stdout
passed = True
checks = {
    "Initialization": "=== INITIALIZED ===" in stdout,
    "Phase C Cerebellar": "Cerebellar: trust->knowledge" in stdout,
    "Phase D PFC": "PFC plan: 3 intents" in stdout,
    "Phase E Assembly": "Frame: trust" in stdout,
    "Phase F+G Realizer": "Realized:" in stdout,
    "Discourse marker": "furthermore" in stdout.lower(),
    "Full pipeline": "RESPONSE:" in stdout,
    "Well-formed": "PASS: Well-formed response!" in stdout,
}

print()
print("=" * 60)
print("TEST RESULTS")
print("=" * 60)
for name, status in checks.items():
    print(f"  {'✅' if status else '❌'} {name}")

if all(checks.values()):
    print()
    print("ALL TESTS PASSED ✅")
else:
    print(f"\n{sum(1 for v in checks.values() if v)}/{len(checks)} passed")
