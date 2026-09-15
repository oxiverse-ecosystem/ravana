import os, sys, hashlib, time
os.environ['RAVANA_OFFLINE'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
for _p in (
    _PROJ,
    os.path.join(_PROJ, "ravana", "src"),
    os.path.join(_PROJ, "ravana_ml", "src"),
    os.path.join(_PROJ, "ravana-v2", "src"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import glob
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.reproducibility import (
    SpikeLog, full_reproducibility_fingerprint, graph_state_fingerprint, engine_fingerprint
)

def _clean_suffix_files(user_suffix='repro_test_debug5'):
    weights_dir = os.path.join(_PROJ, "weights")
    patterns = [
        os.path.join(weights_dir, f"ravana_weights{user_suffix}*.pkl"),
        os.path.join(weights_dir, f"ravana_usermodel{user_suffix}*.pkl"),
        os.path.join(weights_dir, f"ravana_weights{user_suffix}*.db"),
        os.path.join(weights_dir, f"ravana_weights{user_suffix}*.sha"),
    ]
    for pat in patterns:
        for fpath in glob.glob(pat):
            try:
                os.remove(fpath)
            except OSError:
                pass

probes = ['hello', 'what is your name', 'i like open source software', 'do you think privacy is important', 'my cat is named milo', "what is my cat's name", 'the ocean is beautiful', 'do you prefer the sea or the mountains']

results = []
for run_idx in range(2):
    _clean_suffix_files('repro_test_debug5')
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='repro_test_debug5')
    for i, q in enumerate(probes):
        reply = eng.process_turn(q)
    # Compute fingerprint directly
    gf = graph_state_fingerprint(eng.graph)
    ef = engine_fingerprint(eng)
    results.append({'graph': gf, 'engine': ef, 'turn_count': eng.turn_count})

print(f'Run 0: graph={results[0]["graph"][:16]} engine={results[0]["engine"][:16]} turn_count={results[0]["turn_count"]}')
print(f'Run 1: graph={results[1]["graph"][:16]} engine={results[1]["engine"][:16]} turn_count={results[1]["turn_count"]}')

if results[0]['graph'] == results[1]['graph']:
    print('Graph fingerprints MATCH')
else:
    print('Graph fingerprints DIFFER')

# Now also compare the full structure
eng0 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='repro_test_debug5b')
eng1 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='repro_test_debug5c')
_clean_suffix_files('repro_test_debug5b')
_clean_suffix_files('repro_test_debug5c')
eng0 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='repro_test_debug5b')
eng1 = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='repro_test_debug5c')
for i, q in enumerate(probes):
    eng0.process_turn(q)
    eng1.process_turn(q)

# Compare node 0 details
nids0 = sorted(eng0.graph.nodes.keys())
nids1 = sorted(eng1.graph.nodes.keys())
if nids0 != nids1:
    print(f'Node ID lists differ! {len(nids0)} vs {len(nids1)}')
    # Find first diff
    for i, (a, b) in enumerate(zip(nids0, nids1)):
        if a != b:
            print(f'  First diff at index {i}: {a} vs {b}')
            break
else:
    print(f'Node ID lists identical ({len(nids0)} nodes)')
    # Check if the node objects have any differences
    diffs = 0
    for nid in nids0:
        n0 = eng0.graph.nodes[nid]
        n1 = eng1.graph.nodes[nid]
        # Check all attributes
        for attr in ['label', 'prediction_free_energy', 'stability', 'confidence']:
            v0 = getattr(n0, attr, None)
            v1 = getattr(n1, attr, None)
            if isinstance(v0, float) and isinstance(v1, float):
                if abs(v0 - v1) > 1e-10:
                    diffs += 1
                    if diffs <= 3:
                        print(f'  Node {nid}.{attr}: {v0} vs {v1}')
            elif v0 != v1:
                diffs += 1
                if diffs <= 3:
                    print(f'  Node {nid}.{attr}: {v0} vs {v1}')
    print(f'  Total attribute diffs: {diffs}')
