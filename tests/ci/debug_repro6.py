import os, sys, hashlib, time
os.environ['RAVANA_OFFLINE'] = '1'
os.environ['PYTHONHASHSEED'] = '0'

_PROJ = os.path.abspath('.')
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

def _clean_suffix_files(user_suffix='repro_test_debug6'):
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
spike_logs_list = []
for run_idx in range(2):
    _clean_suffix_files('repro_test_debug6')
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix='repro_test_debug6')
    spike_log = SpikeLog()
    for i, q in enumerate(probes):
        reply = eng.process_turn(q)
        spike_log.record(
            turn=i + 1,
            kind="pe",
            data={
                "query": q,
                "total_free_energy": float(getattr(eng.graph, "total_free_energy", 0.0)),
                "node_count": len(eng.graph.nodes),
                "edge_count": len(eng.graph.edges),
            },
        )
        if eng.graph.nodes:
            top_nid = max(
                eng.graph.nodes.keys(),
                key=lambda n: float(getattr(eng.graph.nodes[n], "prediction_free_energy", 0.0)),
            )
            top_node = eng.graph.nodes[top_nid]
            spike_log.record(
                turn=i + 1,
                kind="activation",
                data={
                    "top_fe_node_label": top_node.label,
                    "top_fe_value": float(getattr(top_node, "prediction_free_energy", 0.0)),
                    "top_fe_stability": float(getattr(top_node, "stability", 0.5)),
                },
            )
    fp = full_reproducibility_fingerprint(eng, spike_log)
    results.append(fp)
    spike_logs_list.append(spike_log)
    print(f'Run {run_idx}: turn_count={eng.turn_count}, spike_len={len(spike_log)}, '
          f'graph_nodes={len(eng.graph.nodes)}, graph_edges={len(eng.graph.edges)}, '
          f'graph_fe={eng.graph.total_free_energy}')
    print(f'  spike_hash={fp["spike_log_sha256"]}')
    print(f'  graph_hash={fp["graph_sha256"]}')
    print(f'  engine_hash={fp["engine_sha256"]}')
    print(f'  combined={fp["combined_sha256"]}')

print()
if results[0]['combined_sha256'] == results[1]['combined_sha256']:
    print('PASS: fingerprints match')
else:
    print('FAIL: fingerprints differ')
    for key in ['spike_log_sha256', 'graph_sha256', 'engine_sha256', 'combined_sha256', 'turn_count']:
        v0 = results[0][key]
        v1 = results[1][key]
        match = 'OK' if v0 == v1 else 'DIFF'
        print(f'  {match} {key}:')
        print(f'    A: {v0}')
        print(f'    B: {v1}')

# Compare spike log entries one by one
print('\n--- Spike Log Comparison ---')
for i, (e0, e1) in enumerate(zip(spike_logs_list[0].entries, spike_logs_list[1].entries)):
    if e0 != e1:
        print(f'Entry {i} differs:')
        for key in e0:
            v0 = e0[key]
            v1 = e1[key]
            if v0 != v1:
                print(f'  {key}: {repr(v0)[:80]} vs {repr(v1)[:80]}')
    else:
        pass  # print(f'Entry {i} identical')
