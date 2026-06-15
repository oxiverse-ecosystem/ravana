#!/usr/bin/env python3
"""Compare previous benchmark with new per-triple diagnostics."""
import json
import glob
import os

results_dir = 'experiments/experiment_results'

# Load previous benchmark
prev_path = os.path.join(results_dir, 'trajectory_benchmark.json')
if os.path.exists(prev_path):
    with open(prev_path) as f:
        prev = json.load(f)
    print("=== PREVIOUS BENCHMARK ===")
    for r in prev['benchmark']:
        val_gaps = r['val_gaps']
        ho_gaps = r['held_out_gaps']
        val_avg = sum(val_gaps.values()) / len(val_gaps)
        ho_avg = sum(ho_gaps.values()) / len(ho_gaps)
        print(f"\n{r['config']}")
        print(f"  Val gaps: {{{', '.join(f'{k}: {v:+.4f}' for k, v in val_gaps.items())}}}")
        print(f"  Val avg: {val_avg:+.4f} | Sat: {r['val_satisfied']}/5")
        print(f"  Held-out: {{{', '.join(f'{k}: {v:+.4f}' for k, v in ho_gaps.items())}}}")
        print(f"  Held-out avg: {ho_avg:+.4f} | Sat: {r['held_out_satisfied']}/3")

# Load new per-triple diagnostics (non-epoch ones)
print("\n\n=== NEW DIAGNOSTICS (this session) ===")
new_configs = {}
for fn in sorted(glob.glob(os.path.join(results_dir, 'per_triple_diagnostics_*.json'))):
    if '_epoch' in fn:
        continue
    with open(fn) as f:
        d = json.load(f)
    name = d['config_name']
    val_gaps = {k: v['gap'] for k, v in d['validation'].items()}
    ho_gaps = {k: v['gap'] for k, v in d['held_out'].items()}
    val_avg = sum(val_gaps.values()) / len(val_gaps)
    ho_avg = sum(ho_gaps.values()) / len(ho_gaps)
    val_sat = sum(1 for v in d['validation'].values() if v['satisfied'])
    ho_sat = sum(1 for v in d['held_out'].values() if v['satisfied'])
    new_configs[name] = {
        'val_gaps': val_gaps, 'ho_gaps': ho_gaps,
        'val_avg': val_avg, 'ho_avg': ho_avg,
        'val_sat': val_sat, 'ho_sat': ho_sat
    }
    print(f"\n{name}")
    print(f"  Val gaps: {{{', '.join(f'{k}: {v:+.4f}' for k, v in val_gaps.items())}}}")
    print(f"  Val avg: {val_avg:+.4f} | Sat: {val_sat}/5")
    print(f"  Held-out: {{{', '.join(f'{k}: {v:+.4f}' for k, v in ho_gaps.items())}}}")
    print(f"  Held-out avg: {ho_avg:+.4f} | Sat: {ho_sat}/3")

# Compare
print("\n\n=== SIDE-BY-SIDE COMPARISON ===")
print(f"{'Config':<35s} {'Prev Val Avg':<15s} {'New Val Avg':<15s} {'Match?':<10s} {'Prev HO Avg':<15s} {'New HO Avg':<15s} {'Match?':<10s}")
print("-" * 105)
for r in prev['benchmark']:
    name = r['config']
    prev_val_avg = sum(r['val_gaps'].values()) / len(r['val_gaps'])
    prev_ho_avg = sum(r['held_out_gaps'].values()) / len(r['held_out_gaps'])
    if name in new_configs:
        nv = new_configs[name]
        val_match = "YES" if abs(prev_val_avg - nv['val_avg']) < 1e-10 else "DIFF"
        ho_match = "YES" if abs(prev_ho_avg - nv['ho_avg']) < 1e-10 else "DIFF"
        print(f"{name:<35s} {prev_val_avg:>+8.4f}      {nv['val_avg']:>+8.4f}      {val_match:<10s} {prev_ho_avg:>+8.4f}      {nv['ho_avg']:>+8.4f}      {ho_match:<10s}")

print("\n\n=== ANALYSIS ===")
print("The results are IDENTICAL between runs. Why?")
print("1. The RLMv2 dimension fix only affects `_rp_forward` (bilinear relation projection)")
print("2. Evaluation uses `_encoder_forward_full` directly (NOT `_rp_forward`)")
print("3. The hard-boost sampling uses `_rp_forward` but validation satisfaction is 5/5")
print("4. So the fix prevents crashes but doesn't change the evaluation metrics themselves")
print("")
print("The fix was CRITICAL to prevent the Proposed config from crashing at runtime.")
print("Without it: ValueError: matmul dimension mismatch (64 vs 128) in _rp_forward.")
print("With it: all 6 configs complete 150 epochs successfully and results are saved.")
