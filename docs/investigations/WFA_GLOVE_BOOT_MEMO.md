# WFA GloVe boot memo — default-off implementation, full-harness verification, and remaining hotspot

**Date:** 2026-11-27. **Branch:** `perf/wfa-glove-boot-memo` from `main` @ `3465377`
(PR #556 included). **Isolated worktree:** `C:/Users/Likhith/Documents/projects/ravana_wfa_glove_memo`
with a copied `weights/` store and copied GloVe artifacts; no other agent booted an
engine in this worktree. No gate, threshold, item, prompt or scoring rule was changed.

## 1. What was implemented

`ravana/src/ravana/chat/engine_graph.py` gained a process-level memo behind
**`RAVANA_GLOVE_BOOT_MEMO` (default OFF, only `"1"` enables)**. It caches the
immutable GloVe boot payload `(glove_dim, proj, words, word_index, vecs_proj)`
from the npz warm-start path. Key = `(abspath(npz), st_size, st_mtime_ns, dim)`.
On a hit the engine gets fresh mutable state (`_glove_vecs`, `_glove_vector_cache`,
`_glove_raw_vecs = None`) and shares only the immutable arrays/maps. `_glove_vector`
already does `proj_vecs[idx].astype(np.float32)` before normalising, so lookups
cannot mutate the shared matrix.

**Vector digest guard:** on first load a SHA-256 digest of a deterministic
300-row even sample of the projected matrix plus its shape is stored with the
entry; on every reuse the digest is recomputed and compared. A mutated/stale
shared matrix invalidates the entry and forces a fresh npz load rather than
silently emitting changed vectors. With the flag unset or `"0"`, the code path
does not read or write the memo at all.

This is a performance optimisation only. It does not implement or alter a
cognitive mechanism, so the existing source-monitoring / recollection research
in `docs/investigations/WFA_PRIOR_ACT_TRACE.md` already covers the relevant
brain grounding; **no new research was added and no design changed on that basis.**

## 2. Controls (all executed)

| control | result |
|---|---|
| `tests/unit/test_glove_boot_memo.py` (4 tests, one pytest process, import verified from the worktree) | **4 passed** |
| flag unset / `0`: two boots, no memo read or write, byte-identical projected arrays | PASS |
| flag `1`: second boot reuses base matrix + word index, but has independent mutable dicts | PASS |
| digest guard: in-place corruption of the shared matrix forces a fresh load | PASS |
| key identity: separate entries for different `dim` and after `mtime_ns` changes | PASS |
| production vector equivalence, 4 sequential real-engine boots (`0`, `1`, `1`, `0`) on the real 150 MB cache | 300-word + matrix digest `3e53f1d6c8b34b98f7f905c5…` on **all four**; `PROD_GLOVE_MEMO_EQUIVALENCE=PASS`; 1 memo hit on the second `1` boot |
| script: `scratch/prod_glove_memo_equivalence.py` | log: `scratch/prod_glove_memo_equivalence.log` |

The digest reproduces the pre-existing scratch memo control exactly
(`scratch/ab_glove_vector_equivalence.py`, `GLOVE_MEMO_EQUIVALENCE=PASS`), so
the production flag is vector-identical to the unmodified engine.

## 3. Full harness, GloVe-present store, both flags, 3600 s cap — TIMEOUT

`RAVANA_PRIOR_ACT_TRACE=1 RAVANA_GLOVE_BOOT_MEMO=1 RAVANA_OFFLINE=1 PYTHONHASHSEED=0`,
full 272-item bank, `timeout 3600`, 12:02:22 → 13:02:23:
`WFA_BOTH_EXIT=124`. Log `scratch/wfa_full_both_flags.log` (1,775,125 bytes,
646 `[ground_query] input=` lines, one GloVe load, no memo hit because the run
never reached a second boot, no `--- process-dissociation arms` header,
**no summary file**). This is a timeout, **not** a PASS or a new FAIL.

### Reprofiled after the memo (`scratch/profile_wfa_glove_turns.py`)

Focused 13-turn PD slice through the harness's own items/responders on the same
GloVe store, both flags on: 9.703 s total, 6.24 M calls.
`scratch/wfa_glove_turns_after_memo.stats`. Top cumulative rows:

| function | cumtime | calls |
|---|---:|---:|
| `ConceptGraph.add_edge` | 4.614 s | 6,419 |
| `_hrr_encode_hook` | 4.404 s | 6,419 |
| `DualCodeSpace.encode_fact` | 4.293 s | 6,437 |
| `bind_role` / `hrr_bind` | 3.721 s / 3.346 s | 19,311 |
| `chain_walker._auto_expand_concepts` | 3.075 s | 10 |
| `numpy.fft._raw_fft` | 3.028 s | 57,939 |

**Finding:** after the memo, boot cost is no longer binding for the full bank;
the remaining slowdown is per-edge HRR/FFT encoding during graph growth, which
the GloVe-conditioned expansion amplifies. The ~646 input lines in 3600 s versus
~2,981 lines in 856 s without GloVe is a ~14× throughput gap. A byte-identical
fix for repeated HRR encodes is a **separate** work item; it was not attempted
here.

## 4. Full harness, GloVe-free store, both flags, 3600 s cap — COMPLETED

To obtain the endpoint that the GloVe-store run could not produce, the same
harness and both flags were run once with the gitignored GloVe cache/raw vectors
temporarily hidden by a shell `trap` wrapper
(`scratch/run_full_noglove.sh`; restore verified: cache present, no `.hidden`
left). This is a measurement condition change only — no gate/item/scoring change.

`WFA_NOGLOVE_EXIT=1`, 13:08:45 → 13:23:09, `elapsed_s = 855.5`.
Artifacts: `scratch/wfa_full_both_flags_noglove.log`,
`scratch/learn5_wfa_summary.json`, `scratch/learn5_wfa_probes.jsonl`.

### Actual endpoint (internal-source arm — the arm the prior-act trace affects)

| pool | n | credible correct | discredited correct | unknown correct | anti-tautology |
|---|---:|---:|---:|---:|---|
| primary | 65 | 0.9846 | **1.0000** | 1.0000 | PASS |
| sealed | 67 | 0.9552 | **1.0000** | 1.0000 | PASS |

`internal_source.pass = true`, all 24 internal-source rules true, 132
conditions per arm, `missing_conditions = []`, `flag_enabled = true`,
`flag_restored = true`, `max_per_pool = 0` (full bank).

Comparable last complete no-GloVe run before the prior-act trace
(`scratch/full_B_reonly_summary.json`, finding §Full-bank checks) reported
primary `0.985 / 0.923 / 1.000` and sealed `0.955 / 0.627 / 1.000`:
sealed discredited **0.627 → 1.000**, primary discredited **0.923 → 1.000**
at full-bank scale with unchanged thresholds, items and grader.

### Overall verdict

`pass = false`, exit **1** — **WFA remains FAIL**, unchanged. All 27
non-PD rules and all 7 instrument controls (`G1a`–`G7`) are true, no network
attempts and no saves, `saturation_gate.status = FLAG` (`wfa_valid=100`).
The failing rules are the unchanged PD gates: `R_primary = R_sealed = 0`,
`own_decision_rate = 0`, `pd_unparseable_rate = 1.0`, withdrawal 0 — the same
articulation/PD deficit recorded in the last complete run. No threshold was
relaxed to make the internal-source arm pass.

## 5. Status

- GloVe boot memo: implemented, default-off, vector-identical, unit- and
  real-store-controlled.
- Full verification with both flags: **completed on the GloVe-free store**;
  endpoint above. On the GloVe-present store the capped run timed out and was
  reprofiled; no endpoint is claimed for that condition.
- WFA overall: **FAIL**, unchanged. The prior-act trace is verified at full-bank
  scale but does not rescue the PD/forced-choice gates.
- Next work item: remove the repeated HRR/FFT encoding on `add_edge` with a
  byte-identical A/B, then repeat the GloVe-present full run.

The scratch scripts, profile stats and the completed summary JSON are
force-added (`git add -f`) as evidence. The large run logs and `.prof` binary
stay in the worktree and are referenced above.
