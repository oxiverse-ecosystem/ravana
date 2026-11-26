# FINDING — WFA stall root cause: `_activate_from_input`, not `_ground_query`

Scratch-only investigation on `invest/wfa-ground-query-profile` (branch from
current `main` at `9cd5f2d`). No production file was edited, no gate or
threshold was changed, no production PR is proposed.

## Provenance / instrument validation

- Waited for the parent WFA focus run to clear: `ps -ef | grep measure_learn5_wfa`
  returned 3 PIDs until `WFA_E2B_EXIT=124` at 19:15; **no engine was booted
  before that verification**. No WFA process remained at profile time.
- Profiler: `scratch/profile_ground_query_wfa.py`, run with
  `PYTHONHASHSEED=0 RAVANA_OFFLINE=1 uv run python ...`, with the WFA harness's
  own `mfwf.install_frozen_patches()` (network and `save` blocked).
- Booted ONE `CognitiveChatEngine` against the real weight store:
  `nodes=2167`, `glove_words=400001` (the worktree needs the untracked
  `data/ravana_glove_cache.npz`; first, cache-less run was discarded).
- Three prompts came deterministically from
  `measure_learn5_wfa.build_items()`: gap/primary inclusion (short),
  gap/sealed exclusion (long provenance context), decoy/primary forced-choice
  with provenance (the shape that stalls in `wfa_e2_fresh_rerun.log`).

## cProfile results (current main, sorted by cumulative)

| turn | prompt chars | wall | `_ground_query` calls / cumtime | `_activate_from_input` calls / cumtime |
|---|---:|---:|---:|---:|
| inclusion_short | 220 | 0.623 s | 1 / 0.0243 s | 0 / — |
| exclusion_long_context | 2349 | 0.407 s | 1 / 0.0154 s | 0 / — |
| forced_choice_provenance | 2388 | **13.596 s** | 3 / 0.0247 s (**0.18 %**) | 4 / **10.1019 s (74.3 %)** |

Forced-choice turn: 19,542,301 total function calls. Top rows include
`_activate_from_input` tottime 6.704 s; `len` 6,438,578 calls; `dict.get`
3,536,170; `str.lower` 3,197,410; `startswith` 2,268,117; `endswith`
2,275,055. Secondary hotspot: `_extract_and_store_causal_relations` 1.738 s
with 135,291 `_create_causal_edge` calls (separate issue, not the stall).

## Root cause

`ravana/src/ravana/chat/engine_memory.py:3476`, `_activate_from_input`:

```python
words = re.findall(r"[a-zA-Z']{1,}", text.lower())
for w in words:
    ...
    for nid, node in list(self.graph.nodes.items()):   # O(N) for EVERY word
        label = node.label.lower()                     # lower() per node/word
        if label == w: ...
        elif len(w) >= 3 and (label.startswith(w + " ") or ...): ...
        elif len(label) >= 3 and label in w: ...
```

WFA prompts embed eight trusted-source URLs, so the regex yields **379 word
tokens** for the 2,388-character forced-choice prompt. With 2,167 graph nodes
one invocation performs `379 × 2167 ≈ 821,293` inner iterations; the turn
calls it **4 times** (`engine.py:10851` initial, `:10862` after
`_auto_expand_concepts`, `:10900` after `_extract_and_store_causal_relations`,
`:11817`/`:11821` after enrichment) ≈ 3.3 M iterations — matching the 3,197,410
`str.lower` calls in the profile. Each re-activation rescans the whole graph
from scratch even though only nodes/edges were added.

### Why the static suspects were wrong

- `_ground_query` is 3 calls / 0.025 s cumulative in the stalled turn: it is
  **not** the hotspot. Strategy B (graph-wide dot products) and Strategy D
  (all `_concept_labels` per word) sit at `engine_generation.py:2274` and
  `:2290`, *after* the unconditional `if words: ... return` block at `:2202`.
  For these prompts `words` is non-empty, so B/D are unreachable on this path.
- The log's last line is `[ground_query] query_phrase=... words=...`; the
  silent work that follows is the activation loop, so the print misattributes
  the stall to grounding.

## Boot-free A/B (production method, stub graph, no engine boot)

`scratch/ab_hot_loop.py` calls the **unmodified** production
`CognitiveChatEngine._activate_from_input` as an unbound function on a stub
holding the real 2,167 labels read read-only from `weights/ravana_weights.db`;
prompt is the exact 379-token forced-choice prompt. No `__init__`, no store
write, no network. Candidate arms are local scratch reimplementations.

| arm | median (3 reps) | activations identical to baseline | speedup |
|---|---:|---|---:|
| baseline (production) | 0.4659 s | — | 1.00× |
| hoist label snapshot out of word loop | 0.3672 s | yes | 1.27× |
| hoist + inverted token index + substring lookup | 0.0061 s | yes | **75.8×** |

Scaling of the production method on the same prompt: 250 nodes 0.0508 s,
500 0.1033 s, 1000 0.2149 s, 2000 0.4890 s — linear in node count, confirming
the O(words × nodes) scan.

## Narrow fix recommendation (no gate/threshold change)

In `_activate_from_input` only: build the lowercased label view **once per
call** and an inverted `token -> [(nid, node, label)]` index plus a
single-word-label map for `label in w`; visit only candidate nodes per word
instead of `self.graph.nodes.items()`. The scratch index arm is byte-identical
on the emitted activation list and 75.8× faster on the hot function. Optional
second step: reuse the activation result across the four per-turn calls when
no label changed (the intervening work only adds edges/nodes). Do not touch
`_ground_query`, WFA gates, thresholds, or scoring.

## Limitations

- On a fresh three-turn sequence the long exclusion prompt itself took 0.407 s
  and never entered `_activate_from_input`; the multi-second stall reproduced
  on the forced-choice/provenance turn. Both are URL-heavy prompts from the
  same harness, and the parent focus run timed out with the exclusion line as
  its last print, so I cannot present an in-process profile of that exact
  parent turn. The hot loop, its call count, its growth with graph size, and
  the behavior-preserving fix are measured directly.
- `_create_causal_edge` firing 135,291 times in one turn is a separate
  potential amplifier and should be triaged on its own.
