# RAVANA 9-Issue Investigation — Master Plan
# Generated: 2026-05-23 (updated 2026-06-01)

## EXECUTIVE SUMMARY

Three parallel subagents investigated all 9 issues. Here's the consolidated findings:

| # | Issue | Status | Severity | Root Cause |
|---|-------|--------|----------|------------|
| 1 | Cross-domain transfer 0% | PARTIALLY RESOLVED | MEDIUM | NN bridge works on probe configs; full experiment shows neutral transfer |
| 2 | Relational transfer 0% | PARTIALLY RESOLVED | MEDIUM | Phase 2 NN bridge: 67-91% query success on held-out terms via MiniLM bridge |
| 3 | Concept splitting never triggers | RESOLVED | — | Threshold lowered, signal increased, hotspots persist |
| 4 | Hidden layer LR too slow | RESOLVED | — | GRU gates get direct Hebbian updates (rlm.py:1816-1880), Module.sleep_cycle() called from SWS |
| 5 | Shared currencies incomplete | LARGELY RESOLVED | LOW | CognitiveCurrency + CognitiveCurrencies created; some legacy names remain |
| 6 | No REM vs SWS | RESOLVED | — | Implemented with proper phase separation |
| 7 | Graph optimization Phase 3 | PARTIALLY RESOLVED | LOW | compute_curvature now uses sampling (max_sample=500); still dict adjacency |
| 8 | News-to-MDP pipeline | TODO | MEDIUM | reality_grounding.py exists but no pipeline code |
| 9 | Semantic drift defense | PARTIALLY RESOLVED | LOW | Inline defense exists but limited scope |

---

## DETAILED FINDINGS & FIXES

### ISSUE 1+2: Cross-Domain & Relational Transfer (SAME FIX)

Root Cause: `forward()` has NO multi-hop edge traversal.
- `forward_step()` (line 1307) and `generate()` have hop traversal
- `forward()` (used by experiments) only scores direct concept→token
- "vexol → warm → pleasant" can NEVER be inferred during testing

FIX: Add multi-hop traversal to forward(), mirror forward_step lines 1307-1338.
Also relax hop_decay: 0.4 → 0.6 in forward_step, 0.7 → 0.85 in infer_chain.

Files: rlm.py forward() (~line 366), forward_step() (~line 1312), graph.py infer_chain() (~line 1854)

**Update (2026-06-01):** Phase 2 NN bridge achieves 67-91% query success on held-out terms via MiniLM bridge + composed reasoning. However, the full experiment_cross_domain.py shows neutral transfer (0% top-1, 0% top-10). The high numbers come from optimized probe configurations, not general cross-domain transfer.

### ISSUE 3: Concept Splitting

Root Cause: Three compounding gates make splitting unreachable:
1. Entry: only nodes in contradiction_hotspots (free_energy > 5.0)
2. Signal: +0.5/wrong prediction, -0.3/correct, max ~4.0/cycle
3. Reconcile: clears ALL hotspots at end of each cycle

FIX:
- Lower threshold: 5.0 → 2.0 (graph.py apply_free_energy ~line 1128)
- Increase signal: 0.5 → 1.5 (rlm.py learn ~line 533)
- Persist hotspots: don't clear, keep if energy > 1.0 (graph.py reconcile ~line 1695)
- Scan ALL nodes: add high-drift or high-contradiction nodes (rlm.py _sleep_sws ~line 1073)

**Status: RESOLVED.** Threshold lowered, signal increased, hotspots persist across cycles.

### ISSUE 4: Hidden Layer LR (CRITICAL BUG)

Root Cause: GRU gates (W_z, W_r, W_h) are EFFECTIVELY FROZEN:
- Only use accumulate_free_energy (salience=0.3, net lr=0.001/step)
- Module.sleep_cycle() is NEVER called from RLM.sleep_cycle()
- So _weight_free_energy buffers accumulate forever, never applied to weights

FIX:
1. Add direct Hebbian update for GRU gates using GRU's _trace_x input
2. Call Module.sleep_cycle() from _sleep_sws() to flush accumulated buffers
3. Store GRU combined input during forward for proper outer product

Files: rlm.py learn() (~line 604), _sleep_sws() (~line 1060), module.py Linear.sleep_cycle()

**Status: RESOLVED.** GRU gates now get direct Hebbian updates (rlm.py:1816-1880). Module.sleep_cycle() called from SWS.

### ISSUE 5: Shared Currencies

Root Cause: graph.py apply_prediction_error() (lines 1139-1159) still uses:
- node.pressure_history (should be free_energy_history)
- node.pressure_gradient (should be free_energy_gradient)
- node.contradiction_pressure (should be contradiction_free_energy)

FIX: Rename all 3 occurrences in apply_prediction_error(). Keep backward-compat aliases.
Document distinction: confidence=accuracy, stability=maturity.

**Status: LARGELY RESOLVED.** CognitiveCurrency + CognitiveCurrencies modules created. Some legacy field names remain in graph.py.

### ISSUE 6: REM vs SWS

Already implemented! Two minor gaps:
1. Memory replay (_replay_memories_through_graph) runs AFTER both phases — should be in SWS
2. Module.sleep_cycle() not called (same as Issue 4)

FIX: Move replay into _sleep_sws(). Call Module.sleep_cycle() in SWS.

**Status: RESOLVED.** Proper phase separation implemented.

### ISSUE 7: Graph Optimization Phase 3

Current state: dict adjacency + numpy vectorized. OK for <5K nodes.
Hot paths: _nearest_concept (4×/learn), spread_activation (2×/learn), prune/form edges (sleep).

Priority order:
- Phase 3a: scipy.sparse CSR for spread_activation bulk ops
- Phase 3b: HNSW index for find_similar at 10K+ nodes
- Phase 3c: Numba JIT for inner loops (marginal benefit)

**Status: PARTIALLY RESOLVED.** compute_curvature now uses sampling (max_sample=500) for O(1) per-node cost. Still dict adjacency (not scipy.sparse).

### ISSUE 8: News-to-MDP Pipeline

Zero code exists. Proposed architecture:
1. NewsIngestion: RSS → NER → entity extraction → dissonance scoring
2. NewsToMDP: (state=activations, action=concept_updates, reward=-dissonance)
3. RLM.ingest_news() method
4. Periodic feed loop script

**Status: TODO.** reality_grounding.py exists but no structured pipeline.

### ISSUE 9: Semantic Drift Defense

Inline defense exists (rlm.py:468-483) but gaps:
1. Only checks input_concept + output_concept (ignores all others)
2. drift_magnitude uses genesis_vector but correction pulls to core_vector
3. No periodic full-graph drift scan
4. Lab's attractor_drift() never called from learn()

FIX: Expand to scan all concepts periodically. Add core_vector→genesis_vector anchor.

**Status: PARTIALLY RESOLVED.** Inline defense exists but limited to input/output concept pairs.

---

## IMPLEMENTATION PLAN (by priority)

### TIER 1 — Critical architectural fixes (do first)

1. **Fix GRU gate learning (Issue 4)** — CRITICAL BUG
   - Add direct Hebbian for GRU gates
   - Call Module.sleep_cycle() in SWS
   - Files: rlm.py, module.py
   - **DONE**

2. **Add multi-hop to forward() (Issues 1+2)**
   - Copy hop traversal from forward_step
   - Relax hop decay
   - Files: rlm.py, graph.py
   - **PARTIALLY DONE** — NN bridge works on probe configs; full experiment neutral

3. **Fix concept splitting (Issue 3)**
   - Lower threshold, increase signal, persist hotspots
   - Scan all nodes in SWS
   - Files: rlm.py, graph.py
   - **DONE**

### TIER 2 — Cleanup & consistency

4. **Complete pressure→free_energy rename (Issue 5)**
   - 3 occurrences in graph.py apply_prediction_error
   - Keep backward-compat aliases
   - **LARGELY DONE** — CognitiveCurrency created, some legacy names remain

5. **Wire drift defense comprehensively (Issue 9)**
   - Expand scope beyond input/output concepts
   - Add core_vector stability anchor
   - **PARTIALLY DONE** — inline defense exists

6. **Fix memory replay placement (Issue 6)**
   - Move _replay_memories_through_graph into _sleep_sws
   - **DONE**

### TIER 3 — Infrastructure & new features

7. **Graph optimization Phase 3 (Issue 7)**
   - scipy.sparse first, HNSW later
   - **PARTIALLY DONE** — curvature sampling implemented

8. **News-to-MDP pipeline (Issue 8)**
   - Full new module, lowest priority
   - **TODO**

---

## Update: Phase 2 Composed Reasoning PROVEN (2026-05-31)

The transfer issue (Issues 1+2) is PARTIALLY RESOLVED. Phase 2 NN bridge achieves 91% query success on 12 held-out novel terms. Key: MiniLM full-dim bridge + independent traversals + depth decay + reverse edge inheritance.

**Important caveat:** The held-out transfer experiment (different test set) shows 41% — results are test-set dependent. The 91% figure comes from experiment_reverse_inheritance.py on its specific 12-term set. The full experiment_cross_domain.py shows neutral transfer (0% top-1, 0% top-10). Both numbers are real; the difference is in test-set composition and probe configuration.
