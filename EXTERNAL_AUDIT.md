# RAVANA External Audit — 2026-05-23
Source: LLM collaborator review of RAVANA_STATUS.md

## What's Working
- Core cognitive loop closed: predict → error → pressure → Hebbian → sleep → consolidation → graph restructuring
- Concept splitting triggers (0 → 6/cycle after count-reset + level-guard fixes)
- Memory-weights bridge reshapes ConceptGraph
- Predictive coding with no backprop (Linear.backprop() raises NotImplementedError)
- Empirical honesty: honest reporting of weaknesses alongside strengths
- Generation stabilization: fatigue + repetition penalty + exploratory drive
- Unified packaging: pip install -e ravana/, import ravana as torch
- CognitiveFramework API: perceive → predict → learn → sleep → infer

## What's Still Fragile
1. **Cross-domain transfer = 0.0** — most important open problem
   - Need: stronger inductive biases (role-filler separation, predicate structures)
   - Need: much more training on varied relational patterns
   - Relation types not disentangled from specific entities
   
2. **Shared currencies fragmented** — 6 pressure concepts, 4 confidence, 6 stability
   - Makes sleep consolidation, meaning, Governor incommensurable
   - Need: canonical versions with consistent ranges + conservation laws

3. **Benchmark eval just fixed** — all previous numbers need re-running
   - Need: minimal reproducible test harness (10-task few-shot suite)
   - Run automatically, save results, catch regressions

4. **News-to-MDP pipeline** — limits ecological validity
   - Need: structured events with verifiable outcomes
   - Even toy version (mock RSS with tiny MDPs) would close the loop

5. **Scaling limits** — graph_diagnostics throttled but will be needed
   - HNSW/sparse needed past few thousand nodes
   - RLM 14× slower than MLP (may be fundamental, but settle loop can be optimized)

6. **Paper-results mismatch** — dissonance 0.800→0.200 vs 0.323→0.322

## Near-Term Roadmap (Priority Order)
1. Get relational transfer > 0%
   - Diagnose relation vector separation via geometry dashboard
   - Try symbolic scaffolding (A REL B → B REL C → A REL C)
2. Finish shared-currency refactoring
3. Create formal, repeatable benchmark suite (run on every commit)
4. Build toy "news" MDP (5-10 synthetic events with structured predicates)
5. Fix paper-results mismatch
