# RAVANA — Implementation Plan: Crash Hardening + Grounding / Retrieval Robustness

> Status: **plan only — no code implemented.** Every item is mapped to exact file:line
> loci, a brain-mechanism analogy, and a web-sourced fix rationale, then concrete
> steps + a test to add. The four weak areas (common-fact misses, `ravana`
> collision, nondeterministic web, weak counterfactual) are addressed; the five
> strong behaviours are explicitly preserved.

---

## 0. Diagnosis summary

| Bucket | Symptom | Root cause (file:line) |
|---|---|---|
| **A. Crash** | Intermittent native Windows *access violation* inside a longer session | OpenBLAS/MKL BLAS touched from worker threads while main thread uses BLAS (numpy issue #27989). `socket.setdefaulttimeout(4.0)` at `engine.py:9`; `ThreadPoolExecutor` in `web/learner.py` + `engine.py`. |
| **B. Common facts** | "why is the sky blue" / "what is a cat" / "what is music" → "don't know" | `_seed_kb_definitions(top_n=250)` (`engine.py:1890`) seeds *top-frequency tokens of `teen_seeds.txt`* via **live** Wikipedia/ConceptNet. No curated common-knowledge list, no offline fallback. |
| **C. Nondeterminism** | "what is gravity" correct one run, "don't know" next | Live-search timing + per-API circuit breaker (provider disabled 60s) + per-turn cache clear + background deep-read offload (`web_learning.py:302`). Answer depends on transient network. |
| **D. `ravana` collision** | KB/web can return mythological Ramayana Ravana | Partly fixed by `_seeded_relation_response` (`response_gen.py:2586`, wired `:3948`) but the **underlying leak** remains: any other write path (`_definitions["ravana"]`, `chain_walker` associations, web learner) can still ingest the myth. |
| **E. Weak counterfactual** | "if the sun disappeared tomorrow" → vague "everything that depends on X would shift" | `_PREMISE_PATTERNS` Removal/Absence row (`response_gen.py:3310-3319`) matches `disappear/gone/vanished/removed/destroyed`, returns a vague template, and `_abductive_counterfactual` short-circuits **before** `_causal_forward_simulate` (dispatch `:3177-3188`). "tomorrow" is a red herring (not a frame word, doesn't reroute). |

**Brain analogies used throughout**
- *Controlled semantic cognition* (Lambon Ralph et al., *Nat. Rev. Neurosci.* 2016): ATL hub integrates converging features, but **semantic control** (vlPFC) gates which source wins → source-authority for own-concept (D).
- *Striatal prediction-error learning of facts* (doi:10.1038/s41467-018-03992-5): declarative learning is surprise/utility-gated and **persisted**, not re-fetched every recall (C, B).
- *Hub-and-spoke & convergent feature integration* (Coutanche & Thompson-Schill 2015; Yee/Thompson-Schill semantic-memory review): knowledge is integrated from **converging offline sources** + a canonical core, not one noisy corpus (B).
- *Detached forward models / nearest-possible-world counterfactuals* (Pezzulo & Castelfranchi detachment problem; Byrne; Grush emulators) + *hierarchical predictive coding* (frontoparietal long-range semantic forecasts): counterfactuals are **minimal mutation + forward simulation of a causal model**, not a canned phrase (E).
- *One compute substrate, isolated pathways*: the brain doesn't run competing threadpools on the same substrate → bound BLAS threads, isolate I/O from compute (A).

**Web-sourced fix rationale (NumPy/OpenMP)**
- numpy #27989 "np.matmul crashes in multi-threaded settings on windows": fix = `OPENBLAS_NUM_THREADS=1`, or `ProcessPoolExecutor` instead of `ThreadPoolExecutor`, or `threadpoolctl.threadpool_limits`.
- numpy #11826: cross-backend env vars `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` must be set **before** `import numpy`.
- `faulthandler.enable()` converts a silent AV into a readable traceback.

---

## A. Crash hardening (highest priority — blocks "larger session" use)

**Plan**
1. **New shared bootstrap module** `ravana/src/ravana/_numpy_threading.py` — sets all BLAS-thread env vars *before* numpy import, then installs a default `threadpoolctl` limit:
   ```python
   import os
   for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
       os.environ.setdefault(v, "1")
   os.environ["KMP_INIT_AT_FORK"] = "FALSE"
   os.environ["OMP_DYNAMIC"] = "FALSE"
   try:
       import threadpoolctl
       threadpoolctl.threadpool_limits(limits=1)
   except Exception:
       pass
   ```
2. **`engine.py` top**: insert `import ravana._numpy_threading` *before* `import numpy as np` (currently `engine.py:13`); add `import faulthandler; faulthandler.enable()` and register `faulthandler.dump_traceback_later(30, exit=False)`.
3. **`web/learner.py`**: same env guard at module top; wrap every BLAS call (`_definition_coherence_score`, GloVe cosine, any `np.dot`/`matmul`) in a `threadpoolctl.threadpool_limits(limits=1)` context manager. **Move coherence/scoring off the fetch worker thread** — keep `ThreadPoolExecutor` workers *network-only*; do scoring on the main thread after results return (this is the exact numpy #27989 recommendation).
4. **Optional stronger fix**: switch the web fetch `ThreadPoolExecutor` → `ProcessPoolExecutor` (isolates BLAS per process). Lower-risk first; do (3) then measure, escalate to process pool only if soak still reproduces.
5. **Pin BLAS**: add a known-good `numpy`/`scipy` pin in `pyproject.toml` (avoid the 0.3.28-dev OpenBLAS regression from the issue); log `np.show_config()` once at startup for CI diagnostics.
6. **Soak test** `tests/ci/test_av_soak.py`: build `CognitiveChatEngine` and run `test_sm_grounding_gate.py` **50× in one process** (repeated engine builds + many `process_turn` calls) on Windows CI to reproduce the AV and prove the fix. Add to `.github/workflows/ci.yml` Windows matrix.

---

## B. Common-fact misses → authored, offline core-knowledge seed

**Plan**
1. **New curated file** `data/common_facts.json` — canonical textbook facts keyed by concept, covering the universal gaps: `sky` (blue / Rayleigh scattering), `blue`, `cat` (mammal / pet / carnivore), `music` (sound / rhythm / art), `water`, `sun` (star / light / heat), `gravity` (force / Earth / orbit), `tree`, `dog`, `earth`, etc. Each entry: `{ "definition": "...", "relations": [("cat","mammal","is_a",0.9), ...] }`.
2. **Bootstrap merge** at the `_seed_concepts`/`_seed_kb_definitions` call site (`engine.py:731/738`): seed the *authored* `common_facts.json` **first, offline**, then attempt web/KB enrich, fail-closed (skip misses). Now "why is the sky blue" is a deterministic offline hit.
3. **Core-knowledge priority set**: a `CORE_CONCEPTS` list treated like a "hippocampal core schema" — always grounded regardless of corpus frequency, admitted at high confidence via the existing `provenance.admit_as_fact` (`provenance.py:117`, `CONVERGENCE_FACT_THRESHOLD`).
4. **Test** `tests/unit/test_common_facts.py`: assert "why is the sky blue", "what is a cat", "what is music" return a grounded answer (not `metacognitive_uncertainty`) **offline** (mock network down).

---

## C. Nondeterministic web answers → persisted, deterministic knowledge

**Plan** (mirrors striatal PE / reconsolidation: persist stable facts, only *update* on surprise)
1. **Durable read-through cache**: on any successful web/KB lookup, persist the fact to `_definitions` **and** `storage/db.py` `CognitiveDB` (SQLite) so it survives restarts. `_definitions` becomes the always-on store; web is **only a fallback for genuinely missing** facts. → "what is gravity" is deterministic after first success and offline-repeatable.
2. **Stabilize search layer**:
   - Time-bound the *whole* web call and make it **best-effort**: if it returns nothing, **do not flip to "don't know"** when a persisted fact exists — reuse it.
   - Circuit breaker flips to a deterministic *cached/offline mode* instead of silently dropping providers.
   - Do **not** clear the search cache each turn when persistence exists; key by query hash in the durable store.
3. **Decouple deep-read**: keep snippet-only on the critical path; enqueue deep reads into the existing `core/sleep.py` consolidation so they never cause a "don't know" on the current turn.
4. **Calibration split**: keep `_human_like_uncertainty` for *genuinely unseen*; reuse cached fact (not uncertainty) when the concept was *seen before but just not re-fetched now*.
5. **Test** `tests/unit/test_web_determinism.py`: run identical query twice with a simulated network flap (mock first fail / second succeed and vice-versa); assert the final persisted answer is byte-identical and offline-repeatable.

---

## D. `ravana` collision → protected-namespace grounding (close the leak)

**Plan** (controlled semantic cognition: own-concept source outranks any external source)
1. **Provenance precedence** in `provenance.admit_as_fact` (`provenance.py:117`): add a *protected namespace* — facts about concepts in `_seeded_domain_concepts` (`oxiverse`, `intentforge`, `ravana`) may **only** be admitted from `source in {"seeded","identity"}`; any web/KB derivation for a protected concept is rejected at admission.
2. **Web-learner write-guard** in `web_learning._learn_from_text` / `_sanitize_definition_text`: skip/flag writes whose subject ∈ protected namespace **and** content matches a per-concept collision denylist (e.g. `"Ramayana"`, `"demon king"`, `"epic"` for `ravana`). This closes the "any lookup path" leak the seeded-relation fix left open.
3. **Identity anchor**: ensure `core/identity.py` `IdentityEngine` treats `ravana` as self-concept with fixed attributes and routes identity questions through the seeded path always (extends existing `_AGENT_NOUN` + `self_model_router`).
4. **Test** `tests/unit/test_ravana_collision_guard.py`: feed a mythological Ravana web snippet → assert engine never returns it; assert a protected-namespace web write is rejected by `admit_as_fact`.

---

## E. Weak counterfactual → cascade fallback to causal graph

**Plan** (detached forward model: minimal mutation + simulate the cascade)
1. **Don't short-circuit removal on a graph**: in `_abductive_counterfactual` (`response_gen.py:3245`), when the matched pattern is *Removal/Absence* **and** a causal graph exists for the subject, compute `_causal_forward_simulate(subject, max_steps=6, top_k=4)` and **append** the realized multi-hop chain to the template lead instead of returning the vague line alone:
   ```python
   if pat_is_removal and graph.has_causal_edges(start):
       chains = self._causal_forward_simulate(start, max_steps=6, top_k=4)
       realized = self._realize_chain(chains)   # "light → plants → animals → warmth"
       if realized:
           return (lead + "everything that depends on it would shift: " + realized,
                   "counterfactual_simulation")
       # else fall through to generic template
   ```
   This makes "if the sun disappeared [tomorrow]" chain `sun → light → photosynthesis → plants → animals` exactly like "if gravity stopped" already does.
2. **Generalize the premise table**: add `stopped/ended/ceased` as *cessation* cues that map to causal-graph negation (drop the subject's outgoing edges, forward-sim downstream) — reuse the graph for *any* removal, not a fixed template. Keep the rich `photosynthes` row (preserve the strong behaviour).
3. **Physical-world causal seed**: add a small physics causal schema to `core/event_schema.py seed_default_schemas` (or `causal_schema` `INNATE_PRIORS`): `sun→light`, `sun→heat→life`, `sun→photosynthesis→plants→animals`, `sun→Earth orbit(via gravity)`. Without this the simulator still has nothing to chain for "sun".
4. **Temporal cleanup (optional)**: add `tomorrow/suddenly` to `_CONDITIONAL_FRAME` (`engine.py:4249`) and the web-rewrite drop list (`engine.py:4684-4688`) so time words are stripped before grounding (harmless; "tomorrow" is not the real cause).
5. **Preserve the epistemic hedge** ("if that were true, I'd expect…") — keep the well-calibrated honesty.
6. **Test** `tests/unit/test_counterfactual_cascade.py`: assert "if the sun disappeared" / "if the sun disappeared tomorrow" yields a **multi-hop** chain containing downstream consequences (e.g. `light`/`plants`/`life`), not merely "everything that depends on it would shift".

---

## PRESERVE (explicitly do not regress)
- Counterfactual "if humans could photosynthesize" chain → keep `_PREMISE_PATTERNS` photosynthesis row.
- Embodied metaphor "what color is tuesday" → keep metaphor realization in `core/*`.
- Chit-chat / jokes / empathy → keep `emotion` + `mirror` paths.
- Honest uncertainty calibration → keep `_human_like_uncertainty` + the fail-closed `_forward_model_check` / `_final_emit_guard` stack.

---

## Milestone ordering
- **M0 — A (crash)**: blocks longer sessions; ship with soak test first.
- **M1 — B + C**: authored offline facts + persisted deterministic web → complete *and* stable answers.
- **M2 — D**: protected-namespace own-concept grounding.
- **M3 — E**: counterfactual cascade fallback.
- Each milestone adds its unit/soak test; CI runs `test_sm_grounding_gate.py` + new tests in a Windows soak loop.

## Verification / risks
- Env vars MUST be set before `import numpy` in **every** entrypoint (`engine.py`, `web/learner.py`, any `__main__`). Use the shared bootstrap module.
- Protected namespace is scoped to the 3 own-concepts only — don't suppress normal web learning.
- Counterfactual graph sim needs the **physical causal seed** or "sun" stays vague; include it in M3.
