# Acceptance Ledger

Grades each cognitive module **GREEN / YELLOW / RED** with real numbers.
A module is GREEN only when a named test file covers it AND the CI gate exercises it.
Last refreshed: 2026-09-14 against the live test collection (`pytest --co`).

## Grade key

| Grade | Meaning |
|-------|---------|
| **GREEN** | ≥1 dedicated test file, covered by CI gate, no open defects |
| **YELLOW** | Partially covered or relies on cross-module tests only |
| **RED** | No dedicated tests, or has an open known defect |

---

## Chat engine (`ravana/src/ravana/chat/`)

| Module | File | Test file | CI gate | Grade | Evidence |
|--------|------|-----------|---------|-------|----------|
| CognitiveChatEngine (main orchestrator) | `chat/engine.py` | `tests/unit/test_chat_main.py`, `tests/unit/test_chat_fixes.py`, `tests/unit/test_chat_query_fixes.py` | CI `misc-tests`, `unit-tests (1)-(5)` | **GREEN** | 187 unit test files import and exercise `engine.py` directly |
| GraphMixin / concept graph | `chat/engine_graph.py` | `tests/unit/test_graph.py` | CI `unit-tests` | **GREEN** | `test_relation_vector_determinism` verifies graph edge determinism |
| IdentityEngine | `ravana_grace/core/identity.py` | `tests/unit/test_grace_identity_emotion.py`, `tests/unit/test_identity_query.py` | CI `unit-tests` | **GREEN** | Identity state read live via `identity.get_status()` |
| Emotion (VAD) | `ravana_grace/core/emotion.py` | `tests/unit/test_grace_identity_emotion.py` | CI `unit-tests` | **GREEN** | Valence/Arousal/Dominance triple verified |
| Sleep / consolidation | `ravana_grace/core/sleep.py` | `tests/unit/test_grace_memory_sleep_state.py`, `tests/unit/test_generation.py` (`buffer_facts_graduated`) | CI `unit-tests` | **GREEN** | Sleep-phase replay emits `*_graduated` counters in generation output |
| Hippocampal buffer / episodic binder | `core/hippocampal_buffer.py`, `core/episodic_binder.py` | `tests/unit/test_episodic_binder.py`, `tests/unit/test_episode_injector.py` | CI `unit-tests` | **GREEN** | One-shot binding verified; CA3-style pattern completion |
| Working memory | `core/working_memory.py` | `tests/unit/test_grace_memory_sleep_state.py` | CI `unit-tests` | **GREEN** | Holds episodic traces before replay |
| Personal fact store | `chat/personal_fact_store.py` | `tests/unit/test_personal_fact_store.py` | CI `unit-tests` | **GREEN** | 20 tests: supersede, contradiction, confidence decay |
| Belief store | `chat/belief_store.py` | `tests/unit/test_belief_reasoner.py` | CI `unit-tests` | **GREEN** | Hypothesis update/decay/compute-weight |
| User model (stances / opinions) | `chat/user_model.py` | `tests/unit/test_user_stance_recall.py`, `tests/unit/test_agent_opinion_frame_coverage.py` | CI `unit-tests` | **GREEN** | Stance polarity recall + opinion-frame coverage tested |
| Consistency monitor | `chat/consistency_monitor.py` | `tests/unit/test_consistency_monitor.py` | CI `unit-tests` | **GREEN** | Cross-turn claim contradiction detection |
| Epistemic calibrator | `chat/calibration.py` | `tests/unit/test_epistemic_calibration.py` (28), `tests/test_epistemic_calibration_wiring.py` (15) | CI `unit-tests` + `misc-tests` | **GREEN** | 43 passed in 125.19s. Signed bias, per-band reliability curve, adaptive `theta_withhold` verified against the live engine; audit grep for authored strings returns 0 hits. Known limit (pinned by `test_observation_coverage_is_partial_because_predictions_are_partial`): `process_turn` has 60 returns before the prediction, so a 20-turn conversation yields 10 observations |
| Metacognition / ACC | `chat/metacognition.py` | `tests/unit/test_metacognition_acc.py` (20, dedicated), `tests/test_epistemic_calibration_wiring.py` | CI `unit-tests` | **GREEN** | `fok_confidence` curve + `should_assert` gate + `modality_from_support` ladder covered directly. Previously YELLOW: the wiring suite was the module's only importer and was written for the calibrator, leaving these two functions with no coverage of their own. Gate now driven by the online calibrator rather than the fixed `THETA_WITHDHOLD = 0.30` default (`engine_web_search.py:546`, `response_gen.py:3641`). Sabotage-verified: 5 behavioural breaks (gate inversion, boost removal, curve inversion, gate-ignored-by-`read`, unbounded buffer) each turn the suite red |
| Coherence gate / junk scorer | `chat/coherence_gate.py`, `chat/junk_scorer.py` | `tests/unit/test_coherence_v2.py`, `tests/unit/test_degenerate_gate.py` | CI `unit-tests` | **GREEN** | Coherence threshold + degenerate-topic gating |
| Intent router | `chat/intent_router.py` | `tests/unit/test_cognition_driven_generation.py` | CI `unit-tests` | **GREEN** | Intent → decoder path verified |
| Self-model router | `chat/self_model_router.py` | `tests/unit/test_self_opinion_query_head.py` | CI `unit-tests` | **GREEN** | Self/other boundary routing |
| Support router | `chat/support_router.py` | `tests/unit/test_empathy.py` | CI `unit-tests` | **GREEN** | Empathy selector routes distress to support path |
| Provenance | `chat/provenance.py` | `tests/unit/test_attr_consistency.py` | CI `unit-tests` | **GREEN** | Stance provenance bridge tested |
| Response generation / decoder | `chat/response_gen.py`, `chat/engine_generation.py` | `tests/unit/test_generation.py`, `tests/unit/test_cognition_driven_generation.py` | CI `unit-tests` | **GREEN** | Generation buffer, fact graduation counters |
| Surface realizer | `language/surface_realizer.py` | `tests/unit/test_human_likeness_fixes.py` | CI `unit-tests` | **GREEN** | Pronoun/fallback realizer fixes |
| Decision gate (agentic "hands") | `agent/decision_gate.py` | `tests/unit/test_decision_gate_noun_heuristic.py`, `tests/unit/test_decision_gate_url_pattern.py` | CI `unit-tests` | **GREEN** | Tool-use decision gate verified |
| Tool registry | `agent/tool_registry.py` | `tests/unit/test_decision_gate_noun_heuristic.py` | CI `unit-tests` | **GREEN** | web_search / read_website / run_script / github_cli registered |
| Web learning | `chat/web_learning.py` | `tests/unit/test_yesno_web_routing.py` | CI `unit-tests` | **GREEN** | Web-routing guard for yes-no questions |
| Reproducibility (spike log + fingerprint) | `chat/reproducibility.py` | `tests/ci/test_reproducibility.py` | CI `ci` suite | **GREEN** | 6 tests: same-seed same-fingerprint, spike log ordered, RNG state persists |
| MonitorMixin | `chat/engine_monitor.py` | `tests/unit/test_monitor_observability.py` | CI `unit-tests` | **GREEN** | `test_monitor_report_empty` (line 37) and `test_monitor_report_*` (line 48) call `eng.monitor_report()` directly. Re-graded 2026-09-30: the previous "no dedicated unit test" evidence was false. |
| Hedges / epistemic frames | `chat/hedges.py` | `tests/unit/test_hedge_frames_dehardcoded.py` | CI `unit-tests` | **GREEN** | Dedicated de-hardcoding guard pins the contract so `_HEDGE_FRAMES` cannot return. Re-graded 2026-09-30: the previous evidence cited `EPISEMIC_FRAMES`, a symbol that does not exist anywhere in `ravana/src` or `ravana-v2/src` (`grep -rn "EPISEMIC_FRAMES" --include=*.py` returns nothing). |
| Pronoun fallback map | `language/surface_realizer.py` | `tests/unit/test_human_likeness_fixes.py` | CI `unit-tests` | **YELLOW** | `SurfaceRealizer.PRONOUNS_FALLBACK` (line 81), consumed at line 858. Split into its own row 2026-09-30 — the prior row attached it to hedges with a wrong path and a non-existent symbol. |
| Pet slots | `chat/pet_slots.py` | `tests/unit/test_round_2026_08f_regression.py`, `tests/unit/test_same_turn_profile.py` | CI `misc-tests` | **GREEN** | Ordinal/person-name fix verified |
| Temporal grounding | `core/temporal_grounding.py` | `tests/unit/test_temporal_grounding.py` | CI `unit-tests` | **GREEN** | Relative-date grounding (4 years ago, last month) |
| Deductive extractor | `core/deductive_extractor.py` | `tests/unit/test_deductive_extractor.py` | CI `unit-tests` | **GREEN** | Open-class verb extraction |

---

## GRACE 20-phase governor (`ravana-v2/src/ravana_grace/`)

| Module | File | Test file | Grade | Evidence |
|--------|------|-----------|-------|----------|
| Governor | `core/governor.py` | `tests/unit/test_grace_dual_process_gw.py` | **GREEN** | Orchestration phases A–P verified |
| Dual process (System 1/2) | `core/dual_process.py` | `tests/unit/test_grace_dual_process_gw.py` | **GREEN** | System-1 fast path vs System-2 deliberation |
| Global workspace | `core/global_workspace.py` | `tests/unit/test_grace_dual_process_gw.py` | **GREEN** | Conscious broadcast verified |
| Meta-cognition | `core/meta_cognition.py` | `tests/unit/test_grace_memory_sleep_state.py` | **GREEN** | Epistemic mode transitions |
| Planning | `core/planning.py` | `tests/unit/test_grace_planning_intent.py` | **GREEN** | Plan-step sequencing |
| Adaptation | `core/adaptation.py` | `tests/unit/test_adaptation.py` | **GREEN** | Plasticity modulation |
| Active epistemology | `core/active_epistemology.py` | `tests/unit/test_active_epistemology.py` | **GREEN** | VoI-driven action selection |
| Human memory | `core/human_memory.py` | `tests/unit/test_grace_memory_sleep_state.py` | **GREEN** | Episodic + semantic split |
| Meaning / intrinsic motivation | `core/meaning.py` | `tests/unit/test_meaning.py` | **GREEN** | 32 tests (added 2026-09-27, the first test file matching *meaning* anywhere under `tests/`). Imports through the engine's real path (`from ravana_grace.core.meaning import MeaningEngine, MeaningConfig`, same as `chat/engine.py:245`). Covers `compute_meaning` breakdown + all three weight-response branches, the `max(0, ...)` clamps, effort amplification, the predictive-gain EMA window, the stake/resolve round-trip, `get_expected_meaning` and `get_status`. This suite found a real dead-guard bug in the module — see "How this ledger was verified" |
| Empathy | `core/empathy.py` | `tests/unit/test_empathy.py` | **GREEN** | VAD × cause → response frame |
| Strategy | `core/strategy.py` | `tests/unit/test_grace_planning_intent.py` | **GREEN** | Exploration modes |
| Occam layer | `core/occam_layer.py` | `tests/unit/test_occam_layer.py` | **GREEN** | Dedicated suite exists. Re-graded 2026-09-30: the previous "no standalone test" evidence was false. |
| Predictive world model | `core/predictive_world.py` | `tests/unit/test_predictive_world.py`, `tests/unit/test_predictive_coding_v2.py` | **GREEN** | Two dedicated suites exist. Re-graded 2026-09-30: the previous "no standalone test" evidence was false. |

---

## ML substrate (`ravana_ml/`)

| Module | File | Test file | Grade | Evidence |
|--------|------|-----------|-------|----------|
| ConceptGraph | `ravana_ml/graph.py` | `tests/unit/test_graph.py` | **GREEN** | Node/edge CRUD, relation-vector determinism |
| GloVe embedder | `ravana_ml/nn/embedder.py` | `tests/unit/test_embedder.py` | **GREEN** | 64-D projection verified |
| Free energy | `ravana_ml/nn/free_energy.py` | `tests/unit/test_free_energy.py` | **GREEN** | Prediction error accumulation |
| Neural decoder | `ravana_ml/nn/neural_decoder.py` | `tests/unit/test_generation.py` | **GREEN** | Decoder conditioned on graph walks |

---

## Test infrastructure

| Suite | File count | CI job | Grade |
|-------|-----------|--------|-------|
| `tests/unit/` | 198 test files | `unit-tests` (4 shards) | **GREEN** |
| `tests/` (top-level) | 23 test files | `misc-tests` | **GREEN** |
| `tests/integration/` | 13 test files | `integration-tests` | **YELLOW** |
| `tests/ci/` | 5 test files | `ci-critical` | **GREEN** |
| **Total** | **239 test files** | 4 CI jobs | — |

Collected test count at this refresh: **2631** (`pytest --co -q`, 2026-09-30).

Notes (added 2026-09-30):
- `tests/integration/` was previously absent from this table. It is gated by the named
  `integration-tests` job (`.github/workflows/ci.yml` line 153), and that job is a
  required gate via `ci-status` (line 261). It is graded YELLOW, not GREEN, solely because
  `tests/integration/test_sleep_episodic_replay.py::test_sleep_consolidates_episodic_pairs_to_graph`
  is currently failing — see backlog task 9.
- `unit-tests` shards **4**, not 5: `.github/workflows/ci.yml` line 121 sets
  `shard: [1, 2, 3, 4]`. Earlier ledger text citing `(1)-(5)` was inaccurate.

---

## How this ledger was verified

The 2026-09-14 ledger carried three YELLOW grades whose stated evidence was
**false**: it claimed "no standalone test" for `Occam layer` and
`Predictive world model` and "no dedicated unit test" for `MonitorMixin`,
while `tests/unit/test_occam_layer.py`, `tests/unit/test_predictive_world.py`,
`tests/unit/test_predictive_coding_v2.py` and
`tests/unit/test_monitor_observability.py` all existed in the tree. A grade
whose evidence does not exist is worse than a YELLOW — it hides the coverage.

Re-verified on 2026-09-27 (RAVANA_OFFLINE=1, `.venv-real`):

```
pytest tests/unit/test_occam_layer.py tests/unit/test_predictive_world.py \
       tests/unit/test_predictive_coding_v2.py -q
  -> 27 passed in 47.18s          (16 + 8 + 3 = 27, matches the ledger counts)

pytest tests/unit/test_monitor_observability.py -q
  -> 14 passed in 260.66s         (7 test functions; 2 are parametrized)
```

`Meaning / intrinsic motivation` was the one YELLOW that was **true** — no
test file matching `meaning` existed anywhere under `tests/`
(`find tests -iname '*meaning*'` returned nothing). It is now GREEN with
`tests/unit/test_meaning.py` (32 tests).

### The dead-guard bug the new meaning suite found

Writing `test_meaning.py` from the module's contract — rather than fitting
tests to current behaviour — immediately failed against unmodified source:
`test_inauthentic_high_effort_is_penalised` (2 failed, 30 passed).

`MeaningEngine.compute_meaning` gated its authenticity check on
`effort_multiplier > 1.5`. `effort` is documented as 0–1 in that method's own
docstring, and the multiplier is `1.0 + effort_kappa * effort`; at the default
`effort_kappa=0.5` the maximum reachable multiplier over the entire
documented effort range is **exactly 1.5**. A strict `>` therefore never
matched. `MeaningRecord.authentic` was always `True`, the 0.5 inauthenticity
penalty never fired, and `get_status()["authenticity_rate"]` was
hard-wired to `1.0` in production.

Measured over the effort range with default `MeaningConfig`:

| effort | multiplier | `authentic` (before) | in documented 0–1 domain? |
|--------|-----------|----------------------|---------------------------|
| 0.0 | 1.00 | True | yes |
| 0.5 | 1.25 | True | yes |
| 0.9 | 1.45 | True | yes |
| 1.0 | 1.50 | True | yes (maximum) |
| 1.5 | 1.75 | False | **no** |
| 2.0 | 2.00 | False | **no** |

The guard could only fire outside the domain it was written for. Fixed by
making the bound inclusive (`>= 1.5`); no other behaviour changed, and
`tests/unit/test_grace_memory_sleep_state.py` (35 tests) stays green.

This is the argument for a ledger that cites real nodes: the false YELLOW on
`Meaning` is precisely what left that dead guard unexamined.

---

## How to refresh

```bash
# Count tests per module (maps test files to source modules)
.venv-real/Scripts/python.exe -m pytest tests/ --co -q 2>&1 | tail -1

# Run the CI determinism gate
.venv-real/Scripts/python.exe -m pytest tests/ci/test_reproducibility.py -v

# Run the full misc + unit suites (the real coverage)
.venv-real/Scripts/python.exe -m pytest tests/unit/ -q
.venv-real/Scripts/python.exe -m pytest tests/ --ignore=tests/unit --ignore=tests/ci -q
```

When a module regresses to RED, file a FIX card with the module name and the failing test path. When a module reaches GREEN for 3 consecutive rounds, it can be marked **STABLE** (not yet tracked — future work).

---

*This ledger is a living document, not a one-time audit. Update it when modules or tests are added.*
