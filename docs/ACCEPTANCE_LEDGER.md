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
| MonitorMixin | `chat/engine_monitor.py` | (exercised via engine tests) | CI `misc-tests` (trace-monitors) | **YELLOW** | `monitor_report()` exercised by `--trace-monitors` CLI flag; no dedicated unit test |
| Hedges / epistemic frames | `chat/hedges.py` | (covered by generation tests) | CI `unit-tests` | **YELLOW** | Hardcoded `EPISEMIC_FRAMES` / `PRONOUNS_FALLBACK` in surface_realizer — dehardcode plan in progress |
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
| Meaning / intrinsic motivation | `core/meaning.py` | (covered by engine meaning tests) | **YELLOW** | No dedicated unit test; exercised via `MeaningEngine` in engine boot |
| Empathy | `core/empathy.py` | `tests/unit/test_empathy.py` | **GREEN** | VAD × cause → response frame |
| Strategy | `core/strategy.py` | `tests/unit/test_grace_planning_intent.py` | **GREEN** | Exploration modes |
| Occam layer | `core/occam_layer.py` | (covered by reasoning tests) | **YELLOW** | Hypothesis discipline; no standalone test |
| Predictive world model | `core/predictive_world.py` | (covered by reasoning tests) | **YELLOW** | False-world tester; no standalone test |

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
| `tests/unit/` | 187 test files (~2450 tests) | `unit-tests (1)-(5)` (sharded) | **GREEN** |
| `tests/` (top-level) | 15 test files | `misc-tests` | **GREEN** |
| `tests/ci/` | 5 test files | `ci` suite | **GREEN** |
| **Total** | **207 test files** | 3 CI jobs | **GREEN** |

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
