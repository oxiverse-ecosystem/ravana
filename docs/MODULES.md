# Modules

A map of the three source packages. Paths are relative to each package's `src/`
directory. Only the most load-bearing modules are listed; the full tree is
discoverable under `ravana/src`, `ravana_ml/src`, `ravana-v2/src`.

## `ravana_ml` — ML substrate (`ravana_ml/src/ravana_ml/`)

| Module | Responsibility |
|--------|----------------|
| `tensor.py` | CPU-native autodiff tensor (the `torch`-like drop-in). |
| `nn/` | `module.py`, `functional.py`, `neural_decoder.py` (the chat decoder), `rlm.py` / `rlm_v2*.py` (graph relation learner + verb-offset, entity adapters, sleep). |
| `graph.py` | `ConceptGraph`: typed nodes/edges, `find_similar`, `spread_activation`, pruning. |
| `embedder.py` | GloVe→64-D projection, word embeddings. |
| `ontology.py` | ConceptNet + attribute/Binder encoder, Lancaster probe. |
| `free_energy.py`, `propagation.py`, `plasticity.py` | Free-energy learning core. |
| `currencies.py` / `currency.py` | Value/neuromodulator bookkeeping. |
| `episode_injector.py` | Episodic memory injection. |

## `ravana` — chat engine (`ravana/src/ravana/`)

| Subpackage | Key modules |
|------------|-------------|
| `chat/` | `engine.py` (`CognitiveChatEngine` — orchestrator that composes 8 mixins), `engine_generation.py` (`GenerationMixin`), `engine_graph.py` (`GraphMixin`), `engine_memory.py` (`MemoryMixin`), `engine_monitor.py` (`MonitorMixin`), `engine_persistence.py` (`PersistenceMixin`), `engine_reasoning.py` (`ReasoningMixin`), `engine_self_query.py` (`SelfQueryMixin`), `engine_web_search.py` (`WebSearchMixin`). Plus `brain_regions.py` (brain-repair prepasses), `consistency_monitor.py` (cross-turn self-consistency check), `harm_intent_gate.py` (pre-generation safety classifier), `support_router.py` (advice/wellbeing routing), `personal_fact_store.py` (learned user-profile facts), `user_model.py` (opinions/corrections/stance), `interface.py` (interactive chat entrypoint), `coherence_gate.py`, `junk_scorer.py`, `monitor_gate.py`, `intent_router.py`, `chain_walker.py`, `response_gen.py`, `self_model_router.py`, `salad_classifier.py`, `web_learning.py`, `constants.py`. |
| `core/` | `emotion.py` (VAD), `identity.py`, `meaning.py`, `sleep.py`, `predictive_coding.py`, `hrr_reasoner.py`, `analogy_engine.py`, `causal_schema.py`, `working_memory.py`, `dual_process.py`, `global_workspace.py`, `meta_cognition.py`, `mirror.py` (emotional mirror / theory-of-mind). Reasoning/memory additions: `fact_reasoning.py` (chain-walking + conditionals over episodic memory), `frequency_model.py` (learned word-frequency model, replaces hand word lists), `in_prompt_reasoner.py`, `multi_hop_reasoner.py`, `temporal_cloze.py` (TimeDial-style blank filling), `temporal_grounding.py`, `temporal_reasoner.py` (event ordering / time-logic), `situation_model.py`, `event_schema.py`, `proposition_parser.py`, `question_decomposition.py`, `sub_answer_synthesizer.py`, `hippocampal_buffer.py` (episodic buffer + `FactTriple`), `relation_memory.py`, `attractor_memory.py`, `dual_code_space.py`, `active_inference.py`, `abstraction_engine.py`, `implicature_detector.py`, `quantity_modifier.py`, `coherence.py`, `vsa.py`, `system1.py`, `system2.py`, `belief_store.py`. |
| `language/` | `surface_realizer.py`, `syntactic_cell_assembly.py`, `basal_ganglia.py` (gate), `cerebellar_ngram.py`, `prefrontal_workspace.py`, `verb_lexicon.py`, `register.py`, `schemas.py`. |
| `cognitive/` | Re-exports the GRACE cognitive core from `ravana_grace.core`. |
| `ontology/` | `derived.py`, `graph_typing.py`, `conceptnet.py`, `linggen.py` (LingGen P6 sensorimotor conditioning + `LingGenConditioner`). |
| `web/` | `web_to_graph.py`, `learner.py`, `openie.py` (web → typed edges). |
| `decoder/` | `engine.py`, `predictive_coding_generator.py` (settle path). |
| `graph/` | `engine.py` (`GraphEngine` wrapper). |
| `bootstrap/` | `manager.py`, `pmi_seeder.py` (cold-start seeding). |
| `learn/` | `consolidation.py`, `curiosity.py`. |
| `world/` | world-model state. |

## `ravana_grace` — GRACE governor (`ravana-v2/src/ravana_grace/`)

| Subpackage | Key modules |
|------------|-------------|
| `core/` | `governor.py` (orchestrator), `identity.py`, `emotion.py`, `meaning.py`, `sleep.py`, `global_workspace.py`, `belief_reasoner.py`, `active_epistemology.py`, `meta_cognition.py`, `strategy.py`, `strategy_learning.py`, `planning.py`, `environment.py`, `predictive_world.py`, `reality_friction.py`, `social_epistemology.py`, `surgical_probes.py`, `vector_index.py`, `dual_process.py`, `empathy.py`, `memory.py`, `human_memory.py`, `memory_reconstructor.py`, `resolution.py`, `occam_layer.py`, `hypothesis_generation.py`, `conversational_repair.py`, `dialogue_context.py`, `state.py`, `runtime.py`. |
| `dialogue/` | `dialogue_engine.py`. |
| `agent/` | `ravana_agent_loop.py`, `mode_orchestrator.py`, `version_manager.py`, `test_runner.py`. |
| `interface_agent/` | `scripts/` — LLM interpreter, memory learner, reality grounding, telegram reporter, prompt composer (optional LLM bridges). |
| `probes/` | `constraint_stress.py`, `exploration_pressure.py`, `learning_signal.py`. |
| `research/` | `core_k0/` — agent-loop research harness (k0…k3 experiments). |
| `training/` | training loops for the governor. |

## Scripts (`scripts/`)

See [BENCHMARKS.md](BENCHMARKS.md) for the full list and how to run each.

## Experiments (`experiments/`)

`experiments/` is a package of research harnesses (`experiment_cross_domain.py`,
`benchmark_arc.py`, `experiment_ablation.py`, `experiment_sleep_memory.py`, …)
used by the benchmark/diagnostic scripts. They are not part of the runtime path
but are the reference implementations behind the numbers in
[BENCHMARKS.md](BENCHMARKS.md).
