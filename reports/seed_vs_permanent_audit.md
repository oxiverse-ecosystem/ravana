# RAVANA Seed-vs-Permanent Hardcoding Audit
Date: 2026-07-27. Method: delegated investigation (42 tool calls, read-only)
+ parent verification of key claims. Every file:line below was read, not guessed.
Lens: hardcoded content is acceptable ONLY as INNATE PRIOR — it must (1) grow
via learning from web+conversations, (2) consolidate into weights/persisted
stores, (3) be REUSED from those stores afterwards, with the hardcoded copy
ceasing to be the operative source.

## 1. Executive summary
- SEED-OK ............ 6  (seed exists, learning twin ON, persists, reused)
- SEED-STUCK ......... 5  (seed never consolidates OR learned copy never reloaded/never trained)
- PERMANENT-HARDCODE . 6  (no learning path at all today)
- FINE ............... 7  (closed grammatical classes / schema / capacity — not knowledge)

Single most important finding: THE SOURCE-TRUST LEARNING LOOP IS DEAD
(saved but never loaded; outcome-recorder never called). Second: hippocampal
sleep-consolidation candidates are computed but never consumed by the chat
engine, so buffer facts don't graduate to the neocortical graph through the
designed path.

## 2. Per-item verdicts

### SEED-OK (behaves like innate priors)
| Location | Element | Evidence |
|---|---|---|
| engine.py:906 | use_intent_router = True | Learned router ON; fit artifact data/intent_router.json (semantic/shape/affect/reference centroids + margins) with promoted routes [conditional, factual_yesno, self_directed, self_disclosure]. Regex only for un-promoted routes (engine_web_search.py:1633 _router_says). Grows by re-fit + promotion. |
| engine.py:920-921 + data/pos_model.json | use_learned_pos = True, PosModel | Learned distributional POS is operative; _GRAMMATICAL_CONCEPTS (engine.py:1099) is the documented cold-start backstop only (engine_web_search.py:1566-1580). |
| engine_generation.py:2061 _seed_common_facts | data/common_facts.json curated facts | Explicit CURATED provenance into _definitions + typed graph relations; web learning writes the same stores; graph + definitions persist in snapshot (engine.py:4264 save / 4490+ load). Textbook innate-prior design. |
| engine.py:688 _vad_baseline | EMA mu/sigma affect baseline | The project's canonical adaptive exemplar; updated online (response_gen.py:3998), distribution-driven. |
| engine.py:1513-1554 | Adaptive threshold gate (P2-E) | Distribution-driven replacement for fixed cosine cutoffs; cold-start passes at x>=mu then adapts. The pattern the remaining fixed floats should migrate to. |
| data/salad_classifier.json + engine_generation _final_emit_guard | learned word-salad gate | Learned verdict primary; rule-based _is_word_salad only when learned returns None (per dehardcode promotion log). |

### SEED-STUCK (flag ON or infra built, but the loop is broken)
| Location | Element | What's broken | Absorbing infra |
|---|---|---|---|
| engine.py:914-915, 4365; engine_web_search.py:1009-1035 | Source-trust learner | VERIFIED BY PARENT: _source_trust is saved (engine.py:4365) but NEVER loaded — no state.get('source_trust') in the 4490-4991 load path. AND _record_source_outcome (engine_web_search.py:1035) has ZERO callers, so trust never updates even in-session. _PREFERRED_SNIPPET_SOURCES (engine.py:348) is therefore the permanent operative source despite use_source_trust=True. | Infra exists and is complete — needs (a) call _record_source_outcome at snippet accept/reject sites, (b) restore dict in load(). |
| core/hippocampal_buffer.py:292 get_consolidation_candidates + mark_consolidated | Sleep consolidation of buffer facts | Only reference outside the buffer is a DOCSTRING (language/prefrontal_workspace.py:764). The chat engine's _sleep_consolidate (engine_generation.py:1129) prunes/consolidates GRAPH edges but never drains the hippocampal buffer's high-confidence facts into the graph. Buffer facts persist raw (get_state, hippocampal_buffer.py:338) but never graduate to neocortical weights — REM without memory transfer. | HippocampalReplay (ravana/learn/consolidation.py) + _sleep_consolidate are both present; wire candidates->graph edges->mark_consolidated. |
| synaptic_dynamics.py:379 ConnectorLearner.PROTOTYPE_CONNECTORS | connector→relation prototypes | Learner is instantiated (engine.py:964) and maps via GloVe similarity (good: prototypes are seed, geometry generalizes). But prototypes are never UPDATED from observed discourse and the learner has no persisted fit — same seed every boot. Grows: NO. Consolidates: NO. | Hebbian update of prototype centroids from confirmed parses; persist centroids in snapshot like intent_router.json. |
| engine.py:348 _PREFERRED_SNIPPET_SOURCES | preferred-domain allowlist | Meant as cold-start prior inside _domain_trust (engine_web_search.py:1020) — but because the trust loop is dead (above), it is de facto permanent. Verdict inherits from the source-trust fix. | Source-trust learner (once alive). |
| engine.py:4244-4245 _IRREGULAR_VERBS | irregular verb map | Consulted directly at runtime; no learned morphology twin, no growth from text. English irregulars are near-closed-class (borderline FINE), but the brain learns these from exposure; a corpus-derived lemma table (from the already-shipped Gutenberg corpora in data/) could replace it. | PMISeeder/corpus pipeline (bootstrap/pmi_seeder.py) could emit lemma pairs; NONE wired today. |

### PERMANENT-HARDCODE (no learning path today)
| Location | Element | Why it violates | Absorbing infra |
|---|---|---|---|
| engine.py:237 _COMMON_WORDS | big inline common-word set | Consulted every request; open-class (content words drift by domain); no learned frequency source. | Document-frequency over hippocampal buffer / decoder vocab counts (the df-weighting pattern just shipped in the supersede fix is the template). |
| engine.py:338 _CONDITIONAL_FRAME | hand-authored conditional response frames | Hand-authored generation templates as permanent operative source. | Decoder generation + retrieval of stored phrasings; or centroid-based frame selection (intent-router pattern). |
| engine.py:486 _ATTR_WORDS | attribute-word tuple | Fixed lexical list gating attribute questions. | GloVe neighborhood of learned attribute exemplars (same mechanism as recall-seed concepts in hippocampal_buffer, which already uses GloVe similarity). |
| engine.py:327 _PROP_TO_BINDER | property→binder map | Fixed semantic mapping, never learned. | ConnectorLearner-style prototype+GloVe projection. |
| engine_generation.py:1615 _GENERIC_NOUNS | generic-noun list | Fixed list in emit path. | junk_score (constants.py:407) already computes GloVe-magnitude-based genericity — extend it; the learned twin half-exists. |
| Fixed decision floats not yet migrated to the P2-E gate (e.g. _SNIPPET_PLAUSIBILITY_FLOOR / _DEGENERATE, engine_web_search.py:415-418; assorted 0.3/0.55/0.6 cosine cutoffs) | fixed thresholds | Fixed cutoffs gate behavior forever — rejected on principle. | engine.py:1513 adaptive threshold gate (EMA mu/sigma, z-score) — the in-repo pattern; migrate one cutoff at a time with regression gate. |

### FINE (not knowledge — do not "fix")
- constants.py:126 _QUESTION_WORDS — closed grammatical class.
- constants.py:432 _UNIVERSAL_PURGE — minimal universal pronouns (closed class).
- constants.py:440 _DEFINITION_ASSERTION — universal copula regex (grammar).
- SAVE_SCHEMA_VERSION / state checksums (engine.py:4264+ save path) — schema, not knowledge.
- MAX_DECODER_VOCAB_SIZE and buffer capacity hyperparams — capacity, not knowledge (harness overrides them anyway).
- The new stem chain rstrip("s")rstrip("d")rstrip("e")rstrip("ing") (engine_reasoning.py:1016 + 10 more sites) — crude but GRAMMATICAL morphology (closed-class suffixes), not world knowledge. Acceptable; a learned lemmatizer would be a nice-to-have, not a violation.
- The new appositive-grounding regex (engine.py:~1900 _entity_recall_via_buffer) and df-weighted supersede (engine.py:1691-1760) — pure grammar + distribution-driven statistics; zero fixed thresholds, zero entity knowledge. These PASS the audit lens by construction.

## 3. Learning-loop verification (web fact today -> weights tomorrow?)
Traced through save/load:
- SAVE (engine.py:4264): persists graph, decoder state (word_to_idx/embeds/state_dict), hippocampal_buffer_state (get_state, hippocampal_buffer.py:338 — full fact triples incl. confidence/rehearsal/consolidated), causal_schema_state, relation_memory_state, source_trust (4365), definitions, concept_labels.
- LOAD (engine.py:4490-4991): restores graph (with corruption guard), decoder maps (4591-4593), hippocampal_buffer_state (4824), causal_schema_state (4827), relation_memory_state (4830), decoder_state_dict (4833), concept_labels (4820).
VERDICT: YES for facts/graph/decoder — a fact learned from the web today (written into _definitions + graph + buffer) is reloaded and reused tomorrow without re-searching. Web-search is consulted only when internal recall fails (engine_web_search gates check consolidated memory first, engine_web_search.py:2230-2232).
TWO HOLES: (a) source-trust never reloads (and never learns) — the only saved-but-dead store; (b) buffer facts never mark consolidated=True via the designed sleep path, so their promotion to graph edges rides on other code paths, not the hippocampal-replay design.

## 4. Do NOT touch
- _QUESTION_WORDS, FOLLOW_UP_WORDS, _UNIVERSAL_PURGE, _DEFINITION_ASSERTION — closed classes / universal grammar.
- _LANCASTER_ORDER — fixed encoder contract (changing it corrupts every saved vector).
- SAVE_SCHEMA_VERSION, state checksum machinery.
- _PROTECTED_CONCEPTS — curated namespace guard.
- tests/test_dehardcode_plan.py — regression gate (22 tests, 21/1 baseline), never recreate.
- The OOV / silent-KG safety nets and snippet-PE hard backstop (fail-open guards, not knowledge).

## 5. Priority order for fixing (smallest risk first)
P0: revive the source-trust loop — call _record_source_outcome at the existing
    snippet accept/reject sites; add state.get('source_trust') restore in load.
    Infra 100% built; two small wires. Immediately converts _PREFERRED_SNIPPET_SOURCES
    from PERMANENT to true seed.
P1: wire hippocampal get_consolidation_candidates -> graph edges + mark_consolidated
    inside _sleep_consolidate (engine_generation.py:1129). HippocampalReplay exists.
P1: persist + update ConnectorLearner centroids (intent_router.json is the pattern).
P2: migrate remaining fixed floats to the P2-E adaptive gate, one per regression run.
P2: replace _COMMON_WORDS/_GENERIC_NOUNS with df/junk_score-derived sets;
    _ATTR_WORDS/_PROP_TO_BINDER with GloVe-prototype projections.
