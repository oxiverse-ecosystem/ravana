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

## 6. Brain-Based De-Hardcoding Plan — Applied Verification (2026-07-27)

A full brain-based de-hardcoding plan was reviewed against the CURRENT tree
(post engine.py split into 8 mixins). Findings, with each item traced to a
file:line (read, not assumed):

### Plans A / C / D were ALREADY satisfied by existing infra
- Plan A (adaptive thresholds): `CognitiveChatEngine._adaptive_baselines`
  ({mu,sigma,n} per gate) + `_adaptive_gate(key,x,strict,eta)`
  (engine.py ~L980 / ~L1540) already cover ALL six targets:
  recall_gist(0.6), episodic_cos(0.5), selfq_sim(0.45),
  schema_cos family(0.6/0.4/0.5), phrase_sim(0.75). The plan's cited line
  numbers were pre-split; the gates existed. The ONE real gap was that the
  baselines were never persisted (reset to seed every boot) — fixed: added
  `'adaptive_baselines'` to save() and an overlay-restore in load(), plus a
  `recall_cos`(0.55) gate now drives the recall trigger (engine_memory.py:1101).
  Verified: mu/n survive a full save->reload (1620 tokens preserved).
- Plan C (sensorimotor): `data/lancaster_encoder.npz`
  (LancasterEncoder, GloVe-64 -> 11-D Lancaster norms, 39,707 human-rated
  words, trained by scripts/train_lancaster_probe.py) + `engine_graph.py:
  _lancaster_vector` already drive `_top_sensorimotor_dim` (engine_generation.
  py:1074-1127). `_PROP_TO_BINDER` is already only an EXCLUSION set and
  `_SENSORY_DIM_PHRASE` only a REALIZATION-TEMPLATE fallback — the learned
  probe does the dimension selection. No duplicate SensorimotorProjector built.
- Plan D (junk domains): `junk_scorer.OnlineJunkClassifier` (continually-fit
  logistic, self-labeled from consolidation outcomes, theta derived from data)
  already IS the cerebellar forward-model the plan describes, with
  `_WEBSITE_SHAPE`/keyboard-mash/vowel-less as a non-learnable structural
  backstop. `_JUNK_SNIPPET_DOMAINS` is a separate, deliberate SOURCE-MONITORING
  blocklist (catches crossword/thesaurus/spam/art-title masquerade). It is kept
  — there is NO labeled valid/junk URL corpus to train a replacement, and
  replacing it would regress (source-monitoring failure). Not duplicated.

### Plan B (word lists -> frequency) — IMPLEMENTED
New `ravana/core/frequency_model.py::FrequencyModel`: seeds from the current
hand lists (`_GENERIC_NOUNS`, `TOPIC_SKIP_WORDS`, `_SUBJECT_CONTEXT_WORDS`) so
day-one behavior is identical, then folds observed conversation text into a
running frequency distribution. The high-frequency tail is discovered from
exposure (mental-lexicon frequency effect / Zipf). Wired as `self._freq_models`
(engine.py __init__), with `_is_generic_noun` / `_in_topic_skip` /
`_is_subject_glue` helpers routing the prior raw-set call sites
(engine_generation.py:1615, response_gen.py:5828). Persisted in save/load.
Verified: seed words known at day-one; after >=200 observed tokens a frequent
word (`cat`) is treated as common; counts survive reload. `_COMMON_WORDS` left
to the existing functional_lexicon single-source (already externalized).

### Plan E — Genuine closed classes (left as-is, documented)
QUESTION_WORDS, FOLLOW_UP_WORDS, _UNIVERSAL_PURGE, _DEFINITION_ASSERTION,
_LANCASTER_ORDER (encoder wire-format), SAVE_SCHEMA_VERSION, _PROTECTED_CONCEPTS,
VAD/Identity/GW eta constants, MAX_DECODER_VOCAB_SIZE: closed grammatical
classes / schema / capacity / wire-format — not world knowledge, per the
audit's FINE verdict. _IRREGULAR_VERBS kept as a stable seed (small, and the
brain stores very-high-frequency irregulars as whole-word memories).

### Net result
- Real code added: FrequencyModel + 4 engine helpers + persistence for both
  adaptive baselines and freq models + recall_cos gate.
- Items the plan feared were "permanent hardcoded" were in fact already
  adaptive (A/C/D) — the only true bug was the persistence hole, now fixed.
- No dead/duplicate code introduced; no fixed-threshold wins added.

---

## 9. Remaining Items — closed (2026-07-27 follow-up)

The user traced every remaining static item against the tree (post Plan A/B
commits) and assigned priorities. All were addressed and verified by real
execution (save to reload round-trip + targeted drains), not by inspection
alone. Commits d266373, eef69eb, 80921a9.

### Category 1 - SEED-STUCK (broken learning loops, infra existed)
1. Source-trust loop (P0, Item 1). source_trust was saved but never reloaded
   (state.get('source_trust') absent from load) -> reset to {} every boot;
   _record_source_outcome had 0 callers. FIX: restore in load; wire accept at
   the chosen-snippet return (engine_web_search.py) and reject at the junk-domain
   blocklist continue. VERIFIED: trust dict {spam:0.3, wiki:0.6} survives a full
   save->reload.
2. Hippocampal buffer -> graph (P1, Item 2). get_consolidation_candidates() /
   mark_consolidated() existed but were never called. FIX: drain in
   _sleep_consolidate after replay via _ensure_relation + mark_consolidated.
   VERIFIED: injected "paris is_capital_of france" graduated to a real graph edge
   <Edge 262->263 w=1.0 conf=0.9 is_capital_of>.
3. ConnectorLearner (P1, Item 3). Instantiated from seeds, never saved/loaded.
   FIX: to_dict/from_dict + hebbian_update; persisted in save/load; per-sleep
   Hebbian reinforcement. VERIFIED: round-trip preserves connector->relation map
   and re-affirmation runs without error.

### Category 2 - PERMANENT-HARDCODE (brain-inspired replacement)
4. _RECALL_SEED_CONCEPTS (P2, Item 4). Now _recall_seed_concepts() extends the
   hand anchors with graph concepts near the seeds (GloVe cos >= 0.7), cached and
   refreshed as the graph grows. Used by the recall trigger (engine_memory.py).
5. _IRREGULAR_VERBS (P2, Item 5). Added _learned_lemmas + _base_form() (irregular
   map -> learned map -> phonological CVC/-ied/-ed fallback) + _learn_lemma()
   co-occurrence hook in _observe_language. Persisted. VERIFIED:
   base_form('stopped')->'stop', 'wugged'->'wug'.
6. _PROP_TO_BINDER (P2, Item 6). _prop_binder_exclude() augments the hand map
   with the dominant Lancaster dimension of the property (trained probe
   data/lancaster_encoder.npz) so the binder exclusion generalizes beyond the
   ~11 hand-authored properties. Hand map retained as fallback (day-one stable).

### Category 3 - dead code removed
7. _CONDITIONAL_FRAME (Item 7). No runtime callers -> deleted.
8. _ATTR_WORDS (Item 8). No runtime callers -> deleted (ConceptNet attribute
   probe is the learned replacement).

### Category 4 - pre-existing bug
9. state_checksum mismatch every reload (P0). Root cause: the graph is
   independently ACID-persisted via SQLite and serializes to a string form that
   legitimately differs from the in-memory object. FIX: exclude graph from the
   checksum fingerprint (engine_persistence.py). VERIFIED: no checksum-mismatch
   warning on a full save->reload.

### Regression
tests/test_dehardcode_plan.py: 21 passed, 1 failed. The single failure
(test_meaning_of_life_not_dict_dump) is PRE-EXISTING - it fails identically on
the clean baseline (confirmed via git stash). No new regressions introduced by
these changes.

