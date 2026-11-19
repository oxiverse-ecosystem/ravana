# Documentation staleness audit — 2026-11-19

**Repo:** `C:/Users/Likhith/Documents/projects/ravana` — main `6a2b5fa`, tree
clean at start of audit, remote main identical. Current PR count: **68 PRs
merged**. **Audit owner:** docs/ only (except the three excluded files listed
below) plus root `README.md`.

## Ground truth used for the audit

The current measurement is the honest, gated seven-front **n=60 offline** run
in [BENCHMARK_60_ITEMS.md](BENCHMARK_60_ITEMS.md) §4 (primary 30 + sealed 30,
no rubric changes from the n=27 harness):

| Front | Full-60 (n=60) | n=27 baseline (provenance only) |
|---|---:|---:|
| language understanding / conversation | 93 | 85 |
| grounded factual answers | 17 | 78 |
| opinions / stance | 87 | 70 |
| thinking / reasoning | 8 | 19 |
| speaking / surface realisation | 7 | 37 |
| code generation | 100 (saturated, G6 INVALID) | 100 |
| language learning (online) | 80 | 80 |
| **MEAN** | **56** | 67 |

Web arm at n=60: **timed out at 2400 s with 207 of 434 probes and no valid
summary** — no web score is recorded anywhere by this audit. Reasoning: only
two subjects have a verified 2-hop trusted chain
(`sun→light→photosynthesis`, `water vapour→clouds→rain`); offline thinking is
8/100 at n=60. Definition-based substrate growth was implemented and reverted
(`WRITE_N=0`, `CANDIDATES=0`), recorded in `BRAIN_CAUSAL_SUBSTRATE.md` §10.
Query reformulation was A/B-tested and reverted (gap loop unchanged).
IntentForge is **up** (`/health` HTTP 200), as already recorded in
`RAIN_DEFINITION_GAP.md`.

## Method

Listed every file under `docs/` plus root `README.md`; grepped for numeric
scores (67, 78, 85, 100/100, 86, 19, 37, 41), percentages, `n=27`/`n=60`, PR
counts, commit hashes, and phrases such as *current*, *latest*, *works*,
*fixed*, *remaining open*, *IntentForge*, *reformulation*, *substrate growth*.
Read every matching section in context. Patched stale claims in place with the
measured value plus a 2026-11-19 date qualifier; did **not** invent numbers;
marked unverifiable claims as unverified rather than deleting them. Historical
experiment logs and dated research notes were left intact where they already
carry an explicit `n=27`/commit/date qualifier — rewriting those would falsify
the historical record.

## Files changed (8)

| File | What was stale | What it says now |
|---|---|---|
| `README.md` | First "Benchmark results" section said nine-battery figures were "current as of the latest commits"; "Known limits" said the suite "has 27 items per front", ±17.6, and that instrument gates were "not implemented". | Added a 2026-11-19 staleness note pointing to BENCHMARK_60_ITEMS §4 with the n=60 numbers and the web timeout (no web number); rewrote Known limits to 60 items/front, ±12.3, gates implemented in `scripts/front_gates.py`. The existing validity note on the second benchmark section is retained. |
| `docs/INSTRUMENT_GATES.md` | Section titled "Measured result on current main", base commit `7e55cda`; scores conversation 85 / grounded 78 / opinion 70 / thinking 19 / speaking 37 / code 100 / learning 80, MEAN 67 — all n=27, presented as current. | Retitled "Measured result on `7e55cda` (n=27 — SUPERSEDED)"; added a note that main is `6a2b5fa` (68 PRs merged), that every number below is historical n=27, and that the current result is n=60; kept the n=27 line labelled SUPERSEDED and added the n=60 line (93/17/87/8/7/100/80, mean 56). Stated the web arm at n=60 timed out with no score. |
| `docs/SEVEN_FRONTS_STATUS.md` | "Last full run on a current main" table showing "now" = 100/100/100/0/100/100/100, mean 86 — stale small-probe scores presented as current; "defect pattern across 26 PRs" as an unqualified total. | Added a 2026-11-19 supersession banner with the n=60 offline numbers and web-timeout note, relabelled the status line "historical", and qualified the PR heading (26 PRs as of that round; 68 PRs merged on main as of 2026-11-19). |
| `docs/SEVEN_FRONTS_BASELINE.md` | Status line presented the original small-probe baseline (100/100/67/33/0/0/40, mean 49) as the plan of record without a current-state pointer. | Added a 2026-11-19 supersession banner with the n=60 numbers; relabelled the status line "historical baseline"; explicitly warned not to read the 100/100 figures as current capability claims. |
| `docs/BENCHMARKING_SURVEY.md` | "The scripts currently ship 27 items per front … until it is adopted" — false as of the n=60 adoption, and internally inconsistent with the adjacent "over 60 held-out items" paragraph. | Replaced with: 60-item split adopted and measured (BENCHMARK_60_ITEMS §2/§4), ±12.3 at n=60 vs ±17.6 at n=27, 27-item lists retained only as provenance, with a note that the paragraph dates from the 27-item era. |
| `docs/BENCHMARKS.md` | Presented the nine-battery 0.57 vs 0.23 (+0.34) result as a current capability claim with no validity qualifier. | Appended a qualifier that the nine-battery figures are historical/provenance only, pointing to the README validity note and to BENCHMARK_60_ITEMS §4. |
| `docs/CAUSAL_SUBSTRATE_CODEMAP.md` | Bottom line proposed definition-based substrate growth as "the narrowest correct lever" and quoted thinking 19 offline / 41 web at n=27 as the current front state. | Marked the bottom line superseded; recorded that the definition-promotion pass was implemented, measured as a no-op (`WRITE_N=0`, `CANDIDATES=0`), and reverted (BRAIN_CAUSAL_SUBSTRATE §10); added current n=60 thinking 8/100 and the web-timeout/no-web-number note. |
| `docs/RAIN_DEFINITION_GAP.md` | "The seven-fronts probe marks this BAD, so grounded reads 75 instead of 100" with no indication that the small-probe score is historical. | Qualified it as a small historical probe set and added that current honest n=60 grounded is 17/100 (BENCHMARK_60_ITEMS §4). The existing "CORRECTION: IntentForge is UP" section was already correct and was left unchanged. |

## Known stale claims — disposition

- **Language understanding / grounded near 100 as current** — stale in `SEVEN_FRONTS_STATUS.md`, `SEVEN_FRONTS_BASELINE.md`, `RAIN_DEFINITION_GAP.md`, and the root `README.md` nine-battery tables. Fixed with supersession notes; current values 93 and 17 at n=60.
- **Reasoning works broadly** — `CAUSAL_SUBSTRATE_CODEMAP.md` and the SEVEN_FRONTS docs overstated the front; fixed. `BRAIN_CAUSAL_SUBSTRATE.md` already recorded the correct two-chain scope and n=60 thinking 8/100; left current.
- **Offline 67 / web 78 as current** — `INSTRUMENT_GATES.md` fixed; both are n=27 provenance. Web 78 was the grounded n=27 figure, not a web n=60 result.
- **Web arm measured at n=60** — no doc was found claiming an n=60 web number; `BENCHMARK_60_ITEMS.md` explicitly says the web arm was not run there. Every supersession note added by this audit records the timeout (2400 s, 207/434 probes, no valid summary) and states that no web score is recorded.
- **Definition-based substrate growth works** — `CAUSAL_SUBSTRATE_CODEMAP.md` fixed (it was the one doc proposing it as the lever). `BRAIN_CAUSAL_SUBSTRATE.md` §10 already had the negative result; left current.
- **Query reformulation fixed the gap loop** — no such claim found in owned files. `WHY_QUERY_HEDGE_DIAGNOSIS.md` and `WHY_HOW_CHAIN_REQUIREMENT.md` already say fix attempts were reverted; `INFORMATION_NEED_CODEMAP.md` (excluded, other agent) also records the failed A/B. Left as-is.
- **IntentForge is down** — no owned doc claims this. `RAIN_DEFINITION_GAP.md` already contains "CORRECTION: IntentForge is UP" with a measured HTTP 200 health check; left current.
- **PR counts / commit hashes** — current main is `6a2b5fa`, 68 PRs merged. Claims of "current main" at `7e55cda` (`INSTRUMENT_GATES.md`) and unqualified "26 PRs" (`SEVEN_FRONTS_STATUS.md`) were qualified. Per-capability "shipped in commit X" and historical PR numbers (#315, #328, #388, etc.) are shipping records, not current-main claims, and were left intact. Historical audit/generated reports carry their own dates.

## Unverifiable / deliberately not rewritten

- Historical per-capability commit hashes and PR numbers could not be re-derived as "current" values and are not current-state claims; they are retained as the historical records they are, not deleted.
- The nine-battery numbers in README.md are marked provenance-not-current by the existing validity note; this audit did not attempt to re-run `scripts/evaluate_ravana.py` and does not claim they are reproduced.
- No n=60 web number was invented; the web run produced no valid summary.

## Files checked and left current

Excluded from this audit (owned by other agents — **not touched**):
- `docs/BRAIN_INFORMATION_NEED.md`
- `docs/INFORMATION_NEED_CODEMAP.md` (untracked in git)
- `docs/BENCHMARKING_SYSTEMS_2026.md`

Also created by this audit: `docs/DOCS_AUDIT_2026.md` (this file).

### Benchmark / status / index docs
- `docs/BENCHMARK_60_ITEMS.md` — authoritative n=60 result; current.
- `docs/ACCEPTANCE_LEDGER.md` — dated 2026-09-14 test-coverage grades; not a front-score claim.
- `docs/README.md`, `docs/FAQ.md`, `docs/CONCEPTS.md`, `docs/DEVELOPMENT.md`, `docs/GETTING_STARTED.md`, `docs/TESTING.md`, `docs/TRAINING.md`, `docs/ARCHITECTURE.md`, `docs/MODULES.md`, `docs/API_REFERENCE.md`, `docs/HONEST_REPORTING_STANDARD.md` — descriptive/index/standard docs; no stale current-state score claims.

### Brain / research docs
- `docs/BRAIN_CAUSAL_SUBSTRATE.md`
- `docs/BRAIN_CROSS_FRAME_PRODUCTIVITY.md`
- `docs/BRAIN_CUE_SENSITIVE_CONSTRUCTIONS.md`
- `docs/BRAIN_ENVIRONMENTAL_FEEDBACK.md`
- `docs/BRAIN_EXPLANATION_AND_REASONING.md`
- `docs/BRAIN_LEXICAL_CATEGORY.md`
- `docs/BRAIN_LINEARIZATION_SPEC.md`
- `docs/BRAIN_MECHANISM_ASSEMBLY.md`
- `docs/BRAIN_MESSAGE_PREPARATION.md`
- `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md`
- `docs/BRAIN_NOVEL_TERM_LEARNING.md`
- `docs/BRAIN_OPINION_FORMATION.md`
- `docs/BRAIN_PRODUCTIVE_VOCABULARY.md`
- `docs/BRAIN_REASONING_DESIGN.md`
- `docs/BRAIN_RELATION_TYPING_SPEC.md`
- `docs/BRAIN_TO_RAVANA_LEARNING.md`
- `docs/BRAIN_USAGE_BASED_SELECTION.md`

### Decoder / decoder-diagnostic docs
- `docs/DECODER_AFTER_ATTENTION_FIX.md`
- `docs/DECODER_AUDIT_DISPOSITIONS.md`
- `docs/DECODER_BLOCKED_BY_TWO_GATES.md`
- `docs/DECODER_FIXED_FILLER_REGISTER.md`
- `docs/DECODER_GROUNDEDNESS_BASELINE.md`
- `docs/DECODER_GROUNDING_VETO.md`
- `docs/DECODER_OFF_TOPIC_REGARDLESS_OF_CPG.md`
- `docs/DECODER_ONLY_NUMBERS.md`
- `docs/DECODER_REACHABILITY.md`
- `docs/DECODER_TRAINED_ON_APHORISMS.md`
- `docs/DECODER_VOCAB_MISMATCH.md`

### Generated reports (historical, self-dated)
- `docs/_generated/audit-report.md`
- `docs/_generated/suite-timings.md`

### Capability docs (historical "shipped in commit X" records — accurate as history)
- `docs/CAPABILITY_ACTIVITY_OBJECT_CATEGORY_RECALL_BRIDGE.md`
- `docs/CAPABILITY_ACTIVITY_OBJECT_NOUN_PHRASE.md`
- `docs/CAPABILITY_AFFECT_VERB_ATTITUDE_MINING.md`
- `docs/CAPABILITY_AGENT_OWN_RECALL_DISAMBIGUATION.md`
- `docs/CAPABILITY_AGENT_OWN_STANCE_PERSISTENCE.md`
- `docs/CAPABILITY_AUTOBIOGRAPHICAL_RECALL.md`
- `docs/CAPABILITY_AUTOBIOGRAPHICAL_RECALL_ENTITY_HEAD_BRIDGE.md`
- `docs/CAPABILITY_AUX_VERB_RELATIONSHIP_RECALL.md`
- `docs/CAPABILITY_BARE_COPULA_NAME_MINING_CASE_INSENSITIVE.md`
- `docs/CAPABILITY_BELIEF_SURFACING.md`
- `docs/CAPABILITY_BEREAVEMENT_EMPATHY.md`
- `docs/CAPABILITY_BOUNDED_STANCE_INERTIA.md`
- `docs/CAPABILITY_C4_CEREBELLAR_PHRASE_PATTERN_LEARNER.md`
- `docs/CAPABILITY_CATEGORY_ENUMERATION_RECALL.md`
- `docs/CAPABILITY_CLAUSE_STRUCTURE_TOPIC_DETECTION.md`
- `docs/CAPABILITY_CODE_LEARNING.md`
- `docs/CAPABILITY_COMPOUND_QUERY_DECOMPOSITION.md`
- `docs/CAPABILITY_CONTRACTION_FACT_MINER.md`
- `docs/CAPABILITY_CREATIVE_UNKNOWN_SUBJECT_GROUNDING.md`
- `docs/CAPABILITY_CROSS_TURN_SUBJECT_REPETITION_ANAPHORA.md`
- `docs/CAPABILITY_D4_CONFABULATION_FIX.md`
- `docs/CAPABILITY_DATE_RECALL_PARAPHRASE.md`
- `docs/CAPABILITY_DEGENERATE_HEAD_SKIP.md`
- `docs/CAPABILITY_DETERMINISTIC_GREETING_FALLBACK.md`
- `docs/CAPABILITY_DISTINCT_TOPIC_RECALL.md`
- `docs/CAPABILITY_DURATION_ATTACHES_TO_NEAREST.md`
- `docs/CAPABILITY_DURATION_MINING.md`
- `docs/CAPABILITY_ENTITY_LINKED_NAME_RECALL.md`
- `docs/CAPABILITY_EPISTEMIC_CALIBRATION.md`
- `docs/CAPABILITY_EPISTEMIC_FRAME_FREE_ENERGY.md`
- `docs/CAPABILITY_EVALUATIVE_PREDICATE_POLARITY.md`
- `docs/CAPABILITY_FACTUAL_ATTRIBUTE_VOCABULARY_GROWTH.md`
- `docs/CAPABILITY_FAVORITE_DISCLOSURE_MODIFIER_STRIPPING.md`
- `docs/CAPABILITY_FIRST_PERSON_RECALL_USER_ATTR_GUARD.md`
- `docs/CAPABILITY_FREE_FORM_CONTRADICTION_RECODE.md`
- `docs/CAPABILITY_GENERATIVE_EMITTER.md`
- `docs/CAPABILITY_GENERATIVE_FAREWELL.md`
- `docs/CAPABILITY_GENERATIVE_GREETING.md`
- `docs/CAPABILITY_GENERATIVE_WELLBEING.md`
- `docs/CAPABILITY_GREW_UP_LOCATION_AND_TEMPORAL_NAME_GUARD.md`
- `docs/CAPABILITY_HONEST_UNCERTAINTY_FACTUAL_QUERIES.md`
- `docs/CAPABILITY_LEARNED_CONSTRUCTIONS_PERSIST_REUSE.md`
- `docs/CAPABILITY_META_IDENTITY.md`
- `docs/CAPABILITY_MIND_DISCLOSURE_NORMALIZATION.md`
- `docs/CAPABILITY_MULTI_ACTIVITY_STANCE_KEY.md`
- `docs/CAPABILITY_NONKIN_ROLE_RECALL.md`
- `docs/CAPABILITY_OBJECT_DISAMBIGUATED_DATE_RECALL.md`
- `docs/CAPABILITY_OPEN_ENDED_RELATIONSHIP_RECALL.md`
- `docs/CAPABILITY_POSSESSION_ATTRIBUTE_MINING.md`
- `docs/CAPABILITY_QUERY_INTENT_GATE.md`
- `docs/CAPABILITY_RECALL_RELEVANCE_SCORER.md`
- `docs/CAPABILITY_RECALL_ROUTING_FIXES.md`
- `docs/CAPABILITY_RECENCY_WEIGHTED_FACT_FOLDING.md`
- `docs/CAPABILITY_RELATIONSHIP_ATTR_ENUMERATION_MINING.md`
- `docs/CAPABILITY_RESIDUAL_CLAUSE_RELATIONSHIP_MINING.md`
- `docs/CAPABILITY_RETRACTION_STORE_EVENT.md`
- `docs/CAPABILITY_REVERSAL_ACK_NEW_POLARITY.md`
- `docs/CAPABILITY_SELF_CUED_EPISODIC_RECALL.md`
- `docs/CAPABILITY_SELF_DIRECTED_LEARNING.md`
- `docs/CAPABILITY_SELF_REFERENCE_WEB_SEARCH_GATE.md`
- `docs/CAPABILITY_SEMANTIC_HEAD_NOUN.md`
- `docs/CAPABILITY_SM_UNKNOWN_SUBJECT_GROUNDING.md`
- `docs/CAPABILITY_SOURCE_MONITORING_AFFECTIVE_ECHO.md`
- `docs/CAPABILITY_STANCE_INVERSION_NEW_POLARITY.md`
- `docs/CAPABILITY_STANCE_PROVENANCE.md`
- `docs/CAPABILITY_STATE_CONDITIONED_FAREWELL.md`
- `docs/CAPABILITY_STATE_CONDITIONED_SURFACE_FORM.md`
- `docs/CAPABILITY_STRUCTURED_MESSAGE.md`
- `docs/CAPABILITY_SUPERSEDED_FACT_FILTERING.md`
- `docs/CAPABILITY_TRAILING_ADVERB_STANCE_KEY.md`
- `docs/CAPABILITY_TRAILING_MODIFIER_STRIPPING.md`
- `docs/CAPABILITY_UNIVERSAL_RELEVANCE_GATING.md`
- `docs/CAPABILITY_USER_MODEL_AGGREGATION.md`
- `docs/CAPABILITY_USER_STANCE_RECALL.md`
- `docs/CAPABILITY_WEB_SEARCH_SUMMARISE_INTENTFORGE.md`

### Other docs
- `docs/AGENT_SELF_STANCE.md`
- `docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md`
- `docs/AUTHORED_CAUSAL_SKELETON.md`
- `docs/AUTONOMOUS_LANGUAGE_LEARNING_EXPERIMENT.md`
- `docs/BOOST_STRENGTH_NOT_THE_LEVER.md`
- `docs/CE_DRIFT_IS_NOT_FAILURE.md`
- `docs/CODEBASE_DECODER_AUDIT.md`
- `docs/CONDITIONER_FIX_NO_EFFECT.md`
- `docs/CONDITIONING_COLLAPSES_TO_A_POINT.md`
- `docs/CONDITIONING_INPUT_LACKS_SIGNAL.md`
- `docs/CONSOLIDATION_RESULT.md`
- `docs/CONTRASTIVE_SELF_OPINION.md`
- `docs/COVERAGE_TWO_BUGS.md`
- `docs/CPG_MEASURED_EFFECT.md`
- `docs/CROSS_FRAME_CODEMAP.md`
- `docs/CUE_SELECTION_CODEMAP.md`
- `docs/CYC_TRAP_SOLUTION.md`
- `docs/DATE_GROUNDED_RECALL_YEAR_ANCHOR.md`
- `docs/DEFINITION_PROPOSITION.md`
- `docs/DEFINITION_PROPOSITION_RECALL.md`
- `docs/DELEGATION_AUDIT_DISPATCH_MAP.md`
- `docs/DESIGN_PATH_COMPRESSION.md`
- `docs/DETERMINER_FRAME_REPLICATION.md`
- `docs/EMPIRICAL_GENERALIZATION.md`
- `docs/ENTITY_LOCATION_RECALL.md`
- `docs/ESCALATION_INERT.md`
- `docs/ESCALATION_MEASUREMENT_CONFOUND.md`
- `docs/ESCALATION_VERIFIED.md`
- `docs/EVAL_MEASURES_A_DIFFERENT_PATH.md`
- `docs/EXPLANATION_CHAINS_MEASURED.md`
- `docs/EXTERNAL_TOTAL_AGENT_MEMORY.md`
- `docs/FORMULATOR_RESPECTS_PLAN.md`
- `docs/FULL_RETRAIN_VS_PARTIAL.md`
- `docs/GAP_DRIVEN_WEB_LEARNING.md`
- `docs/GENERALISATION_ABSTRACTION_EXPERIMENT.md`
- `docs/GENERATION_REACHABILITY_200.md`
- `docs/GENERATION_STACK_DECISION.md`
- `docs/GROUNDING_MECHANISM.md`
- `docs/GROUNDING_PASS_TOO_WEAK.md`
- `docs/HARVEST_RESTORED.md`
- `docs/HIDDEN_STATE_COLLAPSES_TO_EOS.md`
- `docs/LANGUAGE_LEARNING_CONFIRMED.md`
- `docs/LANGUAGE_PRODUCTION_PIVOT.md`
- `docs/LEARNING_CURVE.md`
- `docs/LEARNING_CURVE_CONSOLIDATION.md`
- `docs/LIMITATION_RV14_EVENT_POLARITY.md`
- `docs/LINEARIZATION_PROTOTYPE.md`
- `docs/LINGGEN_CE_COMPARISON.md`
- `docs/NEUROLANGUAGE_PRODUCTION_MAPPING.md`
- `docs/NOGO_SUPPRESSION_FAILS.md`
- `docs/NOUN_SUFFIX_MEASURED.md`
- `docs/ONLINE_RELATION_TYPING.md`
- `docs/OPEN_CLASS_VERB_CAPTURE.md`
- `docs/OVERNIGHT_REPORT_2026-11-12.md`
- `docs/PENALTY_SCALE_ROOT_CAUSE.md`
- `docs/PET_NAME_RECALL.md`
- `docs/PLAN_FLAG_ON_MEASURED.md`
- `docs/PLAN_NARRATIVE_POSITIVE.md`
- `docs/PLAN_PREFIX_MEASURED_NEGATIVE.md`
- `docs/PLAN_ROUTE_REPLICATION.md`
- `docs/plan_route_replication_data.json`
- `docs/POS_MISLABEL_ROOT_CAUSE.md`
- `docs/POSSESSION_REATTRIBUTION.md`
- `docs/POST_RETRAINING_RESULT.md`
- `docs/PRE_RETRAINING_BASELINE.md`
- `docs/PRODUCTION_OUTPUT_FINAL.md`
- `docs/PRODUCTION_OUTPUT_RESULT.md`
- `docs/PRODUCTIVE_VOCAB_CODEMAP.md`
- `docs/PROPOSITION_UNION.md`
- `docs/QUANTITY_MEMORY.md`
- `docs/READOUT_CANNOT_PREDICT_CONTENT.md`
- `docs/REFLECTIVE_GAP_CALIBRATION.md`
- `docs/RELATION_GATING_INCONCLUSIVE.md`
- `docs/RELATION_GATING_SETTLED.md`
- `docs/RETRAINING_RESULT.md`
- `docs/RETRIEVAL_GENERATION_VALIDATION.md`
- `docs/RETRIEVAL_VS_GENERATION.md`
- `docs/ROUND_REPORT_2026-08-22T0703Z.md`
- `docs/round-report-2026-11-02T0521Z.md`
- `docs/SCALING_GENERALIZATION.md`
- `docs/SELF_OPINION_RELATIVE_CLAUSE.md`
- `docs/SESSION_GOAL_dim_assumptions.md`
- `docs/SESSION_GOAL_grounded_decoder.md`
- `docs/SESSION_GOAL_verbosity.md`
- `docs/SESSION_GOAL_wrapper.md`
- `docs/STANCE_REVERSAL.md`
- `docs/STOP_GATE_MEASUREMENT.md`
- `docs/SWITCH_IS_OFF.md`
- `docs/SYNTHESIS_TOTAL_AGENT_MEMORY.md`
- `docs/THETA_LOWER_IS_BETTER.md`
- `docs/TOTAL_AGENT_MEMORY_RESEARCH.md`
- `docs/TYPING_ACCURACY_IS_CIRCULAR.md`
- `docs/UNIT_SUITE_STALL_DIAGNOSIS.md`
- `docs/USAGE_SELECTION_CODEMAP.md`
- `docs/VOCAB_BLOCKS_TRAINING.md`
- `docs/WEB_LEARNING_CHAIN_VERIFIED.md`
- `docs/WEB_NODE_ADMISSION_BLOCKED.md`
- `docs/WHAT_THE_DECODER_ROWS_CONTAIN.md`
- `docs/WHICH_ARCHITECTURE.md`
- `docs/WHY_DECODER_NEVER_RUNS.md`
- `docs/WHY_HOW_CHAIN_REQUIREMENT.md`
- `docs/WHY_QUERY_HEDGE_DIAGNOSIS.md`
- `docs/WRAPPER_CONTRACT_DECISION.md`
- `docs/WSM_CONDITIONER_DEAD_AT_GENERATION.md`
- `docs/WSM_IS_NOISE.md`
- `docs/WSM_R2_MEASUREMENT.md`

**Counts:** 8 files changed (7 under `docs/` + root `README.md`); 233 files were present under `docs/` at the start of the audit (232 .md + 1 .json); 224 docs files checked and left current; 3 excluded (all present by end of audit, untouched); 1 audit file created.
