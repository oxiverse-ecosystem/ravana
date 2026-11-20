# RAVANA-LEARN-5 codebase reuse map

**Status:** decision-ready implementation map. This document changes no code, no harness, no gate, no fixture and no other doc.
**Repo / tree:** `C:/Users/Likhith/Documents/projects/ravana`, clean at `3d9c119` at time of mapping. Windows host (bash/MSYS shell).
**Spec source:** `docs/BENCHMARKING_SYSTEMS_2026.md` §0/§3/§4/§6/§7 (defines CAR/WFA/PDR/CLI/RSI, their controls, pass criteria and probe-validity constraints).
**Method:** every claim below is tied to an on-disk file and line. Measured numbers are read from the checked-in artifacts, not recomputed here. This is a *reuse* map: the default verdict is "extend", and a new script is justified only where the existing harness cannot express the required procedure.

---

## 0. TL;DR decisions

| Component | Verdict | Reuse anchor | New code that is genuinely required |
|---|---|---|---|
| **CAR** | **EXTEND existing harness** | `scripts/measure_learning_curve.py` (467 lines) | k-points, held-out paraphrases, delayed cold-reload retest, U/S/F arms, savings formula — all as new functions *in that file* |
| **WFA** | **NEW thin script that imports the frozen replay harness** | `scripts/measure_fronts_web_frozen.py` + `tests/fixtures/web_snapshots/web_fixtures.json` (1087 queries) | gap→read→post procedure + source-attribution judge; do NOT rewrite the provider stubs or gates |
| **PDR** | **NEW thin script that imports WFA's boundary; NEW poison variants of the frozen bundle** | WFA script + `engine_web_search.py` snippet-trust hooks + `learn/explanation.py` provenance gate | poison-sweep fixture variants, poison/clean/disambiguation graders, edge-write census |
| **CLI** | **NEW thin script importing CAR helpers** | `measure_learning_curve.py` helpers + `core/sleep.py` / `learn/consolidation.py` | A→B→A ordering, consolidation on/off ablation, fresh-engine control, forward transfer |
| **RSI** | **NEW script + new item bank + new exact-match grader** | `core/analogy_engine.py`, `core/causal_schema.py`, `core/deductive_reasoning.py` are the *subjects under test*, not harnesses | rule descriptions, k-demonstration curve, structural exact-match grader, permuted-demo control |

**Narrowest order to a measurable result:** **CAR → WFA → PDR → CLI → RSI** (§4). CAR already has a gated, controlled, one-process number; WFA already has a reproducible replay boundary; PDR rides on WFA; CLI needs extra boots and an ablation; RSI needs the most new machinery.

---

## 1. Existing assets, measured, with line evidence

### 1.1 `scripts/measure_learning_curve.py` (467 lines) — the CAR seed

- Detector controls and arms: `CAUSAL_REL` :63, `CAUSAL_MARKERS` :64, `TEACH` :70, `NO_TEACH` :84, `DEF_ONLY` :92, `WEB_UNGATED` :100.
- Reusable helpers: `prepare_data_dir()` :110, `label_map()` :131 (handles the `graph.nodes` dict trap explicitly at :135-138), `causal_snapshot()` :149, `trusted_chain_summary()` :170, `trusted_chain()` :183, `install_trace()` :203, `ask()` :222, `answer_verdict()` :233, `teach_conversational()` :247, `teach_web()` :256.
- Instrument self-check before the curve: `run_detector_controls()` :292 (positive trusted 2-hop, negative definition, direction check); called at :344.
- One-process boot and arms: `main()` :331, boot :336-342, `run_arm()` :352, arms invoked at :425-428 (`TEACH_BOTH`, `NO_TEACH_CONTROL`, `DEF_ONLY_CONTROL`, `WEB_UNGATED_POSITIVE`), `summarize()` :431.
- **Measured result** — `scratch/learning_curve_audit.json` (commit `41ec83a`):
  - `TEACH_BOTH` n=12: `post_answer_chain_rate=0.583`, `chain_rate_delta=0.583`, `new_trusted_edges_total=19`.
  - `NO_TEACH_CONTROL` n=6: `chain_rate_delta=0.0`; `DEF_ONLY_CONTROL` n=6: `0.0`; `WEB_UNGATED_POSITIVE` n=6: `0.5` (proves the instrument *can* detect learning).
  - `detector_controls`: positive trusted 2-hop detected = true, negative definition rejected = true, direction forward ok = true.
  - Caveat: n=12/6 is below the skill's ~20-probe hypothesis threshold; treat as an instrument-validated prototype, not a release number. The +0.583 is a **chain-rate** delta (graph and emitted answer agree here), not yet the §6.1 k-curve.

### 1.2 Frozen web replay — WFA/PDR substrate

- `scripts/capture_web_fixtures.py`: `FIXTURES_PATH` :40, `PROVIDER` :42 (recorded provider chain), `_install_provider_stubs()` :84 with `capture_search()` :94, `capture_kb()` :120, `capture_tool()` :135, `top_up()` :156, `main()` :181.
- `tests/fixtures/web_snapshots/web_fixtures.json`: **1087** query keys, 61 `kb_describes`, 207 `tool_web_search`, `captured_at=2026-11-19T18:40:01Z`.
- `scripts/measure_fronts_web_frozen.py`: `FIXTURES_DIR` :52, `FIXTURES_PATH` :53, `EXPECTED_PROBE_ROWS=434` :54, `NETWORK_ATTEMPTS`/`SAVE_ATTEMPTS`/`PROVIDER_CALLS` counters :56-58, error classes :62/:66/:70, `FIXTURE_QUERY_COUNT` :80, `captured_at`/`provider` :81-82.
- Replay stubs: `_frozen_search()` :85, `_frozen_kb()` :98, `_frozen_tool()` :109; blocked boundary `_blocked_urlopen()` :121, `_blocked_call_api()` :127, `_blocked_save()` :132; provider patch applied at :137-142.
- Harness structure worth reusing: `_seal_split()` :304, `_build()` :313, `_boot()` :706, `_ask()` :715, judges :771-901, `_cold_reload_probe()` :911, `pre_gates()` :967, `post_gates()` :977, `gates_only()` :992, `run_controls()` :1023, `pool_scores()` :1106, `main()` :1243.
- The replay never touches the network and never saves; the frozen-snapshot recipe and its stale-summary / `EXIT=124` failure signatures are in the `ravana-debugging` skill reference `references/frozen-snapshots.md`.

### 1.3 `scripts/front_gates.py` — G1–G6 already implemented

- Thresholds: `SELECTIVITY_MIN=0.30` :21, `POLICY_PASS_MAX=0.25` :22, `SATURATION_LOW=5`/`SATURATION_HIGH=95` :24-25.
- Case banks: `QUERY_FRONTS` :27, `PAIRED_CASES` :38, `CONTROL_CASES` :63, `DECLINE_CASES` :88, `ASSERT_CASES` :100, `DECLINE_THINKING` :114, `ASSERT_THINKING` :119, `GOOD_THINKING` :124.
- Gate functions: `gate_paired()` :165 (G1), `gate_selectivity()` :231 (G2), `gate_symmetric()` :263 (G3), `gate_parity()` :318 (G4), `gate_census()` :391 (G5), `gate_saturation()` :435 (G6), `gate_valid()` :452, `print_table()` :457. Run via `gates_only()` at :992 of the frozen harness (or directly).
- All five components must call these functions, not reimplement them. Case banks are front-specific (`judge_thinking` etc.), so add LEARN-5 paired/control cases rather than reuse the old front strings (see §6 caveat).

### 1.4 `scripts/measure_fronts.py` — sealed-split donor

- `_seal_split()` :221 hashes `sha256("RAVANA-7-v1|<front>|<difficulty>|<subject>")` and sorts to assign primary/sealed pools; `_build()` :230, `_build_code()` :240, `N_MAP` :612, `_boot()` :623, `_cold_reload_probe()` :828, `pre_gates()` :884, `post_gates()` :894, `gates_only()` :909, `run_controls()` :940, `pool_scores()` :1023, `main()` :1160.
- For LEARN-5 change only the hash namespace to `RAVANA-LEARN-v2|<component>|<difficulty>|<subject>` per spec §3; keep the *mechanism*, not the old front rubrics.

---

## 2. Engine observable hooks (where to instrument, not what to rewrite)

### 2.1 Trusted causal chains — `ravana/src/ravana/learn/explanation.py`

`Hop` :52, `ExplanationChain` :69, `_edge_is_causal()` :128, `score_chain()` :147, **`enumerate_chains()` :156** (walker, forward/backward), **`_hop_provenance()` :268**, **`provenance_trusted()` :302**, **`assertable_chain()` :319**, **`grade_chain()` :331**. `grade_chain` already converts `assert`→`decline` on an untrusted hop (confirmed by `tests/unit/test_explanation.py` `TestProvenanceGate` :80-135 and `docs/BRAIN_CAUSAL_SUBSTRATE.md:316`). **Do not add a confidence surrogate; grade the structured chain object.**

### 2.2 Source provenance — `ravana/src/ravana/chat/provenance.py`

Kinds :48-54 (`VERIFIED_WEB`, `CURATED`, `CO_OCCURRENCE`, `AUTO_EXPAND`, `BOOT`, `INFERRED`, `UNKNOWN`), `CONVERGENCE_FACT_THRESHOLD=59`, `provenance_class()` :85, `is_verifiable()` :112, `admit_as_fact()` :117, `provenance_adjusted_confidence()` :126, **`populate_provenance()` :168** (the write-time stamping hook), `save_policy()` :204, `load_policy()` :211. Tests: `tests/unit/test_phase1_provenance.py:40,61`, `tests/unit/test_provenance_persistence.py`.

Write path donors: `web/web_to_graph.py` `WebToGraph` :49, `_provenance()` :91, `_register_label()` :100, `_ensure_node()` :126, `learn_text()` :206, `knowledge_gap()` :300, `fact_count()` :335. `learn/intake.py` `edge_type_for_relation()` ~:260, `get_state/set_state` :1011/:1024. `web/learner.py` `WebLearner` :646, `learn_from_web()` :701, `_learn_from_text()` :812.

**Trust-gate rule:** stamp `source_metadata` (a real `Provenance`) at creation; never lower a threshold to make a curve rise. `authority`/`score` from IntentForge are ranking signals only (spec §4, `docs/BENCHMARKING_SYSTEMS_2026.md:132`).

### 2.3 Definition store

`engine._definitions` is the store: writes at `chat/engine_generation.py:2783`, `chat/engine_graph.py:1561-1567,1627`, `chat/web_learning.py:2159,2172,2254,2262,2294`; reads at `chat/brain_regions.py:565` (`consult_internal` ~:516), `chat/chain_walker.py:2546`, `chat/engine.py:10349`, `chat/response_gen.py:1154,4178,8844,9271,9348`. Proposition extractor: `learn/definition_proposition.py` `proposition_from_definition()` :118 (`_words()` :84, `_head_noun()` :97). Test: `tests/unit/test_definition_proposition.py`.
Use for CAR `DEF_ONLY` control and PDR wrong-sense decoys: a definition must produce **zero** causal hops.

### 2.4 Sleep consolidation — CLI ablation

`core/sleep.py`: `SleepConfig` :20, `SleepConsolidation` :47, `run_cycle()` :66, `_normalize_outgoing_weights()` :224, `_prune_weak_edges()` :236, `_rem_dream_sabotage()` :275. `learn/consolidation.py`: `HippocampalReplay` :18, `run_nrem_cycle()` :55, `sleep_cycle()` :94, `get_state/set_state` :107/:114. Engine wiring: `engine.py` sleep counters :1105, `[sleep]` log :11390, `sdl.run_cycle()` :12076 and :14105; correction consolidation `chat/engine_persistence.py:_consolidate_corrections_in_sleep()` :379. Tests: `tests/integration/test_sleep_episodic_replay.py:26,71,109`.
CLI on/off ablation toggles these; do not modify them.

### 2.5 Correction detection — CAR knowledge-update + CLI/BRAIN feedback

`chat/user_model.py`: fields `detected_correction`, `detected_correction_type`, `correction_severity`, `detected_correction_fact` :1664-1668; `_detect_correction()` :6824 with DIRECT :6843, SENTIMENT_DROP :6851, INDIRECT_REASK :6883, CORRECTION_WITH_FACT :6931/6950; flagged at :6784. `chat/engine_persistence.py`: `_detect_and_handle_correction()` :577, `_process_correction_feedback()` :443, `_weaken_edges_for_response()` :490, fact persistence :583-596. Called in `engine.py` `process_turn()` :8041 (ack path :11951). From `docs/BRAIN_ENVIRONMENTAL_FEEDBACK.md:25`: weak reask/sentiment signal at `engine.py:8154`, `user_model.py:1874-1877`.
**A self-rating is not a teacher** (`ravana-debugging` skill `references/online-learning.md`, `docs/USAGE_SELECTION_CODEMAP.md:162` `_assess_response_quality` at `engine.py:11652`): use correction/reask as a gate, and validate a positive *and* negative metric before reinforcing anything.

### 2.6 Rule / analogy machinery — RSI subjects

- `core/analogy_engine.py`: `AnalogyEngine` :70, `solve_analogy()` :88, `_score_mapping()` :169, `_compute_structural_consistency()` :207, `_find_paths()` :250, `get_best_completion()` :286, `solve_analogy_query()` :295.
- `core/causal_schema.py`: `CausalSchemaLearner` :54, `learn()` :114, `predict()` :158, `get_causal_path()` :202, `explain_causal_chain()` :234, `get_state/set_state` :266/:283. Instantiate schemas as unassertable `schema_hypothesis` — that tag is **not** in `provenance_trusted` (`docs/BRAIN_CAUSAL_SUBSTRATE.md:319`), so shape may select evidence but never supply a trusted link (skill §7a).
- `core/abstraction_engine.py`: `AbstractionEngine` :60, `analyze_abstract_concept()` :131. `core/deductive_reasoning.py`: `RoleMetaruleEngine` :176, `apply()` :188, `_transitive()` :202, `_symmetric()` :236, `_inverse()` :247, `_rule_implication()` :267, `parse_deductive_premises()` :373, `deductive_mc_answer()` :441. `core/deductive_extractor.py` supplies premise extraction.
- Tests: `tests/unit/test_analogical_fix.py`, `tests/integration/test_structural_transfer.py`, `tests/unit/test_deductive_extractor.py`, `tests/unit/test_multi_hop_reasoner.py`.

### 2.7 Snippet trust — PDR hooks

`chat/engine_web_search.py`: `_web_snippet_search()` :186, `_web_direct_answer()` :347 (the hook harnesses stub to `None`), `_best_answer_snippet()` :761, `_snippet_is_structural_junk()` :1145, `_preferred_source()` :1182, `_domain_trust()` :1193, `_record_source_outcome()` :1208, `_source_trust_threshold()` :1230, `_snippet_quality()` :1235, `_snippet_plausibility()` :1335, `_answer_prediction_error()` :1388, `_needs_web_search()` :1925. `chat/web_learning.py`: `learn_from_web()` :438, `close_mechanism_gap()` :661, `_gap_chain_hops()` :820, `_mark_mechanism_gap_resolved()` :859, `_get_web_to_graph()` :2996. Gap detector: `chat/self_directed_learning.py:detect_gaps()` :33, consumed :103/:116.

---

## 3. Component-by-component implementation map

### 3.1 CAR — extend `scripts/measure_learning_curve.py`

**Why extend:** it already has a gated baseline→teach→post design, four arms, a traced `process_turn`, a scratch data_dir, and a validated +0.583 prototype delta. Rewriting loses all of that.

Add functions **in that file**:
- `run_curve(spec, k_points=(0,1,2,4,8), control=None)` — call the existing `teach_conversational()` :247 / `teach_web()` :256 repeatedly at each k; reuse `trusted_chain()` :183 and `answer_verdict()` :233 at every point.
- `held_out_paraphrase_post(spec, k)` — new paraphrase bank keyed by subject (new fixture data only); never re-ask the taught sentence (spec §3, testing effect [29]).
- `delayed_retest(spec, cold_engine)` — retest after interference and a **cold** reload in the same process; a gain that vanishes on cold reload is not learning (spec §3).
- `savings(L, L_t)` — `S=(L−L_t)/L` relearning measure.
- `run_control_arms()` — U (neutral chatter of equal length), S (teaching turns permuted), F (fresh engine for transfer analogues); existing `NO_TEACH`/`DEF_ONLY`/`WEB_UNGATED` are the starting points.
- `knowledge_update_item()` — teach X then corrected X′; grade with `grade_chain()` :331 and the correction store (§2.5); assert new ≥0.70, superseded ≤0.10.

Grade the **structured chain object** (`ExplanationChain` + `grade_chain`), never the prose. Report `post−k0`, `T−U`, delayed retest, savings, sealed pool, and zero gain on untaught distractors (spec §6.1). Wire `front_gates.py` G1–G5 plus G6 saturation before printing any curve.

### 3.2 WFA — new thin `scripts/measure_learn5_wfa.py` importing the frozen harness

**Why a new script:** the frozen harness's `judge_grounded()` :789 is a one-shot string judge; WFA needs gap→read→post-test plus a separate source-attribution number. But the *replay boundary* must be reused byte-for-byte.

Import (do not duplicate): `measure_fronts_web_frozen._frozen_search` :85, `_frozen_kb` :98, `_frozen_tool` :109, blocked functions :121/:127/:132, counters :56-58, `EXPECTED_PROBE_ROWS` pattern :54, `pre_gates/post_gates` :967/:977, `_seal_split` :304.

Add:
- `gap_question(item)` → assert `detect_gaps()` :33 fires (target ≥0.80) and `close_mechanism_gap()` :661 reads exactly one frozen hit.
- `post_paraphrase_answer(item)` → grade with `trusted_chain()`/`grade_chain()`.
- `attribution_verdict(chain, fixture_hit)` → require each hop's `_hop_provenance()` :268 to carry real `source_url` + evidence span from a hit actually read; `authority`/`score` never count. Emit answer accuracy and attribution accuracy as **two numbers** (spec §4 / HALoGEN).
- `abstain_verdict(item)` for the 10 false-premise/unanswerable items: abstain ≥0.90, confident-wrong ≤0.10.
- `summarize_arms()` reporting OFF/ON as a pair with any discrepancy investigated, never scored as a bonus (G4).

Items: 10 fast-changing + 10 noisy/conflicting + 10 false-premise, frozen by the §3 hash rule, drawn from the existing 1087-query fixture (new dated slice only if the frozen bundle cannot supply a genuine multi-hop process paragraph — measure first, see §5).

### 3.3 PDR — new thin script + poisoned fixture variants

**Why new:** neither existing harness has a poison sweep, a single-blocker variant, or a disambiguation variant. Reuse WFA's boundary and grader; do not touch the committed `web_fixtures.json`.

Add `scripts/measure_learn5_pdr.py` with:
- `make_snapshot_variant(base, n_poison ∈ {0,1,3,5}, blocker=False, decoy_kind=...)` writing new fixture files beside (not into) the frozen bundle, with a recorded variant hash.
- `poison_acceptance_rate(variant)` — attacker-chosen answer accepted; target ≤0.05 at 1 snippet, ≤0.10 at 5.
- `edges_written_after_rejected_hit()` — reuse `causal_snapshot()` :149; must be exactly zero, and `admit_as_fact()` :117 / `populate_provenance()` :168 must show no unstamped edge.
- `sense_selected()` — dictionary-decoy / navigation-blurb / process-paragraph variants; only the process paragraph may become an asserted chain (correct sense ≥0.70 with decoys). Report wrong-sense rejection and zero edges from the rejected hit separately.
- `clean_accuracy_guard()` — clean arm within 0.10 of the no-poison baseline, so "disable the web" cannot win (spec §6.3(f)); single-blocker arm must not cause abstention on answerable items.

The repo's own `docs/INFORMATION_NEED_CODEMAP.md` notes the gap loop currently takes the first two hits verbatim: PDR must **expose** that behaviour, not excuse or silently fix it.

### 3.4 CLI — new thin script importing CAR helpers

**Why new:** `run_arm()` :352 in the learning-curve harness is per-spec independent; A→B→A needs ordered state, a topically close/definitionally conflicting B, repeated **cold reloads**, and a consolidation on/off ablation — a different control flow that should not destabilise the CAR instrument.

New `scripts/measure_learn5_cli.py` imports `prepare_data_dir` :110, `causal_snapshot` :149, `trusted_chain` :183, `ask` :222, `answer_verdict` :233 from the CAR harness, and toggles `SleepConsolidation` (§2.4). Add:
- `teach_A()`, `teach_B_conflict()`, `retest_A(label)`, `transfer_B_prime(fresh_engine)`.
- `run_ablation(consolidation_on: bool)` — one cold boot per arm, no `eng.save()`.
- `forgetting(A1, A2, A3)`, `forward_transfer(B′, F)`, `savings(A)`.
- Gates G1–G5 + cold-reload census (G5) before reporting; no trust/promotion gate relaxed to obtain the forgetting bound (spec §6.4(f)).

### 3.5 RSI — new script, new items, new grader

**Why new:** no existing harness grades few-shot rule induction by exact structural match, and no existing item bank holds lexically disjoint verbal causal rules. The `core/` modules in §2.6 are the *systems being tested*, not harnesses.

New `scripts/measure_learn5_rsi.py` with:
- `load_rules()` — 30 novel rule descriptions with a held-out paraphrase per rule; sealed by `sha256("RAVANA-LEARN-v2|RSI|<difficulty>|<subject>")`.
- `run_k_demo(k ∈ {0,1,2,4})` — present k demonstrations, require the missing causal chain for an unseen case.
- `exact_structural_match(chain, gold)` — compare (hops, relation types, direction, intermediate node) on the `ExplanationChain` object; no prose similarity, no credit for verbatim demonstration reproduction.
- `permuted_demo_control()` — relations/demos randomised; must stay ≤ U + 0.10.
- `transfer_delta(fresh_engine)` — lexically disjoint analogue; report over F, not raw.

Reuse `enumerate_chains()` :156 / `grade_chain()` :331 for the produced chain, and `front_gates.py` for G1–G5. Keep prepared `schema_hypothesis` instances unassertable (§2.6).

---

## 4. Narrowest implementation order (soonest measurable result)

1. **CAR** — extend `measure_learning_curve.py`. Lowest new-code cost: the instrument self-check and the +0.583 controlled delta already exist. First deliverable: a real k-curve with held-out paraphrase and cold-reload retest, primary + sealed pools.
2. **WFA** — thin script over the frozen replay boundary. Second number, same week: attribution is a new column on an already reproducible web run. First check the fixture corpus can supply two connected causal hops before adding any new retrieval machinery (§5).
3. **PDR** — poisoned variants of WFA's snapshot; additive on WFA's boundary and the provenance gate. Third number.
4. **CLI** — needs multiple cold boots and a consolidation ablation; more expensive per number, and depends on CAR's helpers and the sleep hooks.
5. **RSI** — most new code (items, exact grader, sealed pool); last. Public ARC-AGI-2 human average 66% is a difficulty reference, not a target (spec §6.5).

Stop rule at each step: if the new component scores ≥95 or ≤5 on both arms, flag it non-informative (G6) and refresh its items — never relax the rubric to move the number.

---

## 5. Avoiding concurrent engine boot and `eng.save()` on weights

- **One cold boot per arm, one process, no concurrent suites.** `measure_learning_curve.py::main()` :331-342 boots once and runs all arms in that process; batch probes, measure boot cost once (warm-cache boot ~5 s; older cold boots ~200 s). The skill is explicit: last writer wins when engines run concurrently and learned state is lost.
- **Isolate the data_dir.** `prepare_data_dir()` :110-122 copies `weights/ravana_weights.pkl`, `.db` and `junk_*` into `scratch/learning_curve_data`; `CognitiveChatEngine(data_dir=DATA_DIR, ...)` at :339. Any save then lands in scratch, never in the authoritative weights.
- **Block saves explicitly anyway.** `measure_fronts_web_frozen.py::_blocked_save()` :132 raises `FrozenSaveError` and appends to `SAVE_ATTEMPTS` :57; `eng.save()` is the persisting call (`engine.py:12856`); there is no `eng._save()`. Assert `save_attempts == 0` before writing any summary.
- **Block the network at the seam.** `_blocked_urlopen()` :121, `_blocked_call_api()` :127 and the provider patch :137-142 raise and count; stub `eng._web_direct_answer = lambda ctx: None` (done at `measure_learning_curve.py:341`). Assert `network_attempts == 0` and `fixture_miss == 0`.
- **Validate the summary before quoting it.** Probe-row count must equal the expected total (`EXPECTED_PROBE_ROWS` :54), summary mtime must be fresh, and the record must carry `offline`, `frozen=true`, `network_attempts=0`, `save_attempts=0`, `fixture_miss=0`, `elapsed_s`. `EXIT=124` or a missing EXIT line means incomplete (skill `references/frozen-snapshots.md`).
- **Seed both boot paths if you add any graph seed.** The saved-weights path runs `_typed_edges_bootstrap()` :2188 and `return`s at :2199 before the cold-start block, while the cold path seeds at :2201+ and `_seed_curriculum_causal()` at :2272; the curriculum seed is therefore called in **both** paths (:2194 and :2272, defined `engine_graph.py:887`). Verify counts on a cold load (`CURRICULUM_EDGES=53`), not by reading source.
- **Never repair only one store.** `weights/ravana_weights.pkl` is authoritative on boot, `.db` is the offline mirror; the next `eng.save()` silently overwrites a one-sided repair. Repair both, then verify with a cold `CognitiveChatEngine` load.

---

## 6. Tests: present and needed

### 6.1 Already present (reuse as regression scaffolding)

| Area | Tests |
|---|---|
| Chain enumeration / provenance gate | `tests/unit/test_explanation.py` `TestEnumeration`:49, `TestProvenanceGate`:80-135 (one untrusted hop sinks the chain) |
| Provenance stamping / persistence | `tests/unit/test_phase1_provenance.py:40,61`, `tests/unit/test_provenance_persistence.py`, `tests/unit/test_conversational_causal_consolidation.py:77,84,92,107,116` |
| Gates G1–G5 | `tests/test_front_gates.py:30-81` (paired rubric, inverted judge, selectivity, symmetric decline, parity phantom credit, saturation/blocking) |
| Gap→read→learn loop | `tests/unit/test_gap_driven_web_learning.py:84,138,166` and bounded/fail-open `TestGapLoopIsBoundedAndFailOpen`:185-209 |
| Snippet trust | `tests/unit/test_web_snippet_source_quality.py:39,50,63,88` |
| Retrieval practice / consolidation | `tests/unit/test_retrieval_practice.py:41-116`, `tests/unit/test_retrieval_practice_writes.py`, `tests/integration/test_sleep_episodic_replay.py:26,71,109` |
| Curriculum / authored scaffold | `tests/unit/test_curriculum_scaffold.py:19,37` |
| Definitions, analogy, transfer, contradiction | `tests/unit/test_definition_proposition.py`, `tests/unit/test_analogical_fix.py`, `tests/integration/test_structural_transfer.py`, `tests/unit/test_contradiction.py`, `tests/unit/test_deductive_extractor.py` |

### 6.2 New tests required (one per load-bearing claim)

- **CAR:** held-out paraphrase graded on chain object, not string; no-teach and def-only stay flat at **every** k; delayed retest survives a cold reload; savings formula; knowledge update asserts X′ and suppresses X ≤0.10; sealed pool reproduces all signs; untaught distractor gain == 0.
- **WFA:** attribution FAILS when a hop lacks real `source_url`+span; `authority`/`score` alone FAIL; abstention ≥0.90 on false premises; OFF/ON parity with no conditional bonus; unstamped edge fails `grade_chain`.
- **PDR:** 0/1/3/5 poison acceptance curve; edges written after a rejected hit == 0; blocker does not abort answerable items; wrong-sense decoy rejected; clean accuracy within 0.10 of baseline.
- **CLI:** A2−A1 ≥ −0.15; consolidation-on minus off ≥0.10 on A3; forward transfer over F ≥0.20; all gains survive cold reload; no trust/promotion gate relaxed.
- **RSI:** exact structural match on (hops, relation types, direction, intermediate); permuted demos ≤ U+0.10; zero credit for verbatim demo reproduction; lexically disjoint transfer; sealed pool within 0.15.
- **Cross-cutting harness tests:** harness refuses to print any score when a gate fails; summary carries the frozen pointers (§5); each probe logs query, raw response, verdict, reason, hand-label; hash assignment is stable across processes (`PYTHONHASHSEED=0`).

---

## 7. Probe-validity caveats (from `docs/BENCHMARKING_SYSTEMS_2026.md` and the debugging skill)

1. **Do not inherit the old front rubrics as graders.** §5 diagnoses code generation=100 as saturated (G6 INVALID), thinking=8 / speaking=7 as floor proxies with the documented `that`/`which` causal-regex defect, opinions=87 as marker emission, grounded=17 as a coverage boundary misread as a learning result. `judge_thinking()` :835, `judge_speaking()` :866, `judge_opinion()` :814 in `measure_fronts.py` are the defective instruments, not reusable rubrics.
2. **Grade the chain object, not the sentence** (spec §5, skill `references/probe-validation.md`): shared intermediate, process predicate on every hop, direction, trusted provenance on every hop, non-empty causal census before trusting a zero, deletion test on a frozen snapshot.
3. **Instrument validity first.** For each new rubric: paired known-good/known-bad (bad < good), a control task the behaviour cannot pass (real − control ≥ 0.30), hand-label 5–10 outputs, and print input/output not just conclusions. The first run of a new rubric is a draft of the instrument.
4. **Sample size is part of the instrument.** The current n=12/6 learning-curve prototype is a hypothesis, not a release result. Use 20 primary + 10 sealed per component; log every probe individually.
5. **Hash-split before any run and reveal sealed once.** Report primary and full-set side by side; a sealed drop is the real baseline, never something to repair by editing a rubric or gate (spec §3, skill §1).
6. **No score conditional on a capability being absent.** OFF/ON reported as a pair; parity is a diagnostic, not a bonus (G4, spec §6.0). Clean accuracy must stay high in the poisoned arm so "disable the web" cannot win.
7. **A self-rating cannot be the teacher** (skill §7). Correction/reask signals are external but weak proxies; report both a positive and a negative metric and adopt only if the negative one moves.
8. **Known graph traps when writing probe helpers:** `ConceptGraph.nodes` is `Dict[int, ConceptNode]` — iterate `.values()` (`measure_learning_curve.py:135-138` shows the safe pattern); `get_edge`/`get_node` take integer ids, not labels; `_concept_keywords` is not refreshed after web learning; web learning can create a surface node (`rains`) that shadows the causal node (`rain`); directive frames and wh-cues ground poorly, so match the taught term lexically from the raw query (longest match) before concluding knowledge is missing.
9. **A chain-level gain is not an answer-level gain** (skill §7a). Count assertable chains from the indexed node **and** check emitted answers on the same subjects; label scaffolded/curriculum knowledge as scaffolding, never as the system learning.
10. **Locate the bottleneck before building more acquisition machinery.** Run the existing OpenIE extractor over the frozen fixture corpus and count distinct causal triples per query; compare with how often the gap-read template fires. If the corpus cannot supply two connected relations, no fifth tweak to the web loop will help (skill §7b).
11. **Pre-registration before the first scored run** (spec §7): freeze the 150 item ids, hash split, rubrics, seeds, snapshot hashes, URL allowlists, thresholds and stopping rule in a reviewed commit; BH q=0.05 across five primary comparisons; Wilson 95% intervals; no non-improvement dropped.

---

## 8. Concrete file list (implementation checklist)

**Modified (all new functions only; never rewrite existing instruments):**
- `scripts/measure_learning_curve.py` — CAR k-curve, paraphrase/delayed-retest/savings/U-S-F arms, knowledge-update items.
- `scripts/measure_fronts_web_frozen.py` — expose replay boundary/gates as importable helpers if needed; do not change its judges or thresholds.
- `scripts/capture_web_fixtures.py` — dated/false-premise/noisy slice capture via existing `top_up()` :156 if 1087-query bundle is insufficient.
- `scripts/front_gates.py` — add LEARN-5 paired/control cases only; keep thresholds and gate functions byte-identical.

**New scripts:**
- `scripts/measure_learn5_wfa.py`
- `scripts/measure_learn5_pdr.py`
- `scripts/measure_learn5_cli.py`
- `scripts/measure_learn5_rsi.py`
- `scripts/measure_learn5_runall.py` — one-process orchestrator: one cold boot per arm, gates first, per-component rows, no MEAN.

**New data/tests (fixtures owned by whoever implements; this map only names them):**
- Held-out paraphrase + knowledge-update bank (CAR), poisoned/blocker/decoy snapshot variants (PDR), verbal rule bank + gold chain objects (RSI), all hash-sealed.
- `tests/unit/test_learn5_car.py`, `test_learn5_wfa.py`, `test_learn5_pdr.py`, `test_learn5_cli.py`, `test_learn5_rsi.py`, `tests/integration/test_learn5_frozen_boundary.py`.
- `docs/LEARN5_ACCEPTANCE_LEDGER.md` — record commit SHA, probe hashes, snapshot hashes, thresholds and the five component rows.
