# Brain-Faithful Implementation Plan — RAVANA Fixes

**Date:** 2026-07-22
**Regression gate:** `python -m pytest tests/test_dehardcode_plan.py -q` — must yield **21 passed, 1 failed** (only `test_meaning_of_life_not_dict_dump`) throughout.
**Cold-start rule:** Every adaptive gate starts with prior == today's value. Engine behaves identically on first run.

---

## Priority Groups

| Priority | Items | Effort | Risk |
|----------|-------|--------|------|
| **P1** | A (working memory), I (dedupe), G (connector wire), H (collapse) | M / L / M / M | High (A is new capability) |
| **P2** | B (consult gap), C (calibration), E (adaptive thresholds) | M / S / M | Med |
| **P3** | D (sensorimotor lexicon), F (shape junk) | L / M | Low |

---

## P1 — Must-Do / High Impact

### A) Same-Turn Working Memory Buffer (Genuine Capability Gap #1)

**Brain mechanism:** Baddeley's episodic buffer (2000) + phonological loop. Within-turn facts are held in a limited-capacity buffer distinct from hippocampal cross-turn consolidation.

**Current code:**
- `process_turn` at engine.py:1397 — the main turn loop. Around line 1443 (`_consult_internal_knowledge`) and line 1459 (`_try_memory_query`) are where internal knowledge and cross-turn memory are checked, but neither checks same-turn asserted facts.
- `WorkingMemory` (engine.py:960) exists as a class instance but is NOT wired into the recall path.
- `_recent_user_turns` (engine.py:1462) stores verbatim text, not extracted propositions.
- `_record_episode` (engine.py:1468) records for cross-turn but too late for same-turn.

**Plan:**

1. **Add a `_working_memory_buffer` to the engine** (in `__init__`, around engine.py:960 where `self.working_memory = WorkingMemory(...)` already exists but is unused for recall). Define it as a dict: `self._wm_facts: Dict[str, Dict] = {}` — maps subject → {"value": str, "turn": int, "source": str}.

2. **Populate the buffer in process_turn** after the self-disclosure / assertion detection paths (around engine.py:1815–1822, the `_process_self_disclosure_stmt` block, and around engine.py:2577 for assertion path). For each detected assertion ("my cat's name is whiskers", "i love stargazing", "ravana is an ai"), extract the subject → value triple and store it in `self._wm_facts`.

3. **Check WM buffer BEFORE cross-turn recall** in `_try_memory_query` (engine_memory.py) and `_consult_internal_knowledge` (engine.py:1443 call). Add a `_check_wm_for_subject(subject)` method that scans `_wm_facts` and returns a match if the current turn's query refers to a subject stated earlier in the SAME turn.

4. **Clear the buffer at turn end** (or keep for 1-2 turns with decay — matching the ~2s phonological loop decay). Use a turn-counter: `self._wm_turn_born = self.turn_count`; facts older than 1 turn are evicted to prevent them from competing with cross-turn consolidation.

5. **Reuse engine_memory.py structures** — `_reconstruct_gist` (engine_memory.py:438) and `_episodic_remember` (engine_memory.py:453) can be repurposed for WM response generation.

**Reused infra:** `WorkingMemory` class (engine.py:960), `PropositionParser` (engine.py:940), `HippocampalBuffer` (engine.py:939) for structure patterns.

**Cold-start:** No cold-start needed (buffer is initially empty — correct behavior).

**Regression gate:** Must stay 21/1. The existing test suite does not test same-turn recall (that is why this is a genuine gap). No existing test should regress.

**Effort:** L (new capability). **Risk:** High (must not weaken cross-turn path). Do NOT route cross-turn queries through the WM buffer — WM is checked FIRST and only for within-turn facts.

---

### I) `_UNIVERSAL_PURGE` and `_DEFINITION_ASSERTION` Deduplication

**Current code:**
- `_UNIVERSAL_PURGE` defined in 9 places: engine.py:158, engine_graph.py:145, engine_reasoning.py:145, engine_memory.py:145, engine_web_search.py:145, engine_generation.py:145, engine_self_query.py:145, engine_persistence.py:145, engine_monitor.py:145.
- `_DEFINITION_ASSERTION` defined in 9 places: engine.py:169, engine_graph.py:156, engine_reasoning.py:156, etc.

**Plan:**

1. **Define once in `constants.py`** (at end of file, after line 422):
   ```python
   _UNIVERSAL_PURGE = {"you", "i", "we", "they", "he", "she", "it", "me", "my", "your",
                       "our", "their", "us", "them", "him", "her", "this", "that"}
   _DEFINITION_ASSERTION = re.compile(
       r"\b(is|are|was|were|be|been|being|means?|refers?\s+to|describes?|"
       r"occurs?|happens?|defined\s+as|represents?|signifies?|constitutes?|"
       r"denotes?)\b", re.IGNORECASE)
   ```

2. **Remove from all 9 modules** — delete lines 145-172 from each mixin. (Note: each mixin may have slightly different line numbers; the catalog says "line 145" for each mixin `_UNIVERSAL_PURGE` and "line 156" for `_DEFINITION_ASSERTION`.)

3. **Add imports** in each mixin and engine.py:
   ```python
   from .constants import _UNIVERSAL_PURGE, _DEFINITION_ASSERTION
   ```
   Some files (e.g., engine_graph.py:1048-1056) already reference `_UNIVERSAL_PURGE` as a module-level name — the import makes it available.

4. **Verify all references** still resolve. `_UNIVERSAL_PURGE` is used in engine.py:3915 and engine_graph.py:1048,1056. `_DEFINITION_ASSERTION` is used in engine_graph.py:1105.

**Reused infra:** `constants.py` — already the shared constant module.

**Cold-start:** N/A — pure refactor, no behavior change.

**Regression gate:** Must stay 21/1. Pure import refactor — no behavior change means no regression.

**Effort:** S. **Risk:** None.

---

### G) Wire ConnectorLearner (Replace `_EDGE_CONNECTORS`)

**Brain mechanism:** Hebbian/STDP learning of temporal contiguity — connector word → relation mapping is learned from co-occurrence statistics.

**Current code:**
- `ConnectorLearner` (synaptic_dynamics.py:379-524) — fully built, tested, but **never imported** by any engine module. Only `chain_walker.py` imports `synaptic_dynamics`.
- `_EDGE_CONNECTORS` (v1: engine.py:219; v2 weighted: engine.py:591) — hardcoded.
- Reverse lookup built at engine.py:1010-1016 — `self._CONNECTOR_TO_REL` built from `_EDGE_CONNECTORS`.

**Plan:**

1. **Import and build `ConnectorLearner` lazily** in `__init__` (around engine.py:936 where `self.plasticity = Plasticity(...)` is created):
   ```python
   from .synaptic_dynamics import ConnectorLearner
   self._connector_learner = ConnectorLearner(glove_fn=self._glove_vector)
   self._connector_learner.initialize(graph_concepts=list(self.graph.nodes.items())
       if hasattr(self.graph, 'nodes') else None)
   ```

2. **Replace reverse-lookup construction** (engine.py:1010-1016) with a method that tries the learned learner first:
   ```python
   # Instead of building from _EDGE_CONNECTORS only:
   self._CONNECTOR_TO_REL = {}
   if self._connector_learner and self._connector_learner._is_initialized:
       self._CONNECTOR_TO_REL = self._connector_learner.get_connector_to_rel()
   # Fallback: build from _EDGE_CONNECTORS for OOV connectors
   for rel_type, tiers in self._EDGE_CONNECTORS.items():
       for entry in tiers:
           options = entry[1] if isinstance(entry, tuple) and len(entry) == 2 else entry[2]
           for opt in options:
               self._CONNECTOR_TO_REL.setdefault(opt, rel_type)
   self._CONNECTOR_SET = set(self._CONNECTOR_TO_REL.keys())
   ```

3. **Wire into edge-building path** (engine_graph.py, around where edges are created from connector words — search for `_CONNECTOR_TO_REL` or connector usage). The engine_graph.py mixin methods that create edges should call `self._connector_learner.get_relation_for_connector(word)` when available, falling back to `self._CONNECTOR_TO_REL.get(word, "semantic")`.

4. **Keep `_EDGE_CONNECTORS` v2 as cold-start prior** — do not delete. It serves as OOV fallback for connector words the learner hasn't seen.

**Reused infra:** `ConnectorLearner` (synaptic_dynamics.py:379), `_EDGE_CONNECTORS` as cold-start, engine_graph.py edge-creation path.

**Cold-start:** `_EDGE_CONNECTORS` v2 (engine.py:591) is the prior. Learner is additive — OOV falls back to the hand map.

**Regression gate:** Must stay 21/1. Existing connections should be identical at bootstrap (learner with no graph data returns prototype seeds which match the hand map).

**Effort:** M. **Risk:** Low-Med (learner already built and tested).

---

### H) Collapse Closed-Class Lists into `functional_lexicon.json`

**Brain mechanism:** The brain does not store 6 separate function-word lists — one functional lexicon is learned from distributional statistics.

**Current code:**
- `_COMMON_WORDS` (engine.py:227)
- `TOPIC_SKIP_WORDS` (engine.py:508)
- `_SUBJECT_CONTEXT_WORDS` (engine.py:527)
- `_CONDITIONAL_FRAME` (engine.py:328)
- `_ATTR_WORDS` (engine.py:476)
- `_GRAMMATICAL_CONCEPTS` (engine.py:1023) — already has `functional_lexicon.py` as learned replacement but `use_learned_pos=True` is ON and `_is_function_word` uses PosModel.

Redundancy: These lists overlap each other and `constants.py`'s `STOP_WORDS`, `FUNCTION_WORDS`, `KNOWN_VERBS`.

**Plan:**

1. **Load `functional_lexicon.json` as the single source** (already loaded via `self._func_lex` at engine.py:880-881). Ensure it covers all categories: common/stop words, topic skip, subject context, conditional frames, attribute words, and grammatical concepts.

2. **For each of the 6 lists, add a getter that delegates** to `functional_lexicon.json` when available:
   ```python
   @property
   def TOPIC_SKIP_WORDS(self):
       if self._func_lex and 'topic_skip' in self._func_lex:
           return self._func_lex['topic_skip']
       return self._FALLBACK_TOPIC_SKIP_WORDS  # keep hand set as fallback
   ```
   This pattern means the hand sets become fallbacks only — they are never the primary source when the fit file is present.

3. **Remove duplicate entries** — the hand sets should be trimmed to ONLY contain words that are NOT in `functional_lexicon.json`. Or more safely: keep them as-is but add a gate so they are only consulted when `self._func_lex` is None.

4. **Do NOT delete the hand sets themselves** — per the catalog ("Keep as OOV backstop"). Only add the delegation layer.

**Reused infra:** `functional_lexicon.py` + `_default_lexicon` (engine.py:880), `_func_lex` (engine.py:880), `PosModel` (use_learned_pos ON, engine.py:901).

**Cold-start:** Hand sets remain as-is when `_func_lex` is None (fit file absent). Identical behavior at bootstrap.

**Regression gate:** Must stay 21/1. At bootstrap, `_func_lex` may or may not be present. If absent, behavior is unchanged (hand fallback). If present, the JSON covers the same words plus more.

**Effort:** M. **Risk:** Low (delegation layer is additive; hand sets never removed).

---

## P2 — Important Improvements

### B) Broaden Consult KB / Advice Knowledge (Genuine Capability Gap #2)

**Brain mechanism:** Semantic memory is acquired from the world, not hardcoded. The existing web-learning pipeline builds the graph and definition store across turns.

**Current code:**
- `consult_internal` in brain_regions.py (called at engine_self_query.py:539) — checks `_definitions` dict and graph edges.
- `_definitions` (engine.py:633) — grows via web learning.
- `WebLearningMixin` — web_learning.py accumulates knowledge.
- `_bg_learning_queue` (engine.py:1166) — topics queued for background learning.

**Plan:**

1. **Extend the definition cache to cover health/programming domains** — when the engine does a web search and gets a high-confidence snippet for a health or programming term, store it in `_definitions` with a domain tag. This is already how web learning works — the fix is ensuring the coverage is broad enough.

2. **Add a "consult fallback to web" path** in `_handle_self_query` (engine_self_query.py:538-564): when `consult_internal` returns None for the subject, AND the subject looks like a domain-knowledge term (not a self/identity query), route to a quick web lookup via `_ground_query` + `_web_search_pipeline` instead of immediately returning None.

3. **No new hardcoded advice lists** — the fix is purely about making sure the WEB-to-INTERNAL pipeline covers more ground before giving up.

4. **Use `_bg_multi_search_max` and `_deep_read_max`** (engine.py:1172, 1185) to broaden searches for failed consult subjects.

**Reused infra:** `_definitions` (engine.py:633), `_ground_query`, `_web_search_pipeline`, `bg_learning_queue` (engine.py:1166).

**Cold-start:** No change — at bootstrap, the engine has no knowledge. This fix accelerates knowledge acquisition.

**Regression gate:** Must stay 21/1. The existing test does not test consult quality (it tests consult existence).

**Effort:** M. **Risk:** Medium (must not reduce the bar for honest uncertainty; only add a fallback path for high-confidence consult).

---

### C) Self-Evaluation / Confidence Calibration Tuning (Genuine Capability Gap #3)

**Brain mechanism:** Koriat's accessibility model — FOK is computed from accessibility of partial information, not a fixed threshold. Calibration is learned from prediction-error history.

**Current code:**
- `MetaCognition` (metacognition.py:76, instantiated at engine.py:1069-1072).
- `confidence_calibration_window=15` (engine.py:1071).
- `_calibration_error` (engine.py:1161).
- `MetaCognitiveConfig` (metacognition.py or grace config).
- `_last_quality_score` (engine.py:3005).
- `_assess_response_quality` (engine.py:3001).

**Plan:**

1. **Make `confidence_calibration_window` adaptive**: start at 15 (cold-start). For each `_assess_response_quality` call, compute the rolling std of quality scores. If quality is stable (std < 0.1 over window), increase window up to max 30. If quality is volatile (std > 0.3), shrink window down to min 5. This mirrors the brain's precision-weighting: when the environment is stable, rely on more history; when volatile, rely on recent samples.

2. **Adjust `_calibration_error` integration**: currently tracked at engine.py:1161 but not used to adjust thresholds. In `_consult_internal_knowledge` or response generation, modulate `theta_withhold` (metacognition.py:29) as a function of `_calibration_error`: when `_calibration_error` is high (overconfident), raise `theta_withhold`; when low (underconfident), lower it. Use EMA with the same cold-start prior (0.30).

3. **Code sites**: metacognition.py:29 (`THETA_WITHDHOLD = 0.30`), engine.py:1071 (`confidence_calibration_window=15`), engine.py:1161 (`self._calibration_error`), engine.py:3001-3005 (quality assessment).

**Reused infra:** `MetaCognition` (metacognition.py), `_calibration_error` (engine.py:1161), `_last_quality_score` (engine.py:3005), `_assess_response_quality` (engine.py:3001).

**Cold-start:** Start with existing `window=15`, `theta_withhold=0.30`. These values produce today's behavior. Only adapt after seeing confidence-vs-accuracy data.

**Regression gate:** Must stay 21/1. The existing test (`test_meaning_of_life_not_dict_dump`) may be affected if it tracks self_evaluation output. Check its assertion — it likely checks that the engine doesn't dict-dump, not a specific threshold value. If it checks a specific confidence threshold, update the assertion to allow adaptive range.

**Effort:** S. **Risk:** Low.

---

### E) Adaptive Thresholds (Replace 6 Fixed Cutoffs)

**Brain mechanism:** Friston free-energy / precision-weighting: gates are distribution-driven (EMA z-score), not fixed.

**Current fixed cutoffs (from the catalog):**

| # | Location | Constant | Current value | Cold-start prior |
|---|---|---|---|---|
| 24 | engine_memory.py:651 | `_RECALL_DETECTION_THRESHOLD` | 0.55 | `{"mu": 0.55, "sigma": 0.15, "n": 0}` |
| 37 | engine_generation.py:465-466 | schema cos thresholds | 0.6/0.4/0.5 (pe-driven) | `{"mu_schema": 0.6, "mu_uncertain": 0.4, "mu_mid": 0.5}` |
| 38 | engine_memory.py:425 | episodic cos threshold | 0.5 | `{"mu": 0.5, "sigma": 0.15, "n": 0}` |
| 39 | engine_self_query.py:336 | `_best_sim >= 0.45` | 0.45 | `{"mu": 0.45, "sigma": 0.15, "n": 0}` |
| 40 | engine_generation.py:1698 | `best_sim > 0.75` | 0.75 | `{"mu": 0.75, "sigma": 0.1, "n": 0}` |
| 41 | engine_generation.py:1155-1157 | drift/prune 0.7/0.05/0.1 | 0.7/0.05/0.1 | Drift already tied to SleepConfig |

**Template (copy `_vad_baseline` engine.py:678):**
```python
self._recall_threshold_baseline = {"mu": 0.55, "sigma": 0.15, "n": 0}
# Usage: z = (current_sim - mu) / max(sigma, 0.01)
# if z > 0 or current_sim > mu: pass (same behavior as today at bootstrap)
# After N observations: mu/sigma updated via EMA
```

**Plan for each:**

1. **Create a baseline dict** next to each threshold (or collect them in one `self._adaptive_baselines` dict).

2. **Replace the scalar comparison** with a z-score gate:
   ```python
   # Old: if sim >= self._RECALL_DETECTION_THRESHOLD:
   # New:
   _bl = self._recall_threshold_baseline
   _z = (sim - _bl["mu"]) / max(_bl["sigma"], 0.01)
   if _z > -0.5:  # roughly equivalent to > mu - 0.5*sigma
   ```
   At cold-start: `mu=0.55, sigma=0.15` → the threshold equivalent 0.55 − 0.5*0.15 = 0.475... That's NOT the same as `sim >= 0.55`. To preserve bootstrap behavior exactly: `if _z >= 0` is equivalent to `sim >= 0.55` at cold-start (because `mu=0.55` and `_z >= 0` means `sim >= mu`). So use `if _z >= 0: # gate fires` at bootstrap.

3. **Update mu/sigma via EMA** on each call. When the gate fires and the result is good (response quality > 0.55), fold the current sim into the baseline:
   ```python
   _eta = 0.05  # learning rate
   _bl["mu"] = (1 - _eta) * _bl["mu"] + _eta * sim
   _bl["sigma"] = (1 - _eta) * _bl["sigma"] + _eta * abs(sim - _bl["mu"])
   _bl["n"] += 1
   ```

4. **For the PE-driven schema thresholds** (engine_generation.py:465-466): these already branch on `pe`. Make them adaptive by maintaining separate baselines per PE regime, or use a single baseline where the threshold = f(EMA of within-schema cosines).

**Reused infra:** `_vad_baseline` pattern (engine.py:678).

**Cold-start values:** See table above. Each baseline starts with `mu == current fixed value` so behavior is identical on first run.

**Regression gate:** Must stay 21/1. The existing tests likely exercise recall detection and schema expansion. At bootstrap, adaptive gates behave identically to fixed thresholds. Only after many turns of observation do they diverge.

**Effort:** M. **Risk:** Low-Med (need to ensure the z-score formula at cold-start maps to the same behavior as the fixed threshold).

---

## P3 — Future / Smaller

### D) Sensorimotor Realization Lexicon (Property→Phrase Mappings)

**Brain mechanism:** Lancaster Sensorimotor Norms provide 11-dim strength ratings for ~40k words. Property verbalization is derived from sensorimotor strength, not hand-coded.

**Current code:**
- `_SENSORY_DIM_PHRASE` (engine.py:290-299): hand map property→(verb, sensory-phrase).
- `_PROP_TO_BINDER` (engine.py:317-327): hand map property→sensory dims.
- `_LANCASTER_ORDER` (engine.py:286): fixed encoder contract — keep.
- Used in engine_generation.py:1130-1141 (`_select_sensorimotor_dim`).

**Plan:**

1. **Build a fit script** (like `experiments/measure_pos_model.py`) that:
   - Loads Lancaster 40k norms (from `data/cache/lancaster_norms.csv` or download from OSF).
   - For each property word in the union of `_SENSORY_DIM_PHRASE` keys and common English property adjectives ("smooth", "loud", "heavy", etc.), compute the top-2 sensorimotor dimensions.
   - Fit a property→phrase map: for each property, select the verb based on the dominant action effector (Hand_arm → "feel"/"hold", Mouth → "taste", etc.) and the phrase based on the dominant perceptual modality.
   - Save as `data/sensorimotor_lexicon.json`.

2. **Write a loader** (like `functional_lexicon.py`): `def default_sensorimotor_lexicon()` → loads from `data/sensorimotor_lexicon.json`, returns dict or None.

3. **Wire into engine** (around engine.py:880 area where `_func_lex` is loaded): add `self._sensorimotor_lex = ...` and use it to replace `_SENSORY_DIM_PHRASE` and `_PROP_TO_BINDER` lookups.

4. **Keep hand maps as fallback** (OOV safety net, identical pattern to all other de-hardcoded elements).

**Reused infra:** `functional_lexicon.py` loader pattern, `_LANCASTER_ORDER` as encoder contract, `_lancaster_vector` (engine.py:1156) already computes sensorimotor projections.

**Cold-start:** Hand maps remain when fit file is absent.

**Regression gate:** Must stay 21/1. Hand maps are preserved as fallback.

**Effort:** L. **Risk:** Low.

---

### F) Shape-Based Junk Detection (Replace `_JUNK_SNIPPET_DOMAINS`)

**Brain mechanism:** Predictive coding — boilerplate is detected by its statistical shape (high vowel ratio, low information density, TLD suffixes), not by a domain blocklist.

**Current code:**
- `_JUNK_SNIPPET_DOMAINS` (engine.py:344) — hardcoded 45-domain blocklist.
- `_WEBSITE_SHAPE` (constants.py:401) — regex for TLD tails.
- `junk_score` (constants.py:407, delegates to `junk_scorer.py`) — already combines shape + learned signals.
- `junk_scorer.py` — self-supervised classifier with shape features.
- `snippet_quality.py` — contrastive snippet PE model.

**Plan:**

1. **Add shape features to `junk_scorer.py`** (around line 76 where `_WEBSITE_SHAPE` is used): add features for vowel ratio (count vowels / count chars), digit density, uppercase ratio, POS diversity. These features already exist conceptually in `_WEBSITE_SHAPE` and `_is_keyboard_mash` (constants.py:53).

2. **Wire shape features into the snippet-quality path** (engine_web_search.py snippet processing, around the `_rate_snippet_quality` or equivalent method). Use `junk_score` from constants.py:407 which already combines shape + learned signals.

3. **Keep `_JUNK_SNIPPET_DOMAINS` as optional cold-start prior** — the catalog says keep as backstop. Add a flag `use_junk_domain_blocklist` that defaults to True but is deprecated. The shape path runs FIRST; the domain list is only checked when shape is uncertain.

4. **No new domain entries** — the fix extends shape detection to replace the domain list, not add to it.

**Reused infra:** `_WEBSITE_SHAPE` (constants.py:401), `junk_score` (constants.py:407), `junk_scorer.py`, `snippet_quality.py`, `_is_keyboard_mash` (constants.py:53).

**Cold-start:** Domain list kept. Shape detection is additive — it runs alongside.

**Regression gate:** Must stay 21/1. No snippet tests check specific domain blocking.

**Effort:** M. **Risk:** Low.

---

## "Do NOT Touch" List (inherited from catalog)

| Element | Reason |
|---------|--------|
| `SAVE_SCHEMA_VERSION` (#29, engine.py:590) | Schema marker |
| `MAX_DECODER_VOCAB_SIZE` (#2, engine.py:226) | Capacity hyperparam |
| `QUESTION_WORDS` (#21, engine.py:485) | Genuine closed class |
| `FOLLOW_UP_WORDS` (#22, engine.py:488) | Genuine closed class |
| `_UNIVERSAL_PURGE` set contents (#35) | Intentionally minimal (only deduplicate definition, not words) |
| `_DEFINITION_ASSERTION` regex pattern | Correct as-is |
| `_PROTECTED_CONCEPTS` (#31, engine.py:643) | Curated project namespace |
| `_LANCASTER_ORDER` (#7, engine.py:286) | Fixed encoder contract |
| `_vad_baseline` (#34, engine.py:678) | **Exemplar pattern** — KEEP and copy |
| `_CATEGORY_OF_SUBJECT` / `_CATEGORY_AFFORDANCES` / `_PROPERTY_CATEGORIES` (#4-6) | OOV safety net behind ConceptNet |
| `_SNIPPET_REJECT_SHAPES` / `_SNIPPET_NOISE` (#13-14) | Hard backstop behind cerebellar model |
| VAD/Identity/GW/Sleep eta constants (#33) | Model config, not knowledge |

---

## Verification Section

### Command
```bash
python -m pytest tests/test_dehardcode_plan.py -q
```

### Expected outcome after each planned change

| Change | Expected | Notes |
|--------|----------|-------|
| Baseline (current HEAD) | **21 passed, 1 failed** | Pre-existing `test_meaning_of_life_not_dict_dump` |
| A) Working memory buffer | **21 passed, 1 failed** | New capability; existing tests don't test same-turn recall |
| I) Deduplicate `_UNIVERSAL_PURGE` / `_DEFINITION_ASSERTION` | **21 passed, 1 failed** | Pure import refactor |
| G) Wire ConnectorLearner | **21 passed, 1 failed** | Cold-start behavior identical; learner adds no degradation |
| H) Collapse closed-class lists | **21 passed, 1 failed** | Delegation layer; fallback preserves behavior |
| B) Broaden consult KB | **21 passed, 1 failed** | No existing test tests consult quality |
| C) Calibration tuning | **21 passed, 1 failed** | Cold-start preserves 0.30 theta_withhold |
| E) Adaptive thresholds (each) | **21 passed, 1 failed** | Each adaptive gate at cold-start equals current fixed value |
| D) Sensorimotor lexicon | **21 passed, 1 failed** | Hand maps preserved as fallback |
| F) Shape junk detection | **21 passed, 1 failed** | Additive; domain list kept as cold-start prior |

### Testing note
The single failure (`test_meaning_of_life_not_dict_dump`) is PRE-EXISTING on HEAD. If any change causes ADDITIONAL failures, revert that change immediately. If a change requires updating a test assertion (e.g., confidence calibration might change a specific self_evaluation value), update the assertion to match the new adaptive behavior and document the exact change.

---

## File Paths of Deliverables

1. `C:\Users\Likhith\Documents\projects\ravana\reports\brain_research_report.md` — brain mechanism research with citations
2. `C:\Users\Likhith\Documents\projects\ravana\reports\brain_fix_plan.md` — this implementation plan
