# Usage-selection codemap — where construction choice is made and how to learn it

**Scope:** read-only investigation at commit `7e55cda` (tree clean). No code changed,
no `eng.save()` called. This maps the machinery that already exists so that
cue-to-frame associations can be learned from usage instead of being hardcoded.
It is the empirical companion to `docs/CUE_SELECTION_CODEMAP.md`,
`docs/BRAIN_CUE_SENSITIVE_CONSTRUCTIONS.md` (CVCC) and
`docs/BRAIN_CROSS_FRAME_PRODUCTIVITY.md` (SCC-RF). Those proposals describe a
scored competition over the construction inventory; this document describes the
code that is actually on disk *now* and the narrowest edit that turns the fixed
cue mapping into a learned table.

**Method note (per `ravana-debugging` skill 1):** no runtime score is reported —
a cold engine boot is ~200 s and this task forbids code changes, so no probe was
launched. Every claim below is a file:line citation to the current source, and
every identifier/call site was cross-checked with ripgrep across `ravana/`.
Where a proposal exists in the brain docs it is labelled as a proposal, not as
measured behaviour.

---

## 0. The path as it exists now (baseline to preserve)

| Item | File:line | Note |
|---|---|---|
| `extract_construction_cue` | `response_gen.py:205-225` | module-level cue reader. Regex cascade: `PART_WHOLE` (`:214-216`), `LOCATIVE` (`:217-219`), `FUNCTION` (`:220-222`), `PROPERTY` (`:223-224`), else `NONE` (`:225`). This is the "fixed cue mapping" named in the task. |
| `find_taught_term` | `response_gen.py:187-202` | longest lexical match of a taught term in the raw query; used only for taught-word cue questions. |
| `_PRODUCTIVE_USE_RE` / `is_productive_use_request` | `response_gen.py:180-184` / `:228-230` | directive detector, single source of intent truth. |
| `_is_productive_cue_request` | `response_gen.py:7521-7536` | gate: directive always true (`:7532-7533`); cue question only if a taught term equals the subject (`:7534-7536`). |
| `_productive_use_response` | `response_gen.py:7538-7618` | the emitter. Full detail below. |
| call site | `response_gen.py:8798-8804` | inside `_generate_response`; gated by the `_definitions` lookup at `:8791-8792`; cue subject resolution at `:8783-8788`; `try/except: pass` fallback at `:8803-8804`. |

Inside `_productive_use_response`:

* intent gate `:7552-7554`; empty input gate `:7555-7556`.
* definition decomposed by a relative-clause regex `:7559-7570`; failure returns
  `None` (fail-closed, `:7563-7567`) and the caller falls through to the definition.
* cue read once at `:7574`; head noun / locative PP / adjectives read at
  `:7575-7587`.
* **the fixed mapping** is the `if/elif` chain at `:7589-7600`:
  `FUNCTION → f"{subject} {predicate}."`; `PART_WHOLE → f"{subject} is part of a {_head}."`;
  `LOCATIVE → f"{subject} is found {_loc}."`; `PROPERTY → f"{subject} is {' and '.join(_adjs)}."`.
* **the repeat counter tie-break** is `:7600-7611`: on `NONE` or an inadmissible
  cue, `_idx = int(_counts.get(subject.lower(), 0)) % 2` (`:7606`), store
  `_idx + 1` (`:7607`), frame 1 is the appositive `A {head} called {subject} {predicate}.`
  (`:7608-7609`), frame 0 the bare predication (`:7610-7611`).
* echo/length gate `:7612-7617` (definition not substring of output and vice
  versa; `>= 4` words); result `(Capitalized, "productive_use")` at `:7618`.

**The single state this path touches** is `self._productive_use_counts`, created
and read/written only at `response_gen.py:7603-7607`. Ripgrep finds no other
occurrence: it is an in-memory engine attribute, absent from every
`get_state`/`set_state` blob, so it does not survive save/load. No RNG is used.

**Critical reachability fact:** this emitter builds its string with f-strings
only. It never calls `SyntacticCellAssembly.bind_to_sentence`, never sets
`syntactic_assembly._last_frame`, never calls `SurfaceRealizer.realize` or
`_apply_variant`, never sets `surface_realizer._last_variant_name`. Therefore
**none** of the construction-learning or realizer-learning state in §1–§2 is
currently written or read by the productive-use path. This is the gap the
learned association must close.

---

## 1. `SyntacticCellAssembly` — the existing construction store

File: `ravana/src/ravana/language/syntactic_cell_assembly.py`.

### Data model
* `SyntacticFrame` dataclass `:23-43`: `subject_concept` `:26`, `verb_concept`
  `:27`, `object_concept` `:28`, `relation_type` `:29`, `verb_phrase` `:32`,
  `weight: float = 0.1` `:40`, `embedded_frame` `:41`.
* `SyntacticCellAssembly.__init__` `:99-118`: `learning_rate = 0.05` `:99-100`,
  `_last_frame` `:101`, `learned_frames: List[SyntacticFrame] = []` `:107`,
  `_max_learned_frames = 512` `:108`. Role/sequence priors `:110-118`.

### How a `(subject, relation, target)` frame key is formed
* `_frame_key(frame)` `:577-579` returns the 4-tuple
  `(frame.subject_concept.lower(), frame.verb_concept.lower(),
  frame.object_concept.lower(), frame.relation_type)` — all three concepts
  lowercased, `relation_type` kept verbatim. **There is no cue field.**
* `find_construction(subject, relation, target)` `:581-589` computes
  `key = (subject.lower(), "", target.lower(), relation)` at `:583` but then
  **does not use `key`**: the loop at `:584-588` compares `subject_concept`,
  `object_concept` and `relation_type` directly and ignores `verb_concept`
  entirely. So the effective retrieval key is `(subject, object, relation)`;
  the `""` verb placeholder at `:583` is dead code. This matters: retrieval
  (`bind_to_sentence`'s M16 reuse, `:232-241`) is verb-blind, while allocation
  and reinforcement are verb-sensitive (next paragraph).
* `bind_to_sentence(subject, relation, target, ...)` `:207-287`: M16 reuse at
  `:232-241` returns a shallow copy of a learned frame if `find_construction`
  hits; otherwise it picks `verb_concept` (`:246`, `_pick_verb_for_relation`
  `:289+`) and `verb_phrase` (`:249`), builds a fresh `SyntacticFrame` at
  `:270-285` and sets `_last_frame` at `:286`.

### How weight changes
* `allocate_construction(frame, weight=0.1)` `:591-614`: clamps the *new* frame's
  `weight` to `[0, 2]` (`:598`); if `_frame_key` matches an existing frame,
  reinforces it by `+learning_rate` (`:600-604`, `existing.weight = min(2.0,
  existing.weight + 0.05)`), sets `_last_frame`, returns `True`, appends nothing;
  otherwise, at capacity `:606-611`, evicts the lowest-weight entry by
  `key=lambda i: self.learned_frames[i].weight`, then appends (`:612-613`).
* `reinforce_construction(frame, success)` `:616-621`: `lr = +0.05 if success
  else -0.025` (`:618`); for every frame with equal `_frame_key`, `weight =
  clamp(weight + lr, 0, 2)` (`:619-621`).
* `learn_from_feedback(frame, user_understood)` `:531-572` is a *different*
  update: `lr = +0.05 / -0.025` (`:538`) applied to the role matrices
  `subject_role`/`object_role`/`verb_role` (`:544-549`), `sequence_patterns`
  (`:551-557`), `verb_phrase_counts` (`:559-562`) and `pattern_weights`
  (`:568-572`). **It does not touch `learned_frames[].weight`.** The end-of-turn
  hook calls this method, not `reinforce_construction` — so today the inventory
  frame weights are never reinforced online.

### Persistence
* `get_state()` `:735-745` serializes `learned_frames` with `asdict` at `:743`
  and `max_learned_frames` at `:744`.
* `set_state(state)` `:747-768`: `learned_frames` restored at `:758-768`,
  filtering to real dataclass fields (`:764-765`), swallowing bad entries
  (`:766-767`), truncating to the last `_max_learned_frames` (`:768`).
* Engine wiring: saved into the pickle blob as `'syntactic_assembly_state'` at
  `engine.py:12973`; restored on cold load at `engine.py:13703-13706`
  (`if sa_state and hasattr(self, 'syntactic_assembly'): self.syntactic_assembly.set_state(sa_state)`).

**Conclusion for §1:** this is the durable, already-persisted home for learned
frame strength. Its key has no cue dimension and `find_construction` ignores the
verb, so a cue→frame table should live as a *side dict inside the same state
blob*, not as a fifth component of `_frame_key` (which would force changes to
every allocation/reinforcement/retrieval caller).

---

## 2. `SurfaceRealizer` — surface-variant learning, and what is reachable

File: `ravana/src/ravana/language/surface_realizer.py`.

| Item | File:line | What it is | Reachable from productive-use path? |
|---|---|---|---|
| `MERGE_VARIANTS` | `:153-164` | 10 named merge orders: `svo`, `left_dislocation`, `pseudo_cleft`, `topic_fronting_for`, `existential`, `cleft`, `as_for_topic`, `adverb_fronted`, `svo_emphatic`, `svo_causal`. | **No.** These are word-order variants, not constructional relations; the productive-use path never calls `realize`. |
| `_variant_weights` | `:180` (init `{v: 1.0}`) | per-variant learned preference in `[0.1, 2.0]`. | No — read only by `_weighted_variant_select` `:793`. |
| `_last_variant_name` / `_recent_variants` | `:181-182` (`_recent_variants` capped at 3 at `:618-621`); set at `:522` | STN fatigue history. | No — set only inside the realizer. |
| `learn_from_feedback(variant_name, success)` | `:213-217` | `lr = +0.05 / -0.025`, clamps `_variant_weights[variant_name]` to `[0.1, 2.0]`. | No call from this path. Called from `engine.py:11743-11745` for the *normal* generation path. |
| `_weighted_variant_select` | `:789-809` | combines base weights, `_variant_weights`, dopamine, fatigue and FE, then **`random.choices(...)` at `:809`**. | Not usable as a deterministic selector (RNG). |
| `_apply_variant(variant, ...)` | `:811+` | applies a named merge order deterministically. | The deterministic precedent: callers that pass an explicit name bypass the sampler. |
| `_select_verb_phrase` | `:1182-1204` | cerebellar phrase prediction then `VerbLexicon.select_verb` fallback. | No. |

**Answer to the reachability question:** none of `learn_from_feedback`,
`_variant_weights`, `_recent_variants` or `_apply_variant` is reachable from
`_productive_use_response` today. They are live only for responses that go
through `SurfaceRealizer.realize` (`:351+`). They are still useful as *design
precedents*: `learn_from_feedback` is the exact `+lr / -lr*0.5` clamp idiom to
mirror, `_recent_variants` is the recency/fatigue idea to demote to a tie-break,
and `_apply_variant(name, ...)` proves that explicit-name selection is
deterministic whereas `_weighted_variant_select` is not (CVCC/SCC-RF both say
"reuse the fatigue idea, not the sampler").

---

## 3. Feedback and correction signals already in the engine

| Signal | File:line | What it carries | Fit for reinforcing a chosen frame |
|---|---|---|---|
| `quality_score = self._assess_response_quality(response, strategy, ctx)` | `engine.py:11652` | per-turn scalar quality of the emitted response; persisted as `self._last_quality_score` `:11656`; folded into `_calib_buf` `:11663-11667` and `calibrator.observe(confidence, quality_score)` `:11693`. | **Cleanest online signal.** Fires every turn, is already the reinforcement gate for both learning paths, and needs no new plumbing. |
| `user_understood = (quality_score >= 0.55)` and the two feedback calls | `engine.py:11738-11745` | boolean derived from that score; drives `syntactic_assembly.learn_from_feedback(_last_frame, user_understood=...)` `:11740-11742` and `surface_realizer.learn_from_feedback(_last_variant_name, success=...)` `:11743-11745`. | This is the exact junction at which a `cue→frame` reinforcement call belongs. |
| `_identity_end_of_turn(..., quality_score)` | `engine.py:12441-12458` | reuses the same threshold `float(quality_score >= 0.55)` as `_resolution_success` for identity update. | Same scalar, different consumer; confirms `0.55` is the house threshold. |
| user_model correction flags | `user_model.py:1856-1897, 2189-2258, 6843-6970` (`detected_correction`, `correction_severity`, `detected_correction_type`, `correction_subject`); reset at `:6968-6972` | **explicit** user correction, with typed severity. | Sparse (only on explicit correction) and it *replaces* the response via `_detect_and_handle_correction`; it is a valuable negative signal but not a per-turn online reward. |
| `_detect_and_handle_correction` → `wire_feedback` | `engine_persistence.py:577-621`; `recall_ranker.py:381-392` | correction type+severity wired to `RecallRanker` threshold adaptation (MRS), *not* to construction weights. | Existing online-learning precedent, but the wrong target store for frames. |
| `_log_monitor_fire(monitor, dropped_clause, reason)` | `response_gen.py:9908-9922`; fired e.g. `engine_generation.py:967`, `chain_walker.py:2355` | structured `{ts, path, monitor, dropped_clause, reason}` record in `_monitor_log`. | Logging only — no numeric reward to back-propagate. |
| `register.adapt_from_feedback(reward, lr)` | `language/register.py:201-209` | knobs pushed toward/away from current register by a reward. | Register-style only; not wired to construction frames. |

**Recommendation:** use `user_understood` (= `quality_score >= 0.55`) at
`engine.py:11739`, already consumed at `:11740-11745`, as the single online
reinforcement signal for the chosen `(cue, word) → frame` association. It is
already computed for the productive-use response (the emitter returns the
`"productive_use"` strategy at `response_gen.py:7618`, and `quality_score` takes
`strategy` as an argument at `engine.py:11652`). Keep the explicit correction
flags as a stronger negative update when `detected_correction` is set, but do
not make frame learning depend on corrections — they are too sparse to shape an
association. Do **not** repurpose `_log_monitor_fire` (no reward) or
`wire_feedback` (wrong store).

---

## 4. Narrowest correct change — learned `cue × word → frame` table

**One insertion point:** `response_gen.py:7574-7611` (the cue read and the fixed
`if/elif` chain). Everything else is a definition or an unchanged gate.

1. **Keep byte-identical:** intent gate `:7552-7556`; definition + predicate
   extraction `:7557-7570`; the four frame *builders* at `:7590-7599` (they
   become candidate factories, their strings do not change); the counter
   fallback `:7600-7611`; the echo/length gate `:7612-7617`; the caller
   `:8791-8804` (the `_definitions` gate at `:8791-8792` is what fails closed for
   untaught words, and the `try/except` at `:8803-8804` is what degrades safely);
   and `extract_construction_cue` itself `:205-225` (it stays the cue reader, so
   cue extraction is not part of the change).

2. **Add a learned table** `cue_frame_weights: Dict[(cue, word), Dict[frame_id, float]]`
   as a *side dict in `SyntacticCellAssembly` state* (§5), seeded empty. The
   candidate frame ids are exactly the five branches already present:
   `FUNCTION`, `PART_WHOLE`, `LOCATIVE`, `PROPERTY`, and `NONE_COUNTER`. Reuse
   the existing `learn_from_feedback` clamp idiom (`+0.05 / -0.025`, bounds
   `[0.1, 2.0]`, `surface_realizer.py:213-217`) for updates.

3. **Replace only the selection rule** at `:7589`: compute the frame by a
   deterministic `select_frame(cue, word)` that (a) reads the weights for
   `(cue, word)`; (b) picks `argmax` over admissible frames; (c) breaks ties by
   a stable order `FUNCTION, PART_WHOLE, LOCATIVE, PROPERTY, NONE_COUNTER`;
   and (d) uses the same *admissibility* tests already in the `if/elif`
   (`_head` not a stopword for `PART_WHOLE`, `_loc` non-empty for `LOCATIVE`,
   `_adjs` non-empty for `PROPERTY`). **No `random.choice`/`random.choices`:**
   the `surface_realizer.py:809` sampler would break reproducibility. Pass the
   chosen id to the unchanged frame builders at `:7590-7611`.

4. **Backward compatibility (load-bearing):** the table starts empty. Define
   `select_frame` so that an **absent** `(cue, word)` entry returns the *built-in*
   preferred frame for that cue — the same frame the `if/elif` chain returns
   today — and `NONE` returns the counter frame. With an empty table the four
   cue answers and the `NONE_COUNTER` alternation are byte-identical to current
   output. Only after a reinforcement writes a non-empty entry can the learned
   weights override the prior. Equivalently: the built-in mapping is the prior
   mean of an empty cell, not a separate hardcoded switch.

5. **Reinforcement, using §3's signal:** inside `_productive_use_response`,
   record `self._last_cue_frame = (cue, word, frame_id)` on acceptance (just
   before `:7618`). At `engine.py:11738-11745`, next to the two existing
   feedback calls, call
   `syntactic_assembly.reinforce_cue_frame(*self._last_cue_frame, success=user_understood)`
   and clear it. `+0.05` on success, `-0.025` on failure, clamped; the update is
   exactly the existing clamp idiom and needs no new reward.

6. **Fail-closed and gates:** keep `_is_productive_cue_request`'s taught-term
   check `:7521-7536`, the caller's `_definitions` gate `:8791-8792`, and the
   echo gate `:7612-7617` unchanged. If a cued frame fails the echo gate, rank
   the next admissible frame (do not relax the gate); if none survives, return
   `None` and let the caller fall through to the definition. The change touches
   no grounding gate (`_sm_response_grounded`, `_is_word_salad`) and no directive
   normalizer (`engine_generation.py:2083-2122`).

**Why narrowest:** one function's selection line changes; no new store, no new
router, no RNG, no change to intent/definition/echo gates, and the existing
five frame builders plus the existing clamp idiom are reused. It satisfies
deterministic selection, fail-closed behaviour for untaught words, and identical
pre-learning output.

---

## 5. Persistence — which store, and cold-load safety

There are three durable weight stores and they drift (`ravana-debugging` skill §3):

| Store | Evidence | Role |
|---|---|---|
| `weights/ravana_weights*.pkl` | path set `engine.py:1080-1092` (`_save_path`); written by `_safe_pickle_dump(state, self._save_path)` at `:13087`, with a `.sha` checksum at `:13089-13096`; read by `ravana_unpickler` at `:13126-13127`. | **Authoritative on boot.** The only blob the engine actually restores into live objects. |
| `weights/ravana_weights*.db` (SQLite) | `_db_path = self._save_path.replace(".pkl", ".db")` `engine.py:1624`; `db.save_metadata('definitions', ...)` at `:13064-13065`. | Mirror / offline metadata fallback. Editing it alone is silently overwritten by the next `eng.save()`. |
| `user_models/` | saved to its own directory at `engine.py:13098+` (separate from the weight pkl). | User profile / correction flags; independent lifecycle. |

**Where the new table lives:** inside the `syntactic_assembly_state` sub-blob of
the pickle — add `'cue_frame_weights'` to `SyntacticFrameAssembly.get_state()`
(`syntactic_cell_assembly.py:735-745`) and read it defensively in `set_state()`
(`:747-768`). It therefore travels on the single authoritative path
`eng.save() → pkl → cold load → set_state → engine.py:13703-13706`; it is *not*
duplicated into the SQLite mirror or `user_models/`, so there is no second copy
to drift.

**Cold-load non-clobber rules:**
* `set_state` must use `state.get('cue_frame_weights', {})` so an older pkl
  (no key) restores an empty table rather than raising — preserving backward
  compatibility and the fixed mapping.
* Follow the `sequence_patterns` precedent (`get_state` `:741` stringifies keys,
  `set_state` `:751-756` re-parses them): stringify the `(cue, word)` tuple key
  on write and split it on read, so the table survives any serialization that
  routes through the SQLite/JSON-style path instead of raw pickle tuples.
* Coerce values to `float`, skip malformed entries in `try/except` (the pattern
  at `:763-767`), and do not truncate against `_max_learned_frames` — that cap
  applies only to `learned_frames` at `:768`.
* Never hand-edit pkl or db separately; repair by loading, calling `eng.save()`
  to rewrite the authoritative snapshot, and verifying with a **cold**
  `CognitiveChatEngine` load (skill §3). Never run two engines concurrently —
  last writer wins and the table is lost.

---

## Insertion-point one-liner

`ravana/src/ravana/chat/response_gen.py:7589` — replace the fixed
`if _cue == "FUNCTION": ... elif ...` selector with
`select_frame(cue, subject)` over an initially empty, persisted
`cue_frame_weights` table, defaulting to today's built-in frame per cue and to
the `:7603-7607` counter for `NONE`; reinforce the chosen `(cue, word, frame)`
at `engine.py:11738-11745` with `user_understood`. Leave `:7552-7587`,
`:7590-7611`, `:7612-7617` and `:8791-8804` unchanged.
