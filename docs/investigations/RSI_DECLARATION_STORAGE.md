# RSI declaration storage: root cause is RETRIEVAL, not storage (2026-11-23)

Status: **static code + web research; no engine process, no pytest** (parent runs
both). This note supersedes the storage-missing hypothesis in
`RSI_LIVE_RULE_INDUCTION.md` §5 in light of the parent's direct live census.

## 0. Corrected headline — storage is PRESENT

The parent's `scratch/rsi_declaration_probe.log` census found, after **two**
`process_turn` calls:

- `('gale','rainbow',causal,curated,source='conversation')`
- `('mist','thaw',causal,curated,source='conversation')`

both with live nodes indexed in `_concept_keywords`. The subsequent query
`"Considering the examples above, what is true of thaw? Options: sunset,
typhoon, mist, hail. Answer briefly."` returned
`"honestly, hail is a bit outside what i know right now..."` — an **option
token** became the subject, and the just-taught trusted edge was never used.

**Therefore: the missing step is not writing the relation. It is (a) resolving
the query subject from the question rather than from the option list, and
(b) admitting an already-trusted taught edge as evidence at the answer gate.**
Editing write provenance further cannot fix this. The fix belongs on the
retrieval/articulation side.

### The writer is identifiable statically

`chain_walker.py:878-930` `_consolidate_asserted_causal_facts(text)`:

- `chain_walker.py:887-891` — refuses interrogatives, extracts asserted causal
  propositions, else returns 0.
- `chain_walker.py:903-916` — builds `Fact(subject=cause, relation="causes",
  obj=effect, confidence=0.9, source_sentence=text)` and
  `Provenance(source="conversation", source_type="user", url="",
  extractor="conversational-assertion", operations=("taught-in-conversation",))`,
  then calls `intake.intake(fact, prov, salience=1.0)`.
- `intake.py:80` maps `source_type="user"` -> edge kind **`curated`**; the edge
  therefore carries `edge_kind='curated'` and `source='conversation'` — an
  exact match to the parent's census.
- `chain_walker.py:917-918` counts `fast_merge`/`novel_write`/`revise` as writes.
- `engine.py:10660-10668` calls it on every turn (Phase 1.6a), wrapped in a
  best-effort `try/except`.

So the trusted write occurs at *generation-adjacent* consolidation, not in the
assertion mirror. This is why an earlier static-only reading could miss it:

- `process_turn` entry `engine.py:8049` -> `_process_turn_impl` `engine.py:8097`
  -> `engine.py:8200`.
- Speech-act classifier defaults a bare declarative with no first/third-person
  marker to **`"question"`** (`prefrontal_workspace.py:1260-1261`); hence
  `_handle_assertion` returns `None` at `response_gen.py:2813` and the
  assertion mirror at `engine.py:11326-11327` is skipped. That skip does NOT
  imply no write: `engine.py:10666` is reached later in the same turn.
- `detect_question_type("gale causes rainbow")` -> `("general", [text])` by the
  regex default (`prefrontal_workspace.py:1188-1189`); the dual-process route is
  chosen at `engine.py:11384`.
- `WebToGraph.learn_text` (`engine_web_search.py:736`, `web_learning.py:756`,
  `web_learning.py:1304`) and `ConceptGraph.add_edge` (`ravana_ml/.../graph.py:1694`)
  are NOT the writers here; the write is through `KnowledgeIntake.intake`.

**Correction owned:** my earlier claim (and the earlier note's §5) that the
teaching assertion "never stores a trace" was wrong in its conclusion. The
correct statements are: the assertion mirror is skipped, a hedge is emitted,
BUT a provenance-trusted edge is still written by the consolidation path.

## 1. Exact trace of the query turn — why an option wins

Harness constructs the question at `scripts/measure_learn5_rsi.py:544-548` and
grades only that response at `:555`. No option-masking is applied to the string
before it reaches `process_turn`.

### 1.1 Subject resolution

1. `engine.py:10788` — `subject, obj = self._extract_topic(user_input, activated)`.
2. `engine_generation.py:2505` `_extract_topic` -> `_ground_query(text)` at `:2514`.
3. `engine_generation.py:2029` `_ground_query`. The PFC qtype is recomputed at
   `:2050`. Because the prompt begins with `"Considering the examples above,
   ..."` (not a sentence-initial `what`), the anchored `what_is` regex
   (`prefrontal_workspace.py:933-935`, `pattern.match`) does not fire; the qtype
   falls to `general`. Exact-label/exact-keyword short-circuits at
   `engine_generation.py:2139-2142` therefore do not pin `thaw`.
4. `engine_generation.py:2186-2191` builds `words` from the WHOLE prompt. The
   option tokens `sunset`, `typhoon`, `mist`, `hail` survive the filters —
   `QUESTION_WORDS`/`TOPIC_SKIP_WORDS`/`_GENERIC_NOUNS` are at `engine.py:776-811`
   and `STOP_WORDS` at `constants.py:438,525` — and none of them is an option
   list. `briefly` is stripped as a trailing modifier (`functional_lexicon.json`
   trailing modifiers), but the options are not.
5. `engine_generation.py:2204-2209` strips trailing modifiers; `:2211-2214`
   then takes `clean_subj = " ".join(words[-3:])` — **the last content words of
   the prompt, i.e. option tokens**.
6. `engine_generation.py:2249` returns that phrase with confidence `0.45` and
   method `"multi_word_unconnected"`. Crucially, the compositional branch at
   `:2251-2267` (which would pick a known/taught word via `known_words`) is
   **unreachable whenever `words` is non-empty** — the `if words:` block at
   `:2198` already returned at `:2249`. That branch is dead code for every real
   query. This is the structural reason a taught node cannot win by familiarity.
7. `engine_generation.py:2515-2519` — `_extract_topic` accepts that phrase:
   confidence `0.45 < 0.5`, but `"multi_word_unconnected"` is **not** in the
   exclusion tuple at `:2518`. The option-derived phrase becomes `ctx.subject`
   at `engine.py:11667-11668`.
8. `response_gen.py:7416-7430` reduces the multi-token subject to a single word;
   `response_gen.py:7436-7440` can further replace it with
   `find_vector_neighbor(subject)`, which returns the highest-cosine graph node
   (`graph/engine.py:1058-1078`, `best_sim=0.45`) — a taught option node is a
   valid hit. The emitted anchor is the option (`hail`), not the interrogated
   subject (`thaw`).

**Verified structurally:** option tokens reach subject resolution and a low-
confidence last-words phrase is accepted as the subject. **NOT verified without
an engine:** which exact step selected `hail` rather than `typhoon`/`mist`
(token-length reduction at `response_gen.py:7426`, `_clean_subject_phrase` at
`engine_graph.py:1518-1544`, or the vector-neighbour fallback at
`response_gen.py:7436-7440`). Parent can settle it in one process by reading the
existing trace prints at `engine_generation.py:2038,2051,2197` and
`self._last_grounding_method` set at `engine.py:10843`.

### 1.2 Candidate evidence and the gate that vetoes it

- A forced-choice decision surface exists but is default-off / fail-closed:
  `use_triplet_candidate` (`engine.py:7506`, `_triplet_mc_answer` `:7526`) and
  `use_deductive_candidate` (`engine.py:7516`, `_deductive_mc_answer`).
  `_triplet_mc_answer` calls `_split_options` (`fact_reasoning.py:73-104`) and
  returns `None` when fewer than two options are parsed (`engine.py:7550-7551`).
  `_split_options` splits letter markers, but its fallback is `[;\n]` **only**
  (`fact_reasoning.py:101-103`), so comma-separated options arrive as ONE item
  and the path abstains. Neither `_ground_query` nor `_extract_topic` calls
  `_split_options` at all.
- Chain assembly over trusted edges does exist — `enumerate_chains`
  (`explanation.py:156`), `_hop_provenance` (`:295-323`), `provenance_trusted`
  (`:326-340`), `assertable_chain` (`:343-352`), `grade_chain` (`:355-390`) —
  but it is reached for `why`/`how` explanation queries, not for a
  `"what is true of X? Options: ..."` prompt.
- The terminal veto is the metacognitive response-conflict monitor at
  `response_gen.py:9021-9047`: `grounded = self._topic_grounded(ctx)`;
  `if not grounded: ... return self._human_like_uncertainty(ctx)`.
  `_topic_grounded` (`response_gen.py:4735-4780`) recognises a stored
  definition (`:4752`), a concept source (`:4754`), and GloVe similarity
  (`:4757-4780`, `best >= 0.45`). **It never consults an incident
  provenance-trusted edge.** A taught `mist -> thaw` edge with
  `edge_kind='curated'` and `source='conversation'` is therefore invisible to
  the gate; the hedge fires and no candidate token is ever emitted. This is
  the retrieval-side counterpart of the storage bug: the edge is written and
  trusted on disk, but not admitted as grounding at answer time.
- Hedge templates: `response_gen.py:7456-7467` (`"honestly, {subj} is a bit
  outside what i know right now."`, etc.) — the exact family in the census.
- `_final_emit_guard` exemption pattern (`engine.py:12903-12926`) already shows
  the gate-preserving idiom: a byte-exact class of provenance-verified content
  is exempted from a censor re-run; thresholds are untouched. Use the same
  idiom for trusted taught edges.

## 2. Minimal gate-preserving hook (design only — files owned by parent)

All three hooks are on the READ side. No write path, threshold, or gate value
changes: `THETA_ASSERT=0.70`, `THETA_HEDGE=0.40` (`explanation.py:47-48`),
`fast_merge_threshold=0.55` (`intake.py:305`), and the `0.45` GloVe bar
(`response_gen.py:4780`) stay exactly as they are.

**H1 — pin the subject from the question, not the options.** At the
`engine.py:10788` call site, when a forced-choice frame is detected, ground
`subject` from the text *before* the `Options:` marker (a structural regex,
already partially present as `fact_reasoning._split_options`). Pass only that
text to `_extract_topic`. Expected effect: `ctx.subject = "thaw"`, not an
option. This does not change any decision gate; it fixes the input to the gate.

**H2 — admit a trusted taught edge as grounding evidence.** In
`_topic_grounded` (`response_gen.py:4735`), add one early-return branch: if the
subject has an incident edge with `edge_kind` in the already-trusted set
(`web_fact`/`definition`/`curriculum`/`curated`) and `source == "conversation"`,
return `True`. This is a NEW ADMITTED EVIDENCE TYPE, not a lowered threshold —
the same shape as the `gated_explanation_chain` exemption at
`engine.py:12924-12925`. Do NOT mark the relation as an asserted fact; a single
provenance-stamped edge is an observation that licenses a candidate.

**H3 — expose the candidate to the answer surface.** Feed the trusted incident
edges into the forced-choice handler (`engine.py:7506-7524`) so a candidate can
be selected, keeping the Wilson/provenance gates and echo stripping. A schema
hypothesis (converged across >=2 independently taught edges) may *propose* a
candidate but must remain `assertable=False`; observed edges admit it
(skill §7a, "schemas propose; sources admit").

Pure, fake-testable versions of H1/H2 and the later-stage schema hypothesis
are provided in `ravana/src/ravana/chat/schema_induction.py` with
`tests/unit/test_schema_induction.py`. They are default-OFF behind
`RAVANA_RSI_DECLARATION_STORAGE=1`, import no engine, and change no gate.
They are NOT wired into the engine: `engine_generation.py` and `response_gen.py`
are owned by other agents. Parent must do the wiring there.

## 3. Brain mechanism — verified citations

Verified against publisher/PsycNet/PubMed/PMC records on 2026-11-23:

1. **Tse, D., Langston, R. F., Kakeyama, M., Bethus, I., Spooner, P. A., Wood,
   E. R., Witter, M. P., & Morris, R. G. M. (2007). Schemas and memory
   consolidation. *Science, 316*(5821), 76–82.** doi:10.1126/science.1135935.
   science.org abstract: *"systems consolidation can occur extremely quickly if
   an associative 'schema' ... has previously been created ... New traces,
   trained for only one trial, then became assimilated and rapidly
   hippocampal-independent."*
2. **McClelland, J. L., McNaughton, B. L., & O'Reilly, R. C. (1995). Why there
   are complementary learning systems in the hippocampus and neocortex.
   *Psychological Review, 102*(3), 419–457.**
   doi:10.1037/0033-295X.102.3.419; PMID 7624455. PsycNet: rapid hippocampal
   storage, slow interleaved neocortical extraction of structure.
3. **Yassa, M. A., & Stark, C. E. L. (2011). Pattern separation in the
   hippocampus. *Trends in Neurosciences, 34*(10), 515–525.**
   doi:10.1016/j.tins.2011.06.006; PMC3183227.DG orthogonalises similar inputs
   so new encoding does not overwrite old traces.
4. **Gentner, D. (1983). Structure-mapping: A theoretical framework for
   analogy. *Cognitive Science, 7*(2), 155–170.**
   doi:10.1207/s15516709cog0702_3 (publisher alias 10.1016/S0364-0213(83)80009-3).
   Wiley abstract: *"Relations between objects, rather than attributes of
   objects, are mapped ... rules depend only on syntactic properties ... not
   on the specific content of the domains."*
5. **Roediger, H. L. III, & Karpicke, J. D. (2006). Test-enhanced learning:
   Taking memory tests improves long-term retention. *Psychological Science,
   17*(3), 249–255.** doi:10.1111/j.1467-9280.2006.01693.x. SAGE/PsycNet:
   delayed retention is greater after retrieval practice than after restudy.
6. **Webb, C. E., Turney, I. C., & Dennis, N. A. (2016). What's the gist? The
   influence of schemas on the neural correlates underlying true and false
   memories. *Neuropsychologia, 93*(Pt A), 61–75.**
   doi:10.1016/j.neuropsychologia.2016.09.023. Verified via PsycNet/PMC5354939:
   schema-congruent lures produce high false-alarm rates; the paper explicitly
   notes schemas "contribute to false memories".

### Support-vs-extrapolation ledger

| Claim used | Source | Verdict | Limit |
|---|---|---|---|
| Fast one-trial assimilation needs a pre-existing schema; a single trace is not a schema | Tse 2007 | **SUPPORT** | Rat paired-associate; does not show a naive system forming the schema from the same traces. The already-written trusted edge is the analogue of the single trace; the relation vocabulary is the analogue of the schema. |
| Store each demo rapidly, extract the regularity gradually | McClelland et al. 1995 | **SUPPORT (conceptual)** | Hippocampus/neocortex -> buffer/graph mapping is an **ANALOGY**, not a homology claim. |
| Keep converged demonstrations distinct until the relation type is abstracted | Yassa & Stark 2011 | **SUPPORT (conceptual)** | Neural pattern separation is not measured in RAVANA; the ledger entry licenses distinct trace records, not a dentate-gyrus claim. |
| Near/far/foil/permuted design tests relation transfer, not lexical overlap | Gentner 1983 | **SUPPORT (framework)** | RAVANA is not claimed to perform human analogy; the harness is an operationalisation. |
| Asking the query after teaching is part of the learning design | Roediger & Karpicke 2006 | **SUPPORT for design; EXTRAPOLATION for RAVANA** | Human testing effect; effect of one extra `process_turn` on this engine unmeasured. |
| Schema congruence alone can create a false link; a hypothesis must stay unassertable | Webb et al. 2016 | **SUPPORT** | Human fMRI/scene paradigm; mapped structurally to "schema proposes, sources admit". |
| The 96.7% hedges are a functional analogue of failed pattern separation | — | **EXTRAPOLATION (weak)** | Observable is a refusal string; no neural mechanism was measured. Call it a failure to *consult* stored evidence, not a DG analogue. |
| `hail` specifically wins over `thaw` via token-length reduction vs vector-neighbour fallback | — | **UNVERIFIED** | Needs the trace prints named in §1.1; do not assert a specific sub-step without it. |

## 4. Measurable predictions and falsifiers

**Predictions (after H1+H2+H3, flag ON, identical state, one process):**

- Query-turn subject is `thaw` (assert on `ctx.subject` / `_last_grounding_method`),
  and the graded answer can name `mist` from the trusted edge.
- Structural near/far target accuracy moves off `0.00` at **k=8**, with
  `growth_headroom > 0` (k=8 minus k=1 positive; the curve is not saturated).
- Foil/negative rejection stays at the current live rates (1.00 foil;
  1.0 negative at k=8) — negative metrics must not move down.
- Fake arm remains 1.00 with no `engine_answer` fields; flag OFF remains
  byte-identical to today's output (differential probe before the A/B).

**Falsifiers (pre-registered):**

- **F1 — storage is exonerated, stop editing it.** k incident trusted edges are
  proven present on the taught path (already true, 2 edges) and the
  forced-choice query still yields no candidate -> bottleneck is downstream.
  This condition is **already met**; do not add more write paths.
- **F2 — downstream of retrieval.** With `ctx.subject` pinned to `thaw` and a
  trusted `mist->thaw` edge admitted as grounding, a forced single-token
  answer surface (echo stripped) still yields ~0 explicit decisions -> the gap
  is in answer emission, not grounding; stop editing `_topic_grounded` and move
  to the articulation/decision surface.
- **F3 — subject fix too broad.** If option masking changes foil/negative
  rejection or any non-forced-choice front, the mask is too aggressive: revert
  H1 and use a narrower frame predicate.
- **F4 — uninformative ceiling.** If `growth_headroom == 0` because the
  criterion set is saturated at k=1, the on-off delta is uninformative; do not
  report it as a learning gain.
- **F5 — schema-only promotion.** If near/far improves on a store with no
  observed trusted edge (hypothesis alone), the source-admission gate leaked;
  revert H3's schema branch (Webb et al. caution).

## 5. Unverified / caveats

- No engine boot and no pytest were run for this note (per task constraint).
  The delivered tests are written but must be run by the parent with
  `PYTHONPATH=<worktree>/ravana/src/ravana/chat:. python -m pytest tests/unit/test_schema_induction.py -q`
  (the test imports the module flat so no `chat`/engine `__init__` runs).
- The identity of the writer is inferred from an exact provenance-stamp match
  (`chain_walker.py:912-916` + `intake.py:80`) against the parent's census, not
  from a live call trace; the parent's `[consolidate]` line would confirm it.
- The exact per-token subject winner (`hail`) is not pinned statically; §1.1
  names the two candidate sub-steps and the trace lines that settle it.
- n = 2 teaching turns in the census; mechanism probe, not suite verdict.
- All six citations were verified via records retrieved on 2026-11-23; DOIs and
  page ranges are as listed. The schema-false-memory caution is supported by
  Webb et al. 2016 and is NOT attributed to Tse et al.'s abstract.
- Files touched: doc + NEW `schema_induction.py` + NEW
  `tests/unit/test_schema_induction.py`. No existing production file was
  edited; no gate, threshold, `response_gen.py`, `models.py`, `engine.py`, or
  `measure_learn5_rsi.py` change.
