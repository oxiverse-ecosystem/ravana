# Productive vocabulary: emitted-definition codemap

Scope: investigation only. Repo `C:/Users/Likhith/Documents/projects/ravana` at `1753dfa`,
tree clean. No code changed, no `eng.save()` called, `RAVANA_OFFLINE=1` throughout.

Question: after teaching "a zorblat is a small blue device that measures air pressure",
"Use the word zorblat in a sentence" returns the stored definition instead of a novel
sentence. This maps *where that happens* and *the narrowest place to fix it*. A separate
agent owns the brain-research rationale.

## Method / instrument validation

Four short offline probes (scripts in the Hermes scratch dir, not the repo):
`probe_vocab_codemap.py` (emit trace + caller stack), `probe_vocab_reach.py`
(composition-call census), `probe_vocab_node.py` (graph-node creation trace),
`probe_vocab_final.py` (macroplanning/linearization call census, flags, MERGE_VARIANTS).

Probe inputs and outputs were printed, not just conclusions:

- Teach call: `learn_novel_term_from_turn("a zorblat is a small blue device that measures air pressure")` -> `LEARN_COUNT 1`.
- Stored value after teach: `'a small blue device that measures air pressure'`.
- Query: `Use the word zorblat in a sentence`.
- Instrumented emit: `strategy='definition_with_assoc'`, text
  `'It seems to me that zorblat is a small blue device that measures air pressure. does that match what you knew?'`.
- `process_turn` final answer: `'It seems to me that zorblat is a small blue device that measures air pressure.'`
  (the trailing question is stripped downstream, consistent with the ravana-debugging
  skill's "what runs AFTER generation" rule).
- Control, **untaught** word: `Use the word quxblarg in a sentence` ->
  `'honestly, quxblarg is a bit outside what i know right now.'` Refusal already works.
- Call census for the taught query: `['_try_surface_realize', 'SA.bind_to_sentence', 'SR.realize']`;
  `MACROPLAN_CALLS []`. No `_plan_narrative_sentence`, no `_generate_creative`,
  no `_generate_with_situation_model`.

Boot was 4.6 s offline (cached 400001 projected vectors, 100D -> 64D); the ~200 s cold
path is the un-cached GloVe read. No weights were written.

---

## 1. Where the query is classified as a use-in-sentence request

**Nowhere.** There is no productive-use intent class.

- `_ground_query` — `engine_generation.py:2016`. The subject-grounding fix **is present
  and verified**: `engine_generation.py:2098-2104` strips the performative frame
  `^(?:use|using)\s+(?:the\s+)?(?:word|term|expression)\s+`, and
  `2106-2110` strips the trailing `in a sentence`. Instrumented result:
  `query_phrase='zorblat'`, `words=['zorblat']`, subject `'zorblat'`
  (first pass `method='multi_word_unconnected', conf=0.45`; later passes
  `method='exact_label', conf=0.95` once the query turn registers the term).
  **Do not redo this fix.**
- PFC question typing — `prefrontal_workspace.py:1037` -> `_detect_question_type_regex`
  at `1133`; `QUESTION_PATTERNS` at `914-1002` contains no use-in-sentence pattern,
  so it falls to `general` at `1189`. With the semantic space wired, Stage 2
  (`1065-1076`) refined this query to `qtype='hypothetical'` — observed in every
  probe. That label is spurious for this request and is *not* used to compose.
- `_is_informational_query` — `engine_generation.py:2627-2677` returns `False` here
  (no `?`, no wh-/info prefix). It is consulted for KB lookup (`response_gen.py:8620`)
  but **not** for the stored-definition fallback, so its `False` does not prevent
  the definition from being emitted.
- `engine.py:673` regex `how to use .* in a sentence` is in the **web-SNIPPET reject
  shapes** list (source monitoring at retrieval time), not a user-query classifier.

Net: the only classification that fires is "subject has a stored definition".

## 2. Which response function emits the definition string, and why

**Emitting function: `ResponseGenMixin._definition_response` — `response_gen.py:7450-7533`.**
It returns the 2-tuple `(text, "definition_with_assoc")` at `response_gen.py:7533`.

Instrumented caller stack (outer -> inner):

```
process_turn                 engine.py:8014   -> _process_turn_impl
_process_turn_impl           engine.py:11571  -> self._generate_response(ctx)
_generate_response           response_gen.py:8615 -> self._definition_response(ctx)
_definition_response         response_gen.py:7504 -> _try_surface_realize(subject, target=defn, ...)
```

`_generate_response` (`response_gen.py:8383+`) checks paths in order; the one that wins is
the **stored-definition fallback at `response_gen.py:8609-8618`**, specifically:

```python
if getattr(self, "_definitions", {}).get(_sl):        # :8614
    _def_resp = self._definition_response(ctx)        # :8615
    if _def_resp:
        return _def_resp                              # :8617
```

Why it emits the definition: the branch tests **only** `_definitions` membership. It has
no notion of what the user wants to *do* with the word. `_definition_response` then
passes the definition itself as the `target` of `_try_surface_realize`
(`response_gen.py:7504-7506`, helper at `chain_walker.py:2325-2362`). The surface
realizer can only reorder/hedge that target — it cannot invent new semantic content.
`"It seems to me that"` is an `EPISTEMIC_FRAMES["low_confidence"]` member
(`surface_realizer.py:137-142`, applied at `:258`); the invented word is OOD, so the
sensorimotor function wired at `engine.py:1975` raises effective free energy and the
realizer hedges. The emitted sentence is therefore a re-worded definition, not a new
composition.

Other `_definition_response` call sites are **not** the emitter for this query:
`response_gen.py:8587-8590` (curated/offline seeds only), `8628` (on-demand KB),
`8670-8672` (later duplicate branch, unreachable after the `8617` return).

## 3. Composition machinery: what exists and what is reachable

| Mechanism | Definition | Reachable from this query? |
|---|---|---|
| `syntactic_cell_assembly.py` — `SyntacticCellAssembly` | class `:46`; `bind_to_sentence` `:207`; verb pick `:289`; `compose_sentence` `:625`; `seed_from_pos` `:183` | **YES**, transitively via `_try_surface_realize` -> `chain_walker.py:2338`. Observed `SA.bind_to_sentence` call. |
| `surface_realizer.py` — `SurfaceRealizer`, `MERGE_VARIANTS` | class `:58`; `MERGE_VARIANTS` `:153-164`; `realize` `:351`; `EPISTEMIC_FRAMES` `:137-142` | **YES**, same path (`SR.realize` observed). 10 variants exist: svo, left_dislocation, pseudo_cleft, topic_fronting_for, existential, cleft, as_for_topic, adverb_fronted, svo_emphatic, svo_causal. They vary word order only; content is still the definition target. |
| `verb_lexicon.py` — `VerbLexicon.select_verb` | class `:82`; `select_verb` `:572`; situation variant `:942`; wired at `response_gen.py:1067-1069` | **YES, transitively** through the realizer's verb selection; not a routing decision here. |
| `learn/macroplanning.py` — `message_from_best_edge`, `build_message` | `:297`, `:418` | **NO.** Probe `MACROPLAN_CALLS []`. |
| `learn/linearization.py` — `message_to_plan`, `plan_is_committed` | `:321`, `:248` | **NO.** Only called from `_plan_narrative_sentence` and the disabled plan-prefix block. |
| `learn/definition_proposition.py` — `proposition_from_definition` | `:118` | **NO.** Only called from `_plan_narrative_sentence` (`:7944`, `:8080`). This is the existing parser that turns a definition into `(relation, target)` — relevant to a fix. |
| `response_gen._plan_narrative_sentence` | `:7930-8124`; graph edges `:7959`, definition Source B `:8068-8101`, formulator at `:8122` | **NO.** Single caller is `response_gen.py:1985` inside `_generate_with_situation_model`, which this turn never reaches (early return at `:8617`). It also needs `_concept_keywords[subject]` (`:7952-7954`) and graph edges; those only appear *after* auto-expansion, and are then synthetic. |
| `response_gen._generate_creative` | `:2921-3005`; graph-membership guard `:2955-2958` | **NO.** No creative-request route for "use in a sentence"; that phrase type is absent from `QUESTION_PATTERNS`. |
| `response_gen._generate_with_situation_model` | `:1564` | **NO.** `use_linggen=False`, `_rg_switch_enabled=False` (defaults at `engine.py:1278`, `engine.py:1307`; confirmed in probe), and the definition return preempts the ACC-defer branch at `:8732-8738`. |

So the **only** composition machinery actually reachable is the definition-fed
surface-realizer chain. It is structurally incapable of novel composition: its content
is the definition string.

## 4. What the engine knows about the learned word (and the constraint)

- `engine._definitions: Dict[str,str]` — `engine.py:953`. The teaching function
  `learn_novel_term_from_turn` — `engine.py:2703-2815` — writes
  `known[term.lower()] = definition[:200]` at `:2784`, after the clean/acceptable/
  quality gates at `:2776-2782`. Exposure count `_term_exposures` at `:2792-2799`.
  **This dict is the sole durable content for an invented word.**
- GloVe — `_glove_vector`, `engine_graph.py:580-619`. Whole-word lookup for `zorblat`
  returns `None` (measured). The node-creation block inside the teaching function
  (`engine.py:2801-2810`) is guarded by `if vec is not None`, so teaching creates **no**
  graph node, no `_concept_labels` entry, no `_concept_keywords` entry.
- BUT: on the *query* turn `_auto_expand_concepts` runs at `engine.py:10496`
  (from `engine.py:10489-10501`), and `chain_walker.py:335-354` builds an OOV
  **subword composite** vector. Measured: `_glove_vector("zor")`, `("blat")`,
  `("zorb")`, `("lat")` all exist while the whole word does not. So a node *does*
  appear post-query: `kw=[259]`, `in_labels=True`, `graph_nodes=1`, `glove=None`.
  Trace: `engine.py:10496 -> chain_walker.py:355 add_node`. This node is a synthetic
  similarity artifact with no authored/typed semantic edges; graph-neighbour
  composition cannot derive "small blue device that measures air pressure" from it.
- POS hints — `_concept_pos = ConceptPosDict()` at `engine.py:1058`. Probe value for
  `zorblat`: `'noun'` (fallback, not an evidence-backed POS tag). `bind_to_sentence`
  and `SurfaceRealizer` consume this map (`:340`, `chain_walker.py:2340`).
- `concept_keywords` — `engine.py:1056`. Empty immediately after teach; `[259]` only
  after the query turn's auto-expansion.

**Constraint:** for an invented word the definition string is the only trustworthy
semantic content. Any productive composer must parse `_definitions[subject]`
(e.g. with `proposition_from_definition`, `learn/definition_proposition.py:118`) and
realize a *new* construction from it. It must not rely on GloVe neighbours, typed
edges, `_concept_labels`, or a POS tag — all absent or synthetic at teach time.

## 5. Narrowest correct insertion point

**`response_gen.py:8609-8618`, at line 8614/8615 — inside `_generate_response`'s
"Fallback to stored `_definitions` or on-demand KB grounding" block, immediately
*BEFORE* `self._definition_response(ctx)` is called.**

Concretely: add an intent guard in that branch, ahead of `:8615`:

```python
if getattr(self, "_definitions", {}).get(_sl):
    if self._is_productive_use_request(ctx.raw_input) and _sl in self._definitions:
        composed = self._productive_use_response(ctx)   # new, narrow helper
        if composed:
            return composed
    _def_resp = self._definition_response(ctx)          # unchanged default
    ...
```

Why this is the narrowest correct point:

1. It is the exact branch that emitted the definition (proven by the instrumented
   caller stack). It already has `ctx.subject` and the definition available.
2. The gate `_sl in self._definitions` is the same condition that makes untaught words
   refuse. Untaught words never enter the branch, so the measured honest-uncertainty
   answer (`'honestly, quxblarg is a bit outside what i know right now.'`) is preserved
   with zero new code. No composition for unknown words.
3. It leaves `what is X`, curated definitions, KB grounding, and every other caller of
   `_definition_response` byte-identical. Do **not** change `_definition_response`
   itself (it is also reached from `response_gen.py:8588`, `8628`, `8670`) and do **not**
   add intent logic to `_ground_query` (already fixed at `engine_generation.py:2098-2104`;
   mixing parsing into grounding would regress the measured subject fix).
4. It is fail-closed: if `_productive_use_response` returns `None` (no parseable
   proposition, salad gate veto, or ungrounded realization), control falls through to
   the existing `_definition_response`, so no turn is lost.

Definition of the new helper (implementation belongs to the code-owning agent): detect
the productive-use directive with the same regex family already used defensively in
`_ground_query` (`engine_generation.py:2102-2104`), but **detect without stripping**;
parse `self._definitions[sl]` via `proposition_from_definition`
(`learn/definition_proposition.py:118`); realize a *new* construction by calling
`syntactic_assembly.bind_to_sentence` + `surface_realizer.realize` with a
non-definitional `discourse_type`/relation (the current default `explain` at
`response_gen.py:7506` is precisely what forces a definitional sentence); then apply
the existing `_sm_response_grounded` / `_is_word_salad` gates used at
`response_gen.py:1986-1989` before emitting. `MERGE_VARIANTS` (`surface_realizer.py:153`)
supplies surface variety.

A routing fix alone will not produce productive use — the definition-gated graph paths
(`_plan_narrative_sentence`, macroplanning, linearization) are off or preempted, and the
only reachable realiser is currently fed the definition as its content. The fix must
change the *content* passed to the realiser, at the insertion point above.
