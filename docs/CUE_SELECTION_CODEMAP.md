# Cue-sensitive construction selection — codemap

Investigation only. No code changed, no `eng.save()` called. Repo
`C:/Users/Likhith/Documents/projects/ravana`, tree clean at `1b8be3d`.
This document maps **existing code and the narrowest insertion point** for
making productive-use frame choice depend on the question cue. It is the
companion to `docs/CROSS_FRAME_CODEMAP.md` (written at `e3a02d8`, before the
two-frame counter existed) and `docs/BRAIN_CROSS_FRAME_PRODUCTIVITY.md`
(SCC-RF proposal). Where those docs describe a proposed mechanism, this one
describes the code that is actually on disk now.

## Method / instrument validation

Static code investigation. Per the `ravana-debugging` skill ("validate the
instrument before you report the number"), no runtime score is reported: a cold
engine boot is ~200 s and this task forbids code changes, so no engine probe
was launched. Instead every claim below was checked mechanically: a scratch
script (`$TMPDIR/validate_cue_codemap.py`) asserts the source text at the exact
file:line cited here (39/39 assertions passed, re-run after correcting one
off-by-one docstring line in the instrument itself). Each identifier and call
site was additionally cross-checked with ripgrep across `ravana/` and
`ravana_ml/`. No word-level claim rests on an output string that was not traced
to its composing line.

## 0. Where productive use lives now

| Item | File:line | Note |
|---|---|---|
| `_PRODUCTIVE_USE_RE` | `response_gen.py:180-184` | module regex: `use/using … word/term` OR `make/write/compose/produce/give/show … sentence/example` |
| `is_productive_use_request` | `response_gen.py:187-189` | module-level predicate; the single source of intent truth |
| predicate call sites | `engine.py:504` (import), `engine.py:11104`, `response_gen.py:2881`, `response_gen.py:7494` | pre-router guard (`:11104`) gates creative/assertion routers at `:11135,:11151,:11170`; action guard at `:2881`; emitter gate at `:7494` |
| `_productive_use_response` | `response_gen.py:7480-7541` | two frames + counter, detailed in §1 |
| call site | `response_gen.py:8708-8720` | inside `_generate_response` (defined `:8475`); only runs when `_defs.get(_sl)` is truthy (`:8708`) |
| directive normalizer | `engine_generation.py:2083-2122` | strips performative frames before grounding; the `make/write … sentence … using X` strip is at `:2101-2106`, `use the word X` at `:2111-2113` — **leave untouched** |

The call site passes exactly three arguments: `(ctx, _sl, _defs[_sl])`
(`response_gen.py:8715-8716`). Everything in §2 is reachable from `ctx` or
`self` at that point.

## 1. The current two-frame counter, exactly

```
7517  _counts = getattr(self, "_productive_use_counts", None)
7518  if _counts is None:
7519      _counts = self._productive_use_counts = {}
7520  _idx = int(_counts.get(subject.lower(), 0)) % 2
7521  _counts[subject.lower()] = _idx + 1
7532  sent = f"A {_head} called {subject} {predicate}."   # frame 1 (guarded)
7534  sent = f"{subject} {predicate}."                    # frame 0 (default)
7537  if text.lower() in sent.lower() or sent.lower() in text.lower(): return None
7539  if len(sent.split()) < 4: return None
7541  return (sent[0].upper() + sent[1:], "productive_use")
```

* The counter is a per-subject **parity**, keyed by `subject.lower()`
  (`:7520-7521`). Trace: missing→`idx=0`, store `1`; then `1%2=1`, store `2`;
  then `2%2=0`, store `1`; so frames alternate `0,1,0,1` and the stored value
  oscillates `1,2,1,2`. It is a modulo-2 toggle, not a growing count.
* Frame selection is `if _idx == 1 and _head and _head not in (...)` (`:7531`):
  frame 1 is the appositive "A `{head}` called `{subject}` `{predicate}."`;
  otherwise frame 0 is bare "`{subject} {predicate}.`". `_head` is the last
  token before the relative-clause match (`:7526-7530`), vetoed for the small
  stopword list `a/an/the/small/blue` at `:7531` — so frame 1 can silently
  degrade to frame 0 for many definitions.
* The functional predicate is **regex-extracted from the definition string**,
  not from a graph relation: relative clause `that|which|who …` at
  `response_gen.py:7500-7503`, joined to at most 5 lowercase words, then
  `len(predicate.split()) < 2 → None` (`:7510`). A definition with no relative
  clause fails closed (`:7504-7508`).
* **Every storage location for that state:** `self._productive_use_counts` is
  created at `response_gen.py:7519` and read/written only at `:7517,:7520,:7521`.
  Ripgrep across the repo finds no other occurrence — it is an in-memory
  engine-instance attribute and is **not** in any `get_state`/`set_state` blob,
  so it does not survive save/load. No RNG is involved.
* State that is *not* read by this path (do not mistake it for the counter):
  `DiscourseState.subject_repetitions` (`surface_realizer.py:53`, populated
  `engine_generation.py:293-311`, consumed `surface_realizer.py:902-903`),
  `SurfaceRealizer._recent_variants` / `_variant_weights` /
  `_last_variant_name` (`surface_realizer.py:180-182`), and
  `SyntacticCellAssembly._last_frame` / `learned_frames` (`:101,:107`).
* Guards that must stay: intent gate `response_gen.py:7494`; empty-input gate
  `:7496-7497`; relative-clause extraction `:7500-7511`; echo gate
  `:7535-7540`; caller's `_definitions` gate `:8708` plus `try/except` fallback
  `:8714-8720` (so untaught words never enter and any failure degrades to the
  unchanged definition path).

## 2. Question cues available at the productive-use call site

All of the following are readable at `response_gen.py:8714` without new
plumbing (`qtype` is a local of the same `_generate_response`, defined at
`:8527` and never reassigned through `:8714`; `ctx` is the argument at `:8715`).

| Cue | File:line | What it gives | Reliability for what / where / part-of |
|---|---|---|---|
| `ctx.raw_input` | field `models.py:69`; used at `:7494`, `:8785`, `:9932` | the literal user string, including the cue phrase | **Most reliable.** Same string the intent predicate and the normalizer already parse. A 3-way regex over it can distinguish `what does X do` / `what is X part of` / `where would you find X` directly, with no dependence on learned classifiers. This is the only source that currently carries the part-of/locative distinction. |
| `ctx.subject` | field `models.py:66`; passed as `_sl` at `:8704,:8716` | the learned word (key into `_definitions`) | Identifies the **word**, not the frame cue. Already the counter key (`:7520`); keep as the tie-break/fallback key. |
| `pfc_workspace.detect_question_type` | call at `response_gen.py:8527`; method `prefrontal_workspace.py:1037-1076`; regex cascade `:1133-1189` | coarse label (`what_is`, `why`, `how`, `compare`, `general`, social types) | Partial. `QUESTION_PATTERNS` (`:914-1002`) has **no** `where` and **no** `part_of` type: `what is X part of?` matches greedy `what_is` (`:933-936`), `where …` falls to `general` (`:1189`). Usable as a weak "functional vs general" prior only, not as the part-of/locative selector. |
| `QTYPE_PRIMARY_RELATION` / `get_primary_relation_for_qtype` | `prefrontal_workspace.py:1017-1035` | maps qtype → `semantic`/`causal`/`contrastive`/`analogical` | No `part_of`, no `locative` entry (validated). There is no separate `QTYPE` constant symbol in the tree; this class dict is what the name refers to. Cannot drive three-way frame choice. |
| `pfc_workspace.classify_speech_act` / `classify_speech_act_rules` | `:1263-1308` / `:1192-1261` | **binary** `question` / `statement` only (docstring `:1265`); wh-word test at `:1229`, aux-inversion `:1232-1235`, `give me` at `:1237`; `_confident_question` at `:1291-1297` | Wrong granularity. Detects interrogativity, not what/where/part-of. Useful only as "is this a question at all", which is already implied by reaching the productive-use branch. |
| `_ground_query` outputs | `engine_generation.py:2016-2017`, returns `(subject, confidence, method)`; method values e.g. `no_pattern` (`:2064`), `multi_word_unconnected` (`:2509-2511`) | grounded subject + strategy label | Not a question cue: `method` describes *how the subject was found*, not the constructional frame. `ctx.subject` already carries its result. |
| interrogative detectors | `ctx.raw_input.strip().endswith('?')` at `response_gen.py:7383,8787,9932`; wh-regex `prefrontal_workspace.py:1229,1293-1294` | question vs non-question | Binary; same limitation as speech-act classification. |

**Verdict:** read the cue from `ctx.raw_input` at the call site (cheap regex on
the three cue shapes). Use `qtype` (`:8527`) only as a secondary prior for the
functional/general distinction. Do not build the selector on
`classify_speech_act`, `QTYPE_PRIMARY_RELATION`, or `_ground_query`: none of
them encodes part-whole or location.

## 3. Relation vocabulary and construction machinery

### What exists

| Item | File:line | Expresses |
|---|---|---|
| `ConceptEdge.RELATION_DEFAULT_WEIGHTS` | `ravana_ml/src/ravana_ml/graph.py:168-181` (read at `:1718`) | 12 generic graph relation weights: `semantic`, `causal`, `temporal`, `analogical`, `contextual`, `inferred`, `negation`, `antonym`, `transitive`, `physical_cause`, `comparison`, `pragmatic`. **No** `part_of`, **no** `locative` key. |
| graph-level remap | `engine.py:782-784` | `part_of → contextual` for spreading; a graph activation label, not a surface verb |
| `VerbLexicon.COMPOSITION_RULES` typed block | `verb_lexicon.py:275-310` | relations with **surface verbs/prepositions**: `is_a`/`isa` → copula "is a"; `part_of` → copula + prepositions "is part of" / "forms part of" (`:291-293,:308-310`); `has_property`/`property` → "is characterized by"/copula+adj (`:294-301`); `capable_of` → "is capable of" (`:302-304`); `used_for` → "is used for"/"is used to" (`:305-307`). This is the functional/part-whole vocabulary. **No locative relation rule.** |
| `VerbLexicon.select_verb` | `verb_lexicon.py:572+` | Hebbian selection of a phrase for a relation; used by the realizer fallback |
| `SurfaceRealizer` typed-relation override | `surface_realizer.py:438-446` | when relation ∈ `is_a/isa/has_property/property/part_of/used_for/capable_of`, the plan's `verb_phrase` is authoritative over a re-derived verb — the hook that preserves a chosen functional/part-of verb |
| `SurfaceRealizer._select_verb_phrase` | `surface_realizer.py:1182-1204` | cerebellar phrase prediction, then `VerbLexicon.select_verb` fallback |
| `SurfaceRealizer.ADVERBIAL_MODIFIERS` | `surface_realizer.py:166-174` | relation→adverb map (only generic relations) |
| `SurfaceRealizer.MERGE_VARIANTS` | `surface_realizer.py:153-164` | 10 named variants: `svo`, `left_dislocation`, `pseudo_cleft`, `topic_fronting_for`, `existential`, `cleft`, `as_for_topic`, `adverb_fronted`, `svo_emphatic`, `svo_causal` |
| variant selection | `surface_realizer.py:729-809` | **RNG** (`random.choices` at `:755,:809`) plus STN fatigue (`:801`). `_apply_variant` (`:811+`) accepts an **explicit variant name**, so deterministic callers can bypass the sampler. |
| `SyntacticCellAssembly.bind_to_sentence` | `syntactic_cell_assembly.py:207-287` | builds a `SyntacticFrame` from `(subject, relation, target)`; `_pick_verb_for_relation` (`:289`), `_pick_verb_phrase` (`:319`); M16 reuse of a learned frame at `:232-241` |
| `SyntacticFrame` | `syntactic_cell_assembly.py:39-41` | fields include `relation_type`, `weight` (M16 usage weight), `verb_phrase`, `verb_concept` |

### Honest gaps for the three cue types

* **Functional (`what does X do`)** — already expressible: the current `predicate`
  is exactly a functional clause (`response_gen.py:7500-7511`), and `used_for` /
  `capable_of` have surface verbs (`verb_lexicon.py:302-307`). This is the only
  cue the present frames really support.
* **Part-whole (`what is X part of`)** — expressible at the construction layer:
  `part_of` typed verb + preposition (`verb_lexicon.py:291-293,308-310`),
  honoured by the realizer override (`surface_realizer.py:444-446`) and by
  `bind_to_sentence`. But it needs a **graph-supported target**; the current
  productive-use path reads only the definition string and never checks a
  graph neighbour, so a part-of frame must fail closed (fall back to the
  counter or to honesty) when no grounded target exists, or it will assert an
  ungrounded part. `ConceptEdge.RELATION_DEFAULT_WEIGHTS` does **not** contain
  `part_of` (graph-level `part_of` is remapped to `contextual` at `engine.py:783`).
* **Locative (`where would you find X`)** — **no surface-verb relation exists**
  in the language layer: no `locative`/`located_in` entry in
  `VerbLexicon.COMPOSITION_RULES` (`:275-310`), none in
  `RELATION_DEFAULT_WEIGHTS` (`graph.py:168-181`), and none in the realizer's
  typed set (`surface_realizer.py:444-445`). Location verbs/prepositions live
  in the personal-fact slots (`pet_slots.py:157-165`, `user_model.py:2726-2728`)
  and are not consumable by this construction path. A locative variant would
  have to be a prepositional clause built from a grounded neighbour, with the
  same fail-closed target requirement; otherwise the cue should fall back to
  the counter. `MERGE_VARIANTS` cannot supply it — those ten variants change
  word order, not the proposition (companion doc §6/S5).

## 4. Narrowest correct change (insertion point)

**Single insertion point: `response_gen.py:7517-7534` — the `_idx` computation
at `:7520` and the two-frame if/else at `:7531-7534`.** Nothing upstream or
downstream needs to move.

1. Keep `:7494` (intent gate), `:7496-7511` (definition + predicate
   extraction), and `:7535-7540` (echo/length gate) byte-identical. Keep the
   caller at `:8708-8720` byte-identical: the `_definitions` gate at `:8708`
   is what preserves untaught-word refusal, and the `try/except` at `:8714-8720`
   is what preserves fail-closed degradation.
2. At `:7520`, before computing `_idx`, derive a cue label from
   `ctx.raw_input` (the same string already read at `:7494`) with a small
   regex over the three cue shapes. When a cue maps to an **admissible** frame
   (functional → frame 0; part-of → a `part_of`-verb frame built via
   `VerbLexicon`/`bind_to_sentence` **only with a grounded target**; locative →
   a prepositional frame only if a grounded neighbour exists), use that frame
   as the first candidate.
3. Keep the repeat counter as **tie-break/fallback**: when the cue is absent,
   ambiguous, or its preferred frame is inadmissible, compute
   `_idx = int(_counts.get(subject.lower(), 0)) % 2` as today; retain the
   counter update at `:7521` on every deployment so two identical requests
   still walk frames `0 → 1` (the O1 property) and no RNG is introduced.
4. Run the chosen frame through the unchanged echo gate (`:7537-7540`); if it
   fails, advance to the next-ranked frame (cue's second choice, then the
   counter frame) rather than relaxing the gate. If all fail, `return None`
   and let the caller fall through to the definition path.
5. Do **not** touch the directive normalizer (`engine_generation.py:2083-2122`)
   or the pre-router guard (`engine.py:11104`); they already deliver this
   request to the call site. Do **not** use `classify_speech_act` or
   `QTYPE_PRIMARY_RELATION` as the primary cue (§2). Do **not** add
   `random.choice`/`random.choices` (the `surface_realizer.py:755,809` sampler
   would break reproducibility).

Why this is narrowest: it changes one function's selection line, reuses the
two existing frames plus the existing typed-relation verbs, leaves the intent
predicate, routing guards, definition gate, and echo gate untouched, and adds
no new state store.

## 5. Existing per-construction scoring / reinforcement state

| State | File:line | Can it hold cue→frame associations? |
|---|---|---|
| `self._productive_use_counts` | `response_gen.py:7517-7521` | Currently holds only the per-subject parity. It is the natural minimal side-dict for a per-word frame-count/cue map, but is **not persisted** (no `get_state`/`set_state` entry) — fine for a within-session tie-break, not for learned associations. |
| `SyntacticCellAssembly.learned_frames` | `syntactic_cell_assembly.py:107`; bounded `_max_learned_frames=512` at `:108` | Yes: each `SyntacticFrame` carries `relation_type` and a usage `weight` (`:39-41`). `find_construction` (`:581-589`) keys on `(subject, "", target, relation)` via `_frame_key` (`:577-579`) — **no cue field**, so storing a cue→frame association needs either a new key component or a side dict in the same state blob. |
| `allocate_construction` / `reinforce_construction` | `syntactic_cell_assembly.py:591-614` / `:616-621` | Register a new frame (evict lowest weight) and Hebbian ±LR update; the write path a successful cue-selected deployment would use. M16 reuse at `:232-241`. |
| M16 persistence | `syntactic_cell_assembly.py:735-768` (`get_state` `:735`, `learned_frames` serialized `:743`, restored `:758-768`); engine load `engine.py:13679-13680` | Survives save/load, so cue→frame weights placed in this blob would persist. Feedback hook at `engine.py:11711-11716`. |
| `SurfaceRealizer._variant_weights` / `learn_from_feedback` / `_last_variant_name` / `_recent_variants` | `surface_realizer.py:180-182`, `:213-217`, `:618-621`, fatigue `:801` | Per-variant learned weights + last-3 history; usable as scoring state, but selection at `:729-809` is RNG-coupled (`random.choices` at `:755,:809`) so it cannot be the deterministic selector as-is. `_apply_variant` (`:811`) with an explicit name is the deterministic precedent. |
| `DiscourseState.subject_repetitions` | `surface_realizer.py:53`; `engine_generation.py:293-311`; consumed `surface_realizer.py:902-903` | Cross-turn "same subject again" counter, deterministic and RNG-free; an alternative slot for the tie-break if the per-subject dict is unwanted. |
| `_response_frame_history` | `engine.py:1111` | Ripgrep finds only the initialisation — dead state; a candidate slot for a per-subject frame-signature log if persistence is wanted. |

**Bottom line:** the only per-construction state the current path actually
reads is `_productive_use_counts`. The durable, already-persisted home for a
cue→frame association is `SyntacticCellAssembly.learned_frames` /
`allocate_construction` / `reinforce_construction` (M16), with the honest
caveat that `_frame_key` has no cue component and `ConceptEdge` / `VerbLexicon`
have no locative relation yet.

## Insertion-point one-liner

`ravana/src/ravana/chat/response_gen.py:7520` — replace the unconditional
`_idx = int(_counts.get(subject.lower(), 0)) % 2` with
"cue-derived preferred frame if admissible, else that same parity expression",
leaving `:7494-7511`, `:7535-7541` (echo gate) and `:8708-8720`
(untaught-word refusal + fail-closed fallback) unchanged.
