# Cross-frame productive-use codemap

Investigation only. No code changed, no `eng.save()` called. Repo
`C:/Users/Likhith/Documents/projects/ravana`, tree clean at `e3a02d8`.
The brain-research rationale belongs to another agent; this document maps only
**code causes and existing machinery**, with file:line evidence. It supersedes
the state described in `docs/PRODUCTIVE_VOCAB_CODEMAP.md` (written at `1753dfa`,
before `_productive_use_response` existed).

## Method / instrument validation

This is a static code investigation. The measured failures were supplied by the
caller, not re-run here: a cold engine boot is ~200 s and the task forbids
code changes, so no probe was launched. Every claim below was validated by
reading the exact source line(s) with `read_file` and cross-checking each
identifier / call site with ripgrep. The one runtime-sensitive claim (why
`give me an example ...` is tagged `statement`) is derived from the full
classifier cascade, read end to end, not guessed from the output string.

**Probe-output → source mapping (hand check):** the string
`ok, noted. anything else on your mind?` has exactly one possible origin in the
tree: lead `other_leads_notopic` = `"ok, noted."` plus follow `follows` =
`"anything else on your mind?"`, composed by `_handle_assertion`. That mapping
was verified against both pools and the composition site before attributing the
failure to that path.

## 0. Why the phrasings diverge (routing order)

`process_turn` runs a pre-router that returns early, *before*
`_generate_response` (where productive use lives). Order in `engine.py`:

| Order | Gate | Lines | Catches |
|---|---|---|---|
| 1 | creative pre-router | `11100-11139` | art noun + gen verb/topic anchor |
| 2 | creative-shape pre-router | `11140-11155` | `write`, `make up`, `invent`, ... |
| 3 | assertion mirror | `11157-11181` | classified `statement` |
| 4 | chit-chat | `11183-11191` | social |
| 5 | action request | `11193-11220` | `_ACTION_VERBS` imperative |
| 6 | main generation | `11571` → `_generate_response` | stored definition fallback |

The productive-use check is **last**: `_generate_response` calls
`_productive_use_response` at `response_gen.py:8670-8675`, inside the stored
definition fallback at `response_gen.py:8657-8679`. All three "bad" phrasings
are intercepted at steps 1-5 and never reach line 8670. Only
`use the word zorblat in a sentence` survives to that line — which is exactly
why it produces the single deterministic frame.

Critically, `_USE_IN_SENTENCE` (`response_gen.py:7452-7455`) **already matches
all three phrasings**:

- `use the word zorblat in a sentence` → `use ... word` (alt 1)
- `give me an example using the word zorblat` → `using ... word` (alt 1)
- `make a sentence with zorblat` / `write a sentence using zorblat` → `make/write ... a sentence` (alt 2)

So the intent regex is not the only problem: the phrasings are recognised at
`_productive_use_response:7471` but that function is unreachable for them.

## 1. Source of `i can't physically make/write a sentence ...`

**Function: `ResponseGenMixin._handle_action_request` — `response_gen.py:3203`.**
The exact template is at **`response_gen.py:3307`**, one of three `leads`
(`3304-3308`), chosen with `random.choice` at `3314` (the observed reply used
the third lead verbatim: `i can't physically {verb}{obj_disp}, since i run as a
chat brain without tools`). `obj_disp` is derived at `3291-3302`.

Trigger chain:

- `_ACTION_VERBS` (`response_gen.py:2830-2845`) contains `make`, `write`,
  `compose`, `describe`, `create` (`2831`, `2836`, `2844`), so both verbs are
  classed as physical/tool actions.
- `_is_action_request` (`response_gen.py:2847`) bare-imperative branch
  (`2873-2918`) returns the first word. `make a sentence with zorblat` starts
  with `make`, which is not an auxiliary (`2880-2883`), not a question
  (`2891-2892`), and no finite verb appears in `words[1:5]` (`2914-2917`), so it
  returns `"make"` at `2918`. Engine `engine.py:11208-11210` then calls
  `_handle_action_request` → refusal.
- `write a sentence using zorblat` is caught **earlier** by the creative-shape
  pre-router: `_creative_shape` (`engine.py:11119-11122`) includes `write`;
  `_is_info_tell` (`engine.py:11145`) does not match; so `engine.py:11146-11147`
  calls `_handle_action_request(user, "write", subject)` → the same refusal.

These are false positives: the user wants a *sentence generated*, not a tool
action in their environment. The informational guard at `response_gen.py:2860-2863`
already exempts `tell me / explain / describe / give an overview / what is /
define` — it simply has no entry for `make/write a sentence/example`.

## 2. Why `give me an example using the word zorblat` → `ok, noted. anything else on your mind?`

**This is the assertion mirror, NOT the correction circuit.**

- String origin: `_handle_assertion` — `response_gen.py:2700-2822`.
  - lead pool `other_leads_notopic` = `"ok, noted."` — `realizer_lexicon.py:120`
    (selected at `response_gen.py:2803` because no clean topic).
  - follow pool `follows` = `"anything else on your mind?"` —
    `realizer_lexicon.py:124`.
  - composition `f"{lead} {follow}"` at `response_gen.py:2821-2822`.
- Trigger: `engine.py:11165-11181` calls `_handle_assertion`; it acts only if
  `pfc_workspace.classify_speech_act(text) == "statement"`
  (`response_gen.py:2722-2725`).
- Why a request is tagged `statement`: the rule cascade **does** return
  `question` for `give me ...` (`prefrontal_workspace.py:1237`). But that hint is
  weak: `_confident_question` at `prefrontal_workspace.py:1291-1297` covers only
  `?`, sentence-initial wh-words, and auxiliary inversion — **not `give me`**.
  With embeddings wired, `SpeechActClassifier.classify`
  (`prefrontal_workspace.py:253-282`, especially `277-282`) returns the nearest
  prototype and ignores the hint, and `classify_speech_act` returns it at
  `1307-1308`. The long declarative-ish utterance is scored `statement`, and
  the assertion ack fires.
- `_handle_assertion`'s own exemptions do not help: wh-word (`2726`), yes/no
  (`2735`), and `_is_informational_query` (`2742`). The latter
  (`engine_generation.py:2627-2671`) requires a wh-/info token at `2639-2641`
  and its `info_patterns` at `2659-2671` include `give an overview`, `tell me`,
  `describe`, `define` — but **not `give me an example`**. No wh-word, no `?`
  → returns `False`.
- Downstream consequence: because `process_turn` returns at `engine.py:11181`,
  `_productive_use_response` is never called — even though alt 1 of
  `_USE_IN_SENTENCE` matches `using the word zorblat`.

**The correction circuit is a different path and is not the source of this
string.** `_detect_and_handle_correction` — `engine_persistence.py:577-659`,
called `engine.py:11829-11849` — emits `thanks for correcting me ...` / `thanks
for the correction ...` / `thanks — i'll be more careful there ...` at
`:644-651`, and its override is already gated to explicit corrections at
`engine.py:11844-11849`. It is the same *failure class* as the skill's
similarity-only re-ask warning, but it did not produce the observed text.

## 3. Existing construction / variation machinery

### SurfaceRealizer

| Item | File:line | Note |
|---|---|---|
| `MERGE_VARIANTS` | `surface_realizer.py:153-164` | 10 variants: `svo`, `left_dislocation`, `pseudo_cleft`, `topic_fronting_for`, `existential`, `cleft`, `as_for_topic`, `adverb_fronted`, `svo_emphatic`, `svo_causal` |
| `_variant_weights` | `surface_realizer.py:180` | per-variant learned weight, init 1.0 |
| `_last_variant_name` / `_recent_variants` | `surface_realizer.py:181-182` | last name + last-3 (STN fatigue) |
| `learn_from_feedback` | `surface_realizer.py:213-217` | Hebbian ±LR on a variant |
| `_select_variant` | `surface_realizer.py:729-787` | **RNG**: `random.choices` at `755`; `_weighted_variant_select` at `773,786` |
| `_weighted_variant_select` | `surface_realizer.py:789-809` | `random.choices` at `809`; fatigue penalty at `801` |
| `_apply_variant` | `surface_realizer.py:811+` | applies a *named* variant — deterministic if the caller supplies the name |
| `_recent_variants` update | `surface_realizer.py:618-621` | rolling last-3 |
| `_generate_epistemic_frame` | `surface_realizer.py:246-270` | **already deterministic** (FE + stable content hash, no RNG) — the in-repo precedent for this fix |

### SyntacticCellAssembly (M16 usage-based constructions)

| Item | File:line | Note |
|---|---|---|
| `_last_frame` / `learned_frames` / `_max_learned_frames` (512) | `syntactic_cell_assembly.py:101,107,108` | bounded inventory |
| `_frame_key` | `syntactic_cell_assembly.py:577-579` | (subject, verb, object, relation) |
| `find_construction` | `syntactic_cell_assembly.py:581-589` | **returns the same frame for the same role triple** — the structural cause of "one frame forever" |
| `allocate_construction` | `syntactic_cell_assembly.py:591-614` | registers/reinforces a frame, evicts lowest weight |
| `reinforce_construction` | `syntactic_cell_assembly.py:616-621` | Hebbian update |
| M16 reuse | `syntactic_cell_assembly.py:232-241` | reuses learned frame verbatim for the same triple |
| persistence / restore | `syntactic_cell_assembly.py:743-768`, `engine.py:13674-13677` | survives save/load |
| feedback hook | `engine.py:11711-11716` | `learn_from_feedback` on both assembly and realizer |

### Deterministic state that can drive a second frame

- **`DiscourseState.subject_repetitions`** — `surface_realizer.py:53`; populated
  as `_cross_turn_repeat` (0/1) at `engine_generation.py:293-311`; consumed as
  `ref_count` at `surface_realizer.py:902-919` where repeated references cycle
  through pronoun options. This is the closest existing "same subject again"
  counter and is RNG-free.
- **`_response_frame_history: Dict[str, List[frozenset]]`** — `engine.py:1111`,
  documented as a template-invariance tracker keyed by subject. Ripgrep finds
  **only the initialisation**: it is currently dead state, but is the natural
  per-conversation slot for a repeat counter / frame signature log.
- `_used_subjects` (`surface_realizer.py:177`, reset at `614-616`),
  `_recent_variants` (`182`), `_variant_weights` (`180`), `_last_frame`
  (`syntactic_cell_assembly.py:101`), `learned_frames` (`107`).

**Two-frame recipe without RNG noise:** key a repeat counter on the normalised
request (or subject), then emit `FRAMES[count % len(FRAMES)]`. Concretely, rebind
the same decomposed functional clause (currently `predicate`, extracted at
`response_gen.py:7477-7488`) into two constructions — SVO
`"Zorblat measures air pressure."` and a topicalised/pseudo-cleft variant such
as `"Air pressure is what Zorblat measures."` — using `SurfaceRealizer.MERGE_VARIANTS`
via `_apply_variant` with an **explicit** variant name (bypassing the RNG in
`_select_variant`), or by allocating two `SyntacticFrame`s with different
`_frame_key`s through `allocate_construction`. The counter can be
`subject_repetitions` (`engine_generation.py:310`) or a new entry in the unused
`_response_frame_history` (`engine.py:1111`). First identical request → index 0;
second → index 1, deterministically, with no `random.*` call.

## 4. Narrowest correct changes

**(a) Broaden the intent.** Widen `_USE_IN_SENTENCE` (`response_gen.py:7452-7455`)
to cover `example` phrasings, and factor a single `_is_productive_use_request(text)`
helper. Because routing intercepts first, also consult that helper at:

- `engine.py:11100-11155` (creative pre-routers) — skip them for productive-use
  requests; note `_creative_intent` at `11113-11116` already matches the word
  `word` in `"the word zorblat"`, so this guard is necessary.
- `engine.py:11165` (assertion check) — skip the assertion mirror.
- `engine_generation.py:2659-2671` (`_is_informational_query` `info_patterns`)
  — add `give me an example` / a productive-use rule.
- `prefrontal_workspace.py:1291-1297` (`_confident_question`) — add `give me`
  so the symbolic `question` hint from `:1237` survives the semantic override.

**(b) Stop the two misfires.**

- Physical action: add a productive-use early exit in `_is_action_request`
  (`response_gen.py:2847`, before the bare-imperative branch at `2873`) so
  `make/write a sentence|example using the word X` never returns an action
  verb; add the same guard at `engine.py:11146` for the `write` creative-shape
  branch. Leave `_handle_action_request:3307` untouched — it is correct for
  genuine tool actions.
- Acknowledgment: the `give me` additions in (a) stop the
  `_handle_assertion` → `other_leads_notopic` + `follows` ack at
  `response_gen.py:2803,2821-2822` from firing, letting the turn reach
  `_productive_use_response` at `response_gen.py:8670`.

**(c) Deterministic frame variation.** In `_productive_use_response`
(`response_gen.py:7457-7496`), replace the single construction at **`:7489`**
(`sent = f"{subject} {predicate}."`) with a small frame list and select by the
repeat counter described in §3. Preserve the echo gate (`7490-7495`) and the
stative fail-closed behaviour (`7481-7488`); do not let the definition string
leak. Keep the generated strings free of load-bearing trailing whitespace
(skill §4) and preserve exact-reproduction tests.

## 5. Per-turn / per-conversation state already available

| State | File:line | Scope | Use for variation? |
|---|---|---|---|
| `subject_repetitions` | `surface_realizer.py:53`; `engine_generation.py:293-311` | cross-turn | yes — deterministic 0/1 per repeated subject |
| `_response_frame_history` | `engine.py:1111` | per-conversation, keyed by subject | yes — currently unused; ideal repeat counter/frame log |
| `_recent_variants` | `surface_realizer.py:182,618-621` | session, last 3 | already a fatigue signal; still paired with RNG at `:809` |
| `_variant_weights` | `surface_realizer.py:180,213-217`; feedback `engine.py:11714-11716` | persistent | biases frame choice by success |
| `_last_variant_name` | `surface_realizer.py:181`; cleared `engine.py:11714-11716` | per-turn | one-step history |
| `_last_frame` | `syntactic_cell_assembly.py:101`; cleared `engine.py:11711-11713` | per-turn | one-step history |
| `learned_frames` | `syntactic_cell_assembly.py:107` | persistent, capped 512 | two frames need distinct `_frame_key`s |
| `_last_responses` (last 10) | `engine.py:11135-11137,11217-11219,11176-11178` | per-conversation | could detect repeated identical requests |
| `turn_count` | used in `_handle_assertion` seed `response_gen.py:2811-2813` | per-conversation | already drives deterministic surface selection |

Note the codebase already prefers **state-conditioned deterministic surface
selection over `random.choice`** in two places: `_generate_epistemic_frame`
(`surface_realizer.py:246-270`) and `_handle_assertion`'s `_state_seed`
(`response_gen.py:2804-2817`). The frame-variation fix should follow the same
pattern rather than adding another RNG draw.
