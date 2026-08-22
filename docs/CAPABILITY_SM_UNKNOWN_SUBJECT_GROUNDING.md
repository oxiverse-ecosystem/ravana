# Capability: unknown-subject guard on the Situation-Model free-decode path (D4 word-salad limitation)

**Status:** shipped (commits `ab43047`, `8105d97`, branch
`auto/round-2026-08-19T1026Z`). **Verified:** regression suite
`tests/unit/test_sm_unknown_subject_grounding.py` passes — **5 passed in
18.36s** (real run, `RAVANA_OFFLINE=1`, `.venv-real`, `dim=64`, `seed=42`).
Hardcoding self-audit clean — the fix adds **zero** authored reply strings; the
only added logic is a structural concept-graph membership test (seed vocabulary,
not content).

## What it does

The Situation-Model path (`strategy situation_model_decoder`) does *free*
generation from the neural decoder, then only a permissive degeneracy check
(`_is_word_salad`, whose `>=3`-novel-word safety valve lets fluent-but-false
text through). Free decoding with no reality constraint is the architecture that
produces **word salad**.

This capability closes the **D4 limitation** logged in the round: the
Situation-Model free-decode path used to emit word salad for a subject RAVANA has
**no durable knowledge of** — e.g. *"tired"* (no definition, no web source, not
in the concept graph). The internal grounding monitor (`_sm_response_grounded`,
Step 1) accepted it because the subject's `associated_concepts` are the **query's
own** GloVe-similar neighbours (*ever, lot, really*), all of which pass the
`sim >= 0.30` check. So the gate falsely concluded *"RAVANA knows about this"*
and let the decoder restate those same associates as a "fact".

After the fix:

- An **UNKNOWN subject** — not in the concept graph, with no stored definition,
  and no web-learned source — can **no longer be grounded by free-association
  similarity alone**. Its utterance is withheld and the path falls back to honest
  uncertainty / reflective response instead of confident garbage.
- A **KNOWN subject** (already in the concept graph — a concept RAVANA has
  actually learned — or with a seeded definition) still grounds a genuine,
  topical answer. The control arm of the test asserts this is *not*
  over-suppressed.
- **Online graph growth re-admits a concept.** If RAVANA later learns a subject
  from chat or web learning, that subject joins the concept graph and is once
  again eligible to ground a free-decode answer. The guard is a live membership
  test, not a frozen blocklist.

This mirrors the brain's **source-monitoring** (Johnson, 1993: a claim built
only from unverified, low-similarity associations is a confabulation, not
knowledge, and should be withheld) and the decomposition path's own **D2 guard**
(`_decomp_grounded`), so the two free-generation monitors share one notion of
"knowing" — there is no second, divergent definition of grounding.

No LLM, no per-topic reply table, no retraining. The concept-graph membership
test is seed vocabulary that RAVANA expands at runtime (`_concept_keywords` /
`_concept_labels` grow as edges are mined), so a concept it later learns *is*
re-admitted.

## How it grew from the conversation

This cycle's chat round (round `2026-08-19T1026Z`) surfaced, among its residual
limitations, that the Situation-Model free-decoder emitted **word salad for
subjects RAVANA knows nothing durable about**. The feature card (`t_6df360c3`,
limitation D4) picked the concrete unknown-subject sub-capability as a gap.

**Root cause / prior behavior.** In `_sm_response_grounded` (Step 1), when a
subject had no definition and no web source, the code fell through to a
GloVe-similarity check against `ctx.associated_concepts` and set
`has_verified_fact = best >= 0.30`. But for an unknown subject the
`associated_concepts` are the **query's own** near-neighbours, which are all
GloVe-similar to the subject — so `best >= 0.30` was almost always true, and the
salad passed. The gate conflated *"words near the subject in the query"* with
*"RAVANA knows something about the subject."*

**Fix (commit `ab43047`).** The similarity check is now **gated** on the subject
being a *known* concept first:

1. The `else` branch that used to run the bare GloVe check is now guarded by a
   `_known` test — `subject in _concept_keywords` OR `subject in
   _concept_labels` (the engine's live concept-graph indices).
2. If `_known` is **False** → `has_verified_fact = False` directly (the unknown
   subject is withheld — no claim has a source).
3. Only if `_known` is **True** does the original GloVe-similarity fall-through
   run — i.e. loose association may only *LEAN* on spreading activation for
   concepts RAVANA has actually learned.

This is exactly parallel to the decomposition path's D2 guard
(`_decomp_grounded`, `response_gen.py:7316`, whose own `_known` block at
`response_gen.py:7360-7363` was the template), so both free-generation monitors
agree on what "known" means.

**Test correction (commit `8105d97`).** The two existing positive grounding
tests asserted the pre-fix *buggy* behavior: they used *"pluto"* as the subject,
but `pluto` is **not** in the concept graph, so the source-monitoring guard
correctly withholds it. Those assertions were encoding the very leak D4 fixes.
They were renamed to use a **KNOWN** subject (`gravity`, which has a seeded
definition) — now they assert genuine knowledge still grounds. A brand-new test
file targets the D4 class directly (below).

**Hardcoding audit.** The diff adds **zero** authored reply strings — only the
structural `_known` membership test plus comments. A grep for long added reply
literals (`git diff ab43047~1 ab43047 -- ravana/src/ | grep "^+" | grep -oE
'"[a-z][^"]{45,}"'`) returns nothing prose-shaped; the longest added string is
the seed-concept set of linguistic function words in the *other* guard
(`_GENERIC_CONCEPTS`, a vocabulary table, not a reply). The control tests prove
real knowledge still grounds, so the change is subtraction-of-a-leak, not
addition-of-a-script.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Situation-Model grounding monitor (declaration + docstring) | `ravana/src/ravana/chat/response_gen.py:3948` |
| Step-1 verified-fact anchor (the D4 `_known` gate) | `ravana/src/ravana/chat/response_gen.py:4015-4059` (guard added at `4021-4059`) |
| `_known` concept-graph membership test | `ravana/src/ravana/chat/response_gen.py:4038-4041` |
| Decomposition-path D2 guard (parallel design) | `ravana/src/ravana/chat/response_gen.py:7316` (`_decomp_grounded`) |
| D2 `_known` block (the template this mirrors) | `ravana/src/ravana/chat/response_gen.py:7360-7363` |
| Concept-graph indices declared | `ravana/src/ravana/graph/engine.py:326-327` (`_concept_labels`, `_concept_keywords`) |
| Indices grown at runtime as concepts are learned | `ravana/src/ravana/graph/engine.py:463-464`, `489-490`, `800-804` |
| Regression suite (D4 class) | `tests/unit/test_sm_unknown_subject_grounding.py` (5 tests, green) |
| Corrected pre-fix positive tests (pluto → gravity) | `tests/unit/test_sm_grounding_gate.py::test_coherent_factual_answer_passes`, `::test_junk_token_reply_still_topically_coherent_passes` |

## Test coverage

`tests/unit/test_sm_unknown_subject_grounding.py` — **5 tests, all GREEN
(18.36s)**:

- `test_unknown_subject_free_decode_salad_is_ungrounded` — the literal D4 class:
  subject *"tired"* (asserted not in `_definitions` / `_concept_sources` /
  `_concept_keywords`) with query-derived similar associates (`ever`, `lot`,
  `really`, …) must be **withheld** (`is False`).
- `test_unknown_subject_with_similar_associates_still_ungrounded` — even with a
  subject (*"perplexed"*) that *has* a GloVe vector and `sim >= 0.30` associates,
  an unknown subject is withheld (guards against the exact pre-fix regression).
- `test_known_subject_with_definition_still_grounds` — control: *"gravity"* (seeded
  definition) still grounds a genuine answer (`is True`).
- `test_known_graph_concept_leans_on_association` — control: *"oxiverse"*
  (bootstrapped into the concept graph) still grounds (`is True`).
- `test_d4_unknown_subject_sm_path_does_not_emit_salad` — **integration**: drives
  the REAL `_generate_with_situation_model` for the unknown subject *"tired"* and
  asserts the internal gate certifies whatever it produced (so free-decode salad
  cannot slip); asserts no `"it ties to"` / `"it ultimately has a relationship
  with"` salad glue remains.

The suite **fails RED** if the gate is stashed out (unknown subject re-admitted),
proving a real fix. The earlier `test_sm_grounding_gate.py` positives were also
corrected to use a known subject (`gravity`) so they no longer encode the leak.

> Note: the feature card's "add a test if a documented behaviour lacks coverage"
> is already satisfied — the 5-test D4 suite + the corrected positives cover both
> the unknown-subject withholding and the known-subject control arm. No further
> test was added in this docs pass; this card's scope is documentation only.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_sm_unknown_subject_grounding.py -v
```

## Known rough edges (honest — logged for a future round)

- **The broader D4 is only partially closed.** This round closed the concrete
  *unknown-subject* sub-capability: a subject with **no** durable knowledge is
  now withheld. The larger D4 class — word salad **anchored on a KNOWN concept**
  (e.g. free-decode junk about *"intentforge"*, which RAVANA does know) — is the
  deferred larger work. The current guard still permits association-spreading for
  known concepts, which is correct source-monitoring, but does not yet punish a
  known concept whose emitted text is itself incoherent beyond the existing
  word-salad / coherence sub-checks.
- The `>= 0.30` association threshold for *known* concepts is a shared constant
  with `_decomp_grounded`; it is data-derived vocabulary, not a hardcoded reply,
  but a future round may want to tune it per-concept.
