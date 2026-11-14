# Production output: what RAVANA actually emits

Date: 2026-11-14
Instrument: `scripts/measure_production_output.py --no-web` (runs the real
`process_turn` entry point, not the decoder in isolation)

## Headline

**97 of 100 corpus subjects get a correct, grounded definition.**

```
sky      The sky appears blue because molecules in the air scatter shorter (blue) wavelengths...
blue     Blue is the colour of the short-wavelength end of the visible spectrum...
cat      A cat is a small domesticated carnivorous mammal (Felis catus)...
dog      A dog is a domesticated descendant of the wolf...
music    Music is sound organized in time — rhythm, melody, and harmony...
sun      The Sun is the star at the centre of the Solar System...
heat     In thermodynamics, heat is defined as energy in transfer...
planet   A planet is a large, rounded astronomical body that is...
children A child is a human being between the stages of birth an...
```

## The two real failures

```
like    we could pick up on definition. anything you'd rather talk about?
being   electronic dragged pending broadcasting switching aristotle's binding vices...
```

Both are function-ish non-concepts with no stored definition — the dispatcher
MISSes and the decoder fills the gap. `like` at least fails closed with an
honest deflection; `being` confabulates.

## Why this number is much better than the eval suggested

`eval_decoder_groundedness.py` calls `_generate_with_situation_model` directly,
bypassing the dispatch layer. Production consults stored knowledge FIRST:

```
consult_internal (brain_regions.py:471)
  1. _definitions[s]      <- hits for ~99% of corpus subjects
  2. hippocampal_buffer
  3. graph typed edges
  4. belief_store
  5. MISS -> decoder / live web
```

So the decoder's salad is the **fallback**, not the primary path. The eval's
"57% gate withholding" measured the fallback path's quality, not the engine's
answers.

## What made this work

| change | effect |
|---|---|
| #347/#348 corpus → graph | 512/512 subjects graph-known |
| **#355 corpus → `_definitions`** | dispatch HIT **74% → 99%**; `_definitions` 314 → 518 |
| #354 definition stutter guard | correct answers no longer withheld on a surface artifact |
| #351 reference-check content filter | the gate no longer passes salad |

The decisive one is **#355**: the graph and `_definitions` are different
indexes, and the dispatcher reads `_definitions` first. Making the corpus
graph-known was not enough.

## Instrument caveat

The salad classifier in `measure_production_output.py` had two false-positive
modes, both fixed and recorded in its docstring:

1. a type-token-ratio cutoff flagged correct long definitions as salad;
2. the codebase's `_is_word_salad` is imported into the `response_gen` mixin and
   is **not** reachable as `eng._is_word_salad`, so a `hasattr` guard silently
   fell through.

The current heuristic (subject head word present + ≥2 content words) still
over-flags inflections (`natural` vs `nature`, `areas` vs `area`). Treat its
counts as approximate; the samples are the evidence.
