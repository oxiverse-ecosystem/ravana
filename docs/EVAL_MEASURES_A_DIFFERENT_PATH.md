# The eval measures a path production does not take

Date: 2026-11-14

## The correction

`scripts/eval_decoder_groundedness.py` calls `eng._generate_with_situation_model(ctx)`
**directly**. That bypasses the dispatch layer in `_process_turn`, which consults
stored knowledge FIRST and only falls through to the decoder when there is
nothing to say.

So the eval's "decoder strategy" arm measures what the decoder does **in
isolation**, not what RAVANA emits. For a subject with a stored definition,
production never reaches the decoder at all.

## Evidence

```
[ground_query] input='what is blue'
  [consult_internal] subject='blue'
  [consult_internal] HIT definition len=97
  [internal_knowledge] subj='blue' text='Blue is the colour of the short-wavelength
                       end of the visible spectrum, between violet and green.'
blue -> Blue is the colour of the short-wavelength end of the visible spectrum,
        between violet and green.
```

Same for `plant`, `moral`, `species`, `world`, `natural`, `cat`, `dog`, `sun`,
`gravity` — every subject with a stored definition gets it emitted verbatim,
with no stutter and no salad.

Meanwhile the eval's decoder arm, on the same subjects, produces:

```
sky   -> Such change like species one used.
blue  -> One many other system used like such.
dog   -> Act often called system used one like.
music -> Little such there often used other.
```

Both are real measurements of different things.

## Why this matters

The "57% gate withholding" figure is the gate rejecting **decoder output that
production would never have requested**. It is a true measurement of the
decoder's quality, but it is not the bottleneck on the engine's actual answers.

The engine's real answers for definition-bearing subjects are the stored
definitions — correct, grounded, and fluent. The decoder is the fallback for
subjects RAVANA has no definition for.

## What to do about it

1. **Add a production instrument.** `scripts/measure_production_output.py` runs
   `process_turn` (the real entry point) and classifies the output. That is the
   number that tracks the goal.
2. **Keep the decoder instrument**, but read it as "decoder quality when
   consulted", not "engine answer quality".
3. **The gate's 57% withholding is still worth fixing** — it is the fallback
   path's quality, and subjects without definitions depend on it. But it is not
   the first thing to look at when asking "are the answers fluent and factual".

## The stutter fix still stands

PR #351's predecessor work included a stutter guard in the narrative path
(`response_gen.py:1043`): when a stored definition already opens with the subject
("A cat is a small domesticated mammal"), prefixing `"{subject} is"` produced
"Cat is A cat is a small ..." and the clause monitor withheld a correct answer.
The guard now emits the definition verbatim. `_definition_response` had this
guard already; the narrative path did not.
