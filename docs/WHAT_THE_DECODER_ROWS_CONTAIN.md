# What the "decoder" rows actually contain

**Status:** traced. This qualifies `DECODER_ONLY_NUMBERS.md`.
**Consequence:** the decoder-only score (on-topic 0.069) is measured on a
**truncated, gated, hedged** string — not on what the network generates.

## The three numbers

From the instrument's per-strategy split (persisted engine, ungated):

```
strategy                    n     on-topic   unsupported   supported   word count
situation_model_decoder   187      0.0508      1.1337       0.0107       8.83
situation_model_narrative 252      0.2227      2.1190       1.7976      36.33
situation_model_syntax      1      0.4444      0.0000       2.0000      15.00
```

Three things stand out:

**1. The decoder produces 8.83 words on average.** `max_steps=24`
(`response_gen.py:1571`). So it is emitting less than half its budget, and the
path requires `len(generated) >= 5` (`:1582`) or the output is discarded
entirely.

**2. `supported assertions` = 0.0107.** Essentially **zero** of the decoder's
assertions are supported by the graph. Compare narrative at 1.7976. The
decoder is not slightly ungrounded — it produces almost no graph-supported
content at all.

**3. `unique ratio` = 1.0000.** Every word is distinct. That is not a quality
signal; it is what a short, hedged, associative string looks like.

## What the emitted string actually is

Tracing `_generate_with_situation_model` (`response_gen.py:1560-1625`), the
returned text has passed through, in order:

1. `neural_decoder.generate(..., max_steps=24, temperature=temp)`
2. a `len(generated) >= 5` floor — shorter outputs are discarded
3. `case_infer(...)` — recasing
4. terminal-punctuation injection
5. `_is_word_salad` / `_is_word_salad_any_sentence` — rejection
6. `_sm_response_grounded(ctx, text)` — the grounding gate
7. on failure, `_grounded_surviving_clauses(...)` — **keeps only the clauses
   that pass**, discards the rest
8. `_epistemic_tag(ctx)` — **prepends a hedge** ("i believe", "i might be
   reaching", "perhaps")

So the string scored as "the decoder" is: a truncated sample, filtered to its
grounded clauses, with a hedge prepended.

## Why this matters for interpretation

**The 0.069 is not the network's score.** It is the score of what survives a
pipeline that removes exactly the ungrounded material. The instrument's
`ungated` flag disables `_sm_response_grounded`, but it cannot disable steps
1-5 or 8 — those are structural.

That explains the otherwise-puzzling combination:

```
on-topic 0.051   but   word-salad flags 0/187   and   unique ratio 1.0000
```

A string that short, with the ungrounded clauses already removed, will look
structurally clean while saying almost nothing supported.

## What it does NOT change

The conclusion that the decoder is the worst component still holds — it is
worse than the narrative path on every axis. But the mechanism is now clearer:

* The decoder emits short, associative, hedged text.
* The gate and clause-filter then strip most of it.
* What remains scores low on supported assertions because there was almost
  nothing supported to begin with.

## The implication for the fix

`_epistemic_tag` prepending a hedge is worth a second look. Measured decoder
output includes:

```
"I believe it is archived versus philosophy comparison. It perhaps broadly is philosophy versus archi..."
"i might be reaching, but There such many psychology learning like one species ethics."
```

The hedge makes fluent nonsense **sound** calibrated. The decoder learned
hedging from the corpus, and `_epistemic_tag` adds more on top. Neither is a
grounding mechanism.

## Related

- `docs/DECODER_ONLY_NUMBERS.md` — the split this qualifies
- `docs/CONDITIONING_INPUT_LACKS_SIGNAL.md` — why the decoder is off-topic
- `docs/GROUNDING_MECHANISM.md` — CPG, which gates at selection time instead
