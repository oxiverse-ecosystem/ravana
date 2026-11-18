# Language learning: what RAVANA actually acquires from conversation

Measured on 2026-11-19 with `RAVANA_OFFLINE=1`, one cold engine process, no
`eng.save()`. Raw output: `scratch/lang_learn.json`.

## Novel vocabulary: WORKS

Three invented terms, each taught in one conversational turn:

| term | teaching sentence | stored definition | next-turn recall |
|---|---|---|---|
| `zorblat` | *a zorblat is a small blue device that measures air pressure* | `a small blue device that measures air pressure` | **correct** |
| `quindle` | *a quindle is the feeling of waiting for rain* | `the feeling of waiting for rain` | **correct** |
| `marrowlight` | *a marrowlight is a lamp that glows only in deep water* | `a lamp that glows only in deep water` | **correct** |

Control: an untaught term (`flibbertigibbet`) returned an honest calibrated gap —
*"honestly, flibbertigibbet is a bit outside what i know right now…"* — so the
recall is not a generic echo.

## Productive use: PARTIAL

`use the word zorblat in a sentence` originally grounded to the subject
`word zorblat`, which is absent from the definition store, so a word the system
had just learned was treated as unknown.

Root cause: the directive-frame normalizer in `engine_generation.py` handled
*"tell me a sentence about X"* but not the performative *"use the word X"*.
Fixed by stripping `use|using (the)? (word|term|expression)` before the
trailing `in a sentence` rule.

After the fix:

```
Q=use the word zorblat in a sentence -> subject='zorblat'
   A=It seems to me that zorblat is a small blue device that measures air pressure.
Q=tell me a sentence about water     -> subject='water'
   A=Water is a transparent, tasteless, odourless liquid (H2O) essential for all known life.
```

The learned word now appears in a sentence, but the sentence is the **stored
definition** rather than a freshly composed usage. True generative use —
producing a novel sentence that deploys the word in a new context — is **not
demonstrated**. That needs a construction-based realiser (Goldberg), not
another subject-resolution fix.

## What this means

- **Vocabulary acquisition from conversation is real**: 3/3 taught terms stored
  and recalled, 1/1 untaught term declined.
- **Definition recall is not the same as language generation.** The system can
  learn *what a word means*; it cannot yet reliably *use* it productively.

## Note on the front score

`measure_fronts.py`'s "language learning" front teaches **code functions**
(`def name(x): return x + 1`) and tests next-turn recall of the function. That
is symbol learning, not natural-language vocabulary. It is not evidence for or
against the results above.
