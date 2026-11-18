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

## Productive use: NOW WORKS FOR FUNCTIONAL DEFINITIONS (2/3)

`use the word zorblat in a sentence` originally grounded to the subject
`word zorblat`, which is absent from the definition store, so a word the system
had just learned was treated as unknown.

Root cause: the directive-frame normalizer in `engine_generation.py` handled
*"tell me a sentence about X"* but not the performative *"use the word X"*.
Fixed by stripping `use|using (the)? (word|term|expression)` before the
trailing `in a sentence` rule.

After that fix the subject resolved, but the reply was the stored definition.
A second fix (`_productive_use_response` in `response_gen.py`) rebinds a
definition's FUNCTIONAL clause under a new construction with the learned word
as subject, and echo-gates the result. Measured:

```
use the word zorblat in a sentence      -> Zorblat measures air pressure.
use the word marrowlight in a sentence  -> Marrowlight glows only in deep water.
use the word quindle in a sentence      -> It seems to me that quindle is the feeling of waiting for rain.
use the word flibbertigibbet ...        -> hmm, i'm not totally sure about flibbertigibbet yet...
```

Two of three learned words now get a **new construction** — the definitional
copula + head NP is gone, and the functional predicate is reused as the matrix
predicate. The third (`quindle`, a stative "feeling of…" definition) has no
functional clause to rebind and **fails closed** to the definition rather than
inventing content. The untaught word is still declined.

Brain basis: productive vocabulary use is construction deployment, not
definition recall (Goldberg 2006; Carey 2010; Schacter & Addis 2007). See
`docs/BRAIN_PRODUCTIVE_VOCABULARY.md` for the full cited mechanism and
`docs/PRODUCTIVE_VOCAB_CODEMAP.md` for the emitting path.

**Remaining limit:** cross-frame productivity is not demonstrated — each word
is realized with one new construction, not two different ones across two
requests, and only definitions with a relative clause can be rebound. Four
regression tests cover the measured cases.

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
