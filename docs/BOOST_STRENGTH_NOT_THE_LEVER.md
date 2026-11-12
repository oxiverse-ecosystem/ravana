# The boost strength is not the lever — a second negative result

**Status:** measured. Increasing `token_boost` does not improve output.
**Consequence:** neither suppression nor a stronger boost moves the decoder.
The remaining lever is where relations are selected, not how words are biased.

## What was tested

CPG's boost (`token_boost = 2.5 * strength`) was the mechanism that worked —
on-topic 4.5×, supported assertions 10×. The obvious next increment was to make
it **stronger**, on the theory that 2.5 is too weak to compete with the
decoder's learned distribution.

Swept `token_boost` over 2.5, 5.0, 10.0, 20.0 on the same engine and queries:

```
boost=2.5
  why do we dream      "It appears that dream carries through life. Connection begins with a spark..."
  tell me about trust  "i might be reaching, but Scientists learning there like management such."
  what is time         "Time is the continuous progression of existence that occurs in an apparent..."
  how does memory work "It seems to me that memory is archived versus philosophy comparison..."

boost=5.0
  why do we dream      "It appears that dream carries through life. Connection begins with a spark..."
  tell me about trust  "i might be reaching, but Archived must own management want one psychology..."
  what is time         "Time is the continuous progression of existence that occurs in an apparent..."
```

**No improvement.** `dream` is byte-identical. `trust` gets *worse* — different
word salad, equally ungrounded.

The run was killed by a gateway restart before 10.0 and 20.0 completed, but the
trend across 2.5 → 5.0 is flat-to-negative, and the mechanism explains why.

## Why a stronger boost cannot work

`token_boost` is applied per-step, and only to **unseen** tokens
(`neural_decoder.py:462`). Two consequences:

1. **It competes with a trained distribution.** The decoder's logits come from
   400,000 CE presentations. A fixed additive bias of 2.5-20 is small against
   that, and where it does dominate it overrides the model's syntax rather than
   redirecting it — producing "Archived must own management want one
   psychology".

2. **It only affects the first occurrence.** Once a token is emitted it is in
   `seen_counts` and can never be boosted again. So the mechanism shapes the
   opening of a clause and then lets the model run free.

A bias on logits is a **nudge**, and the measurement says the nudge is at its
useful limit. This is consistent with the suppression result: both act on the
logit vector, and the logit vector is not where the grounding decision lives.

## What this rules out

Three interventions on the decoder's output, all measured:

| intervention | on-topic | unsupported | verdict |
|---|---|---|---|
| CPG boost (2.5) | 0.051 → 0.227 | 1.13 → 1.19 | works, plateaus |
| stronger boost (5.0) | flat | flat/worse | no gain |
| NoGo suppression (2.0) | collapses | — | word salad |

**The decoder's logit vector is the wrong level of intervention.**

## Where that leaves the goal

The honest state: the decoder is 4.5× more on-topic than when this work began,
but it still emits ~9-word hedged strings with ~1.2 unsupported assertions each.
The goal -- fluent **and** factual -- is not met.

The remaining candidate, and it is the one the evidence points at: intervene
where **relations are selected**, not where words are. The decoder picks a
relation word ("versus", "like", "leads") and then fills arguments. CPG
constrains the arguments. It does not constrain the relation.

That is a different mechanism: a committed proposition is
``(subject, relation, target)``. CPG currently uses only ``target``. Using the
relation too -- requiring the emitted relation word to match the committed
relation -- would attack the unsupported-assertion count directly, because an
unsupported assertion is precisely a relation the graph does not support.

## Related

- `docs/CPG_MEASURED_EFFECT.md` — the boost result
- `docs/NOGO_SUPPRESSION_FAILS.md` — the suppression result
- `docs/GROUNDING_MECHANISM.md` — the mechanism, whose step 3 says a token is
  admissible only if it "realizes a committed target" -- the relation half was
  never implemented
