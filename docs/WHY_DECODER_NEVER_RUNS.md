# Why the neural decoder never runs

**Status:** traced, not inferred.
**Short answer:** it is gated on `use_linggen`, which is **never set to `True`**
anywhere in the codebase. The path exists and is reachable; the flag is off.

## The chain

```
every query
  ↓
_consult_internal_knowledge          <- fires 16/16 (100%)
  ↓ HIT -> return                      (13.5% of turns, line 8711)
  ↓ MISS
_web_direct_answer (69%)
_definition_response (44%)
_decomposition_generation_path (25%)
_reflective_response (6%)
  ↓
_generate_response(ctx)              <- line 11225, the single funnel
  ↓
use_linggen == False                 <- NEVER True
  ↓
template/ventral path produces the reply   <- 69% of turns end at 11671
```

## The gate

`response_gen.py:7493`:

```python
if (getattr(self, "use_linggen", False)
        and not (getattr(ctx, 'decomposition', None)
                 and getattr(ctx.decomposition, 'sub_questions', None))):
    sm_res = self._generate_with_situation_model(ctx)
```

`use_linggen` is assigned in exactly one place:

```
engine.py:1278    self.use_linggen = False
```

It is read in 5 places, written in 2, and **promoted nowhere**. Nothing in the
repo — not config, not a CLI flag, not a promotion gate — ever sets it `True`.

## The docstring is stale

`_generate_response` (`response_gen.py:7189`) claims:

> Removed from response path:
> - Neural decoder (bag-of-words, CE ~3.9, never produced coherent output)

That was true when written. It is no longer true — the decoder was
subsequently trained (CE 3.93 -> 1.570, top-1 0% -> 62.1%) and the path was
re-added behind `use_linggen`. The docstring was never updated.

**Do not trust that docstring.** The code is the source of truth.

## What this means

The decoder is not "unreachable behind 60 early returns". It is **one flag
away** from running, and that flag is deliberately kept off because the
promotion gate says the decoder does not yet beat the template path.

Measured gate result:

```
decoder_ce  3.8325
template_ce 3.7692
promote_use_linggen: false
```

The decoder loses on cross-entropy, so the flag stays off. That is the gate
working as designed.

## The real question

Not "how do we make the decoder reachable" — it already is. It is:

> **When should the decoder be allowed to compete?**

Two sub-questions:

1. **Is the gate measuring the right thing?** It compares CE on held-out text.
   CE rewards fluency, not correctness or groundedness. A decoder can have
   lower CE and still be less trustworthy.
2. **Should the decoder compete per-query rather than globally?** The brain
   does not have one global "retrieval vs generation" switch — it selects per
   utterance, based on prediction error and confidence.

See the companion research on the brain's retrieval/generation switch.

## Related

- `docs/GENERATION_REACHABILITY_200.md` — the 200-query dispatch audit
- `docs/DECODER_REACHABILITY.md` — the earlier, narrower finding
- `docs/LINGGEN_CE_COMPARISON.md` — the promotion gate measurements
