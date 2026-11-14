# Plan-prefix linearization: a measured negative

**Status:** prototype built, wired behind a default-off flag, and **measured
as not working**. Kept for the diagnostic, not shipped.

Instrument: `scripts/measure_plan_vs_boost.py` -- 30 graph-discovered subjects,
persisted engine, three arms, paired.

## Result

```
arm             n  on_topic   unsup   func   verb   frame
------------------------------------------------------------------------
baseline       10    0.4592   1.400  0.437   1.00   0.900
word_boost      7    0.4082   0.714  0.391   0.71   0.714
plan           11    0.4153   1.091  0.442   0.82   0.758
------------------------------------------------------------------------
delta vs baseline
word_boost          -0.0510  -0.686 -0.046  -0.29  -0.186
plan                -0.0439  -0.309 +0.005  -0.18  -0.142
```

**Both steering mechanisms lose to the decoder's own distribution.** The plan
arm is worse than baseline on on-topic (−0.044), no better on frame (0.758 vs
0.900), and worse on verb rate. The word boost reduces unsupported
assertions (−0.686) but at a cost to every other axis.

(`n` differs per arm because an arm that returns no output is not counted.
Baseline produced 10 scored outputs of 30 subjects; plan 11; word_boost 7.)

## What the samples show

The plan prefix does what it was designed to do -- the plan leads:

```
music    PLAN: "Music has rhythm one there refer other used such."
ocean    PLAN: "Ocean is located in sea other such often there like."
```

The frame is there (`Music has rhythm`), and the continuation is salad. The
decoder cannot CONTINUE a grammatical frame; it can only be started by one.
Feeding the plan teacher-forced through the GRU puts the frame in the hidden
state, but the model has no learned mapping from "I am mid-frame" to "the next
word should be a determiner or a preposition".

That is the same capacity limit as every other decoder-side intervention: the
256-hidden GRU trained on 512 short definitions has not learned grammatical
continuation under an externally-imposed frame.

## Why this is a real negative, not a bug

The wiring is verified working:

* `NeuralDecoder.generate(prefix=...)` feeds the prefix through the GRU and
  returns only the continuation (`neural_decoder.py`).
* The step-0 content mask is suppressed when a prefix is present, so the
  prefix's function words are not masked out.
* `plan_is_committed` gates the plan, so an uncommitted plan is not realized.
* The emitted text is plan + continuation.

And the plans themselves are mostly good: `music has rhythm`, `ocean is
located in sea`. The failure is downstream of the message layer.

## What this rules out

Three interventions on the decoder's output, all measured:

| intervention | on-topic | unsupported | frame | verdict |
|---|---|---|---|---|
| baseline | 0.459 | 1.400 | 0.900 | — |
| word boost (9.0) | 0.408 | 0.714 | 0.714 | worse on 4 of 5 axes |
| plan prefix | 0.415 | 1.091 | 0.758 | worse on 3 of 5 axes |

The decoder's own distribution is the best of the three. **There is no
decoder-side intervention that improves on it.** That is a strong result: it
means the remaining lever is not how the decoder is steered, but whether the
decoder should be generating at all.

## Reconciling with `BRAIN_LINEARIZATION_SPEC.md` §7.5

The research spec's §7.5 says: *"Do not treat this as a decoder-capacity
problem. The GRU is 256-hidden... the narrative path produces grammatical text
with no neural network at all. The deficit is upstream."*

This measurement partly contradicts that, and the contradiction is worth
recording rather than smoothing over:

* **The spec is right that the deficit is not in the message.** The plans are
  good (`music has rhythm`, `ocean is located in sea`), and the failure is
  that the decoder cannot continue them.
* **But "not a capacity problem" is too strong.** A 256-hidden GRU trained on
  512 definitions has not learned to continue an externally-imposed frame.
  That IS a capacity/training limitation -- the spec's own §7.2 predicts the
  logit boost must fail, and the same argument applies to a prefix: the model
  has no learned mapping from "mid-frame" to "next function word".
* **The spec's §7.3 falsifiers anticipated this.** F2 says "if function-word
  ratio does not rise, the decoder is ignoring the conditioning". Here the
  function-word ratio DID rise (the plan's own frame words are in the output),
  and the continuation still failed. So the mechanism is not being ignored --
  it is being unable to use.

The practical consequence is the spec's own conclusion, reached from the other
direction: **the narrative path (no neural network) is the formulator this
system actually has.** §7.5's "do not treat this as a decoder-capacity
problem" should be read as "do not expect decoder-side steering to fix it",
which this measurement supports.

## The honest conclusion

`docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md` §5.5 said: "Do not treat the decoder
as the fix for a graph-coverage problem." This measurement extends that: the
decoder is not the fix for a *generation* problem either.

The brain-faithful reading is the one `docs/RETRIEVAL_VS_GENERATION.md` §1
already argued: the brain's conservative response to uncertainty is to fall
back on the most-practiced STORED form, not to compose something novel. A
256-hidden GRU trained on 512 definitions is not a composition engine, and no
conditioning scheme changes that.

So the recommendation is:

1. **Leave the decoder as a fallback, and stop trying to make it good.** Keep
   `_rg_switch_enabled` / `_plan_prefix_enabled` off.
2. **Invest in retrieval and concept acquisition** -- the 99/100 production
   number is the real capability, and extending the 325 definitions is worth
   more than any decoder work.
3. **If generation is ever needed**, it requires a decoder trained WITH
   conditioning on a message -- not a fitted projection bolted onto a
   CE-trained GRU. That is a retraining project, not a wiring project.
4. **Use the narrative/syntax path as the formulator.** It already produces
   grammatical text with no network; the plan representation could condition
   IT rather than the GRU. That is untested and is the one remaining
   brain-faithful option.

## Reproduce

```
uv run python scripts/measure_plan_vs_boost.py --max-subjects 30
uv run python scripts/prototype_linearization.py gravity ocean music
```

## Related

- `docs/LINEARIZATION_PROTOTYPE.md` -- the module and its unit-level success
- `docs/ANCHOR_BOOST_DESTROYS_SYNTAX.md` -- the word-boost negative
- `docs/RETRIEVAL_VS_GENERATION.md` -- the race model this result supports
- `docs/PRODUCTION_OUTPUT_FINAL.md` -- why retrieval is the real capability
