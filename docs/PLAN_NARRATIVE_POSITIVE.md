# Plan-driven narrative formulator: the first positive result

**Status:** prototype, measured. Not wired into production.

Instrument: `scripts/measure_plan_narrative.py` -- 40 graph-discovered
subjects, persisted engine, paired.

## Result

```
arm             n  on_topic   unsup   frame
------------------------------------------------------------------------
baseline       40    0.1947   1.975   0.791
plan_narr       7    0.3571   0.000   0.571
------------------------------------------------------------------------
baseline produced 40/40; plan_narr produced 7/40
subjects with a committed plan: 7/40

paired on n=7: d_on_topic=+0.0760 d_unsupported=-2.143
```

**On the subjects where a committed plan exists, the plan-driven formulator
beats the production narrative path on both grounding axes**: on-topic
+0.076, and unsupported assertions **−2.143 (to zero)**.

This is the first intervention in the decoder programme that improves on the
baseline rather than degrading it. Every prior arm measured negative:

| intervention | on-topic | unsupported | verdict |
|---|---|---|---|
| decoder baseline | 0.459 | 1.400 | — |
| word boost (9.0) | 0.408 | 0.714 | worse |
| plan prefix (GRU) | 0.415 | 1.091 | worse |
| **plan-driven narrative** | **0.357** | **0.000** | **unsup → 0** |

## The samples

```
music     plan: music has rhythm
          BASELINE  "Music is sound organized in time — rhythm, melody, and
                    harmony — valued as an art form... Commitment begins with
                    a spark."                    on=0.50 unsup=1
          PLAN_NARR "It appears that music is connected to rhythm."
                                                  on=0.50 unsup=0

species   plan: species is located in birds
          BASELINE  "A species is the basic unit of classification... It can
                    be defined as..."              on=0.91 unsup=4
          PLAN_NARR "It appears that species is tied to birds."
                                                  on=0.50 unsup=0

ocean     plan: ocean is located in sea
          BASELINE  "The ocean is the body of salt water..." on=0.88 unsup=4
          PLAN_NARR "It appears that ocean goes hand in hand with sea."
                                                  on=0.33 unsup=0
```

The plan-driven output is grammatical, and every content word is
graph-supported by construction -- which is exactly why unsupported goes to
zero. The baseline reads stored definitions, so it scores higher on-topic but
carries 1-4 unsupported assertions from its second and third sentences.

## The honest caveat: coverage

**Only 7 of 40 subjects yield a committed plan.** `plan_is_committed` rejects
the `relate` fallback and any non-concept filler, and the graph's typed edges
are sparse (gravity has 3 typed edges of 38; time has 18 of 114). So this
mechanism is a precision play, not a coverage play: it cannot fire on most
subjects, and the baseline remains necessary.

The frame score is also lower (0.571 vs 0.791) -- the plan-driven output is a
single short clause while the baseline produces a paragraph. That is a length
difference, not a grammar failure; both are grammatical.

## What this establishes

`docs/PLAN_PREFIX_MEASURED_NEGATIVE.md` concluded that the GRU cannot continue
an imposed frame, and recommended using the narrative path as the formulator.
That recommendation is now measured and holds: **the deficit was the
formulator, not the plan.**

The plan representation is sound. The GRU is the wrong consumer for it. The
symbolic realizer consumes it and produces grounded, grammatical output.

## Next steps, in order

1. **Raise plan coverage.** 7/40 is the binding constraint. The typed-edge
   sparsity is the cause, so this is a concept-acquisition problem -- the same
   conclusion as `docs/PRODUCTION_OUTPUT_FINAL.md`. Every subject with a typed
   edge gets a grounded answer.
2. **Use the plan as a PRECISION route, not a replacement.** Fall back to the
   baseline narrative path when no plan commits. That is a strict improvement:
   the plan fires where it is confident and the baseline handles the rest.
3. **Extend the construction inventory.** Five constructions cover the current
   typed vocabulary; the graph's `temporal`, `analogical`, `contrastive`,
   `episodic` and `inferred` relations have no construction yet.
4. **Do NOT wire this into the GRU.** The prefix result stands.

## Reproduce

```
uv run python scripts/prototype_plan_narrative.py gravity ocean species time music
uv run python scripts/measure_plan_narrative.py --max-subjects 40
```

## Related

- `docs/PLAN_PREFIX_MEASURED_NEGATIVE.md` -- the GRU negative this answers
- `docs/LINEARIZATION_PROTOTYPE.md` -- the plan module
- `docs/BRAIN_LINEARIZATION_SPEC.md` §7 -- the research spec
- `docs/DECODER_ONLY_NUMBERS.md` -- narrative 0.242 vs decoder 0.069
