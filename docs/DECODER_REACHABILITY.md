# Why the neural decoder is unreachable

**Status:** verified by tracing the live engine, not inferred.
**Consequence:** the LingGen promotion gate has never measured the decoder.

## The finding

`_generate_with_situation_model` — the only path that runs the neural decoder —
sits at `response_gen.py:6909`. Every competing generation path returns before
it:

| line | path | strategy returned |
|---|---|---|
| 6719 | `_decomposition_generation_path` | `decomposed_what_is` |
| 6745, 6772, 6785, 6827 | `_definition_response` | `definition_response` |
| 6800 | `_decomposition_generation_path` | `decomposed_what_is` |
| 6823, 6896 | `_reflective_response` | `reflective_uncertainty` |
| ~6328 | `_metacognitive_uncertainty` | `metacognitive_uncertainty` |
| **6909** | **`_generate_with_situation_model`** | **`situation_model_decoder`** |

Instrumenting `_generate_with_situation_model` on the gate's own probe set shows
it is **never called**:

```
Q: tell me about love        [PATH] strategy='definition_response'
Q: describe the ocean        [PATH] strategy='decomposed_what_is'
Q: how do you learn?         [PATH] strategy='metacognitive_uncertainty'
```

`[SM called]` never prints. The decoder does not lose the gate — it never
enters the race.

## Why this makes the gate meaningless

`measure_linggen_ce.py` compares the template path against the decoder path by
flipping `use_linggen` and re-running the same query. But when the decoder path
is unreachable, flipping the flag changes nothing: the same upstream path
returns the same string. That is exactly why:

- 4 of 10 probes were **byte-identical** across the two paths (the probe now
  detects and excludes these, reporting `N_skipped_identical: 4`), and
- the remaining 6 were the template path's *fallback pools*, not decoder output.

So `decoder_ce` and `template_ce` were scored on text that came from the same
code path both times. The comparison could not have produced any other result.

## The ordering is deliberate, and wrong for this purpose

The ordering is not an accident — it is a safety hierarchy. KB definitions and
decomposition are more reliable than free generation, so they are tried first
and the decoder is the last resort. That is the right order for *producing a
reply*.

It is the wrong order for *measuring the decoder*. A gate that can only observe
the decoder on queries where every other path has already failed is a gate that
measures the fallback distribution.

## What a real gate needs

To measure the decoder, the probe must force the decoder path and nothing else:

1. **Bypass the upstream returns** for the measurement — call
   `_generate_with_situation_model` directly, or gate the earlier paths off.
2. **Score it against a reference it can be compared on** — the query's content
   words, expanded through the graph neighbourhood (already implemented in
   `measure_linggen_ce.py::_reference_ids`).
3. **Report the reachability rate** — what fraction of real queries ever reach
   the decoder at all. If that number is ~0, the promotion question is moot:
   the flag cannot change behaviour even if set.

## Note on the "thanks — i'll be more careful there" replies

Those are the correction acknowledgment (`engine_persistence.py:651`), produced
when `_detect_correction` flags a re-ask. The gate runs each query twice per
path, so the second run can see the same query as the previous turn and trip
the re-ask detector (overlap > 0.6, `user_model.py:6862`). That is a *probe*
artifact, not a product bug — but it means the gate's own harness can steer the
engine into a correction state mid-measurement.

## Verdict

The decoder is not "worse than the template path". It is **unmeasured**. Any
promotion decision taken on the current gate is unsupported, in either
direction.
