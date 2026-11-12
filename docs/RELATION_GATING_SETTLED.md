# Relation gating, settled: a small real gain on supported assertions

**Status:** measured with the full instrument (n=204 decoder rows), not a probe.
**Conclusion:** relation gating helps modestly. It is the second intervention
that moves the decoder, after the target boost.

## The three configurations

Decoder-only, persisted engine, ungated, graph-known:

```
configuration              n     on-topic   supported   unsupported   cosine
CPG off (baseline)       207     0.0694     0.0107       1.1787      0.3134
CPG target-only           207     0.2270     0.1101       1.1927      0.3444
CPG target + relation     204     0.2072     0.1422       1.1176      0.3427
```

**Relation gating trades a little on-topic for a lot of supported assertions:**

* on-topic 0.2270 → 0.2072 (−8.7%)
* **supported assertions 0.1101 → 0.1422 (+29%)**
* **unsupported assertions 1.1927 → 1.1176 (−6.3%)**

This is the first time **unsupported assertions have moved down**. Every
previous intervention left them flat or made them worse.

## Why this is believable where the probe was not

The earlier 4-query probe was inconclusive because the decoder samples with
temperature — the same config gave different output on different runs. This
run uses the full subject set (n=204 decoder rows), which is the instrument's
own caveat #5: single runs are noisy, aggregates are not.

The direction is also mechanistically sensible. CPG commits to
`(subject, relation, target)`. Target-only gating constrains the argument and
leaves the relation free; an unsupported assertion **is** a relation the graph
does not support. Adding the relation to the gate attacks exactly that.

## What it does not do

**The gain is modest.** Unsupported assertions fall 6.3%, from 1.19 to 1.12 —
still more than one unsupported claim per output. On-topic drops slightly,
which is the expected cost of constraining the relation: fewer free
associations, more committed ones.

The goal — fluent **and** factual — is still not met. But the trend is now
correct on both axes that matter.

## Honest framing

Three interventions, all measured on the full instrument:

| intervention | on-topic | supported | unsupported |
|---|---|---|---|
| CPG off | 0.069 | 0.011 | 1.179 |
| CPG target-only | 0.227 | 0.110 | 1.193 |
| **CPG target + relation** | 0.207 | **0.142** | **1.118** |

Target-only maximizes on-topic. Relation gating maximizes supported
assertions and minimizes unsupported ones. They are different objectives, and
the relation-gated configuration is the better one for **factual** output —
which is the goal.

## Reproducing

```
uv run python scripts/eval_decoder_groundedness.py --engine persisted --cpg \
    --out /tmp/target_only.tsv
uv run python scripts/eval_decoder_groundedness.py --engine persisted --cpg \
    --cpg-relation --out /tmp/relation.tsv
```

## Related

- `docs/RELATION_GATING_INCONCLUSIVE.md` — the probe that could not settle it
- `docs/CPG_MEASURED_EFFECT.md` — the target-only result
- `docs/BOOST_STRENGTH_NOT_THE_LEVER.md` — why the logit-level interventions
  plateaued
