# Explanation chains: the mechanism works, the graph does not

**Status:** assembler built and measured. `learn/explanation.py`,
`scripts/measure_explanation_chains.py`. **Not wired into the response path** —
the measurement below is why.

## What was built

Stages 2-3 of `docs/BRAIN_EXPLANATION_AND_REASONING.md` §5.2: enumerate simple
paths of 2-4 hops over `relation_type == "causal"` edges only, score by
`Σ log(w·c) − λ·len − μ·unprovenanced`, then grade with a separate monitor
(coverage, typed fraction, provenance, gaps, margin over the runner-up).

## Measured on the live graph

```
graph nodes=259  edges=380  causal=66
subjects scanned: 259
subjects with a 2-4 hop causal chain: 17
verdicts: {'assert': 14, 'hedge': 0, 'decline': 3}

air              air -> sound -> hear            adequacy=1.000 -> assert
animal           animal -> plants -> animals      adequacy=1.000 -> assert
earth            earth -> sun -> light           adequacy=1.000 -> assert
fall             fall -> vibrate -> air           adequacy=1.000 -> assert
gravity          gravity -> earth -> life         adequacy=1.000 -> assert
light            light -> sun -> light            adequacy=1.000 -> assert
photosynthesis   photosynthesis -> plants -> animals
plant            plant -> photosynthesis -> plants
```

## The mechanism is fine; the graph is wrong

Every "explanation" above is graph-adjacent but not an explanation:

```
gravity -> earth        (gravity does not cause the earth)
air -> sound            (air does not cause sound)
earth -> sun -> light   (the earth does not cause the sun)
light -> sun -> light   (a cycle, and not causal)
```

The edges exist with high weight and confidence:

```
gravity  causal -> earth   w=0.85 c=0.90
gravity  causal -> orbit   w=0.80 c=0.90
air      causal -> sound   w=0.90 c=0.90
earth    causal -> sun     w=0.80 c=0.90
```

So the graph's causal edges are **not truthful**, and no chain assembler can
repair that. This is the same finding as `what causes rain` → "Rain leads to
heat, and that in turn leads to climate": the grounding metric measures
graph-support, not truth, so a wrong edge scores as well as a right one.

## The gate has no truth term, and it shows

`adequacy=1.000` for `air -> sound -> hear`. The monitor grades *structure*
(typed fraction, provenance, margin) and every component is maximal because the
edges carry high confidence. **A structural monitor cannot detect a false
chain** — the same limitation as the groundedness metric, one level up.

This is the fourth instance in this project of a proxy that cannot see the
property it is supposed to measure:

| proxy | claimed | independent |
|---|---|---|
| relation typing | 75.5% | 0/32 |
| determiner frame | 0.787 | 0.730 |
| noun-suffix table | 0.803 | 0.608 (worse than baseline) |
| explanation adequacy | 1.000 on a false chain | no truth term exists |

## What this means for front 4 (thinking, 33/100)

Building the chain route on this graph would **increase** confident wrong
output. The honest options, in order:

1. **Do not wire it.** Report the finding. Front 4 stays at 33, but no new
   false explanations are emitted.
2. **Wire it behind a truth source.** Only assert a chain whose hops come from
   a provenance the system trusts (a stored definition, a vetted web fact) —
   not from the co-occurrence-derived graph edges that produced
   `gravity -> earth`.
3. **Fix the graph's causal edges at the source.** Highest leverage, largest
   scope: the edges are learned by co-occurrence, which cannot distinguish
   "co-occurs with" from "causes".

Option 2 is the only one that both ships and is honest. Option 3 is the real
fix and is where the web-relation work (#378) was heading.

## Reproduce

```
uv run python scripts/measure_explanation_chains.py
```

## Related

- `docs/BRAIN_EXPLANATION_AND_REASONING.md` §5.2 — the spec this implements
- `docs/SEVEN_FRONTS_BASELINE.md` — front 4 baseline
- `docs/WEB_LEARNING_CHAIN_VERIFIED.md` — the web path that can supply
  provenance-trusted causal edges
