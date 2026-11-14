# Coverage: two bugs, not sparsity

**Status:** measured. The "typed-edge sparsity" hypothesis was wrong; the
constraint was two mapping bugs.

Instrument: funnel over 40 graph-discovered subjects, persisted engine.

## The funnel, before

```
40 subjects
 ├─ 5  no outgoing edges
 ├─ 16 no TYPED edge (only 'semantic')   <-- assumed to be the bottleneck
 └─ 19 have a typed edge
      ├─ 7  plan built but NOT committed
      └─ 12 committed                      <-- measured as "7/40 coverage"
```

Earlier notes called this "typed-edge sparsity". It was not. **19/40 subjects
had typed edges**, and 7 of those 19 were being thrown away by the commitment
gate.

## Bug 1: the graph's relation vocabulary is adjectival

The graph emits `causal`, `analogical`, `contrastive`, `temporal`, `episodic`,
`inferred`. `_construction_for` only knew the verbal spellings from
`graph_typing.TYPED_RELATION_TYPES` (`causes`, `is_a`, `has_property`).

So every causal/analogical edge fell through to the `relate` fallback, and
`plan_is_committed` -- correctly -- rejects `relate` as a vacuous proximity
claim:

```
gravity  causal edge to 'earth'  ->  "gravity is related to earth"   REJECTED
water    causal edge to 'life'   ->  "water is related to heavy"     REJECTED
light    causal edge to 'sun'    ->  "light is related to soft"      REJECTED
```

The relation was there in the graph and lost in the mapping. Fixing the
vocabulary took coverage from 12/19 to **19/19**.

## Bug 2: target selection picked hubs, not edges

`spreading_activation` redistributes a fraction of the TOTAL field to every
candidate, so a high-degree neighbour wins on connectivity rather than on the
quality of its own edge. And the winning binding's relation was not necessarily
the strongest relation the graph offered.

Fixed with `select_target` / `message_from_best_edge`: the target is chosen by
EDGE strength, and the relation comes from that SAME edge, so the
(subject, relation, target) triple is mutually consistent. This is also more
brain-faithful -- lemma selection competes among lemmas for a role the message
has already committed to, not among every neighbour for the whole message.

## Result, wired end-to-end (40 subjects)

```
plan route n=15  on_topic=0.3267  unsupported=0.0667
fallback   n=12  on_topic=0.3232  unsupported=0.9167
```

| metric | before this change | after |
|---|---|---|
| plan-route firings | 6 | **15** |
| plan-route unsupported | 0.143 | **0.067** |
| fallback unsupported | 0.600 | 0.917 |
| on-topic (plan vs fallback) | 0.344 vs 0.340 | 0.327 vs 0.323 |

**Unsupported assertions on the plan route: 0.067 vs 0.917 for the fallback --
a 13.7x reduction**, at equal on-topic.

And the sentences are now semantically right:

```
gravity contributes to orbit
light prompts photosynthesis
memory drives data
love leads to life
ocean forms sea
music relates to rhythm
```

Before: `water leads to heavy`, `light leads to soft`, `trust is like
management`.

## What is still true

- **16/40 subjects have no typed edge at all.** That IS sparsity, and it is
  now the real bottleneck. Fixing it is concept acquisition, not code.
- **5/40 have no edges.**
- The plan route still fires on only ~half of subjects. It is a precision
  route, not a replacement.
- `trust is different from must` and `able is connected to enough` are still
  weak -- the graph's edges for those subjects are weak, and no selection rule
  can repair a weak edge.

## Reproduce

```
uv run python scripts/measure_plan_narrative.py --max-subjects 40
```

## Related

- `docs/PLAN_NARRATIVE_POSITIVE.md` -- the original positive result
- `docs/PLAN_PREFIX_MEASURED_NEGATIVE.md` -- the GRU negative
- `ravana/src/ravana/learn/macroplanning.py` -- `select_target`
- `ravana/src/ravana/learn/linearization.py` -- `_construction_for`
