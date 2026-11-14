# Two proposition sources: the union reaches 94.4%

**Status:** measured and wired. `scripts/measure_proposition_union.py`.

## The two sources

The plan route needs a `(relation, target)` pair. There are two independent
ways to get one:

- **A. typed graph edge** -- the strongest non-vacuous edge, target resolved
  from the graph.
- **B. stored definition** -- the copula read off the definition text
  (`learn/definition_proposition.py`), target resolved to a graph concept.

They fail differently: A needs a typed edge, B needs a stored definition whose
predicate noun is in the graph.

## Coverage

**On the 324 stored definitions** (the subjects RAVANA can answer):

```
typed-edge route      269  (83.0%)
definition route      222  (68.5%)
union                 306  (94.4%)
  both sources        185
  typed edge only      84
  definition only      37

union on_topic=0.5266  unsupported=0.0621
```

The sources are genuinely complementary: 84 subjects have only a typed edge,
37 have only a definition. The union is 11 points above the better single
source.

**On all 2,695 graph-discovered subjects:** union 44.4% (typed 43.4%,
definition 7.8%). The definition route adds little there because most of those
subjects have no stored definition -- definitions are the scarcer resource.

## Wired result

`_plan_narrative_sentence` now falls back to the definition source when no
typed edge commits. On 60 eval subjects:

```
before:  plan n=15  unsupported 0.133   fallback n=9   unsupported 0.667
after:   plan n=23  unsupported 0.0435  fallback n=19  unsupported 1.4211
```

Coverage 15 -> 23, and unsupported on the plan route 0.133 -> **0.0435**.

```
music      is a rhythm          (was "music are a rhythm")
language   is a primary
abstract   is similar to objects
abstraction precedes a abstract
```

## Two bugs this uncovered

1. **`music are a rhythm`** -- `VerbLexicon` SAMPLES the copula from a pool
   (`is` / `are`) without reference to the subject. The realizer now resolves
   the copula deterministically from the subject's number.
2. **`music am a rhythm`** -- the copula pool contained `am`, which only
   agrees with "I". Removed; the engine never makes a concept the subject of
   its own definition.

## Honest limits

- `trust differs from a must`, `able is opposite to a enough`,
  `above feeds into feet` -- the graph's edges for these subjects are weak,
  and no selection rule repairs a weak edge.
- `abstraction precedes a abstract` -- a tautology that passed because the
  target IS a graph concept. Needs a subject≠target check in the definition
  route.
- on_topic is measured against the graph, not truth.
- The definition route still needs a stored definition. Subjects with neither
  a definition nor a typed edge get the honest hedge, which is correct.

## Reproduce

```
uv run python scripts/measure_proposition_union.py
```

## Related

- `docs/DEFINITION_PROPOSITION_RECALL.md` -- the definition parser (80% recall)
- `docs/PLAN_NARRATIVE_POSITIVE.md` -- the plan route
- `docs/TYPING_ACCURACY_IS_CIRCULAR.md` -- why the vector proxy is not used
