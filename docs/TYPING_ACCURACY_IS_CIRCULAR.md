# Online relation typing: the earlier accuracy was circular

**Status:** the 75.5% number was an artifact. Independent measurement gives
**59.4%**, and **0/13** on the relation types that matter.

Instrument: `scripts/measure_typing_independent.py` -- 32 hand-labelled
(subject, target, relation) triples, labelled by reading the words.

## Why the earlier number was wrong

`docs/ONLINE_RELATION_TYPING.md` reported 75.5% recovery on "known typed
edges". Those labels are not independent: `_correct_relation_types`
(`chain_walker.py:1479`) runs at seed time and re-labels edges using the SAME
prototype mechanism that `_infer_relation_type` uses. The measurement was
scoring the mechanism against its own output.

The persisted graph confirms there is no independent label source:

```
isa 0, has_property 1, capable_of 0, used_for 0, part_of 1
```

Essentially no ConceptNet-typed edges survive. So "known typed edge" meant
"an edge the prototype mechanism typed".

## Independent result

```
relation         n  correct  mean_conf  examples
-----------------------------------------------------------------------------
is_a             7    7/7     0.286  dog->mammal=semantic, robin->bird=semantic
part_of          4    4/4     0.300  wheel->car=semantic, leaf->tree=semantic
has_property     4    4/4     0.300  ice->cold=semantic, glass->fragile=semantic
capable_of       2    2/2     0.300  bird->fly=semantic, fish->swim=semantic
used_for         2    2/2     0.300  knife->cut=semantic, pen->write=semantic
causes           4    0/4     0.292  fire->smoke=semantic, rain->flood=semantic
temporal         3    0/3     0.293  spring->summer=contrastive, seed->sapling=semantic
analogical       3    0/3     0.233  heart->pump=semantic, brain->computer=semantic
contrastive      3    0/3     0.300  hot->cold=semantic, light->dark=semantic
-----------------------------------------------------------------------------
TOTAL           32   19/32   (0.594)
```

**The 19 "correct" are all `is_a`/`part_of`/`has_property`/`capable_of`/
`used_for` -- which map to `semantic`, the mechanism's catch-all.** They are
correct only because the label was aliased to the default. Every relation that
requires actual discrimination scored **zero**:

- `fire -> smoke` typed `semantic` (should be causal)
- `spring -> summer` typed `contrastive` (should be temporal)
- `heart -> pump` typed `semantic` (should be analogical)
- `hot -> cold` typed `semantic` (should be contrastive)

## The cause: a vocabulary mismatch, not a tuning problem

The prototype inventory is `['analogical', 'causal', 'contrastive',
'semantic', 'temporal']`. The graph's typed vocabulary is `isa`, `has_property`,
`capable_of`, `used_for`, `part_of` -- **none of which are in the inventory.**

So the mechanism cannot express "is a kind of" or "has property" at all. It
can only separate causal / temporal / analogical / contrastive from the
catch-all. And on those four it scores 0/13.

The mean confidence is ~0.29 for everything, correct or not. **Confidence does
not separate right from wrong** on independent labels -- so the 0.35 threshold
is not measuring what `docs/ONLINE_RELATION_TYPING.md` claimed.

## What this means for the plan route

The plan route's numbers were measured with the broken typing wired in (see
the bug below), so they are an upper bound. Current wired state after fixing
the wiring bug:

```
plan route n=19  on_topic=0.3368  unsupported=0.0526
fallback   n=7   on_topic=0.3769  unsupported=0.7143
```

`unsupported=0.053` measures *graph-supportedness*, not truth. A mis-typed
relation still uses graph-neighbour words, so the metric cannot see the error.
The samples show it: `time reflects early`, `trust is similar to must`.

## The wiring bug this uncovered

`_infer_relation_type` lives on `ChainWalkerMixin`, which `ResponseGenMixin`
inherits -- so it is on the engine. My first wiring looked for
`self.chain_walker`, an attribute that does not exist (the name appears only
in comments). `getattr` returned `None` and **every inference was silently
skipped**, so the plan route was running with online typing disabled while
appearing to be enabled. Fixed to call `self._infer_relation_type` directly.

This is the same class of bug the repo documents repeatedly: a mechanism that
is wired and inert.

## What to do

1. **Expand the prototype inventory** to the graph's typed vocabulary
   (`isa`, `has_property`, `capable_of`, `used_for`, `part_of`) alongside the
   four it has. Without this, no amount of threshold tuning helps.
2. **Re-fit the threshold on independent labels**, and report a margin
   histogram (correct vs incorrect) -- if margin does not separate them, the
   gate is not measuring anything.
3. **Do not trust `unsupported=0` as truth.** It is graph-supportedness.
4. Keep the query-time-only rule: an inferred type must never be persisted.

## Reproduce

```
uv run python scripts/measure_typing_independent.py
```

## Related

- `docs/ONLINE_RELATION_TYPING.md` -- the superseded (circular) measurement
- `docs/BRAIN_RELATION_TYPING_SPEC.md` -- the research spec, which predicted
  exactly this ("the graph's typed vocabulary is absent from the current 5-type
  inventory, so isa edges can only be typed by accidental vector alignment")
- `ravana/src/ravana/chat/chain_walker.py:608` -- `_init_relation_prototypes`
- `ravana/src/ravana/chat/chain_walker.py:1479` -- `_correct_relation_types`
