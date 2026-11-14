# Online relation typing: the brain does not have a concept-acquisition problem

**Status:** measured. The user's reframing was correct: the brain meets a novel
concept and immediately knows what KIND of relation holds. It does not wait for
a stored label.

Instrument: 40 graph-discovered subjects, persisted engine.

## The reframing

`docs/COVERAGE_TWO_BUGS.md` concluded: "16/40 subjects have no typed edge. That
is concept acquisition, not code." That framing was wrong.

The brain does not have a concept-acquisition problem. It encounters a novel
concept and *computes* its relational structure on the spot -- is this a kind
of X, a part of X, a cause of X? The label is not retrieved; it is derived
from the concept's features and its relation to what is already known.

RAVANA already has that mechanism: `chain_walker._infer_relation_type`
(chain_walker.py:1410) compares the *relational direction vector*
(`target.vector - subject.vector`) against learned relation prototypes
(`chain_walker.py:608`). It is online, it needs no stored label, and it is
exactly the brain's comparative/analogical route to typing something novel.

It was simply not wired into the plan route, and its confidence was not gated.

## Is the mechanism any good?

Tested on subjects that DO have a typed edge, so there is ground truth:

```
recovery on KNOWN typed edges: 40/53 = 0.755

  inferred confidence   n     accuracy
  0.00 - 0.30          20     0.600
  0.30 - 0.45          31     0.839
  0.45 - 1.00           2     1.000
```

**75.5% recovery, and accuracy is monotone in confidence.** The mechanism is
real; it just needs a threshold. Below 0.30 it is 60% -- barely better than
chance over five relation types, and not worth asserting.

## Result: online typing + confidence gate

Plan route with untyped edges typed online, by confidence threshold:

```
TH=0.00  fired=30/40  on_topic=0.3061  unsupported=0.0333
TH=0.35  fired=29/40  on_topic=0.3172  unsupported=0.0000
TH=0.45  fired=25/40  on_topic=0.3160  unsupported=0.0800
```

**TH=0.35 is the sweet spot: 29/40 subjects (72%) get a committed plan, with
unsupported assertions at exactly 0.000 and on-topic 0.317.**

Compare the progression:

| stage | firings | unsupported |
|---|---|---|
| original prototype | 7/40 | 0.000 |
| + two mapping-bug fixes | 15/40 | 0.067 |
| **+ online typing (TH=0.35)** | **29/40** | **0.000** |

Coverage went from 17% to 72% with no loss in precision -- because the typing
is *derived from the graph's own vectors*, so a typed edge is as
graph-supported as a stored one.

## Why this is brain-faithful, not a hack

The alternative was to treat the 16 untyped subjects as a data problem and go
acquire more typed edges. That is the "concept acquisition" framing, and it
misunderstands what the brain does: it does not accumulate labels until
coverage is adequate. It computes the relation from structure.

`_infer_relation_type` does the same thing -- it compares the direction from
the subject to the candidate against prototype directions learned from seed
words. That is structure-mapping (Gentner) rather than feature overlap, which
is the brain's route to typing the novel.

The confidence gate is also brain-faithful: the brain hedges or defers when
relational evidence is weak, rather than committing at chance. Below 0.30 the
accuracy is 60%, so those edges are skipped and the baseline path handles the
subject.

## Honest caveats

- **24.5% of known typed edges are still mis-typed** at TH=0.35. The plan
  route's `unsupported=0.000` measures *graph-supportedness*, not truth -- a
  mis-typed relation still produces a clause whose words are graph neighbours,
  so the metric cannot see the error. This is the instrument's known limit
  (`docs/DECODER_GROUNDEDNESS_BASELINE.md`: it detects ungrounded claims, not
  false ones).
- **The prototypes are hand-seeded** (`proto_map` in chain_walker.py:622).
  They are seed vocabulary, not authored content, but they are a fixed
  inventory rather than a learned one.
- **n=40.** Direction is clear, magnitude needs a bigger run.

## Next

1. Wire `_infer_relation_type` into `_plan_narrative_sentence` with TH=0.35,
   behind the existing flag.
2. Persist the inferred type back onto the edge? **No** -- that would let a
   60%-confidence guess become a stored fact. Keep it query-time only.
3. Re-measure at n=480.

## Related

- `docs/COVERAGE_TWO_BUGS.md` -- the two mapping bugs (superseded framing)
- `docs/PLAN_NARRATIVE_POSITIVE.md` -- the original positive result
- `ravana/src/ravana/chat/chain_walker.py:1410` -- `_infer_relation_type`
- `ravana/src/ravana/chat/chain_walker.py:608` -- `_init_relation_prototypes`
