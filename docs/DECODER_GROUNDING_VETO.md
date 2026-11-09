# Why the decoder never speaks

**Status:** verified by tracing the live engine with the trained decoder loaded.
**Conclusion:** the decoder generates fluently and is vetoed by a single gate,
because the subjects it is asked about are not in the concept graph.

## The chain, end to end

For `how heavy is a rumour` with the trained decoder and `use_linggen=True`:

```
[_topic_grounded]  subj='heavy rumour'        -> False
[SM entered]       sit_vec=set coh=0.73 active=8
[gen]              24 tokens / 24 words (need >=5)
                   "holder herbalism nukala kick quantifies skywatching ..."
[word_salad]       -> False
[word_salad_any]   -> False
[ground]           'It seems to me that heavy rumour connects to wat...' -> False
[ground]           'I believe it shapes water. It builds on conditio...' -> False
[result]           None
```

Every stage passes except the last. The decoder produces 24 words -- far above
the `len(generated) >= 5` threshold -- and both degeneracy checks pass. It is
`_sm_response_grounded` that withholds the output, twice.

## Why the gate says no

`_sm_response_grounded` (`response_gen.py:3751`) requires condition (1): the
subject must have a verified factual anchor -- a stored definition, a
web-learned source, or graph membership plus a GloVe-close association.

Measured against the live graph:

```
subject              keywords  labels  definitions  sources
heavy rumour         False     False   False        False
echoes fade          False     False   False        False
taste of distance    False     False   False        False
silence              False     False   False        False
love                 False     False   False        False
time                 True      True    True         False
```

Every abstract subject is absent from the graph. This is the gate working as
designed -- the D4 comment in the source is explicit that an unknown subject must
not be grounded by free-association similarity alone (source monitoring, Johnson
1993). The decoder generates, and the monitor correctly says "you have no source
for that claim."

## The isolation probe set confirms it

20 abstract, web-unfriendly queries chosen so nothing upstream could answer them:

```
situation-model hits: 0 / 20
```

Strategies that won instead:

```
web_direct_answer           6
decomposed_what_is/how      3
metacognitive_uncertainty   5
analogy / absurd handlers   6
```

So it is not only the grounding gate. Three layers intercept before the decoder
is reached at all:

1. **Web / KB** -- answers factual queries directly (`web_direct_answer`)
2. **Self-model / analogy / decomposition / absurd-handler** -- handles the rest
3. **Uncertainty templates** -- catches what is left

And on the queries that DO reach the situation-model path, the grounding gate
vetoes the output because the subject is not in the graph.

## What this means

The promotion gate cannot be evaluated on any existing probe set, because the
decoder never produces a competing reply. `promote_use_linggen: false` is
**uninformative, not negative** -- there is nothing to compare.

Two independent blockers, both about graph coverage rather than decoder quality:

- **Reachability** -- the decoder is behind web, KB, analogy, decomposition and
  uncertainty paths, and loses to all of them.
- **Grounding** -- when it does run, its output is withheld because the subject
  has no verified anchor in the graph.

Both resolve the same way: as RAVANA learns concepts into the graph, more
queries reach the decoder and more of its output passes the gate. That makes the
original goal -- real-time learning from conversation -- the actual prerequisite
for template-free generation, rather than a separate workstream.

## Note on the generated text

The 24-word sample is fluent word salad: grammatical local transitions, no
semantic commitment. That is what a 256-hidden GRU trained on 14k sentences
produces. It is not a bug, and it is not fixable by plumbing -- but it is also
not being emitted, because the gate withholds it.
