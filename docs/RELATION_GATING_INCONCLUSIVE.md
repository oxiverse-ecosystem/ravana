# Relation gating: implemented, inconclusive, and why

**Status:** implemented and tested, but the measurement is **not conclusive**.
**Consequence:** shipped off by default. The evidence does not support turning
it on.

## What was built

CPG commits to a proposition `(subject, relation, target)` but used only the
`target`. Since an unsupported assertion is precisely a **relation** the graph
does not support, relation-level gating boosts the surface words that realize
the committed relation too.

`CPGConfig.relation_words` maps a graph `relation_type` to its surface words.
Empty by default.

## The measurement problem

Same engine, same queries, two configurations — run twice:

```
run 1  TARGET-ONLY
  why do we dream      "It appears that dream carries through life. Connection begins..."
  tell me about trust  "i might be reaching, but Scientists there must management one learning."

run 1  RELATION
  why do we dream      "i might be reaching, but Archived history learning like psychology such own..."
  tell me about trust  "It seems to me that trust sparks respect freedom. Trust is earned through..."

run 2  TARGET-ONLY
  why do we dream      "i might be reaching, but Archived there one like learning life history suc..."
  tell me about trust  "i might be reaching, but Archived ethics such must one management like psy..."

run 2  RELATION
  why do we dream      "i might be reaching, but However there one such like learning."
  tell me about trust  "It seems to me that trust drives respect freedom. Trust is earned through..."
```

**The same configuration gives different output on different runs.** The
decoder samples with temperature, so single runs are noise — this is caveat #5
in the instrument's own docstring, and I walked straight into it.

Any comparison of two 4-query probes is therefore meaningless. A real verdict
needs the instrument (`--cpg`, hundreds of subjects), not an eyeball test.

## What can be said

* `trust` improved under relation gating in both runs — "trust drives respect
  freedom" / "trust sparks respect freedom" vs word salad. That is suggestive
  and consistent.
* `dream` did not improve.
* No configuration is uniformly better.

## Why this is the right thing to test next

The reasoning that led here is sound, independent of the noisy measurement:

1. CPG's boost moved on-topic 4.5× but left **unsupported assertions flat**.
2. Suppression (NoGo) failed — it removed supporting vocabulary.
3. A stronger boost failed — the logit vector is the wrong level.
4. An unsupported assertion **is** a relation the graph does not support.

So the relation is the remaining untested part of the committed proposition.
That reasoning stands even though this probe could not confirm the effect.

## What would settle it

Run the instrument with `--cpg` and a relation-gated config over the full
subject set, and compare `unsupported` and `supported` counts. That is a
~10-minute run and it is the only measurement with enough n to see through the
sampling noise.

Until that runs, relation gating stays **off by default** and this file records
it as an untested hypothesis with a plausible mechanism.

## Related

- `docs/BOOST_STRENGTH_NOT_THE_LEVER.md` — the previous negative result
- `docs/NOGO_SUPPRESSION_FAILS.md` — the suppression failure
- `docs/CPG_MEASURED_EFFECT.md` — the boost-only result that stands
