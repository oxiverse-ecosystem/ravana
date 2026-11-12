# Escalation: verified

**Status:** the escalation path (`_cpg_escalate`) works as specified. It ships
**OFF** because it needs a large-n measurement before enabling, not because it
is broken.

## What it does

Step 3 of `docs/GROUNDING_MECHANISM.md`: when no proposition clears the
criterion, decline to compose and escalate to honest uncertainty instead of
generating unguided.

## Verified on a subject genuinely absent from the graph

```
absent from graph: True

ESC OFF  -> "I believe word is opposite to definition."
ESC ON   -> "honestly, xqzvwnonalword is a bit outside what i know right now what's your sense of it"
```

The OFF output is exactly the failure the mechanism targets: fluent,
confident, and false. The ON output declines to assert.

## A false alarm worth recording

My first probe used `zzzunknown` and saw 10 propositions clear theta for a
subject that "does not exist". That looked like the early-return guard was
broken. It was not -- **`zzzunknown` was in the graph**:

```
zzzunknown   nids=[407]  node label='zzzunknown'  outgoing=8
qxzblorp     nids=[420]  node label='qxzblorp'    outgoing=1
```

An earlier eval run's `_auto_expand_concepts` had minted nodes for those
subjects. The escalation path was correct; my test subject was contaminated.

**Lesson:** "unknown subject" probes must check `_concept_keywords` first.
The eval instrument mints nodes for whatever it queries, so any subject it has
ever touched is graph-known.

## Why it ships OFF

The code path works and the mechanism is correct. What is missing is the
measurement: on the full 446-subject instrument, how often does escalation
fire, and does it improve the groundedness metrics rather than just replacing
one kind of output with another?

Enabling it without that measurement would be a change made on a 2-query probe,
which is the mistake the instrument was built to prevent.

## What would settle it

Run `scripts/eval_decoder_groundedness.py --cpg --cpg-escalate` against the
446-subject instrument and compare all three axes against the `--cpg` run.
If escalation improves unsupported without collapsing on-topic, enable it.

## Related

- `docs/GROUNDING_MECHANISM.md` — step 3, the escalation path
- `docs/THETA_LOWER_IS_BETTER.md` — the criterion this interacts with
- PR #328 — ships `_cpg_escalate` off by default
