# Session goal: make the decoder fluent AND factual

## The problem, measured

RAVANA's neural decoder produces **fluent nonsense**:

```
'how does memory work'
  template path: "honestly, Memory is a bit outside what i know"
  decoder:       "It paves the way for brain. It is opposite to data. It sort..."

'why do we dream'
  template path: "honestly, Dream is a bit outside what i know"
  decoder:       "It paves the way to trust. It is connected to mind. It..."
```

Grammatical, confident, and about nothing. The decoder is currently switched
off (`_rg_switch_enabled = False`) because of this.

## The root cause, already identified

The training objective is **pure next-token cross-entropy**:

```
scripts/train.py    ce = nd._avg_cross_entropy
```

CE rewards *fluency*. Nothing in training rewards *saying something true*. So
the decoder learned to produce well-formed sentences with no constraint from
what RAVANA actually knows.

There is a runtime gate (`_sm_response_grounded`) that rejects off-subject
output — but a gate makes the decoder **rarer**, not **better**.

## The goal

**Decoder output that is both fluent and factually grounded in the graph.**

Not "make the gate stricter". Not "leave the decoder off". The decoder has to
become worth listening to.

## Non-negotiables

* **Every design decision must be grounded in cited neuroscience.** The user's
  standing instruction: for any decision about how this should work, research
  how the human brain does it and implement that. No ad-hoc engineering.
* **Measure before and after.** We have been burned twice by acting on
  assumptions. `scripts/eval_decoder_groundedness.py` is the instrument; it
  must show movement.
* **Do not weaken existing tests or gates.**
* **Keep the switch OFF until the measurement says otherwise.** Enabling a bad
  decoder makes RAVANA worse.

## Work order

**1. Measure (in progress).** `scripts/eval_decoder_groundedness.py` scores
decoder output on three axes: fluency, on-topicness, and **unsupported
assertions** (relation words attached to content words that are not graph
neighbours). This is the baseline.

**2. Research (in progress).** How does the brain constrain production so
utterances are both fluent and true? Candidates the literature points at:
the perceptual loop / self-monitoring in Levelt's model, ACC conflict
monitoring and the ERN, predictive-processing verification before articulation,
and the semantic constraint that is damaged in the fluent-but-empty speech of
semantic dementia.

**3. Implement the mechanism.** Shape depends on the research. Likely
directions:
   - a grounding term in the training objective (not just CE), or
   - a verify-then-articulate loop at decode time, or
   - both.
   The brain almost certainly does both: it learns constrained production
   *and* monitors its own output.

**4. Re-measure.** The unsupported-assertion count must fall while fluency
holds. If it does not, the mechanism is wrong — report that plainly.

**5. Only then consider enabling `_rg_switch_enabled`.**

## Acceptance

* `scripts/eval_decoder_groundedness.py` runs and reports a baseline table.
* The chosen mechanism is implemented with a citation for why it has that
  shape.
* Re-measurement shows fewer unsupported assertions at equal or better
  fluency.
* `tests/unit` stays green.
* A PR is opened, CI polled, and merged if green.

## Out of scope

* The retrieval paths (KB / web / decomposition) — they handle 69% of turns
  well and are not the problem.
* The ~40 outer early returns — mapped and understood, not the blocker.
