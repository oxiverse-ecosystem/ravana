# The switch is wired and OFF: the decoder loses, and that is the finding

**Status:** the gate is implemented, wired, verified to fire, and **disabled by
default** because the measurement says the decoder should not win yet.

## What was built

`_rg_switch_enabled` (default `False`) gates `_rg_decide` at the top of
`_generate_response`. When on and the rule says COMPOSE, the decoder gets the
turn before the stored paths.

Placement was the whole problem. The first attempt put the gate just above the
definition lookup, but an unconditional `return self._reflective_response(ctx)`
sits above that point — so the gate was unreachable for exactly the turns it
exists to affect. Verified: `_rg_decide` fired **0 times**.

Moved above the decomposition block. Now, over 4 queries:

```
what is gravity      _generate_response=0  _rg_decide=0
tell me about trust  _generate_response=1  _rg_decide=1
what is respect      _generate_response=1  _rg_decide=1
what is love         _generate_response=0  _rg_decide=0
```

The gate fires on precisely the turns that reach the funnel.

## The measurement that decides it

Same engine, same queries, only the switch differs:

| query | switch OFF (template) | switch ON (decoder) |
|---|---|---|
| how does memory work | *"honestly, Memory is a bit outside what i know"* | **"It paves the way for brain. It is opposite to data. It sort..."** |
| why do we dream | *"honestly, Dream is a bit outside what i know"* | *"hmm, i'm not totally sure about Dream yet..."* |
| what is respect | correct KB definition | correct KB definition (unchanged) |

**The decoder produces fluent nonsense where the template path gave an honest
admission of ignorance.**

That is not a wiring bug. The decoder was trained to CE 1.570 / top-1 62.1%,
but CE rewards *fluency*, not *groundedness* — and "It is opposite to data" is
fluent. The template path's "I don't know" is worth more than a confident
sentence about nothing.

## Why the switch stays off

The rule is sound and the wiring is correct. What is missing is a decoder that
deserves the turn. Enabling the gate now would make RAVANA **worse** on exactly
the queries it currently handles honestly.

This is the same conclusion the CE promotion gate reached independently
(`promote_use_linggen: false`, decoder CE 3.8325 vs template CE 3.7692). Two
different methods, same answer.

## What would change this

The gate is a knob, and the practice term (§10.7) is designed for exactly this:
as the decoder answers more often *and passes the grounding gate*, its threshold
drops and it earns reachability. But it cannot earn that while it produces
word salad — so the ordering of work is:

1. **Improve decoder groundedness**, not fluency. The grounding gate
   (`_sm_response_grounded`) is the right instrument; the training objective
   is the problem.
2. **Then** let the practice term measure it.
3. **Then** consider enabling `_rg_switch_enabled`.

Enabling the switch before step 1 is how a system starts confabulating.

## Verification

- `tests/unit` — **2999 passed, 22 skipped, 0 failed**
- `tests/unit/test_retrieval_generation.py` — 26 passed
- CI on PR #315 — **10 pass, 0 fail**
- Switch OFF is byte-identical to before the change (measured)

## Related

- `docs/RETRIEVAL_VS_GENERATION.md` — the research and the rule
- `docs/RETRIEVAL_GENERATION_VALIDATION.md` — the rule on live signals
- `docs/DECODER_BLOCKED_BY_TWO_GATES.md` — why the decoder never ran
