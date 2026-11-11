# Session goal: wire `_adapt_verbosity_for_user` (the length lever)

## Context

`_adapt_verbosity_for_user` (`engine.py:13374`) is documented as *"roadmap §7
deliverable A"* — adaptive language complexity based on user familiarity and
arousal. It has **zero callers**.

It is defined on `CognitiveChatEngine` itself, so it is **not** dead code in a
dead mixin. It is simply never invoked.

This matters because **length alignment is the one style channel we have never
been able to move**. Measured repeatedly: RAVANA replies ~51 words to a user
writing ~22, and the alignment metric moves **+0.000**. The verbosity adapter
is the missing lever.

## What it does

```python
familiarity = um.infer_user_knows(subject)
goal        = um.last_goal
arousal     = um.emotional_state.get('arousal', 0.3)

target_intents = 3
if arousal < 0.25:  target_intents = 2   # calm user -> concise
elif arousal > 0.6: target_intents = 4   # excited user -> more engagement
# then familiarity trims further
```

It returns a modified `DiscoursePlan`. The docstring cites Baddeley & Hitch
1974 (7±2) for the capacity argument.

## The task

Wire it into the generation path so it actually runs.

**Do not change its logic.** The behaviour is already specified and cited.
The task is reachability.

### Where to look

The PFC discourse-plan path is in `response_gen.py`. Find where a
`DiscoursePlan` is built and where it is consumed, and call the adapter
between those two points.

Note: `_adapt_verbosity_for_user` takes `(plan, subject)`. `subject` may be
`None` on some paths — check how the caller obtains it and fail open if it is
absent (return the plan unchanged), matching the engine's usual fail-open
contract.

### Constraints

* **Do not change behaviour when familiarity/arousal are neutral.** The
  adapter must be a no-op for a user it knows nothing about.
* **Do not weaken any test.** If a test fails, the wiring is wrong.
* **Do not touch the ~60 early returns** in `_process_turn_impl`. That is a
  separate piece of work (a reachability audit is running concurrently).
  If the adapter cannot be reached without touching those, **stop and report**
  rather than widening scope.

### Verify

1. The adapter is called on a generating path. Instrument it temporarily or
   add a counter — prove it fires.
2. `tests/unit/test_style_conditioning.py` still passes.
3. The two integration suites still pass 17/0:
   `tests/integration/test_ravana_chat_pipeline.py`
   `tests/integration/test_unified_semantic_loop.py`
4. A reply for a high-familiarity user is shorter than for a low-familiarity
   user on the same query. If you cannot demonstrate this, say so plainly —
   do not claim the lever works because the call count is non-zero.

## Acceptance

`goal_cmd` must exit 0:

```
uv run --directory "C:/Users/Likhith/Documents/projects/ravana" python "C:/Users/Likhith/Documents/projects/ravana/scripts/check_verbosity_wiring.py"
```

That script asserts the adapter is reachable and reports the call count. It
must show **calls > 0** on a generating path.

## Out of scope

* The reachability audit of `_process_turn_impl` (~60 early returns) — running
  separately, and deliberately not touched here.
* Changing the adapter's thresholds or logic.
* The style-conditioning channels themselves (#303 already landed those).
