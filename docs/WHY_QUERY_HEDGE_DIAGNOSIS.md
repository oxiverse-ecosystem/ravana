# Why `why do you think it rains` cannot be answered

**Status: DIAGNOSED, NOT FIXED.** Three fix attempts were reverted because each
traded one wrong answer for another. This note records the mechanism so the
next attempt does not repeat them.

## The symptom

`why do you think it rains` returns the ignorance hedge:

> the closest i get to rains is rain and maybe fall — and from what i've seen,
> drops of liquid water falling from the sky.

`rain` **has** a definition (`drops of liquid water falling from the sky`) and
`consult_internal` **HITs** it on the same turn. The hedge is wrong.

## The mechanism (measured)

1. **The subject is inflected.** Grounding yields `rains`; the stored key is
   `rain`. Every lookup that tests the subject verbatim misses.

2. **`has_clean_definition` tests the subject verbatim**
   (`response_gen.py:8578`):

   ```python
   has_clean_definition = bool(
       subject and subject.lower() in getattr(self, '_definitions', {}))
   ```

   `rains` is not a key, so the query enters the **no-definition branch**.

3. **That branch ends in `_reflective_response`**, which is documented as
   *"half-known concepts (associations, no real definition)"*. It has **no
   definition lookup at all**, so any subject reaching it gets an ignorance
   hedge.

4. **`why` is unconditionally in `try_first`** (`response_gen.py:8402`), so the
   decomposition path runs before the definition check is reached.

## Why the three fixes failed

| attempt | what it did | result |
|---|---|---|
| 1 | skip decomposition for any defined subject | fixed this query, **regressed** `what would happen if the moon disappeared` to a flat definition |
| 2 | make `_reflective_response` return None for defined subjects | routed into `_generate_creative` → **"Rains carry through a rain"** (a tautology from the DMN free-association connector list) |
| 3 | singularize at `has_clean_definition` | same creative path → same tautology |

**The shared lesson:** the definition branch and the creative branch both sit
behind the same gate, and the creative branch (`_generate_creative`,
`response_gen.py:2921`) emits free-association lines built from a connector list
(`carry`, `whispers to`, `opens into`). Any fix that merely makes the subject
look "defined" without ensuring the **definition** is what gets emitted will
land there. The hedge is unpleasant; the creative line is worse.

## What a real fix needs

Do not patch the routing. Make the **definition branch answer `why` directly**:
when the query is `why <defined-subject>`, emit the definition *as* the answer
to the causal question (the definition of rain does describe how rain forms),
and keep the counterfactual route for hypotheticals. Concretely:

- gate on the **question type** (`why`/`how` vs `what would happen if`), not on
  whether the subject is defined;
- ensure the definition branch is entered, not the creative one;
- verify `moon disappeared` still gets a counterfactual.

## Related open defects

- `what would happen if the moon disappeared` returns a flat definition
  ("Moon is the only natural satellite of Earth.") instead of a counterfactual.
  **Pre-existing on clean main** (`e080ed3`), confirmed by reverting.
- The neural decoder's conditioning is near-null; six unchecked consumers were
  found on 2026-11-15.
