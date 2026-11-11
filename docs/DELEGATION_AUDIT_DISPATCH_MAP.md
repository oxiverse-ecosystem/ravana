# Delegation audit: the dispatch map, and one claim I could not reproduce

**Status:** the codebase map is accurate and useful. One empirical claim in it
could not be reproduced and is recorded here as **unverified** rather than
adopted.

## What was delegated

Two parallel subagents after the 200-query audit found 69% of turns reach the
generation tail but the decoder still never fires:

1. Map the exact dispatch order with `file:line` citations.
2. Research the brain's retrieve-vs-generate switch and propose a rule.

The research deliverable is `docs/RETRIEVAL_VS_GENERATION.md` (635 lines) and
was implemented in `ravana/src/ravana/learn/retrieval_generation.py` (PR #315,
merged `d3e250c`).

## The dispatch map is accurate

I spot-checked it against source. Two layers, ~40 outer gates and ~25 inner
ones:

- **Outer** (`_process_turn_impl`, `engine.py:7769`) — an ordered early-return
  chain. `_consult_internal_knowledge` is gate #10 at `engine.py:8703`; HIT
  returns at 8704, MISS falls through to `_try_memory_query` (8719).
- **Inner** (`_generate_response`, `response_gen.py:7189`) — social, humor,
  self-model, emotional, counterfactual, decomposition, curated definition, web
  direct answer, abstract gate, stored definition, KB grounding, the
  counterfactual/reflective fallback, the ACC/ERN `_topic_grounded` veto, then
  four `use_linggen` gates, then the PFC dorsal/ventral split, then
  `_graph_fallback_response` as the terminal case.

It also correctly lists the real gates and thresholds (`_reg_curiosity >= 0.5`,
`knowledge_confidence < 0.4`, `_topic_grounded` at 0.45/0.60,
`_is_word_salad` / `_unique_ratio >= 0.35` on decoder output). Useful reference.

## The claim I could not reproduce

The agent reported:

> "Confirmed: the decoder IS reachable — but only when
> `_reflective_response` returns `None`."
>
> ```
> Q='why do we sleep'          → decoder:1, dorsal:1, reasoning_loop:1
> Q='how do black holes form'  → decoder:1, dorsal:1, reasoning_loop:1
> ```

Its trace shows those runs produced `strat='dorsal_reasoned'` — the **dorsal**
path, not the reflective-declines path. So the result came from a different
route than the one it was attributed to.

I tried to reproduce both readings, loading the real persisted state
(`use_linggen: True`):

**Forcing `_reflective_response` to return `None`:**

```
why do we sleep       sm=0 decoder=0 refl_called=0  -> "Sleep leads to disease."
how do black holes form sm=0 decoder=0 refl_called=0 -> "honestly, that is a bit outside..."
tell me about the ocean sm=0 decoder=0 refl_called=0  -> "The ocean is the body of salt water..."
```

`refl_called=0` — the reflective gate was never even reached. Those turns
return upstream (KB hit, honest-uncertainty gate, web answer).

**Instrumenting the dorsal and ventral paths directly:**

```
why do we sleep       sm=0 decoder=0 dorsal=0 ventral=0
how do black holes form sm=0 decoder=0 dorsal=0 ventral=0
tell me about the ocean sm=0 decoder=0 dorsal=0 ventral=0
```

Zero across the board. I could not make the decoder fire on these queries by
any route I tried.

## What this means

The agent's *map* is right and I have adopted it as a reference. Its
*empirical claim* about decoder reachability is **unverified** — I could not
reproduce it, and its own trace attributes the result to a different code path
than the one it named.

This does not change the merged conclusion. The switch rule is implemented,
wired, verified to fire on the turns that reach the funnel, and **disabled**
because the measurement shows the decoder produces fluent nonsense where the
template path gives an honest "I don't know":

```
how does memory work
  OFF: "honestly, Memory is a bit outside what i know"
  ON:  "It paves the way for brain. It is opposite to data. It sort..."
```

If the agent's claim is right and there *is* a dorsal route that reaches the
decoder, that would be worth knowing — it would mean the decoder runs
sometimes in production. But it needs reproducing before it is acted on.

## Lesson

A subagent's self-report is not evidence. Its map was verifiable by reading
source and checked out; its measurement was not, and did not. Verify the
claim that changes the decision, not the one that is merely convenient.
