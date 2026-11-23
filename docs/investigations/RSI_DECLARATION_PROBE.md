
## Read-side fix verification (same probe, 2026-11-23)

Two changes on branch `fix/rsi-query-subject`:
1. forced-choice option tails and leading framing clauses are stripped before
   topic extraction (`CognitiveChatEngine._topic_input_for_prompt`);
2. a trusted incident edge with `source=conversation` is admitted as topical
   grounding in `ResponseGenMixin._topic_grounded` (new evidence type, no
   similarity threshold changed).

Result: the query subject is now **`thaw`** (was `hail`, then `above`) — the
option-token bug is fixed. The emitted answer is still chatter:
`"true thaw — that's a fun image! Are you imagining a scenario..."`, with no
candidate and no use of the trusted `mist→thaw` edge.

This meets the falsifier stated in `RSI_DECLARATION_STORAGE.md`: trusted edges
are present and the query subject is correct, yet no candidate is emitted.
**The remaining bottleneck is answer construction/articulation, not storage or
subject resolution.** Do not tune the read-side gates further; the next change
must make the answer builder consult incident trusted edges when selecting a
forced-choice candidate, with its own falsifier and no threshold change.
