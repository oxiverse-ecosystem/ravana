
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

# Direct probe: taught relations ARE stored; the query path drops them

`scratch/rsi_declaration_probe.py` / `scratch/rsi_declaration_probe.log`.
Isolated copy of production weights, one engine, offline, fixed hash seed.
Exact RSI demo phrasing plus the query prompt; then a full incident-edge census.

## Before teaching

- `gale`: 0 nodes, no edges.
- `thaw`: 0 nodes, no edges.
- `mist`: 0 nodes, no edges.
- `rainbow`: 69 incident edges, only one trusted (`dispersion→rainbow`,
  curriculum) — unrelated to the demonstration.

## After two teaching turns

| turn | engine emission |
|---|---|
| `gale causes rainbow` | "hmm, i'm not totally sure about that yet…" |
| `mist causes thaw` | "i don't really have a solid grasp on that so far…" |
| `what is true of thaw? Options: sunset, typhoon, mist, hail…` | "honestly, **hail** is a bit outside what i know right now…" |

**But the graph now contains:**

- `gale → rainbow`, `causal`, `edge_kind=curated`, `source=conversation`
- `mist → thaw`, `causal`, `edge_kind=curated`, `source=conversation`

all with live nodes indexed in `_concept_keywords`.

## Conclusion

The hedge emission is only the *surface* reaction. The engine **does** write
the taught relation as a trusted, directional, provenance-stamped edge — so
RSI's near/far 0.00 is **not** a storage failure. The failure is downstream,
at query time: the answer path resolved the wrong subject (`hail`, one of the
options, instead of `thaw`) and did not consult the just-taught edge.

This corrects the earlier hypothesis in `RSI_LIVE_RULE_INDUCTION.md` (F1
storage-first). The next fix must target query-time subject resolution and
evidence gathering from trusted incident edges, with the same brain grounding
(schema assimilation / pattern completion) and no threshold changes.

Falsifier for the new direction: if a forced-choice query supplies `thaw`
explicitly and the engine still emits no candidate while the trusted edge is
present, the bottleneck is articulation rather than subject resolution.
