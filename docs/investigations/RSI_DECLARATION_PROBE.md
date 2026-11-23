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
