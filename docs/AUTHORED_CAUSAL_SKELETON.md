# Where `rain → heat → climate` actually comes from

**Status:** root cause found. Not a co-occurrence artefact, and not the online
typer. `docs/EXPLANATION_CHAINS_MEASURED.md` blamed "co-occurrence-derived"
edges; that was **wrong** and this note corrects it.

## The wrong hypothesis

The explanation-chain measurement showed chains like `gravity -> earth ->
life`, `air -> sound -> hear`, `earth -> sun -> light`, and concluded the
graph's causal edges are co-occurrence noise. The edges did carry
`edge_kind='unknown'`, which looked consistent.

## What the code actually does

`engine_reasoning.py:3185` `_seed_physics_causal()` authors a **physics causal
skeleton offline** and writes it into the live graph:

```python
("sun", "light", "causal", 0.95)
("sun", "heat", "causal", 0.9)
("heat", "climate", "causal", 0.8)
("earth", "life", "causal", 0.8)
("gravity", "earth", "causal", 0.8)
("air", "sound", "causal", 0.9)
("tree", "fall", "vibrate", "air", "sound")   # the "tree falls" counterfactual
```

Its own docstring is explicit that these are *"universal causal priors
(innate-like), authored offline"*. So:

- `rain → heat → climate` is not learned from anything. `heat → climate` is an
  authored prior; `rain` gets wired to `heat` by the nearest-neighbour
  auto-expansion, and the chain reads as if it were knowledge.
- Every "false chain" in `EXPLANATION_CHAINS_MEASURED.md` is a walk over this
  authored skeleton, not over co-occurrence noise.

## Why the provenance gate did not catch it

`_seed_physics_causal` calls `graph.add_edge(...)` directly and **never stamps
`source_metadata`**. So the authored priors carry `edge_kind='unknown'` --
indistinguishable, to the gate, from co-occurrence noise.

That is the real defect: **the system's own hand-written knowledge is
unlabelled, so it cannot be told apart from learned association.** The gate
declines all of it, which is safe but loses the priors the simulator needs.

## The fix that follows

Stamp authored priors at the source with a provenance class of their own --
`curriculum` -- which `learn/explanation.py::provenance_trusted` already
accepts. Then:

- the explanation gate can assert `sun → light → photosynthesis` (a real prior),
- and still decline `gravity → earth` if that edge is in fact an authored
  mistake rather than a prior.

That second point needs a judgement call per edge, not a blanket stamp. The
skeleton contains at least one edge that is not a prior at all:
`("gravity", "earth", "causal", 0.8)` -- gravity does not cause the earth.

## Corrections to earlier notes

| earlier claim | actual |
|---|---|
| "the graph's causal edges are co-occurrence-derived and not truthful" | they are **authored offline** by `_seed_physics_causal` |
| "all 380 persisted edges predate the provenance work" | true, but the seeded priors are written at runtime and never stamped at all |
| "no chain assembler can repair that" | an assembler cannot, but the fix is a provenance stamp plus reviewing the authored list |

## Reproduce

```
uv run python -c "import ravana.chat.engine_reasoning as m; ..."
grep -n '_seed_physics_causal' ravana/src/ravana/chat/engine_reasoning.py
```
