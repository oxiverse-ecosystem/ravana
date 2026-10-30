# Capability: generative emitter (online-expandable construction bank)

**Status:** shipped (commit `f9e1cb07`, branch
`auto/round-2026-10-30T0331Z`).
**Verified:** in-process probe (dim=64, offline): import + instantiation +
`learn_construction` + `compose` all succeed; `compose` returns state-driven
output (`x i lean for it`) for a state with one stance. Source lines cited
below.

## What it does

`GenerativeEmitter` (in `generative_emitter.py`) replaces template-shaped
emission (frame table + slot fill) with a mechanism that PRODUCES surface form
from learned structure. Constructions are stored in an online-expandable bank
(seed = small compositional rules, i.e. slot names, NOT full sentences) and are
selected/composed from live cognitive state (identity, stances, facts, beliefs).
No LLM, no retraining: a new construction can be added during a single
conversation via `learn_construction`.

- `ravana/src/ravana/chat/generative_emitter.py:GenerativeEmitter.__init__`
  seeds four compositional constructions (slot-name sequences) and an empty
  `learned` list; RNG seeded by engine seed.
- `ravana/src/ravana/chat/generative_emitter.py:GenerativeEmitter.compose`
  selects a construction by state hash (same state -> same form; different
  state -> different form), realizes each slot from state, joins realized
  slot text; returns `None` when any slot cannot be realized.
- `ravana/src/ravana/chat/generative_emitter.py:GenerativeEmitter.learn_construction`
  appends a new construction to `learned` if not already present in either
  `constructions` or `learned` (online growth, no retrain).
- `ravana/src/ravana/chat/generative_emitter.py:GenerativeEmitter._realize_slot`
  maps slot names (stance_topic, stance_lean, identity_strength, fact_value,
  fact_attr, belief_text) to state-derived text; `_last_slot_value` resolves
  dependent slots (stance_lean depends on stance_topic) deterministically.
- Wired into the engine at
  `ravana/src/ravana/chat/engine.py:1231-1236`: `self.generative_emitter =
  GenerativeEmitter(seed=seed)` inside a try/except that degrades to `None`.
- Imported (for availability) in
  `ravana/src/ravana/chat/response_gen.py:73`.

## How it grew from the conversation

This is the GenerativeEmitter capability: take the last limitation (template-
shaped emission) and expose an online-expandable, state-driven surface-form
producer. Seed = compositional slot-name rules; content tokens come from state.
Growth happens at runtime via `learn_construction`, with no retrain and no LLM.

## Source citations

- `ravana/src/ravana/chat/generative_emitter.py:1` — module docstring.
- `ravana/src/ravana/chat/generative_emitter.py:21` — `class GenerativeEmitter`.
- `ravana/src/ravana/chat/generative_emitter.py:22` — `__init__(self, seed=0)`.
- `ravana/src/ravana/chat/generative_emitter.py:101` — `compose`.
- `ravana/src/ravana/chat/generative_emitter.py:120` — `learn_construction`.
- `ravana/src/ravana/chat/engine.py:1231` — wiring.
- `ravana/src/ravana/chat/response_gen.py:73` — import.