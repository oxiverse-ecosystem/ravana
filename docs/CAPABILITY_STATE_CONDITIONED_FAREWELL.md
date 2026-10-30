# Capability: state-conditioned farewell selection (state_seed in _compose_farewell)

**Status:** shipped (commit `5ce0f0f0`, branch
`auto/round-2026-10-30T0716Z`).
**Verified:** module imports clean; `_compose_farewell` derives `_state_seed`
from `turn_count` + `identity.strength` and selects via `_pick`, matching the
existing `state_seed` pattern in `response_gen`.

## What it does

`_compose_farewell` (response_gen.py) previously selected farewell words and
warmth markers with `random.choice` over authored pools, so the same cognitive
state could render as different farewells across turns and runs. This capability
replaces that draw with a state-conditioned `_pick`: the index is computed from
RAVANA's live state — `turn_count * 31 + round(identity.strength * 1000)` —
so the same cognitive state yields the same farewell, while different states can
yield different forms. The relationship-depth branch (`_rel_depth > 0.6`) is
preserved; only the selection mechanism changes.

No authored reply strings were added; the change is in *how* an existing pool is
sampled, not *what* it contains.

- `ravana/src/ravana/chat/response_gen.py:3233` — `_compose_farewell` signature
  `(valence, arousal)`.
- `ravana/src/ravana/chat/response_gen.py:3245-3252` — `_state_seed` derivation
  from `turn_count` + `identity.strength`.
- `ravana/src/ravana/chat/response_gen.py:3253-3256` — `_pick(pool)` helper
  returning `pool[_state_seed % len(pool)]`.
- `ravana/src/ravana/chat/response_gen.py:2064` — call site in the farewell
  routing path.

## How it grew from the conversation

This mirrors the state-conditioned surface form work in the previous round
(`CAPABILITY_STATE_CONDITIONED_SURFACE_FORM.md`): the `state_seed` knob was
added to `RealizerLexicon.realize()` first (pivot), then wired to live
cognitive state (capability). Here the same pattern is applied to
`_compose_farewell`, removing non-determinism from the farewell surface layer so
output is reproducible and the only source of variety is RAVANA's actual
cognitive state, not a random draw.
" content_file="C:\Users\Likhith\Documents\Projects\ravana\docs\CAPABILITY_STATE_CONDITIONED_FAREWELL.md"