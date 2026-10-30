# Capability: deterministic greeting fallback (pivot: remove random.choice pools)

**Status:** shipped (commit `96e51665`, branch
`auto/round-2026-10-30T0448Z`).
**Verified:** module imports clean; `_compose_greeting` still calls the
generative helper first and falls through to the (now deterministic) pool.

## What it does

`_compose_greeting` (in `response_gen.py`) used to build its fallback from
`random.choice` over authored pools of greeting words and topic invites.
Those pools are authorial reply strings selected non-deterministically, which
hides the real (generative) path and makes output non-reproducible. This
capability collapses each pool to a single deterministic fallback value, so
the only authored material left is the honest, fixed frame — no randomness,
no selection.

- `ravana/src/ravana/chat/response_gen.py::_compose_greeting` — valence > 0.6
  branch: `greeting_word = "hello"`, `topic_invite = "how can i help you today?"`.
- valence < 0.4 branch: `greeting_word = "hello"`,
  `topic_invite = "what's on your mind?"`.
- else branch: `greeting_word = "hello"`, `topic_invite = "how can i help you?"`.
- The generative path (`_compose_greeting_generative`) is still attempted
  first and its result returned when non-empty; the deterministic pool is
  only reached on failure/short output.

No authored reply strings were added; authored pools were removed.

## How it grew from the conversation

Mirror of pivot task 3 (humor pools): the same hardcoding-drift trap
(random.choice over authored pools) was present in the greeting fallback. The
fix is subtraction — delete the pools, keep one deterministic frame — so the
generative path is the only source of varied output and the fallback is an
honest, reproducible floor.

## Source citations

- `ravana/src/ravana/chat/response_gen.py:3108` — `_compose_greeting`.
- `ravana/src/ravana/chat/response_gen.py:3088` — `_compose_greeting_generative`.
