# Capability: state-conditioned surface form (state_seed in RealizerLexicon.realize())

**Status:** shipped (commits `76f98b2a` pivot + `45ef66e3` capability,
branch `auto/round-2026-10-30T0602Z`).
**Verified:** module imports clean; `RealizerLexicon.realize()` accepts
`state_seed` and produces deterministic output for the same seed.

## What it does

`RealizerLexicon.realize()` previously selected a surface form with
`random.choice` (or a `rng` passed by the caller), so the same cognitive
state could render as different words across turns and runs. This capability
adds a `state_seed: Optional[int]` parameter: when provided, a fresh
`random.Random(state_seed)` is used instead, making selection a pure
function of the seed.

The capability commit then wires a real cognitive seed into the call site in
`_handle_assertion` (response_gen.py): the seed is derived from RAVANA's live
state — `turn_count * 31 + round(identity.strength * 1000)` — so the same
cognitive state yields the same surface realization, while different states
can yield different forms. No authored reply strings were added; the change
is in *how* an existing pool is sampled, not *what* it contains.

- `ravana/src/ravana/chat/realizer_lexicon.py:166` — `RealizerLexicon.realize()`
  signature gains `state_seed: Optional[int] = None`.
- `ravana/src/ravana/chat/realizer_lexicon.py:178-182` — when `state_seed is not None`,
  `rng = random.Random(state_seed)`; otherwise falls back to `rng or random`.
- `ravana/src/ravana/chat/response_gen.py:2591-2603` — seed derivation from
  `turn_count` + `identity.strength`, passed to `realize("user_leads"/...`,
  `"backchannels"`, `"follows"`).

## How it grew from the conversation

Pivot step (commit `76f98b2a`) added the `state_seed` knob to `realize()` so
the realizer could track cognition instead of a dice roll; capability step
(commit `45ef66e3`) wired that knob to live cognitive state in
`_handle_assertion`. This mirrors the deterministic-greeting-fallback work:
remove non-determinism from the surface layer so output is reproducible and
the only source of variety is RAVANA's actual cognitive state, not a random
draw.

## Source citations

- `ravana/src/ravana/chat/realizer_lexicon.py:166` — `realize()`.
- `ravana/src/ravana/chat/response_gen.py:2582` — `_handle_assertion` (enclosing method).
- `ravana/src/ravana/chat/response_gen.py:2591-2603` — seed derivation + wired calls.
