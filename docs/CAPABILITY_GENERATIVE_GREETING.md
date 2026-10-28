# Capability: generative greeting (P1 settle-generator path)

**Status:** shipped (commit `fae13bd3`, branch
`auto/round-2026-10-28T0602Z`).
**Verified:** `_compose_greeting_generative` is a pure generator path that
returns `None` on any failure (decoder/attractor unavailable), so the existing
pool-based greeting remains the honest fallback. Source lines cited below.

## What it does

`_compose_greeting` (in `response_gen.py`) used to build a greeting purely from
a valence/arousal-keyed pool of authored words. This capability adds a first
attempt at genuine generation: it tries a settle-generator
(`PredictiveCodingGenerator`) grounded in the typed concept graph via
`_grounded_anchor_trajectory("greeting")`, and only falls back to the existing
pool when the decoder/attractor is unavailable or produces too little output.

- `ravana/src/ravana/chat/response_gen.py::_compose_greeting_generative`
  attempts the generative path: obtains the settle generator, retrieves a
  grounding trajectory for the concept `"greeting"`, builds conditioning
  embeddings from the decoder's entries for `greeting`/`hello`/`hi`, runs
  `gen.generate(...)`, and decodes indices back to words. Returns `None` on any
  failure (no generator, no trajectory, no embeddings, short output, exception).
- `_compose_greeting(self, valence, arousal)` calls the generative helper first
  and returns its result when non-empty; otherwise falls through to the
  original valence/arousal pool composition.

No authored reply strings were added; the fallback pool is unchanged.

## How it grew from the conversation

This is the P1 (narrow-domain) step of the Language Production Pivot: take one
narrow domain (greetings) and route it through the settle generator grounded in
the concept graph, with an honest fallback to the existing pool so behaviour
never regresses when the decoder/attractor is unavailable. The helper is
intentionally defensive (try/except returning None) so missing infrastructure
degrades gracefully rather than crashing.

## Source citations

- `ravana/src/ravana/chat/response_gen.py:3088` — `_compose_greeting_generative`.
- `ravana/src/ravana/chat/response_gen.py:3116` — `_compose_greeting` calls the helper first.
- `ravana/src/ravana/chat/response_gen.py:806` — `_grounded_anchor_trajectory`.
- `ravana/src/ravana/chat/response_gen.py:828` — `_get_settle_generator`.
