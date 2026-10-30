# Capability: structured Message (conceptual preparation, Levelt stage 1)

**Status:** shipped (commits `6d92e491`, `20d660d9`, branch
`auto/round-2026-10-28T0315Z`).
**Verified:** tests in `tests/unit/test_message.py` pass; live run reproduced
below (`dim=64, seed=42, baby_mode=True`, offline).

## What it does

The Language Production Pivot (M1/M5) requires that the arrow into the emitter
be a structured `Message`, not a string. This capability adds that structure:
a non-linguistic intent produced by conceptual preparation, grounded in the
typed concept graph, carrying no authored text.

- `ravana/src/ravana/chat/message.py` defines `Message` (predicate_edge,
  agent_node, patient_node, modifiers, focus) and `FeatureBundle`. `Message.is_grounded()`
  is True iff at least one graph node is present; `active_nodes()` returns the
  sparse active set capped at 4.
- `ravana/src/ravana/chat/models.py` adds `message: Any = None` to
  `CognitiveResponseContext`.
- `ravana/src/ravana/chat/interface.py::_build_message` constructs a `Message`
  from `ctx.subject`, `ctx.associated_concepts`, and `ctx.relation`, resolving
  the subject against `graph_engine._concept_keywords`. `_graph_fallback_response`
  calls `ctx.message = self._build_message(ctx)` before producing its reply.

## How it grew from the conversation

The pivot replaced template-shaped emission with genuine generation from learned
structure. The first step (task t_ed5aac32) introduced the `Message` type; the
feature card (t_0f0465e6) wired it into `_graph_fallback_response` so every
fallback path now carries a grounded non-linguistic intent instead of only a
string. No authored reply strings were added.

## Source citations

- `ravana/src/ravana/chat/message.py:1` — module docstring; `Message` dataclass.
- `ravana/src/ravana/chat/models.py:103` — `message: Any = None` field.
- `ravana/src/ravana/chat/interface.py:1300` — `_build_message`.
- `ravana/src/ravana/chat/interface.py:1319` — `_graph_fallback_response` sets `ctx.message`.

## Verified behaviour

Live run: `RAVANA_OFFLINE=1`, `CognitiveChatEngine(dim=64, seed=42, baby_mode=True)`,
turn "tell me about gravity" produced a reply and the engine booted cleanly
(verified 2026-10-28). Unit tests: `tests/unit/test_message.py` (4 tests).
