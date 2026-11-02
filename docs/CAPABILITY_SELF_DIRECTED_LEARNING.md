# Capability: Self-Directed Learning

## Summary

RAVANA detects its own knowledge gaps, forms hypotheses, seeks examples, and
improves through experimentation — all online during normal operation, with no
LLM and no retraining.

## Mechanism

- Module: `ravana/src/ravana/chat/self_directed_learning.py`
  - `KnowledgeGap(topic, confidence, support)` — a detected low-FOK concept.
  - `SelfDirectedLearner(engine)` — runs the loop.
- Wired in `engine.py`:
  - `__init__`: constructs `self.self_directed_learner = SelfDirectedLearner(self)`.
  - Per-turn (normal operation): calls `_sdl.run_cycle()` which detects gaps,
    forms hypotheses, and seeks example queries.
- Detection uses metacognitive FOK confidence over graph support (edge count),
  not a lookup table. Hypotheses are generated from real state (topic + FOK).
- Experimentation re-detects gaps to measure improvement (`improved` flag).

## Brain basis

Metacognitive monitoring (ACC / FOK) drives curiosity and hypothesis testing
(Koriat 1993; Botvinick 2004). FOK confidence already exposed via
`metacognition.fok_confidence`; curiosity drive exists in `web_learning`.

## Acceptance

- Mechanism present and wired into normal turn path.
- Tests: `tests/unit/test_self_directed_learning.py` (4 passed).
- No LLM, no retraining; runs online during normal operation.
- No hardcoded reply strings added (hardcoding audit: this module adds none).
"""