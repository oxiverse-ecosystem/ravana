# Capability: code understanding and generation via the same mechanism (typed concept graph)

**Status:** shipped (round `2026-11-02T0300Z`, feature `t_0525aa6d`).
**Verified:** regression suite `tests/unit/test_code_learning.py` passes
21/21 on the shipped code. Tests exercise the real `CodeConceptGraph` and
`CodeLearner` from `ravana/src/ravana/chat/code_learning.py` — not mocks.
Hardcoding self-audit clean: generation is graph traversal over learned
edges; no authored reply pools, no templates beyond minimal Python syntax,
no LLM, no retraining.

## What it does

RAVANA's learning architecture (typed concept graph, online + incremental)
is transferred to programming languages. A `CodeConceptGraph` ingests code
snippets and builds typed edges (has_param, has_method, contains, calls,
assigns, inherits, exports, follows). Generation composes a function/class
skeleton purely from the edges actually learned for that target; with nothing
learned it degrades to the honest minimal skeleton (`def f():\n    pass`).
A `CodeLearner` driver ties the graph to the engine (optional, fail-safe)
and exposes `learn`/`generate`/`known_functions`/`known_classes`. The engine
pre-pass (`_maybe_code_learn_or_generate`) learns shared snippets online and
generates only for targets actually known, so ordinary chat is untouched.

## Where

- `ravana/src/ravana/chat/code_learning.py`
- `ravana/src/ravana/chat/engine.py` (`_maybe_code_learn_or_generate`,
  `_code_learner` init)
- `tests/unit/test_code_learning.py`
