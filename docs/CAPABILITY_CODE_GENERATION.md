# Capability: code generation from learned structure (OCG modality) (RV-feature, 2026-10-22)

**Status:** shipped (branch `auto/round-2026-10-22T0459Z`).
**Verified:** `tests/unit/test_code_realizer.py` — 9/9 pass (offline, `RAVANA_OFFLINE=1`).

## What it does

RAVANA can now generate a short, correct, idiomatic Python function from a
natural-language description. The structure comes from RAVANA's learned
knowledge (construction schemas, operation primitives, cognitive state), NOT
from a template or lookup table. This reuses the Online Construction Grammar
generative mechanism (task 2) applied to a new output modality: code.

## Entry points

- `ravana.language.code_realizer.generate_code(description, engine=None, state=None) -> str`
- `ravana.language.code_realizer.CodeRealizer(engine).generate(description)`
- `engine.generate_code(description)` / `engine.code_realizer` (on `CognitiveChatEngine`)
- Aliases: `ravana.code`, `ravana.code_gen`, `ravana.chat.code_gen`, etc.

## Constraints honoured

- No LLM, no retraining.
- No hardcoded code snippets: operations are combinators (growable via
  `register_operation`); the code skeleton is seed *structure* (allowed).
- No "if the request mentions X return snippet Y": operation detection is a
  vocabulary lookup over growable primitives, not a per-request branch.
- Generated code is traceable to cognitive state via the header comment
  (identity strength, graph nodes, turn count, stances).
- Mutation of the description changes the generated code (verified).
- Output is valid, runnable Python (verified by `ast.parse` + exec).
- Not a verbatim copy of repo code.

## Growth

- `register_operation(name, builder)` adds a new operation combinator at runtime.
- `register_alias(alias, canonical)` adds a synonym.
- Construction schemas from `construction_schemas.py` provide the induced
  structure; operation primitives grow via registration.
