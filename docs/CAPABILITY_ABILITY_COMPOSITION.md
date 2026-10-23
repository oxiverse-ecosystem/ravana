# Capability: abilities by composition (RV-feature, 2026-10-22)

**Status:** shipped (branch `auto/round-2026-10-22T0459Z`).
**Verified:** `tests/unit/test_ability_composition.py` — 5/5 pass (offline, `RAVANA_OFFLINE=1`).

## What it does

RAVANA can now acquire a genuinely new ability in-session by composing
abilities it already has, the way a brain learns a new skill from existing
primitives. An ability is a *composition* (pipeline) over existing primitives;
adding an ability = learning a new composition, not writing a new branch.

No LLM, no retraining, no new hardcoded dispatch branches per ability. A new
ability is acquirable from instruction or demonstration, online, and RAVANA can
explain which primitives it composed and in what order.

## Entry points

- `ravana.agent.ability_composer.compose_ability(name, primitives, ...)`
- `ravana.agent.ability_composer.learn_ability(name, primitives, ...)`
- `ravana.agent.ability_composer.acquire_ability(name, primitives, ...)`
- `ravana.agent.ability_composer.learn_from_instruction(text, name=None)`
- `ravana.agent.ability_composer.learn_from_demonstration(name, steps)`
- `ravana.agent.ability_composer.run_ability(name, *args)` / `execute_ability`
- `ravana.agent.ability_composer.explain_ability(name)` (returns primitive names)
- `ravana.agent.ability_composer.register_primitive(name, fn)` (grow primitives)
- `engine.compose_ability`, `engine.learn_ability`, `engine.run_ability`,
  `engine.explain_ability`, `engine.register_primitive`, `engine.ability_composer`
- Aliases: `ravana.agent.abilities`, `ravana.agent.composed_abilities`,
  `ravana.chat.ability_composer`, `ravana.language.ability_composer`,
  `ravana.ability_composer`, `ravana.abilities`.

## Growth

- `register_primitive(name, fn)` adds a new primitive combinator at runtime.
- `compose_ability(name, primitives)` learns a new composition (pipeline).
- `learn_from_instruction` / `learn_from_demonstration` acquire from text/demo.
- Compositions can include other composed abilities (recursive composition).

## Constraints honoured

- No LLM, no retraining.
- No hardcoded dispatch branches per ability: the composition is data in a store.
- Primitives are seed structure (growable); removing one degrades gracefully.
- Explainable: `explain_ability` returns the composed primitive names.
- Traceable: each `ComposedAbility` records `learned_from` and `description`.
