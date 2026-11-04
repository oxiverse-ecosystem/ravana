# Capability: growing factual-attribute vocabulary (learn_factual_attribute)

**Status:** shipped (commit `25014121`, branch `auto/round-2026-11-04T0849Z`).
**Verified:** live in-process probe (real engine output, `dim=64, seed=42,
baby_mode=True`, offline) reproduced below. No LLM, no retrain, online/incremental.

## What it does

RAVANA treats a small seed set of words as **factual attributes** — `capital`,
`population`, `inventor`, `founded`, `currency`, ... When a user asks for a
specific fact about a named entity that RAVANA holds no stored fact for
(*"capital of X"*, *"population of X"*, *"who invented X"*), it abstains with
honest uncertainty instead of confabulating via analogical reasoning. See
`engine_self_query.py` lines ~2530-2565.

Previously that vocabulary was a **frozen set** baked into the code. This
capability makes it a **growing seed**: `learn_factual_attribute(word)` extends
the set at runtime and persists the addition to `data/factual_attribute_words.json`,
which is loaded back on init. Returns `True` only when a new word was added
(duplicates and empty/None inputs return `False`). No code change, no retraining,
no LLM required to teach RAVANA a new factual attribute (e.g. `gdp_per_capita`).

## How it grew from the conversation

The parent feature card (`t_85d9a216`) picked the last limitation: the
factual-attribute vocabulary was frozen, so RAVANA could not learn a new kind
of factual query without a code edit. The fix mirrors the seed-vs-hardcode rule
used for `pet_slots` / `possession_attrs`: the seed lives in a store RAVANA can
extend (`_factual_attribute_words` set + `data/factual_attribute_words.json`),
loaded lazily in `_consult_internal_knowledge` (`engine_self_query.py:2536-2560`)
and grown via `learn_factual_attribute` (`engine_self_query.py:2354`).

## API (real, verified)

- `CognitiveChatEngine.learn_factual_attribute(word: str) -> bool`
  (`ravana/src/ravana/chat/engine_self_query.py:2354`)
- Durable store: `data/factual_attribute_words.json` (relative to `_data_dir`,
  default `data`), written as a sorted JSON list.

## Live probe (real output)

```
learn capital first: True
learn new:           True   (gdp_per_capita)
after has new:       True
learn dup:           False
empty:               False
None:                False
store exists:        True
```

`learn_factual_attribute` returns `True` only on a genuinely new word; the
durable store file is created on first learn.
