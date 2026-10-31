# Capability: favorite-disclosure key modifier stripping (no leading/trailing adverb residue)

**Status:** shipped (round `2026-10-30T1824Z`, feature `t_ecfba7a1`).
**Verified:** engine imports cleanly (`CognitiveChatEngine` import chain intact). New
regression coverage added at `tests/unit/test_favorite_disclosure_modifier_stripping.py`.

## What it does

When a user discloses a favorite with leading/trailing modifier adverbs —
*\"my favorite actually old books is X\"* — the mined favorite key used to keep
the modifier, producing a key like `actually old books` instead of `old books`/`books`.
The next recall about that favorite then read a different (duplicate) slot and
missed the stored value.

The fix routes the favorite-disclosure object through the shared
`slot_naming` chokepoint (`strip_leading_modifiers` + `strip_trailing_modifiers`,
`slot_naming.py`) so mined favorite keys drop leading/trailing modifier adverbs
the same way stance/opinion slot keys do (FIX-RV-12). The modifier vocabulary is
the closed-class seed in `data/functional_lexicon.json` (`leading_modifiers` /
`trailing_modifiers`), loaded via `FunctionalLexicon` and runtime-expandable —
no per-topic table, no retraining.

Concretely:

- *\"my favorite actually old books is X\"* → favorite key **`old books`** (not
  `actually old books`).
- *\"my favorite books actually is X\"* → favorite key **`books`** (trailing
  `actually` stripped).

The seed vocabulary and the strip helpers never empty the list, so a phrase that
IS a single modifier keeps its token (fails open). If the lexicon import is
unavailable, the seed sets in `slot_naming.py` are used, so behavior never
regresses below the seed.

## How it grew from the conversation

The stance-key trim work (FIX-RV-12, `slot_naming.py`) established a shared
chokepoint so the stance miner and recall paths name the SAME slot. The
favorite-disclosure path in `engine_reasoning.py` was still building its key
directly from the regex capture (`m.group(1).strip(" .!?")`), bypassing
`slot_naming`, so modifiers survived into the favorite key even though the same
modifiers were already stripped from stance keys.

The feature card (`t_ecfba7a1`) wired `strip_leading_modifiers` /
`strip_trailing_modifiers` into that favorite-disclosure branch so favorite keys
agree with the stance/opinion slot naming by construction.

**Root cause / prior behavior.** `engine_reasoning.py` (the `ReasoningMixin`
favorite branch) parsed `("favorite", m.group(1).strip(" .!?"), m.group(2)...)`
with no modifier trim, so `actually old books` stayed verbatim as the key.

**Fix (round `2026-10-30T1824Z`).** Import `strip_leading_modifiers` and
`strip_trailing_modifiers` from `.slot_naming`, tokenize the captured object,
trim leading then trailing modifiers, and re-join; fall back to the raw captured
string if the result is empty or the import fails (fails open).

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Favorite-disclosure branch (the change) | `ravana/src/ravana/chat/engine_reasoning.py:2179` (approx) |
| Shared strip helpers | `ravana/src/ravana/chat/slot_naming.py` (`strip_leading_modifiers`, `strip_trailing_modifiers`) |
| Modifier vocabulary (seed + fit-file) | `ravana/src/ravana/chat/slot_naming.py` (`leading_modifiers`, `trailing_modifiers`) |
| Functional lexicon fit file | `data/functional_lexicon.json` |
| Regression tests | `tests/unit/test_favorite_disclosure_modifier_stripping.py` |

## Test coverage

`tests/unit/test_favorite_disclosure_modifier_stripping.py` exercises the
`slot_naming` helpers that the favorite branch now calls, plus an end-to-end
probe through `CognitiveChatEngine.process_turn` verifying the mined favorite key
drops the modifier. Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_favorite_disclosure_modifier_stripping.py -v
```
