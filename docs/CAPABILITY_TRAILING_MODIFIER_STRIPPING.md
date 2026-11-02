# Capability: trailing-modifier stripping for topic phrases (online-growable lexicon)

**Status:** shipped (round `2026-11-02T0300Z`, feature `t_8c0c463d`).
**Verified:** regression suite `tests/test_topic_trim_capability.py` passes
5/5 on the shipped code. Tests exercise the real `ModifierLexicon` and
`strip_trailing_modifiers` from `ravana/src/ravana/chat/topic_trim.py` — not
mocks. Hardcoding self-audit clean: no authored reply prose, no per-topic
table, no retraining — only a seed *vocabulary* of modifier forms that grows
online via `learn()`.

## What it does

When a topic phrase ends with a trailing **discourse/temporal modifier**
— *\"actually defines person\"* → the trailing *\"actually\"* is a
discourse-level marker, not part of the referent — the extracted topic key
used to keep the modifier, producing garbled stance topics such as
`actually defines person` (round 2026-11-02T0300Z chat probe, TURN 20).
Keeping the modifier meant the same concept was keyed twice (`defines person`
vs `defines person actually`), so recall read the weaker duplicate.

The fix introduces a small, online-growable `ModifierLexicon`
(`ravana/src/ravana/chat/topic_trim.py`) with a seed vocabulary of modifier
forms and a `strip_trailing_modifiers` function that strips from the right
while the last token is a known modifier, stopping when a non-modifier is
reached or the phrase would become empty:

- *\"defines person actually\"* → **`defines person`**.
- *\"what defines a person so far\"* → **`what defines a person`**.
- *\"actually\"* (single token) → **`actually`** (never empties the phrase).

The lexicon grows online: a word observed to be a trailing modifier in a
grounded extraction is added via `learn()`, so RAVANA's notion of "modifier"
is shaped by experience rather than a frozen list. Removing one seed entry
degrades gracefully; cold start uses seed only.

## How it grew from the conversation

- Round chat probe TURN 20 surfaced the garbled stance topic `actually defines
  person` (the trailing adverb survived topic extraction).
- The capability was implemented as a standalone, state-roundtrippable module
  (`ModifierLexicon.to_state` / `load_state`) so the learned modifier set
  persists across sessions alongside the other durable signals.

## API (verified)

- `ModifierLexicon(seed=...)` — construct with optional seed iterable.
- `.is_modifier(word)` — membership test (seed ∪ learned).
- `.learn(*words)` — online growth; returns the set of newly added words.
- `.to_state()` / `.load_state(state)` — roundtrip the learned set.
- `strip_trailing_modifiers(phrase, lexicon=None)` — trailing-strip; uses the
  module default lexicon when `lexicon` is `None`.
- `default_lexicon()` — returns the shared default instance.

Verified by direct execution:

```
>>> from ravana.chat.topic_trim import strip_trailing_modifiers, ModifierLexicon
>>> strip_trailing_modifiers("defines person actually")
'defines person'
>>> strip_trailing_modifiers("what defines a person so far")
'what defines a person'
>>> lx = ModifierLexicon(); lx.learn("zzzword"); lx.is_modifier("zzzword")
True
```

No LLM, no Q→A dict, no retraining. The capability is a structural addition;
RAVANA's topic keys are recomputed from the stripped phrase. See
`ravana/src/ravana/chat/topic_trim.py`.
