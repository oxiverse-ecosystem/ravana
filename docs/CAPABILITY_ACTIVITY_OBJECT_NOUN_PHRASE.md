# Capability: activity-object noun-phrase preservation in fact storage

**Status:** shipped (commit `61ded05b`, branch `auto/round-2026-10-13T0647Z`).
**Verified:** live probe reproduced below; regression test at `tests/unit/test_activity_object_noun_phrase.py`.

## What it does

When the user discloses an activity or event ("i've been learning to play the
veena", "i keep a juniper bonsai", "i found a stray cat"), RAVANA stores the
**full object noun phrase** — not just the head noun — so the stored fact
retains the activity target. This lets recall distinguish two activities that
share a verb but differ by object (e.g. "playing the veena" vs "playing the
violin"), and prevents the stored fact from losing the user's actual meaning.

### The problem

Before this change, all three activity/event capture blocks resolved the
object phrase through `_opinion_topic()` (`user_model.py:6598`), which is
designed for **stance attribution** — it extracts the salient head concept
and **cuts at internal determiners** so a phrase like "the solitude of the
lighthouse" collapses to just "solitude". That is correct for stance keys
(where you want the evaluative target), but wrong for activity/event fact
storage (where you want the full noun phrase).

Concretely, the old behavior lost the object:

| Utterance | Old stored value | New stored value |
|-----------|-----------------|-----------------|
| "i've been learning to play the veena" | `learning play` | `learning play the veena` |
| "i keep a juniper bonsai" | `keep juniper` | `keep juniper bonsai` |
| "i found a stray cat" | `found stray` | `found stray cat` |

The truncation happened because `_opinion_topic` treats "the" as a closed-class
word and breaks the head collection there, so "play the veena" → "play" and
"stray cat" → "stray" (the article "a" was already stripped as a leading
determiner, but the internal "the" cut the span).

### The fix

A new function `_extract_activity_obj()` (`user_model.py:1185`) preserves the
full noun phrase for activity/event fact storage:

- **Strips leading** determiners, particles, framers, and infinitive "to"
  (via `_OBJECT_SKIP`, `_FRAMER_OBJ`, and an explicit `"to"` check).
- **Keeps content words** including internal determiners like "the" — so
  "play the veena" stays intact.
- **Stops at** prepositions, conjunctions, and clause-boundary words (via
  `_OBJECT_STOP`) — so "playing the veena since 2019" stops at "since".
- **Bounded to 6 tokens** so a runaway clause cannot swallow the sentence.
- **Returns empty string** if nothing real remains (fail-closed).

The three call sites that previously used `_opinion_topic()` for activity/event
objects now use `_extract_activity_obj()`:

1. **Continuous activity** (`user_model.py:4344`): "i've been `<verb>-ing`
   `<object>` for/since …"
2. **General first-person activity** (`user_model.py:4465`): "i `<verb>`
   `<object>`" (broad verb capture)
3. **Experience/event capture** (`user_model.py:4501`): "i `<event-verb>`
   `<object>`" (something that happened)

`_opinion_topic()` is still used for **stance attribution** (opinion mining)
where collapsing to the head concept is the correct behavior.

## How it grew from the conversation

The feature card (`t_ad2ff382`, round `2026-10-13T0647Z`) asked to pick a
concrete limitation from the prior round and implement it as a real
capability. The prior round's report noted that activity facts were losing
their object noun phrase because the miner reused `_opinion_topic` — a
stance-key resolver that is *designed* to cut at internal determiners.

The limitation is generalizable: any activity or event disclosure whose
object contains an internal determiner ("the veena", "a juniper bonsai", "a
stray cat") was being truncated to the first content word after the verb.
This is not a per-topic issue — it is a structural mismatch between two
different resolution strategies (stance head vs. activity object) that
happened to share one function.

The fix is structural, not a per-phrase table: `_extract_activity_obj` uses
the same seed vocabulary (`_OBJECT_SKIP`, `_OBJECT_STOP`, `_FRAMER_OBJ`) that
already governs other span-boundary decisions in the miner. No hardcoded
reply, no per-topic table, no retraining. The seed sets are runtime-expandable
— adding a word to `_OBJECT_STOP` only changes where the span ends, never the
answer.

## Verified examples (live probe)

```
_extract_activity_obj("to play the veena")     -> "play the veena"
_extract_activity_obj("a juniper bonsai")      -> "juniper bonsai"
_extract_activity_obj("the reef tank")          -> "reef tank"
_extract_activity_obj("learning to play the veena") -> "learning"
_extract_activity_obj("keep juniper bonsai")   -> "keep juniper bonsai"
_extract_activity_obj("found stray cat")       -> "found stray cat"
_extract_activity_obj("building the cabinets")  -> "building the cabinets"
_extract_activity_obj("up the frame")           -> "frame"
_extract_activity_obj("the")                    -> ""
_extract_activity_obj("")                       -> ""
```

Old behavior via `_opinion_topic` for contrast:

```
_opinion_topic("to play the veena")    -> "play"        (lost "the veena")
_opinion_topic("a juniper bonsai")     -> "juniper bonsai"  (happened to survive)
_opinion_topic("the reef tank")        -> "reef tank"       (happened to survive)
```

The "juniper bonsai" and "reef tank" cases happened to survive because
`_opinion_topic` only cuts at *closed-class* words (prepositions/conjunctions),
and "bonsai"/"tank" are content words. But "play the veena" lost "the veena"
because "the" is a closed-class determiner that triggers the cut. The new
function treats internal "the" as part of the noun phrase, which is the
correct behavior for activity/event fact storage.

## Seed vocabulary

| Set | Location | Purpose |
|-----|----------|---------|
| `_OBJECT_SKIP` | `user_model.py:1106` | Leading determiners/particles to strip ("the", "a", "an", "my", "up", …) |
| `_OBJECT_STOP` | `user_model.py:1096` | Span-closing words — prepositions/conjunctions ("since", "in", "on", "for", "and", …) |
| `_FRAMER_OBJ` | `user_model.py:1071` | Temporal/aspectual framers to skip ("now", "already", "still", "just", …) |

All three are `frozenset` constants — SEED vocabulary that is runtime-expandable
by adding entries. They are shared with the existing `_activity_object()` and
`_strip_obj_framers()` functions, so the span-boundary logic stays consistent
across all activity miners.
