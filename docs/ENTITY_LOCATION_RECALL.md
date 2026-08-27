# Entity-Keyed Location Recall

How RAVANA captures and later *surfaces* a named thing's whereabouts —
e.g. *"the slow coal is moored at bingley"* → *"where's the slow coal moored?"*
→ *"the slow coal is at bingley."*

This capability was added in round **2026-08-10T0813Z** (limitation #1) and is
verified against the live engine (`dim=64`, offline). No LLM, no retrain, no
authored reply pools — the answer is read from the durable `PersonalFactStore`
and rendered with a single short templated frame.

## What it does

1. **Mines a possession/vehicle's location as a structured, entity-keyed fact.**
   *"the slow coal is moored at bingley for the winter"* is stored as
   `('slow coal', 'location', 'bingley')` — the *entity* is the subject, not `i`.
   This lets a later correction supersede the old value instead of stacking a
   contradiction.
2. **Surfaces that fact on a "where is X" query** for any named entity, in any
   of several phrasings (`where's the slow coal moored?`, `where is the slow
   coal?`, `where's the slow coal?`, `what is the slow coal's location?`).
3. **Reflects corrections.** After *"actually the slow coal is moored at saltaire
   now"*, the same query answers *"the slow coal is at saltaire."*
4. **Fails closed on unknown places.** *"where is paris?"* (no stored entity
   fact) returns an honest uncertainty — it never fabricates a `the paris is at
   X` answer and never leaks an unrelated stored place.
5. **Does not hijack the user's own location.** *"where do i live?"* is answered
   by the biographical path (`you live in hexham.`), not swallowed as an entity
   whereabouts.

## How it grew from the conversation (source citations)

### Mining — `ravana/chat/user_model.py`, `UserModel.mine_personal_facts`

The possession-location miner (regex `_pos_loc`) captures an entity + place:

```python
# ravana/src/ravana/chat/user_model.py  (~L579)
_pos_loc = re.search(
    r"\b(?:my|the|a|an|our|their|his|her)\b\s*"
    r"([\w'-]+(?:\s+[\w'-]+){0,3})"
    r"\s+(?:is|was|are|were|sits|lies|stays|remains)\s+"
    r"(?:moored|berthed|anchored|docked|based|parked|stationed|"
    r"kept|stored|housed|tied up|wintered)\s+"
    r"(?:at|in|on)\s+([\w'-]+(?:\s+[\w'-]+){0,3})", text)
```

A **leading-hedge strip** (commit `31290cb`) resolves corrections like
*"actually the slow coal is moored at saltaire"* to the SAME entity
(`"slow coal"`) instead of a fragment (`"ctually the slow coal"`). The hedge set
is **seed vocabulary** RAVANA can grow at runtime — missing a hedge degrades to a
separate entity, never a wrong answer:

```python
# ravana/src/ravana/chat/user_model.py  (~L595)
_HEDGE = ("actually", "now", "well", "so", "but", "right",
          "okay", "ok", "and", "then", "still")
_ent_words = _ent.split()
while len(_ent_words) > 1 and _ent_words[0] in _HEDGE:
    _ent_words = _ent_words[1:]
_ent = " ".join(_ent_words)
```

The stored fact is `(subject, "location", place)` with a `superseded` flag, so a
correction marks the prior value stale rather than deleting it.

### Surfacing — `ravana/chat/engine.py`, `CognitiveChatEngine._structured_recall`

A new branch `(1c-i)` (commit `3210f46`) matches location queries for a named
entity and resolves them against the live `PersonalFactStore`:

```python
# ravana/src/ravana/chat/engine.py  (~L2410)
_ent_loc_a = re.search(
    r"\bwhere(?:'s|'re|s)?\s+(?:is|are|was|were\s+)?"
    r"(?:the|my|our|their|his|her|a|an|this|that|these|those)?\s*"
    r"([a-z][a-z'\\- ]{1,40}?)\s*"
    r"(?:moored|berthed|anchored|docked|based|parked|stationed|"
    r"kept|stored|housed|tied\s+up|wintered|located|situated)?\s*"
    r"(?:at|in|on)?\s*\??\s*$", q)
_ent_loc_b = re.search(
    r"\bwhat\s+is\s+(?:the|my|our|their|a|an|this|that)?\s*"
    r"([a-z][a-z'\\- ]{1,40}?)'s\s+location\s*\??\s*$", q)
```

Resolution is **generic across entities/places** via a suffix-window match:
the query phrase (or any trailing part of it) is tested for the stored subject
as a whole word, longest subject winning — so *"the slow coal narrowboat"*
still resolves to the stored subject `"slow coal"`. The reply is a single
state-driven frame `f"the {_subj} is at {_place}."`; the real content comes
from the store, not from a script. Unknown entities return `None` and fall
through to the honest uncertainty path.

## Verified behavior (live probe, dim=64, offline)

```
>> the slow coal is moored at bingley
   (stored: ('slow coal','location','bingley'))
>> where's the slow coal moored?      -> 'the slow coal is at bingley.'
>> where is the slow coal?            -> 'the slow coal is at bingley.'
>> where's the slow coal?             -> 'the slow coal is at bingley.'
>> what is the slow coal's location?  -> 'the slow coal is at bingley.'
>> actually the slow coal is moored at saltaire now
   (stored: ('slow coal','location','saltaire'); bingley superseded)
>> where's the slow coal moored?      -> 'the slow coal is at saltaire.'
>> i live in hexham
   noted — i'll remember you live in hexham.
>> where do i live?                   -> 'you live in hexham.'
>> where is paris?                    -> honest uncertainty (no place asserted,
                                        no 'bingley' leaked)
```

> Note: the mining turn's own surface acknowledgement is an uncertainty frame
> (`i don't really have a solid grasp on slow coal moored so far...`) even though
> the fact is stored (proven by the recall answers above). The fact lands in the
> store; only the *acknowledgement* wording is non-committal.

## Tests

`tests/unit/test_round_2026_08_10T0813_possession_location.py` (8 tests, all
passing; run `RAVANA_OFFLINE=1 pytest tests/unit/test_round_2026_08_10T0813_possession_location.py -q`):

- `test_possession_location_captured_and_trimmed` — entity + place stored.
- `test_possession_location_correction_supersedes` — saltaire active, bingley superseded.
- `test_possession_location_leading_hedge_not_new_entity` — *"actually the slow coal"* → same entity.
- `test_entity_location_recall_surfaced_not_echo` — all four phrasings surface the structured fact, not the episodic echo.
- `test_entity_location_recall_multiple_entities` — `slow coal`→bingley, `van`→leeds independently.
- `test_entity_location_recall_correction_composes` — recall reflects the supersede.
- `test_entity_location_recall_unknown_fails_closed` — no fabricated/leaked place (wording-independent assertion).
- `test_entity_location_recall_user_location_not_hijacked` — `where do i live?` answered by the user path.

## Design properties

- **Seed-driven, learnable.** The capability lives in the stores + miners, not
  in reply tables. A user can correct a stored location at runtime and the
  correction is reflected in recall — no retrain, no LLM.
- **Fail-closed.** Unknown places return honest uncertainty; the engine never
  fabricates a location.
- **Generic, not enumerated.** Any entity with a stored `('X','location','Y')`
  fact answers; there is no per-place or per-entity table and no authored reply
  pool (hardcoding audit: clean).
