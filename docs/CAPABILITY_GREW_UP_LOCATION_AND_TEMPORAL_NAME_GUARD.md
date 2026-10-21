# Capability: "I grew up in…" location capture + temporal-adverb name guard (T12 / T28)

**Status:** shipped (commit `a46653f3`, branch `auto/round-2026-10-20T0433Z`).
**Verified:** 7/7 tests pass in
`tests/unit/test_round_2026_10_20_t28_temporal_name.py` (offline, `RAVANA_OFFLINE=1`);
live end-to-end probes below (real engine output, `dim=64, seed=42, baby_mode=True`,
offline). Hardcoding self-audit clean — no authored reply prose, no per-topic answer
table, no retraining, no LLM.

## What it does

Two defects from the round `2026-10-20T0433Z` are fixed in this commit — one a
**location** extraction gap, one a **name** extraction gap:

1. **T12 — "i grew up in <place>" is now a location fact, not a garbage event.**
   The disclosure *\"i grew up in kodaikanal in the mountains\"* is stored as
   `('i', 'grew up', 'kodaikanal')` (with `user_location = "kodaikanal"`) and the
   trailing *\"in the mountains\"* descriptor is stripped. Before the fix,
   **no** location fact was stored and a garbage `('i', 'event:grow', 'grew kodaikanal')`
   fact was written instead (the location miner never recognised the irregular past
   *\"grew\"*, and the event/open-class miners had no phrasal-verb guard for
   *grow up*).

2. **T28 — a temporal adverb is never captured as a personal name.**
   The disclosure *\"i had a huge fight with my best friend today\"* produces
   **zero** facts — specifically it does **not** store the defective
   `('i', 'friend today', 'today')` fact that leaked into unrelated chitchat recall
   (*\"how are you doing today\" → \"your friend is today\"*). A genuine lowercase
   name followed by a temporal tail (*\"my friend rhea called yesterday\"*) is still
   mined correctly as `('i', 'friend rhea', 'called yesterday')`.

Both share one design spine: a **closed-class seed vocabulary** of grammatical words
(location verbs, natural-feature descriptors, temporal adverbs) that the miner consults
when deciding what may occupy a head/name slot, grown at runtime via
`learn_temporal_adverb` / `learn_residence_verb` / `learn_locative_preposition` in
`pet_slots.py` — so a word RAVANA has never seen joins the class without a code change.
No per-topic, per-place, or per-name logic.

## How it grew from the conversation (source citations)

The parent chat round surfaced two independent limitations (T12 and T28); the feature
card (`t_79423d00`) picked both because they share the same root: the relationship /
location miner and the name-span gate had no closed-class test for irregular
inflections and temporal adverbs.

### T12a — location verb inflection (the core fix)

`UserModel.mine_personal_facts` (`ravana/src/ravana/chat/user_model.py`) holds the
location miner `m_loc`, whose verb class was **stem + optional regular suffix**:

```python
# user_model.py:2534  (m_loc regex — residence verb + locative preposition + place)
m_loc = re.search(
    r"\bi\s+(?:live|stay|am|is|are|was|were|been|move|remain|"
    r"reside|settle|relocate|born|grow\s+up|grew\s+up|"
    r"grown\s+up|growing\s+up)(?:s|es|d|ed|ing)?\s+"
    r"(?:in|near|at|from|to|onto)\s+"
    r"([A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*){0,7})",
    q_clean, re.IGNORECASE)
```

Before this fix the verb class was only `grow\s+up` — the **irregular past** *\"grew\"*
was missing (`grew != grow+ed`), so *\"i grew up in kodaikanal\"* matched `m_loc`
neither. The token fell through to the **EVENT verb miner** (`user_model.py:4776`),
which had no phrasal-verb guard for *grow up*, and stored the garbage fact
`('i', 'event:grow', 'grew kodaikanal')` — consuming the place entirely.

The fix adds the full irregular+perfect+progressive inflection set
(`grow up | grew up | grown up | growing up`) to the regex verb class — **structural
inflection coverage**, not per-verb enumeration. Because the optional suffix class
`(?:s|es|d|ed|ing)?` still rides atop it, every form is reachable from the same closed
set. No LLM, no per-city list.

### T12b — phrasal-verb guards on the activity miners (defense-in-depth)

Even though `m_loc` now owns *\"grew up\"*, the activity/event miners would still
re-mine it if the location path ever regressed. Two guards were added so the
residence-verb + *up*-particle shape is rejected upstream:

```python
# user_model.py:4778  (EVENT miner — skip "grow up"/"grew up" as an experience event)
if _verb in ("grow", "grew") and _em.group(2).strip().lower().startswith("up"):
    continue
```

```python
# user_model.py:4928  (OPEN-CLASS miner — skip "grow up"/"grew up" as a does:-fact)
if (_obj_toks and _is_residence_verb(_verb)
        and _obj_toks[0] == "up"
        and _verb in ("grow", "grew", "grown", "growing")):
    continue
```

Both reuse the **shared** `is_residence_verb` / `is_locative_preposition` vocabulary in
`pet_slots.py` (the same one the location miner consults), so all three paths agree on
what counts as a residence disclosure *by construction* — no duplicated verb list.

### T12c — natural-feature lexicon (strip trailing descriptors)

When `m_loc` now captures *\"kodaikanal in the mountains\"* the trailing descriptor is
not a toponym. A shared closed-class lexicon `_NATURAL_FEATURES`
(`user_model.py:2597`, used by both the `_named_loc` path at `user_model.py:2628` and the
`m_loc` branch at `user_model.py:2683`) strips trailing geographic modifiers *before*
the fact is stored:

```python
# user_model.py:2597
_NATURAL_FEATURES = {
    "mountains", "mountain", "hills", "hill", "coast", "coastline",
    "river", "rivers", "valley", "valleys", "range", "ranges",
    "highlands", "lowlands", "plateau", "plateaus", "desert",
    ...
}
```

A determiner-led common-noun fragment (*\"the mountains\"*, *\"the hills\"*) is rejected
by the trailing toponym override guard at `user_model.py:2758` (`_is_desc` checks the
first word is a determiner and the last is in `_NATURAL_FEATURES`) — so the real
preceding toponym is kept instead of being clobbered by *\"the mountains\"*.

### T12d — verb-preserving attribute (correct recall phrasing)

The location is stored under the attribute **\"grew up\"** (not \"location\") when the
disclosure verb is *grow up* (`user_model.py:2637` and `user_model.py:2694`), so a
later recall renders *\"you grew up in kodaikanal\"* instead of the wrong
*\"you live in kodaikanal\"*. The verb is read from the matched text, not hardcoded per
place.

### T28 — temporal-adverb name guard

`mine_personal_facts` has a lowercase-name fallback that accepts any non-function,
non-verb token as a personal name. Temporal adverbs (*today, now, always, ...*) had no
closed-class gate, so *\"my best friend today\"* captured *today* as the NAME. The fix
adds `is_temporal_adverb()` to the **name-span `_close` gate** — the exact same gate
that already consults `is_function_word`:

```python
# user_model.py:11   (import)
from .pet_slots import is_temporal_adverb
```

```python
# user_model.py:3999  (name-span _close check — gates what may occupy a NAME slot)
_close = (
    _tc in _pet_slots._PRONOUN_STOP
    or _tc in _OBJECT_STOP
    or is_activity_verb(_tc)
    or is_relation_verb(_tc)
    or is_aux_verb(_tc)
    or is_temporal_adverb(_tc)        # ← T28
    or (_name_candidate_toks
        and _tc.endswith(("ed", "ing")))
)
```

`is_temporal_adverb` lives in `pet_slots.py` (`pet_slots.py:265`), the shared
slot-naming module, backed by the seed set `_TEMPORAL_SEED`
(`pet_slots.py:232`) plus the runtime-grown `_TEMPORAL_LEARNED` (`pet_slots.py:245`)
populated by `learn_temporal_adverb` (`pet_slots.py:248`) — structurally identical to
the species / function-word seeds. Removing an entry degrades gracefully (the word is
simply allowed again, exactly the cold-start behaviour).

## Design compliance

- **Seed knowledge only.** `_TEMPORAL_SEED`, `_RESIDENCE_SEED`, `_NATURAL_FEATURES`
  are closed-class word sets (grammatical word classes: adverbs, residence verbs,
  geographic descriptors) — data, not reply prose, never rendered to the user.
  Expandable at runtime via `learn_temporal_adverb` / `learn_residence_verb`; removing
  an entry degrades gracefully, no per-topic table, no answer dict.
- **Online / incremental, no retraining.** Facts are mined live from each turn; the
  seed sets grow from real utterances. A new temporal adverb or residence verb becomes
  rejected/known the moment it is disclosed.
- **Fail-closed.** No matched residence verb → no location fact. A temporal adverb
  that survives the gate still gets rejected as a NAME by construction. No fact is
  ever fabricated.
- **Zero authored reply prose.** Reply strings (if recalled) are f-string renders of
  live stored state. Hardcoding audit: `is_temporal_adverb` and the *grow up* verb
  class are word-set entries, not reply strings.
- **No LLM.** Pure regex + closed-class vocabulary lookup + the existing concept
  graph recall path.

## Live verification (fresh engine, offline)

Real output, engine `dim=64, seed=42, baby_mode=True`, `RAVANA_OFFLINE=1`. Stored
facts read from the live `personal_facts.facts` store:

```
MINE   'i grew up in kodaikanal in the mountains'
stored -> ('i','grew up','kodaikanal'), ('i','location','kodaikanal')
user_location = kodaikanal
garbage event:grow fact present: False

MINE   'i grew up in aldermoor'
stored -> ('i','grew up','aldermoor'), ('i','location','aldermoor')
user_location = aldermoor

MINE   'i had a huge fight with my best friend today'
stored -> (no facts)                       # T28: temporal adverb rejected as name
defective ('i','friend today','today') present: False

MINE   'my friend rhea called yesterday'
stored -> ('i','friend rhea','called yesterday')   # rhea preserved
rhea in fact: True

MINE   'i live in berlin'  (regression: regular residence verb still works)
stored -> ('i','location','berlin')
user_location = berlin
```

## Tests

`tests/unit/test_round_2026_10_20_t28_temporal_name.py` — 11 tests, all pass
(`.venv-real`, `RAVANA_OFFLINE=1`, 19.42s):

- `test_temporal_adverb_not_captured_as_name` (5 parametrised disclosures) — no fact
  uses a temporal adverb as the NAME slot in the kin-relation-without-verb path.
- `test_friend_today_defect_absent` — the exact defective fact
  `('i', 'friend today', 'today')` is never mined.
- `test_real_name_still_mined_with_temporal_tail` — a genuine lowercase name (*rhea*)
  following a temporal adverb (*yesterday*) is still captured under the relation.
- `test_grew_up_stores_location_not_garbage_event` — *\"grew up in\"* stores
  `('i','grew up',<place>)`, strips the descriptor, and stores no `event:grow` garbage.
- `test_grew_up_plain_place_no_descriptor` — bare *\"grew up in X\"* dual-stores
  verb + location attribute.
- `test_grew_up_descriptor_stripped_via_natural_features` — *\"the himalayas\"* place
  head kept, no garbage event.
- `test_lived_in_still_works` — regression: regular residence verbs unaffected.

### Coverage gap (honest — now closed)

The T12 **location-extraction** behavior — *\"grew up\" → `('i','grew up',<place>)`*
with the natural-feature descriptor stripped and no `event:grow` garbage fact —
**previously had no dedicated regression test**. It has now been added to
`tests/unit/test_round_2026_10_20_t28_temporal_name.py` as four T12 cases (11-test
suite total, green): `test_grew_up_stores_location_not_garbage_event`,
`test_grew_up_plain_place_no_descriptor`, `test_grew_up_descriptor_stripped_via_natural_features`,
and `test_lived_in_still_works` (regression ensuring regular residence verbs are unaffected).

## Caveats

- The location miner stores the verb-preserved attribute **\"grew up\"** for recall
  phrasing; a query that explicitly asks *\"where did you grow up\"* reads this
  attribute, while a generic *\"where do you live\"* reads the \"location\" attribute.
  Both are mined from the same disclosure (verified above).
- `is_temporal_adverb` only fires inside the name-span `_close` gate; it does not
  suppress legitimate temporal predicates stored as VALUES (e.g.
  *\"my friend rhea called yesterday\"* → `('i', 'friend rhea', 'called yesterday')`
  is correctly kept, because *yesterday* is the VALUE not the NAME).
