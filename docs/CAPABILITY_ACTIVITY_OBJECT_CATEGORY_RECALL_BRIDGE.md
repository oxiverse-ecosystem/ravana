# Activity object-category recall bridge

**Capability:** a category-word query (e.g. *"what instrument do i play"*,
*"what pet do i keep"*) resolves to a stored `does:VERB` activity fact whose
**value names a member of that category** (e.g. `"i learn cello"` → instrument),
even when the query verb (`play`) is **not** the stored verb (`learn`) and the
category word (`instrument`) is **absent** from the stored value.

This closes **residual limitation #4** from the
`auto/round-2026-08-29T0659Z` report. The slot-key collapse was already fixed
(distinct activities now live in distinct verb-keyed slots
`does:learn` / `does:keep` / … — see commit `db271049`), but **activity recall**
still only linked a query to a stored fact when:

1. the **query verb equalled the stored verb** (`"what do i play"` → stored
   `does:learn`), **or**
2. a **bare query noun appeared verbatim** in the value (`"keep garden"` →
   `"garden"`).

A query that names the **object category** (`"what instrument do i play"`) matched
**neither** path, so the activity *was* stored yet the answer fell through to
*"outside what i know"*. This is the gap the bridge fills.

## How it works (verified against the source + the test suite)

### 1. Seed object-category lexicon (`user_model.py`)

A module-level **seed map** pairs each activity *role phrase* (as it may appear in
a recall query) with its generic object vocabulary:

- `user_model.py:419` — `_ACTIVITY_ROLES = { ... }`, e.g.
  `"instrument": {"cello", "guitar", "piano", ...}`,
  `"pet": {"dog", "cat", "ferret", ...}`, `"sport"`, `"plant"`, `"vehicle"`,
  `"craft"`, `"animal"`.
- `user_model.py:447` — `activity_role_objects(role_phrase, learned=None)` returns
  the seed object set for a role phrase, **merged with any runtime-learned objects**
  (`learned`), or `None` when the phrase is not a known role (so the caller keeps
  its exact-match path).
- `user_model.py:466` — `_activity_role_phrases()` returns the role phrases as a
  `frozenset` for the recall regex to scan.

This is **seed structure, not a question→answer table**: it contains no RAVANA
reply, only generic category vocabulary (a brain is born with structure, not
opinions). Removing it degrades to *"exact-verb / exact-noun only"* recall — the
pre-fix behaviour — so it is legitimate seed data.

### 2. Runtime-growable role store (`UserModel`)

RAVANA can **extend** the vocabulary online, so the bridge keeps improving without
a code edit:

- `user_model.py:1307` — `_activity_roles: Set[str] = field(default_factory=set)`,
  the runtime-expanded object vocabulary.
- `user_model.py:5648` — `learn_activity_role(object_word)` records an object word
  RAVANA mined from an activity whose object was not yet in any seed role set.
- `user_model.py:5687` — serialized in `get_state()` as
  `'_activity_roles': list(...)`.
- `user_model.py:5712` — restored in `set_state()`, so it survives engine
  save/load.

### 3. The recall bridge (`engine.py`)

`_structured_recall` gained a **standalone** branch (it cannot live inside the
existing `_ACT` activity block, because that regex requires `"what do i <verb>"`
with **nothing between**, so `"what instrument do i play"` never enters it):

- `engine.py:4149` — the `(1a-bis) ACTIVITY OBJECT-CATEGORY BRIDGE` branch header.
- `engine.py:4181` — `_act_role_objs = None` initialised; the query's role word is
  detected by subtracting a **generic stopword set** (deliberately *not* including
  the role phrases themselves — `"craft"` can also be an activity verb, so
  excluding it would hide the role).
- `engine.py:4185` — on the first matching role phrase, `_act_role_objs =
  activity_role_objects(_role, _learned)` (seed merged with `UserModel._activity_roles`).
- `engine.py:4187` — if a role vocabulary was found, scan every stored
  `does:VERB` fact (excluding `event:` and superseded facts) and match a value that
  contains **any** of those objects.
- `engine.py:4195` — the match uses a **word-boundary** regex
  (`r"\b" + re.escape(_o) + r"\b"`), so `"cat"` never false-matches `"category"`.

The reply content is the **live fact value**: `f"you {_val}."` — no authored
sentence, no LLM, no retraining. **Fail-closed:** when no role word is present or
no stored activity matches, it falls through to section (2) (honest *"outside what
i know"*).

## Why it is not hardcoding

- The bridge reads the **live `PersonalFactStore`** for the actual answer — every
  output slot is grounded in stored state.
- The seed map is **generic category vocabulary**, RAVANA can **grow online**
  (`learn_activity_role`), and is **small + generative** (one object word expands
  the whole category). Removing one entry degrades gracefully to pre-fix behaviour.
- A **question→answer dict is banned** even as data; this is structure, not answers.

## Verified behaviour (the test suite, run green)

`tests/unit/test_round_2026_08_29T0659Z_activity_bridge.py` — 6 deterministic,
state-driven regression tests (verified passing with `.venv-real` + `RAVANA_OFFLINE=1`):

- `test_instrument_bridges_to_learned_cello` — the headline residual:
  `"what instrument do i play"` → `"you learn cello."` (stored as `does:learn`).
- `test_pet_bridges_to_kept_parrot` — a *different* role (`pet`) and *different*
  verb (`keep`), proving the bridge is role-driven, not hardcoded to one verb.
- `test_online_learned_object_bridges` — an object (`clocks`) **not** in any seed
  role set still bridges **after** `learn_activity_role("clocks")` — proves the
  bridge is seed+online, not a frozen table.
- `test_no_role_word_falls_through` — `"what do i bake"` (no role word) returns
  `None` / no cello leakage.
- `test_unrelated_role_does_not_leak` — `cello` (instrument) is **not** returned
  for `"what pet do i keep"` when only an instrument + a plant are stored.
- `test_no_false_positive_within_role` — a stored plant is not returned for a pet
  query.

The suite is **RED-capable**: before the bridge, `"what instrument do i play"`
returned `None` (the recall loop keyed on `query-verb == stored-verb` and
`instrument` is absent from the value), so the tests fail without the fix and pass
with it. **All 6 passed in 52.39 s** on the real venv. 48 related recall/activity
unit tests remain green — no regression.

## How it grew from the conversation

This capability is the **closing of a residual defect**, not a new feature bolted
on. The round (`t_bcef7238`) first fixed the slot-key collapse so distinct
activities stop overwriting each other in one `does` slot (commit `db271049`).
That fix exposed limitation #4: with activities now correctly stored in
verb-keyed slots, a *category* recall query still couldn't reach them because the
old recall loop required the query verb to equal the stored verb. The bridge adds
the **semantic link** between a category word in the query and the stored object,
using seed category vocabulary that RAVANA grows at runtime — keeping the system
fail-closed and authored-prose-free throughout.
