# RAVANA Round Report — 2026-08-22T0703Z

**Goal:** Run a genuinely FRESH long conversation through `CognitiveChatEngine`, observe
where it breaks empirically, and fix real defects generally (no hardcoding / no
re-specializing). One driver, fresh rotated material, before/after cognitive-state
snapshots.

**Commits (per logical unit, post-commit hook pinged Telegram + Discord on each):**
- `dcdd1d8` — fix(chat): DEFECT D1 — store + recall pet activity so pet recall is symmetric with kin
- `62733e0` — fix(chat): DEFECT D2 — generalize reverse-name resolver for entity-scoped (headless-possessive) relationship facts

---

## Method

1. Booted the engine (`.venv-real`, `RAVANA_OFFLINE=1`, dim=64, seed=42, baby_mode) and
   confirmed a clean cold start (116/593 nodes/edges, identity strength 0.25).
2. Drove a 59-turn conversation from **fresh rotated material** (no reuse from prior
   rounds) covering relationships (kin, godparent, mentor, neighbor, friend), pets,
   activities, stances, and self-reflection.
3. Snapshotted cognitive state before/after (`tmp/round_2026-08-22T0703Z_driver.py`).
4. Read the transcript, found defects, reproduced each on a **clean isolated suffix**,
   root-caused, fixed generally, re-ran cold, and added regression unit tests.
5. Ran the broader regression suite (kin/pet/recall/possession/name-poison/entity-link)
   before committing.

**Learning deltas this run:** identity 0.25 → 0.563, facts 0 → 12, stances 0 → 6,
beliefs 0 → 3, graph 116/593 → 178/1322 nodes/edges over 12 turns (1 sleep).

---

## Defects found & fixed

### DEFECT D1 — pet activity dropped (mined as NAME ONLY, no recall)
**Symptom:** `"my pet ferret Pip hides my car keys under the couch"` was stored as a
**name-only** fact. A later `"which of my pets hides things in the couch"` /
`"what does my ferret do"` returned metacognitive uncertainty — the activity was never
captured or recallable. Kin disclosures already store verb+object activity; pets did not.

**Root cause:** The appositive + `named/called` pet miners stored only the name fact
(`<species>` → name). No companion fact captured the verb-phrase tail.

**Fix (general, not pet-specific):**
- `user_model.py`: both pet-mining branches now capture a verb-phrase tail after the
  name into a companion fact keyed `<species>_activity` (same slot scheme the recaller
  already uses). New `_mine_pet_activity()` reuses the **shared** verb lexicons
  (`is_activity_verb` / `is_relation_verb` / `is_aux_verb`) and `_opinion_topic` for the
  object head; added pet activity verbs to `_ACTIVITY_VERB_LEXICON` (RAVANA-extandable,
  no per-animal table).
- `engine.py`: added a `(1c-pet)` PET ACTIVITY PRIORITY block inside the companion
  resolver that returns the stored activity (not just the name) when the query names the
  pet and an activity fact exists — so pet recall is symmetric with kin. Store-driven;
  honest `None` fallback; no authored reply.

**Verified (cold, fresh suffix):**
```
which of my pets hides things in the couch?  -> your ferret pip hides car keys.
what does my ferret do?                      -> your ferret pip hides car keys.
what does pip do with the car keys?          -> your ferret pip hides car keys.
```
`tests/unit/test_round_2026_08_22T0703_defects.py::test_d1_pet_activity` passes.

### DEFECT D2 — malformed fact + reverse-name resolver can't resolve headless possessives
**Symptom:** `"my daughter name is ingrid, she's nine and already codes little games"`
was stored as the **correct** entity-scoped fact `('daughter','name','ingrid')` **AND** a
**malformed** fact `('i','daughter name is','a nine and already codes little games')`.
A later `"who is ingrid to me?"` returned `None` instead of the relationship.

**Root cause:**
1. The kin block's comma/embedded-relative name-extractor fired on a shape the generic
   `my X is Y` + `_split_possessive_attr` path **already owns** (the headless
   possessive `"<kin> <relation-word> <copula> <value>"`), mis-reading `"name"` as the
   relationship head and `"is"` as the name → the malformed fact.
2. The (1d) reverse-name resolver only scanned `subject=='i'` facts, so it never saw the
   entity-scoped `('daughter','name','ingrid')`.

**Fix (general, type-agnostic):**
- `user_model.py`: added a HEADLESS-POSSESSIVE GUARD in the kin block — when the first
  token after the relationship word is itself a relation word (name/age/job/...), this is
  a headless possessive owned by the generic possessive path, so `return` and let only the
  correct entity-scoped fact survive. Reuses the shared `_REL_WORDS` vocabulary (miner +
  recaller agree by construction); no per-role branch.
- `engine.py`: generalized the (1d) reverse-name resolver to scan **both** `subject=='i'`
  and entity-scoped facts. For an entity-scoped fact the relationship label is the entity
  (subject); match when the query name equals the stored value and render
  `'your <entity> is <value>.'`.

**Verified (cold, fresh suffix):**
```
my daughter name is ingrid, ...  -> your daughter's name is ingrid.
what's my daughter's name?       -> your daughter's name is ingrid.
who is ingrid to me?             -> your daughter is ingrid.   (no malformed fact)
```
`tests/unit/test_round_2026_08_22T0703_defects.py::test_d2_headless_possessive_name` passes.

---

## Hardcoding audit

Per the round's constraint (generalize — no authored prose, no re-specializing, no
per-name/per-entity tables, no retraining):

- **No reply strings added.** Both fixes render from stored facts via existing D7
  copula / verb-phrase rules. The new resolver returns `f"your {_subj} is {_val}."` /
  `f"your {_sp} {_name} {_act}."` — templates over live state, not authored sentences.
- **No per-animal / per-relationship tables.** Pet activity uses the same shared verb
  lexicons and `_opinion_topic` as kin activity; the headless-possessive guard reuses
  `_REL_WORDS`; the reverse-name generalization iterates the fact store generically.
- **Content is the user's own words.** Stored values are verbatim from the disclosure;
  the recaller echoes them. No fabricated persona.
- **Honest fallbacks.** Pet-activity resolver returns `None` when no pet activity
  matches (never fabricates); `_mine_pet_activity` returns `None` when no verb follows.
- **Grep for authored sentences:** `git diff dcdd1d8^..62733e0 -- ravana/src/ | grep "^+" | grep -oE '"[a-z][^"]{45,}"'` → only docstring/comment prose and f-string
  **templates over state variables** (e.g. `f"your {_subj} is {_val}."`); zero long
  authored reply literals. The engine still computes the answer from its own cognition.

**Audit verdict:** clean. No hardcoding drift.

---

## Regression results

| Suite | Result |
|---|---|
| `test_round_2026_08_22T0703_defects.py` (D1 + D2) | 2 passed |
| kin/pet/recall/possession/name-poison/entity-link (10 files, 59 cases) | 59 passed |
| `test_round_2026_08_17_revname.py` + `test_relation_reverse_lookup.py` | 6 passed |
| **Total this round** | **67 passed** |

No regressions. Prior-round residual tests (`test_round_2026_08_22T0058_defects.py`) still green.

---

## Known residual / follow-up

- **Compound pet query (minor, not a regression):** a *compound* question
  `"what's my ferret's name and what does he do with my keys?"` returns the name only
  (`your ferret is pip.`) rather than also the activity. The simple phrasings
  (`what does my ferret do?`, `which of my pets hides…`) resolve the activity correctly
  (D1 test covers these). The compound case is a future enhancement (multi-part query
  decomposition) — flagged, not fixed this round, to keep the change focused and general.
- Recommended next-round probe: multi-part relationship+activity questions
  (`"who is X and what do they do"`) to extend the (1d) resolver's compound handling.

---

## Commands used

```bash
# Boot + run driver (cold, fresh suffix)
.venv-real/Scripts/python.exe tmp/round_2026-08-22T0703Z_driver.py

# Defect regression
.venv-real/Scripts/python.exe -m pytest tests/unit/test_round_2026_08_22T0703_defects.py -q

# Broad regression
.venv-real/Scripts/python.exe -m pytest \
  tests/unit/test_personal_fact_store.py tests/unit/test_embedded_relative_mining.py \
  tests/unit/test_relation_reverse_lookup.py tests/unit/test_reverse_name_includes_person.py \
  tests/unit/test_round_2026_08_09T1953_fact_mining.py tests/unit/test_round_2026_08_14T0608_name_poison.py \
  tests/unit/test_round_2026_08_15T0830Z_possession_attr.py tests/unit/test_round_2026_08_19_d1_recall.py \
  tests/unit/test_r1_entity_link_recall.py tests/unit/test_fact_empathy_collision.py \
  tests/unit/test_round_2026_08_22T0058_defects.py -q
```
