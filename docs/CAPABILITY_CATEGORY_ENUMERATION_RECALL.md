# Capability: category-aware enumeration recall (list the people/pets you know)

**Status:** shipped (commit `295224a`, branch `auto/round-2026-08-16T1745Z`).
**Verified:** regression tests in
`tests/test_round_2026_08_16_1745_enum.py` pass (5/5, real engine
`dim=64, seed=42, baby_mode=True`, offline). Live end-to-end probe reproduced
below (real engine output). Hardcoding self-audit clean.

## What it does

When the user asks RAVANA to **LIST the entities it has learned in a category**
— *"name everyone in my family"*, *"name all my pets"*, *"who have i told you
about"*, *"list the people in my family"* — RAVANA **scans its LIVE
PersonalFactStore** and enumerates every relationship fact it mined (relatives)
plus every pet fact, rendering each as a real stored clause.

These queries have **no specific cue word** (no "indira", no "mochi"), so the
existing cued-recall branches (which require a named entity) never fired and the
query fell through to a generic acknowledgement (`noted.`) or a nonsense echo
(`right, people family.`). Measured as probe turn **T59** of round
2026-08-16T1745Z:

> `U: name everyone in my family you've heard about` → `R: noted.`

Every listed name, relationship, and detail is read from a runtime fact RAVANA
mined — never authored prose, no per-person table. Category membership is
decided by the **shared** lexicon helpers (`relation_attrs.is_relation_attribute`
/ `pet_slots.is_pet_attribute`) — the same functions the miner and cued-recall
use — so all three paths agree on what counts as a "relative" / "pet" by
construction (the slot-key-collision lesson from `pet_slots.py`). No LLM, no
retraining, fully online/incremental.

Real engine output (fresh persona taught a grandmother + brother + cat + dog):

```
Q: name everyone in my family you've heard about
A: you've told me about: your grandmother indira weaves baskets;
   your brother arjun climbs mountains; your cat is mochi; your dog is biscuit.

Q: name all my pets
A: (same — lists both pets by species + name)

Q: who have i told you about
A: (same — lists relatives AND pets)

Q: list the people in my family
A: (same — recognizes the enumeration intent under rotated phrasing)
```

The **same** entity set renders for every enumeration phrasing because the list is
built by scanning the store, not by replying to the words. A different
relationship pair (uncle / niece) is covered by a dedicated test, proving no
hardcoded branch.

**Honest empty state.** When the intent is recognized but nothing is stored, it
returns a true-state message (`you haven't told me about any family or pets
yet.`) rather than `None` (which would have fallen through to the wrong `noted.`
ack). Real engine output (brand-new engine):

```
Q(empty): name everyone in my family
A(empty): you haven't told me about any family or pets yet.
```

## How it grew from the conversation

The chat round of this cycle (round `2026-08-16T1745Z`) surfaced, among its
residual limitations, that the **category-list** class of questions was
unhandled — queries that ask for the *set* of learned entities, distinct from a
single cued recall ("what's my cat's name"). The feature card (`t_f1dae1aa`)
picked it as a concrete, generalizable capability gap.

**Root cause / prior behavior.** Enumeration queries had no specific cue word,
so `_structured_recall`'s cued-recall branches (1b) never matched and the query
fell through to the graceful-uncertainty fallback, emitting `noted.` (T59) or a
nonsense echo. RAVANA in fact held the mined facts but had **no path to
enumerate them**.

**Fix (commit `295224a`).**

1. A regex in `_structured_recall` (section `0c`,
   `engine.py:2458-2505`) detects the enumeration intent (name/list everyone in
   my family, name all my pets, who have i told you about, list the people in my
   family, …) — grammatical phrasings, **no per-topic phrase list**.
2. On a match it delegates to a new `_enumerate_entities`
   (`engine.py:3483-3559`) that scans the LIVE `PersonalFactStore`, collects
   every non-superseded relationship fact (`subject == "i"`, attribute passes
   `is_relation_attribute`) and pet fact (`is_pet_attribute`), deduplicates,
   and renders each as `your <combined-attr> <value>` (relative) or
   `your <species> is <name>` (pet).
3. The relationship lexicon that was a buried **local variable inside the miner**
   (`user_model.py` `_KIN`) was extracted into a single source-of-truth module
   `relation_attrs.py` — mirroring `pet_slots.py` — with seed vocabulary +
   runtime-learnable `learn_relation` + `is_relation_attribute` /
   `base_relation` / `render_relation`. The miner, the cued-recall renderers,
   AND the new enumerator now resolve "is this a relative?" through ONE function.

**Hardcoding audit.** `_enumerate_entities` contains no authored reply prose and
no per-topic answer table — only connective scaffolding (`you've told me
about:` / the honest empty-state string) around slots read from state at call
time (each stored name, relationship, detail, species). The diff grep for
`"[^"]{45,}"` returns only the intent regex and the honest empty-state message.
The 5 regression tests assert real stored content appears (each disclosed name
must be listed) and that `noted.` does **not** leak — i.e. they test the store
membership, not a verbatim reply string.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Enumeration intent detection (regex, section `0c`) | `ravana/src/ravana/chat/engine.py:2458-2505` |
| Branch entry in `_structured_recall` (returns the enumeration) | `ravana/src/ravana/chat/engine.py:2502-2504` |
| Store-scanning reply builder | `ravana/src/ravana/chat/engine.py:3483-3553` |
| Skip superseded facts + dedupe by category+value | `ravana/src/ravana/chat/engine.py:3525-3544` |
| Honest empty state (true zero-state) | `ravana/src/ravana/chat/engine.py:3516,3548` |
| Shared relationship lexicon (seed + `learn_relation`) | `ravana/src/ravana/chat/relation_attrs.py` |
| `is_relation_attribute` / `base_relation` / `render_relation` | `ravana/src/ravana/chat/relation_attrs.py:103-139` |
| Regression tests | `tests/test_round_2026_08_16_1745_enum.py` |

## Test coverage

`tests/test_round_2026_08_16_1745_enum.py` (5 tests, all pass):

- `test_enumerate_family_via_process_turn` — after disclosing two relatives,
  `name everyone in my family` lists BOTH by name (content from the store), not
  `noted.`.
- `test_enumerate_family_other_relationship_word` — a DIFFERENT relationship pair
  (uncle / niece) proves generalization, not a hardcoded `grandmother`/`brother`
  branch.
- `test_enumerate_pets_via_process_turn` — `name all my pets` lists each animal
  by species + name.
- `test_enumerate_everyone_mentioned` — `who have i told you about` enumerates
  the disclosed relatives AND pets.
- `test_enumerate_family_empty_is_honest` — a brand-new user gets no fabricated
  family list; `indira`/`arjun` (never disclosed) do not appear.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_round_2026_08_16_1745_enum.py -v
```

Full suites stayed green with the capability in place (unit 1883 passed / 22
skipped; non-unit 161 passed / 4 skipped; the 5 new tests **fail** if the
capability is stashed out, proving a real RED→GREEN).

## Known scope (honest — out of scope for this round)

- Scope is **relatives + pets** (the two categories the miner already produces
  as structured facts). Extending to other mined categories (e.g. "what do i
  keep" → hobbies) would reuse the same scan pattern as a follow-up capability.
- The capability lists what was DISCLOSED and MINED; a relative mentioned only in
  passing without a structured fact is not enumerated (same store boundary as
  every other recall path — honest by design).
