# Capability: non-kin relationship recall (mentor / teacher / coach / friend …)

**Status:** shipped (commit `abf9169`, branch `auto/round-2026-08-17T1730Z`). NOT pushed.
**Feature card:** `t_3347f32f` (round `2026-08-17T1730Z`; handoff limitation #2).
**Verified:** the two round tests in
`tests/unit/test_round_2026_08_17T1126_fixes.py` pass
(`test_role_word_not_misstored_as_pet_species`,
`test_combined_attr_role_recall_full_name_and_activity`); a live in-process probe on
this branch reproduced every example below (real engine output, `dim=64, seed=42,
baby_mode=True`, offline). Hardcoding self-audit clean (no authored reply prose, no
per-role answer table — only connective scaffolding around slots read from the live
`PersonalFactStore`).

## What it does

A disclosure like *"my mentor Dr. Okonkwo taught me astronomy"* is now mined into a
**single correct combined-attr relationship fact** (`'i', 'mentor dr. okonkwo',
'taught astronomy'`) and recalled **in full** from any open phrasing — *"who is my
mentor?"*, *"tell me about my mentor"*, *"what does my mentor do?"*,
*"who is my mentor and what did they teach?"* — returning
`your mentor dr. okonkwo taught astronomy.` with the **full name + activity**, no
truncation, no doubled output.

This generalizes the existing open-ended relationship-recall capability
(`docs/CAPABILITY_OPEN_ENDED_RELATIONSHIP_RECALL.md`) to **non-kin ROLE words**
(mentor, teacher, coach, tutor, friend, neighbour, boss, manager, colleague,
coworker, roommate, landlord, rival, enemy, guardian, carer, …) — relationships the
user discloses about themselves exactly like family, now mined + recalled the same
way as *"my brother Arjun"*.

Real engine output (fresh engine, taught *"my mentor Dr. Okonkwo taught me
astronomy when i was a teenager"*, then queried):

```
Q: who is my mentor?
A: "your mentor dr. okonkwo taught astronomy."

Q: tell me about my mentor
A: "your mentor dr. okonkwo taught astronomy."

Q: what does my mentor do?
A: "your mentor dr. okonkwo taught astronomy."

Q: who is my mentor and what did they teach?
A: "your mentor dr. okonkwo taught astronomy."

Q: who is my mentor and what did they teach me?
A: "your mentor dr. okonkwo taught astronomy."
```

The stored fact set after the disclosure (real probe, filtered to active facts):

```
('i', 'mentor dr. okonkwo', 'taught astronomy')
```

The bogus pet/species fact `('i', 'mentor', 'dr')` that this round fixed is **absent**
— see *How it grew*.

**Generalizes** to teacher / coach / friend / neighbour (all now in the shared seed;
verified by the round test rotating across these roles). No hardcoded reply, no
retraining; the role vocabulary is seed that RAVANA also grows at runtime via
`learn_relation` (see *Hardcoding audit*).

## Fail-closed

The capability rides the same open-ended recall branch (1d) documented in
`CAPABILITY_OPEN_ENDED_RELATIONSHIP_RECALL.md`, so the same fail-closed behaviour
holds: an unknown relationship with no stored fact falls through to honest
uncertainty, and an interrogative/recall frame gate keeps declarative disclosures
(*"my friend is hurting"*) on the empathy path. The only change this round makes is
ensuring the **role word is not silently mis-stored as a pet species** at mine time,
so the correct fact is the one and only thing recall can find.

## How it grew from the conversation

The chat round of this cycle (round `2026-08-17T1730Z`) surfaced a residual
limitation: the open-ended relationship-recall capability (built in
`2026-08-17T1126Z`) worked for **kin** but broke for **non-kin roles**. A disclosure
*"my mentor Dr. Okonkwo taught me astronomy"* produced a **bogus** fact
`('i', 'mentor', 'dr')` that (a) truncated recall to *"your mentor is dr."* and (b)
doubled the open-ended output, because the correct combined-attr fact and the bogus
pet fact both matched the recall scan.

### Root cause — slot collision between the pet miner and the role miner

The **appositive-pet miner** runs **BEFORE** the role miner in
`UserModel.mine_personal_facts` (`ravana/src/ravana/chat/user_model.py`,
appositive-pet branch at `1281-1301`). It matches the shape *"my &lt;species&gt;
&lt;ProperNoun&gt;"* and stores a pet fact — unless the species word is a known
relationship, in which case it bails out via its `relation_of()` guard
(`user_model.py:1288-1292`):

```python
from .relation_attrs import relation_of as _app_rel_of
...
if _app_rel_of(_sp) is not None:
    continue
```

But the non-kin ROLE words (mentor, teacher, coach, friend, …) lived **only** in the
role miner's **LOCAL** `_ROLE_WORDS` set (`user_model.py`, deleted in this commit at
the old block `1478-1487`) and were registered into `relation_attrs` **LATER**, via
`learn_relation` (`user_model.py:1497`) — which executes **after** the pet miner has
already run. So at pet-mining time `relation_of("mentor")` returned `None`, the guard
did **not** fire, and *"my mentor Dr. Okonkwo"* was matched as
`<species=mentor> <ProperNoun=Dr.>` → the bogus fact `('i', 'mentor', 'dr')`.

This is the **slot-collision lesson** from prior rounds (one relationship vocabulary,
not two): the pet miner and the role miner maintained *separate* ideas of what a
relationship word was, so they disagreed at exactly the wrong moment.

### Fix — single source of truth

Make the **shared** `relation_attrs._RELATION_SEED` the **single source of truth** for
*every* relationship word, kin and non-kin, and delete the duplicate local list:

1. **Folded the non-kin ROLE words into `_RELATION_SEED`**
   (`ravana/src/ravana/chat/relation_attrs.py:65-101`). Now `relation_of("mentor")`
   returns `"mentor"` at pet-mining time, so the pet miner's guard (`user_model.py:1288-1292`)
   rejects the role word **before** storing. Only the correct combined-attr fact
   `('i', 'mentor dr. okonkwo', 'taught astronomy')` remains. The comment block
   (`relation_attrs.py:65-77`) spells out the single-source-of-truth contract so
   the miner, the pet miner's guard, and the recaller all agree on what counts as a
   relationship *by construction*.

2. **Deleted the duplicate `_ROLE_WORDS` local set** from the role miner
   (`user_model.py`, old `1478-1487`). The role miner's head-word guard now reads the
   **shared** lexicon: `if _kin in _KIN or _ra_of(_kin) is not None`
   (`user_model.py:1481`). The runtime-growth call `learn_relation(_kin)` is retained
   (`user_model.py:1497`), so a brand-new role word RAVANA has never heard
   (*"my bhabhi neha paints"*) is still addressable for later recall without any code
   change.

Verified storage (real run of the regression test
`test_role_word_not_misstored_as_pet_species`):

```
mine("my mentor Dr. Okonkwo taught me astronomy when i was a teenager")
  -> bogus ('i', 'mentor', 'dr')  : ABSENT
  -> correct ('i', 'mentor dr. okonkwo', 'taught astronomy') : PRESENT
```

## Hardcoding audit (summary)

Every change this round is seed structure or a structural guard — **no authored reply
prose, no `random.choice` reply pools, no keyword→response tables, no Q→A dict**:

- The non-kin role entries in `_RELATION_SEED` (`relation_attrs.py:78-100`) are a
  **seed vocabulary** RAVANA can extend at runtime via `learn_relation`
  (`relation_attrs.py:107-126`); removing any entry degrades gracefully.
- `if _app_rel_of(_sp) is not None: continue` (`user_model.py:1291`) — structural
  guard reading the shared lexicon.
- `if _kin in _KIN or _ra_of(_kin) is not None` (`user_model.py:1481`) — structural
  guard reading the shared lexicon.
- The recall render (`your {attr} {val}.` / `your {attr} is {val}.`) is unchanged
  from the open-ended recall branch and reads live store values.

**Seed-vs-hardcoding:** the role lexicon is SEED (RAVANA-extendable at runtime; single
source of truth shared by three code paths). The deciding test ("can RAVANA change
this by itself?") → YES for the lexicon; the recall content comes entirely from the
store. PASS. **No retraining:** all changes are online/incremental.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Shared relationship seed — non-kin ROLE words folded in (single source of truth) | `ravana/src/ravana/chat/relation_attrs.py:65-101` (entries `78-100`; contract comment `65-77`) |
| `relation_of` (shared lexicon lookup the pet guard + role guard both use) | `ravana/src/ravana/chat/relation_attrs.py:129-140` |
| `learn_relation` (runtime-growth path — new role words become addressable) | `ravana/src/ravana/chat/relation_attrs.py:107-126` |
| Pet miner appositive branch (runs BEFORE role miner) | `ravana/src/ravana/chat/user_model.py:1281-1301` |
| Pet miner `relation_of()` guard that now rejects role words | `ravana/src/ravana/chat/user_model.py:1288-1292` (`continue` at `1291`) |
| Role miner head-word guard now reads shared lexicon (local `_ROLE_WORDS` deleted) | `ravana/src/ravana/chat/user_model.py:1481` |
| Role miner runtime-growth (`learn_relation(_kin)` retained) | `ravana/src/ravana/chat/user_model.py:1497` |
| Open-ended recall branch (1d) that surfaces the stored fact — unchanged this round | `ravana/src/ravana/chat/engine.py:2892-3012` |

## Test coverage

Two round tests in `tests/unit/test_round_2026_08_17T1126_fixes.py` (both pass; real
run: `2 passed in 16.95s`, offline):

- `test_role_word_not_misstored_as_pet_species` (`251`) — disclose *"my mentor Dr.
  Okonkwo taught me astronomy when i was a teenager"*; asserts the bogus pet fact
  `('i', 'mentor', 'dr')` is **not** stored and the correct combined-attr fact
  `('i', 'mentor dr. okonkwo', 'taught astronomy')` **is**. Both fail without the
  seed change (the bogus fact reappears, the correct one is absent).
- `test_combined_attr_role_recall_full_name_and_activity` (`280`) — disclose the
  mentor fact, then assert 5 open/interrogative/activity phrasings all return the
  full name + activity (`dr. okonkwo` + `taught astronomy`), contain **no**
  truncated *"your mentor is dr"*, and emit **exactly one** `your mentor` clause
  (no doubled output). All five assertions fail without the fix.

The documented behaviour (non-kin role mine + full recall) is therefore **already
covered** by the parent commit's regression tests; no additional test was required
for this docs round.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_round_2026_08_17T1126_fixes.py -v
```
