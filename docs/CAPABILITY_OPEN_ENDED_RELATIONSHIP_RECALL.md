# Capability: open-ended relationship / person recall

**Status:** shipped (commit `8581ee3`, branch `auto/round-2026-08-17T1126Z`). NOT pushed.
**Feature card:** `t_1a4a3938` (round `2026-08-17T1126Z`).
**Verified:** the two round tests in `tests/unit/test_round_2026_08_17T1126_fixes.py`
pass (`test_miner_stores_named_relationship_lowercase_name`,
`test_open_ended_relationship_recall`); a live in-process probe on this branch
reproduced every example below (real engine output, `dim=64, seed=42,
baby_mode=True`, offline). Hardcoding self-audit clean (no authored reply prose,
no per-person answer table — only connective scaffolding around slots read from
the live `PersonalFactStore`).

> NOTE ON HASHES: the round's internal feature report cited `4b9f210` for the
> recall commit. That hash is wrong. The actual recall commit is `8581ee3`, and
> the miner commit is `44be502`. All citations below were re-checked against
> `git show` and the live source, not copied from the report.

## What it does

When the user asks RAVANA to **REPORT what it knows about a named
relationship or person in the open** — *"tell me about my grandmother"*,
*"who is my grandmother?"*, *"what does my grandmother do?"*, *"what do you know
about my brother"*, *"describe my niece priya"* — RAVANA now answers from the
**same relationship facts it mined from conversation**, regardless of how the
query is phrased. Previously these returned `None` / metacognitive-uncertainty
because every prior resolver keyed only on a **bare name** (*"who is indira"*) or
a **specific verb** (*"what does X do"*), never on the **relationship word itself**
when phrased openly.

Two cooperating fixes make this work end-to-end:

1. **The miner now STORES the fact at all** (commit `44be502`). The old D7
   relationship-activity miner required the disclosed **Name to be capitalized**
   (`[A-Z][A-Za-z]*`). Real chat names are lowercase
   (*"my grandmother indira bakes bread"*), so the capitalized group matched
   nothing, the name-less fallback fired, and its kin+verb slot (*"indira"*) was
   not an activity verb — so **the fact was never stored**. Without storage there
   was nothing to recall. The fix (a membership-based token scan) stores the fact
   regardless of name casing (see *How it grew*).

2. **A new open-ended recall branch (1d)** in `_structured_recall` resolves the
   open phrasings above by detecting a **relationship word** from the SHARED
   `relation_attrs` lexicon (plus an optional name token or pet entity) and
   scanning the live store for every matching fact.

Real engine output (fresh engine, taught *"my grandmother indira bakes sourdough
bread every sunday"*, *"my brother theo fixes bicycles for the neighborhood
kids"*, *"my niece priya weaves baskets"*, then queried):

```
Q: tell me about my grandmother
A: "your grandmother indira bakes sourdough bread."

Q: who is my grandmother?
A: "your grandmother indira bakes sourdough bread."

Q: what does my grandmother do?
A: "your grandmother indira bakes sourdough bread."

Q: what do you know about my brother
A: "your brother theo fixes bicycles."

Q: describe my niece priya
A: "your niece priya weaves baskets."

Q: who is theo?            # bare NAME, no relationship word in the query
A: "your brother theo fixes bicycles."
```

**Pets are covered too** (branch (1d) `(c)` keys on a stored pet **attribute**,
not the relationship lexicon). With *"my cat is pixel"* / *"my dog is biscuit"*
mined into the pet-name slots:

```
Q: tell me about my cat
A: "your cat is pixel."

Q: who is my cat?
A: "your cat is pixel."
```

## Fail-closed

The new branch returns `None` when nothing maps. An **unknown relative gets honest
uncertainty, never a fabricated bio**, and a relation **without a stored fact**
falls through to the normal pipeline:

```
Q: tell me about my uncle fred      # never disclosed
A: None                             # honest — no fabricated bio
```

**Declarative disclosures are NOT hijacked.** This resolver *reports* stored
knowledge, so it is gated on an **interrogative / recall frame** (the query ends
in `?` or starts with a question word: *what/who/which/…/tell/describe/…*). A
declarative mention must reach the empathy / fact-mining paths untouched:

```
Q: my friend is hurting             # genuine distress
A: None                             # correctly NOT answered as a fact echo;
                                     # reaches the empathy router

Q: my grandmother bakes bread       # declarative disclosure
A: None                             # correctly left to fact-mining, not echoed
```

This gate was added because the first cut fired on any `"my <relation> <token>"`
utterance and swallowed distress (*"my friend is hurting"* returned
*"your friend is hurting."*), failing `test_genuine_distress_still_routes_to_empathy`.

No LLM, no per-person answer table, no retraining. The capability is entirely
store-driven: a user can disclose or correct a relationship/pet fact at runtime
and this path reflects it. RAVANA can revise any stored fact through normal
conversation, satisfying the seed + online-learning constraints.

## Known rough edges (honest — logged for a future round)

- **One relationship word per query.** The salient-token scan records the *first*
  relation word (and the first bare name) it sees, then returns every fact whose
  base relation matches that one word. A compound query that names two relations
  at once — *"tell me about my brother and sister"* — returns only the facts for
  the first matched relation (*"your brother theo fixes bicycles."*), not a
  joined list. This is a scan limitation, not a correctness bug; single-relation
  queries (the common case) are exact.
- The capability renders whatever the **miner** stored, so it inherits the
  miner's fact quality (combined-attr key `"<kin> <name>"`, verb-phrase object
  capped at 5 tokens). The recall path is correct; the upstream stored key shapes
  the output.

## How it grew from the conversation

The chat round of this cycle (round `2026-08-17T1126Z`) closed its **residual
limitation #1**: open-ended kin/relationship phrasing (*"tell me about my
grandmother"*) did not resolve even though *"what does my grandmother do"* was
assumed to work. Investigation during the feature turn showed the gap was in fact
**worse** than logged — the D7 miner wasn't even *storing* the named-relationship
fact when the name was lowercase, so there was nothing to recall.

### Root cause A — miner dropped named-relationship facts (commit `44be502`)

`UserModel._D7` relationship miner (`ravana/src/ravana/chat/user_model.py`,
FIX block at `1349`, code `1366-1396`) used a positional/capitalization-sensitive
regex:

```python
re.search(r"\bmy\s+([a-z][a-z]+)\s+([A-Z][A-Za-z]*(?:\s+[A-Z][A-Za-z]*)?)?\s*([a-z][a-z]+)\s+(.+?)…", q_clean)
```

The optional Name group `(?P<name>[A-Z][A-Za-z]*)` only matched a **capitalized**
name. In real chat the name is lowercase (*"indira"*), so the group matched
nothing, the name-less fallback fired, and its kin+verb slot (*"indira"*) was not
in the activity-verb lexicon → **the fact was never stored**.

**Fix:** replace the positional regex with a **membership-based token scan**
(after `"my <kin>"`, scan tokens left-to-right for the first token that is an
activity verb in the shared `_REL_ACTIVITY_VERB_FORMS` lexicon
`user_model.py:1338-1342`; tokens before it are the Name, tokens after are the
object). Structural — one verb lexicon, **no per-name table, no case assumption**
— generalizes to any name casing/length. Content comes from the user's own words.

Verified storage (real run of the regression test):

```
mine("my grandmother indira bakes sourdough bread every sunday")
  -> {('i', 'grandmother indira'): 'bakes sourdough bread'}
mine("my sister climbs rocks")          -> {('i', 'sister'): 'climbs rocks'}
mine("my brother theo fixes bicycles")  -> {('i', 'brother theo'): 'fixes bicycles'}
```

### Root cause B — no open-ended recall path (commit `8581ee3`)

Even for facts that *were* stored, `_structured_recall` only keyed relationship
recall on a bare name (*"who is indira"*) or a specific verb (*"what does X do"*).
There was no branch that keyed on the **relationship word** when phrased openly.

**Fix:** new branch **(1d)** in `_structured_recall`
(`ravana/src/ravana/chat/engine.py`, `2892-3012`):

1. Build salient query tokens (drop the closed-class `_QR_STOP` set,
   `engine.py:2936`).
2. Detect a **relationship word** via the **SHARED** `relation_attrs` lexicon
   (`relation_of` / `is_relation_attribute` / `base_relation`, imported at
   `engine.py:2901-2905`) — so the miner, the category-enumerator, and this
   recaller agree on what a "relative" is by construction — and/or a **name
   token** and/or a **pet entity** (`is_pet_attribute` / `base_species` from
   `pet_slots.py`, `engine.py:2906-2910`; `is_activity_verb` from `user_model.py`,
   `2914`).
3. **Gate** on an interrogative / recall frame (`_or_is_q`, `engine.py:2959`) so
   declarative disclosures are not hijacked (see Fail-closed).
4. Scan the live `PersonalFactStore` (`pf.facts`) for **every** `subject=="i"`
   fact that (a) is a relationship attr whose `base_relation` matches, (b) carries
   the named person (trailing combined-attr name, or name in value), or (c) is a
   pet of the queried species (`engine.py:2966-3009`).
5. Render all matches with the existing **D7 copula rule** — verb-phrase values
   drop the copula (`your {attr} {verbphrase}.`), noun-phrase values keep it
   (`your {attr} is {value}.`), `engine.py:3010-3012`.

Fail-closed: `_bits` is empty → falls through, returning `None` (`engine.py:3012`).

## Hardcoding audit (summary)

Every reply-producing string added this round is connective scaffolding around
state read at call time — **no authored reply prose, no `random.choice` reply
pools, no keyword→response tables, no Q→A dict**:

- `_mk = re.search(r"\bmy\s+([a-z][a-z]+)\b\s*(.*)", …)` (miner) — structural
  token scan of the user's own disclosure.
- `_t.lower().strip(".,!?") in _REL_ACTIVITY_VERB_FORMS` — membership test
  against a seed verb lexicon RAVANA can extend at runtime.
- `f"your {_attr} {_val}."` / `f"your {_attr} is {_val}."` — recall render;
  `_attr`/`_val` are live store values.
- `_QR_STOP` frozenset — closed-class stopwords (structural vocabulary, not reply
  text).
- `_or_rel_of` / `_or_is_rel` / `_or_base_rel` / `_or_is_pet` / `_or_base_sp` /
  `_or_is_act` — single-source-of-truth imports from the shared lexicon modules.

**Seed-vs-hardcoding:** the verb / relationship / pet lexicons are SEED structure
(RAVANA-extendable at runtime via `learn_relation` / `learn_species`; removing an
entry degrades gracefully). The render frames read live store values. Deciding
test ("can RAVANA change this by itself?") → YES for the lexicons; the recall
content comes entirely from the store. PASS. **No retraining:** all changes are
online/incremental.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| D7 miner token-scan fix (membership, not case) | `ravana/src/ravana/chat/user_model.py:1349-1396` (`_mk` scan at `1366`; verb membership at `1374`; combined-attr store at `1395-1396`) |
| Shared activity-verb lexicon | `ravana/src/ravana/chat/user_model.py:1338-1342` (`_REL_ACTIVITY_VERB_FORMS`) |
| Open-ended recall branch (1d) | `ravana/src/ravana/chat/engine.py:2892-3012` |
| Salient-token stopword set `_QR_STOP` | `ravana/src/ravana/chat/engine.py:2936` |
| Interrogative / recall-frame gate `_or_is_q` | `ravana/src/ravana/chat/engine.py:2959` |
| Shared lexicon imports (relation + pet + verb) | `ravana/src/ravana/chat/engine.py:2901-2915` |
| Live store scan + D7 copula render | `ravana/src/ravana/chat/engine.py:2966-3012` |
| Fail-closed (`return None` when no match) | `ravana/src/ravana/chat/engine.py:3012` |
| Shared relation lexicon (single source of truth) | `ravana/src/ravana/chat/relation_attrs.py:89,103,120` |
| Shared pet lexicon (single source of truth) | `ravana/src/ravana/chat/pet_slots.py:84,89` |
| `is_activity_verb` | `ravana/src/ravana/chat/user_model.py:63` |

## Test coverage

Two round tests in `tests/unit/test_round_2026_08_17T1126_fixes.py` (both pass;
the file is green — 5/5 with the three earlier round tests):

- `test_miner_stores_named_relationship_lowercase_name` (`85`) — disclose
  *"my grandmother indira bakes sourdough bread every sunday"*; asserts the
  combined-attr fact `('i', 'grandmother indira')` is stored with value
  `bakes sourdough bread` — directly exercising the case-insensitive miner fix.
- `test_open_ended_relationship_recall` (`96`) — disclose grandmother + brother,
  then assert open phrasing (*"tell me about my grandmother"*), interrogative
  (*"who is my grandmother?"*), activity-asking (*"what does my grandmother do?"*),
  bare-name (*"who is theo?"*) all return the stored fact, and an unknown relative
  (*"tell me about my uncle fred"*) fails closed (`None`).

A third test covers the documented pet path, which was verified live but lacked a
regression test before this doc:

- `test_open_ended_recall_includes_pet_path` — disclose *"my cat is pixel"* /
  *"my dog is biscuit"*, then assert *"tell me about my cat"* / *"who is my cat?"*
  resolve to *"your cat is pixel."* via branch (1d)`(c)`.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_round_2026_08_17T1126_fixes.py -v
```

The broader recall/stance/relationship suite stayed green at the round (46
related-suite tests passed; the broad non-unit suite ran 69 passed / 0 failures
after the interrogative-gate fix re-closed the distress-empathy regression).
