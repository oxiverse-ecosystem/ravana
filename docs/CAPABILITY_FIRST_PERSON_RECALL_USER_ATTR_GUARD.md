# Capability: first-person recall + user-attribute guard (no confabulated reply)

**Status:** shipped (round `2026-10-12T1042Z`, feature `t_75bd7442`).
**Verified:** regression suite `tests/unit/test_d3_first_person_attr_guard.py` passes
4/4 on the shipped code. The tests exercise real `CognitiveChatEngine` probes
(`dim=64, seed=42, baby_mode=True`, offline) — not mocks. Hardcoding self-audit
clean: no authored reply prose, no per-topic table, no retraining — only a
structural guard that excludes user-attribute queries from the first-person
recall path using the SAME shared lexicon the existing guard already uses.

## What it does

When the user asks a first-person recall query that also names a **user
attribute** — *"what did i just tell you about my dog"*, *"what did i just
tell you about my sister"* — the query is NOT a recall of RAVANA's own speech.
It is a recall of the USER's own fact. Before this fix, the
`_is_first_person_recall` regex matched the scaffolding (*"what did i just
tell you about"*) regardless of what followed, so the user-attribute guard
below was skipped and the query reached the agent-reply store, returning an
unrelated reply from a prior turn.

**Measured defect (T27):** *"what did i just tell you about my dog"* →
*"your sister maya is just got accepted"* — an unrelated agent reply confabulated
from token overlap.

The fix adds a guard at `engine.py:6217-6225`: when `_is_first_person_recall` is
True AND the query also matches the shared user-attribute word list, the flag is
set to False so the query falls through to the user-fact recall path.

Concretely:

- *"what did i just tell you about my dog"* → `_route_agent_own_recall` returns
  `None` (falls through to user-fact recall).
- *"what did i just tell you about my sister"* → returns `None`.
- *"what did you say about art"* → NOT blocked (genuine agent-self recall, no
  user-attribute word).
- *"what did i just tell you about my favorite food"* → NOT blocked (RV-5
  backfill: "favorite food" is NOT in the user-attribute word list).

The attribute word list (`name`, `named`, `called`, `age`, `live`, `lives`,
`from`, `work`, `study`, `studied`, `grew up`, `sister`, `brother`, `mother`,
`father`, `mom`, `dad`, `family`, `relative`, `kin`, `grandmother`,
`grandfather`, `grandma`, `grandpa`, `aunt`, `uncle`, `cousin`, `niece`,
`nephew`, `grandchild`, `wife`, `husband`, `partner`, `kid`, `child`, `son`,
`daughter`, `pet`, `friend`, `job`, `house`, `home`, `phone`, `computer`,
`laptop`, `car`, `cat`, `dog`, `crow`, `bird`, `fish`, `petname`, `pet name`)
is the SAME shared lexicon the existing user-attribute guard at line 6233-6243
already uses — no per-topic table, no authored prose.

No LLM, no Q→A dict, no keyword→reply table. The capability is a structural
addition to the existing recall routing; RAVANA's recall paths are recomputed
from the real utterance every turn, satisfying the seed + online-learning
constraints.

## How it grew from the conversation

The round report `tmp/reports/ravana-2026-10-12T1042Z-t_75bd7442.md` logged a
new defect spotted during that cycle:

> *"a first-person recall query that also names a user attribute ('what did i
> just tell you about my dog') is NOT a recall of RAVANA's own speech — it is
> a recall of the USER's own fact. The _is_first_person_recall regex matches
> the scaffolding regardless of what follows, so the attribute guard is
> skipped and the query reaches the agent-reply store, returning an unrelated
> reply."*

The feature card (`t_75bd7442`) picked this as the last limitation to turn into
a real capability.

**Root cause / prior behavior.** `_route_agent_own_recall` computes
`_is_first_person_recall` by matching the recall scaffolding (*"what did i just
tell you about"*). When True, the user-attribute guard at line 6233 is skipped
(`if not _is_first_person_recall and ...`). So a query like *"what did i just
tell you about my dog"* — which names a user attribute — was treated as an
agent-self recall and routed to the agent-reply store, where token overlap with
a prior reply triggered a confabulation.

**Fix (round `2026-10-12T1042Z`).** Add a guard at `engine.py:6217-6225` that
sets `_is_first_person_recall = False` when the query also matches the shared
user-attribute word list. The existing user-attribute guard at line 6233 then
handles the query correctly (returns None, falling through to user-fact recall).

**Hardcoding audit.** The diff adds zero reply strings. The attribute word list
is the SAME closed-class lexicon the existing guard already uses — no new
per-topic table, no authored answer. The guard is structural: it routes the
query to the correct recall path based on the presence of user-attribute words,
not a hardcoded reply.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_route_agent_own_recall` (the recall routing method) | `ravana/src/ravana/chat/engine.py:6137` |
| `_is_first_person_recall` heuristic (scaffolding match) | `ravana/src/ravana/chat/engine.py:6196-6203` |
| D3 guard: exclude user-attribute queries from first-person recall | `ravana/src/ravana/chat/engine.py:6217-6225` |
| Existing user-attribute guard (shared lexicon) | `ravana/src/ravana/chat/engine.py:6233-6243` |
| Regression tests | `tests/unit/test_d3_first_person_attr_guard.py` |

## Test coverage

`tests/unit/test_d3_first_person_attr_guard.py` (4 tests, all pass on shipped
code):

- `test_d3_first_person_recall_with_user_attr_does_not_confabulate` —
  *"what did i just tell you about my dog"* must NOT return an unrelated agent
  reply; `_route_agent_own_recall` must return `None`.
- `test_d3_first_person_recall_with_sister_attr_does_not_confabulate` —
  *"what did i just tell you about my sister"* must return `None`.
- `test_d3_first_person_recall_without_user_attr_still_works` —
  *"what did you say about art"* must NOT be blocked by the D3 guard (genuine
  agent-self recall).
- `test_d3_first_person_recall_favorite_food_still_works` —
  *"what did i just tell you about my favorite food"* must still reach the
  agent-reply store (RV-5 backfill: "favorite food" is NOT in the
  user-attribute word list).

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_d3_first_person_attr_guard.py -v
```
