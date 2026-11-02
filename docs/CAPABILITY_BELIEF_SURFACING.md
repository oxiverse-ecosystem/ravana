# Capability: belief surfacing on belief-recall queries

**Status:** shipped (round `2026-11-02T0311Z`, feature `t_97335d25`).
**Verified:** live engine run surfaces `you told me ...` on several belief
phrasings (see parent feature card). Source:
`ravana/src/ravana/chat/engine_self_query.py:1688-1702`.

## What it does

When the user asks what RAVANA believes / what they said they believe
(e.g. *"what do you believe about X?"*, *"do you recall my belief that Y?"*),
RAVANA now surfaces the **stored conviction** from its `BeliefStore` whose
proposition best overlaps the query topic, instead of returning a generic
fallback. The reply is composed from live belief state:

- `you told me <proposition>.`

The selection is structural: collect the belief whose proposition has the
highest token overlap with the query tokens; only surface it when at least one
token overlaps. There is no per-topic table, no authored prose, no retraining
— the content comes entirely from the durable `BeliefStore`.

## How it grew from the conversation

- A chat probe surfaced that belief-recall phrasings (queries containing
  `believe`/`belief`/`conviction` plus a recall frame like `what`/`which`/
  `do you`/`did i`/`tell me`/`recall`/`remember`) were not answered from the
  stored belief state, even though the belief had been recorded earlier.
- The fix added a recall handler at the top of `_route_self_query`
  (`engine_self_query.py:1688-1702`) that reads `self.belief_store.beliefs`
  (a dict of `(_subj, _pred) -> (_val, _conf, _turn)`) and returns the
  best-overlapping proposition wrapped as `you told me {proposition}.`.

## API (verified)

The capability is reached through the normal turn path:

```python
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix="demo")
# (record a belief during a turn, e.g. via a statement the miner captures)
r = eng.process_turn("what do you believe about the ocean?")
# r may be "you told me ..." when a matching belief exists, else None/fallback
```

Verified by direct execution of the source path
(`ravana/src/ravana/chat/engine_self_query.py:1688-1702`): the handler
iterates `getattr(self.belief_store, "beliefs", {})`, computes token overlap
between the lowercased query and each proposition, and returns the best match
when overlap >= 1.

No LLM, no Q→A dict, no retraining. The capability is a structural
addition; RAVANA's beliefs are read live from `BeliefStore`.
