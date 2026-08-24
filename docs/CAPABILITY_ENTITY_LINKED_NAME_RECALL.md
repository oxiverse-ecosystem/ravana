# Capability: entity-linked (paraphrased / multi-word) name recall

**Status:** shipped (commits `9fb123e` + `ee0d34e`, branch `auto/round-2026-08-18T1340Z`). NOT pushed.
**Feature card:** `t_977d9eef` (round `2026-08-18T1340Z`).
**Verified:** `tests/unit/test_r1_entity_link_recall.py` passes (4/4). A live
in-process probe on this branch reproduced every example below (real engine
output, `dim=64, seed=42, baby_mode=True`, offline). Hardcoding self-audit
clean — the only added reply string is a templated
`f"your {entity}'s name is {val}."` where both slots are read from the stored
`PersonalFactStore`.

## What it does

When the user asks RAVANA for the **NAME of a possession they previously named**,
and phrases that possession with **different words than the stored entity key**
(*"that sourdough culture on my counter"* for a stored *"sourdough starter"*),
RAVANA now resolves the paraphrase to the **stored entity** via cross-lemma
GloVe cosine linking and reports its name:

```
taught:  "my best friend's name is Tomas and he's a chef in Lisbon"
taught:  "i keep a sourdough starter i named doris"
Q:       "what did i name that sourdough culture on my counter?"
A:       "your sourdough starter's name is doris."
```

This is the R1 capability of round `2026-08-18T1340Z`: **multi-word / paraphrased
entity cued-recall resolution**. Before the fix the same paraphrase returned the
**best-friend's name** (`"tomas"`) — it fell through to an unrelated `i`-scoped
name fact (the R1 confabulation) — because:

1. the personal-facts fold that *builds the entity index* silently **dropped**
   possession-NAME facts, and
2. the name-recall resolver scanned only `subject=="i"` facts, so a
   possession-name query was answered from the wrong store.

### How the paraphrase is bridged (no synonym table, no LLM, no retrain)

The linker `_link_recall_entity` (`engine_memory.py:359-411`) scores the
query's multi-word entity phrase against every stored entity key using the
**same seed GloVe embeddings the rest of the engine reasons over**. Three
properties keep it honest:

- **Store-driven, generalized.** Embeddings come from the runtime-acquired
  projector, so the linker resolves entities RAVANA has learned at runtime, not
  a hardcoded alias list. If the user teaches RAVANA a new word, the linker
  generalizes to it.
- **RAVANA confabulation bar (safety gate).** A stored key is only a candidate
  if its **head word appears verbatim in the query** (`engine_memory.py:382-385`).
  This stops a query naming Entity A from being aliased onto an unrelated
  stored Entity B just because a tail word is loosely similar. The shared head
  word is the genuine lexical common ground; the cosine linker then bridges the
  *paraphrased* tail word ("culture" ↔ "starter").
- **Fail-closed.** If nothing clears the cosine bar
  (`_RECALL_ENTITY_LINK_COS = 0.30`, `engine_memory.py:357`), the linker returns
  `None` and the caller answers with honest uncertainty — it never fabricates a
  fact.

Two cooperating fixes make the end-to-end path work:

1. **Possession-NAME facts are folded into the entity index**
   (`engine_memory.py:529-530`, commit `9fb123e`). The old skip tuple excluded
   `name` for *all* entities, which dropped
   `('sourdough starter','name','doris')` from the index and was the **true
   root cause** of R1.
2. **An entity-scoped name-recall branch in `_structured_recall`**
   (`engine.py:2426-2468`, commit `ee0d34e`) runs **before** the `subject=="i"`
   scanners. It detects a NAME query, gathers the stored entity-scoped name
   facts (`subject != "i"`, `attr == "name"`, not superseded), links the query
   entity phrase to the right key via `_link_recall_entity`, and renders
   `your {entity}'s name is {val}.`.

## Fail-closed (honest abstention, no leak)

```
taught:  "i keep a sourdough starter i named doris"
Q:       "what did i name that garden gnome on my shelf?"   # never disclosed
A:       (no "doris" — honest miss; never aliases onto a stored entity)

taught:  "i keep a sourdough starter i named doris"
Q:       "what is my name?"                                  # generic self-name
A:       (no "doris" — own-name query is NOT hijacked by a possession)
```

The path is gated on an interrogative/recall frame (`engine.py:2443-2445`) and
returns `None` when the linker yields no key, so an unknown possession gets
honest uncertainty and a generic self-name query stays with the `i` profile.

**Adjacent possession still resolves to its own fact** (no regression):

```
taught:  "my best friend's name is Tomas and he's a chef in Lisbon"
taught:  "i keep a sourdough starter i named doris"
Q:       "what's my best friend's name?"
A:       (contains "tomas", not "doris")
```

No LLM, no per-entity answer table, no retraining. The capability is entirely
store-driven: a user can disclose or correct a possession name at runtime and
this path reflects it. RAVANA revises any stored fact through normal
conversation, satisfying the seed + online-learning constraints.

## Known rough edges (honest — logged for a future round)

- **Two-word paraphrase only.** The linker requires the query to contain at
  least two content words (`engine_memory.py:372-374`) and a verbatim head-word
  match (`engine_memory.py:382-385`). A fully opaque paraphrase that shares
  *no* head word with the stored key (e.g. "that bubbling thing" for
  "sourdough starter") will not link and fails closed — by design, to avoid
  confabulation.
- **Name attribute only.** The entity-scoped recall branch is narrow: it fires
  for `attr == "name"`. Other entity-scoped attributes (e.g. a possession's
  `madeof`, `does`) are handled by the separate possession-attribute cued-recall
  path; this branch does not generalize to them.
- The capability renders whatever the **miner** stored, so it inherits miner
  fact quality. The recall path is correct; the upstream stored key shapes the
  output.

## How it grew from the conversation

The chat round of this cycle (round `2026-08-18T1340Z`) closed its **residual
limitation R1**: a cued recall that paraphrased a multi-word entity did not
resolve to the right stored fact. Investigation during the feature turn showed
two distinct defects, not one:

### Root cause A — possession-NAME facts dropped from the entity index (commit `9fb123e`)

In `_retrieve_episodic` (`engine_memory.py`), the personal-facts fold that
builds `_entity_idx` had a skip tuple that excluded `name` for **all** entities:

```python
elif _attr and _attr not in ("name", "location", "does", "event",
                             "is", "favorite", "likes", "background"):
    _entity_idx.setdefault(_key[0], {})[_attr] = _val
```

The intent was to keep the `i` *biographical* profile (name/location/does/…) out
of the entity index. But the skip also threw away **possession-NAME facts keyed
under the ENTITY** — `('sourdough starter','name','doris')` — so the index never
contained them and the paraphrase had nothing to link to. This was the **true
root cause** of R1.

**Fix** (`engine_memory.py:529-530`): split the condition so that a
possession-NAME fact (`_ent != "i" and _attr == "name"`) is folded into the
index, while the `i`-profile attrs stay excluded:

```python
elif _ent and _ent != "i" and _attr == "name":
    _entity_idx.setdefault(_ent, {})[_attr] = _val
elif _ent and _ent != "i" and _attr and _attr not in (
        "location", "does", "event", "is", "favorite",
        "likes", "background"):
    _entity_idx.setdefault(_ent, {})[_attr] = _val
```

The same `_retrieve_episodic` path also seeds the link attempt later
(`engine_memory.py:705-708`): when a query named a specific entity (`_named`)
but the verbatim scan missed, it tries `_link_recall_entity` across the whole
entity index before the generic-self fallback — and leaves `_ent_hit = None`
(fail closed) if even the linker can't resolve it.

### Root cause B — no entity-scoped name recall; `i`-scoped scanner answered (commit `ee0d34e`)

Even with the fact folded, `_structured_recall`'s name resolution scanned only
`subject=="i"` facts. A possession-name query ("sourdough culture") therefore
matched the *first* `i`-scoped name fact it found (the best-friend's name) —
the documented R1 confabulation.

**Fix** (new branch (0a), `engine.py:2426-2468`): before the `i`-scoped
scanners, detect a NAME query, collect the stored entity-scoped name facts, link
the query phrase to the right key, and render it:

```python
_name_q = bool(re.search(r"\b(name|named|called)\b", q)) and bool(
    re.search(r"\?$|\b(what|who|which|tell|do|does|did|how)\b", q))
if _name_q and pf is not None:
    _ent_name_keys = [
        _k for _k in pf.facts.keys()
        if isinstance(_k, tuple) and len(_k) == 3
        and _k[0] not in ("i", "me", "my", "you")
        and _k[1] == "name"
        and not getattr(pf.facts[_k], "superseded", False)
    ]
    if _ent_name_keys:
        _stored_entities = {_k[0] for _k in _ent_name_keys}
        _linked_ent = self._link_recall_entity(q, _stored_entities)
        if _linked_ent is not None:
            for _k in _ent_name_keys:
                if _k[0] == _linked_ent:
                    _v = getattr(pf.facts[_k], "value", _k[2])
                    return f"your {_linked_ent}'s name is {_v}."
```

The branch returns `None` (falls through) when the linker yields no key, so an
unresolved-but-specific query is never aliased onto the `i` profile.

### The cross-lemma linker itself (`engine_memory.py:357-430`, commit `9fb123e`)

`_link_recall_entity` (`:359`) builds the query's content-word phrase, filters
candidate stored keys by verbatim head-word overlap (`:382-385`), then scores
each remaining key by max mean-cosine against every query word-window via
`_phrase_sim_to_query` (`:419`) using `_mean_vec` (`:412`). It returns the best
key only if it clears `_RECALL_ENTITY_LINK_COS` (`:357`, `0.30`).

## Hardcoding audit (summary)

Every reply-producing string added this round is connective scaffolding around
state read at call time — **no authored reply prose, no `random.choice` reply
pools, no keyword→response tables, no Q→A dict**:

- `f"your {_linked_ent}'s name is {_v}."` (`engine.py:2466`) — both `{_linked_ent}`
  and `{_v}` are read from the stored `PersonalFactStore` at call time.
- `_link_recall_entity` / `_mean_vec` / `_phrase_sim_to_query`
  (`engine_memory.py:359,412,419`) — pure semantic computation over the runtime
  GloVe projector; no authored language.
- `_RECALL_ENTITY_LINK_COS = 0.30` (`engine_memory.py:357`) — a measured
  threshold (above the noise floor), not a per-topic constant.
- The `i`-profile attr exclusion tuple (`engine_memory.py:537-540`) — structural
  vocabulary, not reply text.

**Seed-vs-hardcoding:** the GloVe embeddings are the *same seed* the engine
reasons over everywhere; the linker generalizes to runtime-learned words via the
already-acquired projector (no fixed synonym table, so no "frozen vocabulary"
dodge applies). The only render string is templated around live store slots.
Deciding test ("can RAVANA change this by itself?") → the capability reflects any
disclosed/corrected fact through normal conversation; the render content comes
entirely from the store. PASS. **No retraining:** all changes are
online/incremental.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Entity-scoped name recall branch (0a) | `ravana/src/ravana/chat/engine.py:2426-2468` (detect `:2443-2445`; gather keys `:2451-2457`; link `:2460`; render `:2466`) |
| Cross-lemma linker `_link_recall_entity` | `ravana/src/ravana/chat/engine_memory.py:359-411` |
| Link cosine bar `_RECALL_ENTITY_LINK_COS` | `ravana/src/ravana/chat/engine_memory.py:357` |
| Verbatim head-word safety gate | `ravana/src/ravana/chat/engine_memory.py:382-385` |
| `_mean_vec` / `_phrase_sim_to_query` | `ravana/src/ravana/chat/engine_memory.py:412,419` |
| Possession-NAME fold into entity index | `ravana/src/ravana/chat/engine_memory.py:529-530` |
| Link attempt in `_retrieve_episodic` | `ravana/src/ravana/chat/engine_memory.py:705-708` |
| Interrogative/recall-frame gate | `ravana/src/ravana/chat/engine.py:2443-2445` |

## Test coverage

`tests/unit/test_r1_entity_link_recall.py` (4 tests, all pass):

- `test_r1_paraphrased_entity_resolves_to_stored_name` — disclose best-friend +
  sourdough starter; *"what did i name that sourdough culture on my counter?"*
  returns `"doris"` and does **not** leak `"tomas"`.
- `test_r1_best_friend_still_resolves` — *"what's my best friend's name?"*
  still returns `"tomas"`.
- `test_r1_unknown_entity_does_not_confabulate` — *"what did i name that garden
  gnome on my shelf?"* (never disclosed) does **not** return `"doris"`.
- `test_r1_own_name_query_not_hijacked` — *"what is my name?"* does **not**
  return `"doris"`.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_r1_entity_link_recall.py -v
```

The broader recall/stance/relationship suite stayed green at the round (the
parent feature card reports `test_r1_entity_link_recall.py` 4 passed,
`test_recall_confabulation_2026g` + `test_memory_reconstructor` 18 passed).
