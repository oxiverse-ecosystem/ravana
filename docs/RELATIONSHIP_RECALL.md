# Relationship recall — answering relation-word questions from the structured store

RAVANA can now answer questions that name a *relationship word* for a person you
have told it about — "what does my brother do for work?", "what is my sister's
job?", "what's my brother's name?" — by reading the fact it stored about that
person, not by echoing an unrelated past turn. This page documents that
capability: what it does, how it grew out of a real conversational gap, the real
code paths, and how it is verified.

All claims below were checked against the source on branch
`auto/round-2026-08-13T1656Z` at commits `5fe61cb` and `1038b44`. Line numbers
cite that tree.

## What it does

When you disclose a relationship — *"my brother dev works as a paramedic in
leeds"* — RAVANA stores the relationship **under the person's name** (the fact
key is `(subject="dev", attribute="relationship", value="brother")`). That is
the right place for it: Dev is the entity, "brother" is how Dev relates to you.

The problem this capability solves: once that fact exists, a query that names
the *relation word* instead of the *name* — *"what does my brother do for
work?"* — had nothing to look up under "brother" and fell through to an
episodic echo of some earlier turn. This capability adds a **forward relationship
index** that resolves the spoken relation word back to the person's name, then
reads that person's structured facts:

```text
disclosure: "my brother dev works as a paramedic in leeds."
            → store: (dev, relationship, brother)  [attr=relationship, val=brother]
query:      "what does my brother do for work?"
            → resolve_relation("brother") == "dev"
            → read dev's 'does' fact → "works as a paramedic"
            → reply: "your brother works as a paramedic."
```

It handles two word orders:

- **possessive attribute** — "what's my brother's job", "what is my sister job",
  "what is my brother's name" — matched by the `_ENT_ATTR` branch in
  `engine._structured_recall` (`engine.py:2470-2485`).
- **verb-before-noun** — "what does my brother do for work", "what does my
  sister do" — matched by the `_DO_ENT` branch (`engine.py:2493-2506`), because
  the possessive pattern requires the attribute word to *follow* the relation
  noun and misses this order.

If the relationship was never disclosed, both branches return `None` and the
engine abstains (fail-closed) rather than confabulating — see the verification
and limits below.

## How it grew — the residual gap

Round `t_214d9a50` (auto cycle 2026-08-13T1656Z) picked up residual limitation
#1 from the round report: relation-word queries echoed an unrelated episode even
though the structured fact existed.

Root cause: the disclosure miner correctly stores the relationship *under the
person's name* (`user_model.py:602-605`, guarded by the shared `_RELATION_VOCAB`
seed set). So the relationship is keyed by the entity ("dev"), with
`attr="relationship"` and `val="brother"`. The recall resolver for
"who is X to me" (the *reverse* index, name → relationship) was already in
place, but nothing answered the *forward* direction: given the relation word
"brother" that the user speaks in a query, recover the entity name so the
caller can read that entity's `role` / `does` / `name` facts. Queries naming the
relation word found nothing under "brother" and fell through to the episodic
echo.

The fix added the forward half:

1. `PersonalFactStore.resolve_relation(relword) -> entity` — a durable index
   over the existing fact store (`personal_fact_store.py:155-184`). It scans for
   any active (non-superseded) fact with `attr="relationship"` and
   `value == relword` whose subject is not `"i"`, and returns the most
   confidence × recency one. No per-relation table, no authored answer.
2. The relation vocabulary was hoisted to a single module-level seed set,
   `_RELATION_VOCAB` (`user_model.py:30-37`), so the miner and the recall
   resolver can never drift apart.
3. The engine's `_structured_recall` was wired to call `resolve_relation` in both
   the possessive and verb-before-noun branches
   (`engine.py:2471`, `engine.py:2499`).

## Mechanism (real code paths)

### 1. The disclosure is stored under the person's name

In `UserModel.mine_personal_facts`, a pattern matches
`my <relation> <Name> <rest>` (`user_model.py:595-605`):

```python
if _rel in _RELATION_VOCAB and _nm and _nm not in (
        "the", "a", "an", "is", "are", "was", "were", "and"):
    # store the relationship itself (subject=name entity)
    _put_fact_ent(_nm, "relationship", _rel, 0.6)
```

So "my brother dev …" stores `(dev, relationship, brother)` at confidence 0.6.
`_RELATION_VOCAB` (`user_model.py:30-37`) is the shared seed set covering
friend/sister/brother/mother/father/…/pet/dog/cat/bird/… and is RAVANA-extendable
at runtime as new disclosures arrive.

### 2. The forward index

`PersonalFactStore.resolve_relation` (`personal_fact_store.py:155-184`):

```python
def resolve_relation(self, relation: str) -> Optional[str]:
    rel = (relation or "").lower().strip()
    if not rel:
        return None
    best = None
    best_score = -1.0
    for (s, a, v), f in self.facts.items():
        if (a == "relationship" and not f.superseded
                and v == rel and s != "i"):
            sc = self._decay_score(f)
            if sc > best_score:
                best_score = sc
                best = s
    return best
```

It fails closed (returns `None`) when no active fact maps the relation word to an
entity — so callers never fabricate. If two people share a relation (two
sisters), the most confidence × recency one wins and the caller may render an
appropriate reply.

### 3. The engine recall wiring

Possessive attribute phrasing (`engine.py:2470-2485`):

```python
if _eattr in ("name", "does", "role", "is"):
    _ename = pf.resolve_relation(_ent)
    if _ename:
        for _k, _f in pf.facts.items():
            if not (isinstance(_k, tuple) and len(_k) == 3):
                continue
            if _k[0] == _ename and _k[1] in _want \
                    and not getattr(_f, "superseded", False):
                _v = _f.value
                if _eattr == "name":
                    return f"your {_ent}'s name is {_v}."
                if _eattr == "does":
                    return f"your {_ent} does {_v}."
                if _eattr == "is":
                    return f"your {_ent} is {_v}."
                return f"your {_ent}'s {_eattr} is {_v}."
```

Verb-before-noun phrasing (`engine.py:2493-2506`):

```python
_DO_ENT = re.search(
    r"\b(?:what|who)\s+(?:does|do|did)\s+my\s+([a-z][a-z]+)\s+"
    r"(do|does|did|work|for\s+work|for\s+a\s+living|do\s+for\s+a\s+"
    r"living|do\s+for\s+work|job|is|study|studies|teach|teaches)\b", q)
if _DO_ENT and pf is not None:
    _ent = _DO_ENT.group(1).lower().strip()
    _ename = pf.resolve_relation(_ent)
    if _ename:
        for _k, _f in pf.facts.items():
            if not (isinstance(_k, tuple) and len(_k) == 3):
                continue
            if _k[0] == _ename and _k[1] in ("role", "does") \
                    and not getattr(_f, "superseded", False):
                return f"your {_ent} {_f.value}."
```

The reply keeps the user's framing ("your brother …") while every content word
comes from the resolved person's structured fact. The reverse lookup ("who is dev
to me") routes through the *same* store from the opposite direction, so the two
never disagree.

### Why this is seed structure, not hardcoding

- `resolve_relation` is a structural scan over the durable store — no
  question→answer dict, no authored prose. A hardcoding audit of the feature diff
  found **zero authored reply strings**; the only added prose is f-string frames
  (`f"your {_ent} {_f.value}."`) that render only store state.
- `_RELATION_VOCAB` (`user_model.py:30-37`) is a small seed set of relationship
  words, the same class as the correction/opinion cue lists already in the file.
  It names no person and carries no reply; RAVANA extends it at runtime as new
  disclosures arrive. Removing an entry degrades gracefully. A question→answer
  dict would be banned even in a data file — this is structure, not answers.

No LLM or retraining is involved; the capability is online and incremental — a
relationship disclosed in one turn is queryable in the next.

## Verification

Covered by `tests/unit/test_round_2026_08_13T1656_entity_attr_recall.py`
(7 tests, **7 passed in 26.13s** on the round branch with `RAVANA_OFFLINE=1`):

- `test_store_resolve_relation_brother` — "my brother dev works as a paramedic"
  stores `(dev, relationship, brother)` and `resolve_relation("brother") == "dev"`.
- `test_store_resolve_relation_unknown_returns_none` — an undisclosed relation
  returns `None`; an entity stored under its own name (pet "raccoons") is not
  mistaken for a relation word.
- `test_store_resolve_relation_sister` — `resolve_relation("sister") == "meera"`.
- `test_brother_job_from_store_not_episode` — "what does my brother do for work?"
  returns a store-derived answer containing "paramedic" (not an episode), and the
  possessive synonym "what is my brother's job?" also resolves.
- `test_sister_job_from_store_not_episode` — "what is my sister job?" and the
  possessive "what's my sister's job?" both resolve to "marine biologist".
- `test_relation_word_query_returns_none_when_unstored` — no brother ever
  disclosed → `None` (honest fail-closed, not a confab).
- `test_reverse_and_forward_agree` — "who is dev to me" and "what does my brother
  do for work" reference the same stored entity with content from the store.

Six of the seven fail on the pre-feature code (no `resolve_relation`; the queries
return `None`/an episode). The surrounding suites stay green: 30
relational+recall tests, 22 dehardcode tests, and 7 fact-mining tests all passed
in the feature card's run.

## Limits

- A relation-word query resolves against the **structured fact store**. If the
  relationship was never disclosed, both branches return `None` and the engine
  abstains rather than guessing (verified by
  `test_relation_word_query_returns_none_when_unstored`).
- When several people share a relation word (two sisters), the most confidence ×
  recency one wins; the reply does not yet disambiguate "which sister" by a
  spoken qualifier. That is a deliberate simplification, not a gap in the index.
- The capability answers *biographical* relation-word questions (role/name/what
  they do). It does not yet cover *relational stance* ("do you like my brother?")
  — that path reads the stance store, not the relationship index.
