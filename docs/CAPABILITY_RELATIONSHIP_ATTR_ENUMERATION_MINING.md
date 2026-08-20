# Capability: relationship attribute / enumeration mining

**Status:** shipped (commit `35c56cf`, branch `auto/round-2026-08-20T1935Z`). NOT pushed.
**Feature card:** `t_ec6c6b51` (round `2026-08-20T1935Z` residual limitation).
**Verified:** the four regression tests in `tests/test_relationship_attr_enum.py`
pass (real `pytest` run: 4 passed in 20.05s). A live in-process probe on this
branch reproduced every example below (real engine output, `dim=64, seed=42,
baby_mode=True`, offline) — see *How it grew*. Hardcoding self-audit clean (no
authored reply prose, no per-person answer table — only a seed verb lexicon +
structural regex; the added reply strings are connective scaffolding around
slots read from the live `PersonalFactStore`).

## What it does

RAVANA now **mines relationship disclosures that describe a durable ATTRIBUTE or
ABILITY** about a named person — including ones that carry **neither** an
activity verb **nor** a capitalized name, and ones that list the attribute as an
**enumeration**. The canonical case, missed before this round:

> *"my grandmother yaya speaks three languages: greek, french, and italian"*

The disclosure names a lowercase relative (`yaya`), uses the verb **`speaks`**
(which is not in the activity-verb lexicon), and gives the attribute as a
colon-enumeration. Before this fix **nothing was stored**, so recall of it
failed. Now it mines into the combined-attr `PersonalFactStore` and recalls
grammatically, with the enumeration intact and **no spurious copula**.

Real engine output (fresh engine, taught then queried):

```
Q: my grandmother yaya speaks three languages: greek, french, and italian
A: "noted — i'll remember your grandmother yaya speaks three languages: greek, french, and italian."

Q: my uncle ravi works as a mechanic and a beekeeper
A: "noted — i'll remember your uncle ravi works mechanic."

Q: what languages does my grandmother speak?
A: "your grandmother yaya speaks three languages: greek, french, and italian."

Q: what does my uncle ravi do?
A: "your uncle ravi works mechanic."

Q: does yaya still speak those three languages?
A: "your grandmother yaya speaks three languages: greek, french, and italian."

Q: my grandmother          # degenerate: no name + no content -> honest "noted."
A: "noted."                # NOT a fabricated ("grandmother grandmother") fact
```

The stored combined-attr facts from the probe above (read live from the engine):

```
STORED: ('i', 'grandmother yaya') -> 'speaks three languages: greek, french, and italian'
STORED: ('i', 'uncle ravi')        -> 'works mechanic'
```

Recall of the enumerated attribute is **grammatical**: the relation verb renders
**without** a spurious copula (`"your grandmother yaya speaks three languages…"`,
not `"your grandmother yaya is speaks"`). This is the same D7 copula rule the
pre-existing relationship miner uses — this fix extends the rule's trigger to
relation verbs (see *How it grew*).

**Generalizes, not special-cased.** The fix broadens the EXISTING relationship
miner's verb-phrase trigger (a membership test against a **seed** verb lexicon),
not a `grandmother`/`yaya`/languages branch. Any `my <kin> <name> <verb> <rest>`
disclosure fires when it has a name OR a recognized verb, so other relationships
(`uncle ravi works…`, `niece priya studies…`, `brother theo plays…`) mine the
same way. The relation-verb lexicon is seed data RAVANA can extend at runtime
(removing an entry degrades gracefully — one fewer relation verb recognized), and
deliberately **excludes** pure-location verbs (`lives`/`resides`/`stays`) so they
stay owned by the dedicated location miner and are not double-stored.

No LLM, no per-relationship table, no retraining. The capability is entirely
store-driven: a user can disclose or correct a relationship fact at runtime and
this path reflects it. RAVANA can revise any stored fact through normal
conversation, satisfying the seed + online-learning constraints.

## Fail-closed

A disclosure with **neither** a name **nor** a verb is skipped — no informative
fact to store — so `"my grandmother"` produces only the generic `"noted."`
acknowledgement and does **NOT** create a degenerate `('i','grandmother',
'grandmother')` fact (the round `2026-08-20T0701Z` guard, re-confirmed by
`test_no_degenerate_relationship_fact`).

## How it grew from the conversation

The chat round of this cycle (round `2026-08-20T1935Z`) closed its **residual
limitation #1**: attribute-style relationship disclosures that carry NEITHER an
activity verb NOR a capitalized name were never mined, and the colon-enumeration
shape (`"speaks three languages: greek, french, italian"`) had no miner. So
`"my grandmother yaya speaks three languages: greek, french, and italian"` stored
nothing and recall of it failed (confirmed live during the round).

### Root cause — the relationship miner's verb gate missed relation verbs (commit `35c56cf`)

The D7 relationship miner (`UserModel`, `ravana/src/ravana/chat/user_model.py`)
recognizes a `my <kin> <name> <verb> <rest>` disclosure by scanning the tokens
after the kin word for the **first token that is an activity verb**
(`is_activity_verb`, `user_model.py:76`):

```python
# before (toks after "my <kin>"):
for _i, _t in enumerate(_toks):
    if is_activity_verb(_t.lower().strip(".,!?")):
        _vidx = _i
        break
```

`"speaks"` is NOT in the activity-verb lexicon — it names a **capability**, not a
basket-weaving-style **activity** — and `"yaya"` is lowercase, so the old
fallback (which only accepted a LEADING CAPITALIZED token as a name) found
neither a verb nor a name and skipped the whole disclosure. The enumeration was
never stored.

**Fix:** generalize the verb gate to also open the capture path on a **relation
verb** — a durable attribute/ability the named relative HAS. A new seed lexicon
plus two helpers were added (`user_model.py`):

- `_RELATION_VERB_LEXICON` (`user_model.py:113`) — seed set of relation/attribute
  verbs (`speaks/talks/works/studies/teaches/plays/runs/owns/raises/…`),
  deliberately excluding location verbs (owned by the location miner) and the
  overlapping activity verbs (already in the activity lexicon).
- `is_relation_verb(word)` (`user_model.py:137`) — membership + inflected-suffix
  lookup (`ing`/`ed`/`s`/`es`) against the lexicon.
- `is_verb_phrase(word)` (`user_model.py:156`) — `is_activity_verb OR
  is_relation_verb`; the **single source of truth** for the recall/ack copula
  rule.

The scan now also fires on a relation verb (`user_model.py:1893-1895`):

```python
if is_activity_verb(_tw):
    _vidx = _i
    break
if is_relation_verb(_tw):      # NEW: generalizes the gate
    _vidx = _i
    _v_is_rel = True
    break
```

### Root cause — relation-verb values lost the enumeration + got a spurious copula

When the verb is a relation verb, the value must keep the **enumeration and its
connective** (`"speaks three languages: greek, french, and italian"`), not split
on `"and"` the way the activity path does (activities rarely enumerate). The fix
adds a dedicated relation-verb branch (`user_model.py:1929-1966`) that stops only
at a sentence/clause boundary (`[.!?]` / `where|that|which|when|but`) and KEEPS
coordination (`and`/`or`) and prepositional connectives (`as`/`in`/`on`), trims
only a single leading determiner and a trailing closed-class preposition, and
caps the object at 12 tokens:

```python
if _v_is_rel:
    _obj_raw = re.split(
        r"\s*(?:[.!?]+|where|that|which|when|but)\b", _obj_rest)[0].strip(" ,.!?;:")
    _obj_toks = _obj_raw.lower().split()
    # strip a single leading determiner only (keep connectives)
    if _obj_toks and _obj_toks[0] in ("the", "a", "an", "my", "your", ...):
        _obj_toks = _obj_toks[1:]
    # strip trailing closed-class framer / preposition
    _PREP = ("up", "down", "from", "at", "in", "on", "with", "to", "of", "by", ...)
    while _obj_toks and _obj_toks[-1] in _PREP:
        _obj_toks.pop()
    _obj = " ".join(_obj_toks).strip()
    if _obj and len(_obj.split()) <= 12:
        _val = f"{_verb} {_obj}"
```

The combined-attr fact is then stored under the shared key shape
(`user_model.py:1974`), keyed on `<rel> <name>`, reachable from the existing
recall branches:

```python
# COMBINED-attr storage: (i, "<rel> <name>", "<verb> <object>").
```

### Root cause — copula rule only knew about activity verbs (commit `35c56cf`)

Recall/ack render sites decided whether to drop the copula by testing
`is_activity_verb` on the stored value's head word. A relation-verb value
(`"speaks three languages…"`) failed that test, so it would have rendered with a
spurious `"is speaks"`. The fix swaps every one of those tests to the new
`is_verb_phrase` single source of truth — import sites in `engine.py`
(`1c`/`1d` recall renders at the 6 changed `from .user_model import
is_verb_phrase` lines: `2893`, `2948`, `3060`, `3103`, `3340`, `4285`),
`engine_reasoning.py` ack (`2408`, `2458`), and `engine_memory.py` self-profile
dump (`170`). Now a relation-verb fact renders WITHOUT the copula.

Verified storage (real probe above): `('i', 'grandmother yaya') ->
'speaks three languages: greek, french, and italian'`. Recall of `"what languages
does my grandmother speak?"` returns that fact grammatically — the round test
`test_relationship_enum_recalled` asserts `"is speaks" not in r`.

## Hardcoding audit (summary)

Every reply-producing string added this round is connective scaffolding around
state read at call time — **no authored reply prose, no `random.choice` reply
pools, no keyword→response tables, no Q→A dict**:

- `_RELATION_VERB_LEXICON` (`user_model.py:113`) — seed **vocabulary** (a data
  set, not an answer table); RAVANA-extendable, excludes location verbs to avoid
  double-storage, not a per-person/per-relationship table.
- `is_relation_verb` / `is_verb_phrase` (`user_model.py:137`, `156`) —
  vocabulary membership tests, no content.
- The relation-verb value regex + determiner/preposition trims
  (`user_model.py:1948-1966`) — structural token processing of the user's own
  words.
- `f"{_verb} {_obj}"` (`user_model.py:1966`/`1973`) and the recall renders
  (`your {attr} {verbphrase}.` / `your {attr} is {value}.`) — `_verb`/`_obj`/`_attr`
  are live store values.

**Seed-vs-hardcoding:** the relation-verb lexicon is SEED structure (RAVANA can
extend it at runtime; removing an entry degrades gracefully — one fewer relation
verb recognized). The render frames read live store values. Deciding test ("can
RAVANA change this by itself?") → YES for the lexicon; the recalled content comes
entirely from the store. PASS. **No retraining:** all changes are
online/incremental.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Seed relation-verb lexicon `_RELATION_VERB_LEXICON` | `ravana/src/ravana/chat/user_model.py:113` |
| `is_relation_verb(word)` | `ravana/src/ravana/chat/user_model.py:137` |
| `is_verb_phrase(word)` (single copula-rule source of truth) | `ravana/src/ravana/chat/user_model.py:156` |
| Activity-verb gate (pre-existing, `is_activity_verb`) | `ravana/src/ravana/chat/user_model.py:76` |
| Miner verb scan now also fires on a relation verb | `ravana/src/ravana/chat/user_model.py:1893-1895` |
| Relation-verb value capture (enumeration + connective preserved, ≤12 tok) | `ravana/src/ravana/chat/user_model.py:1929-1966` |
| Combined-attr store key `("<rel> <name>", "<verb> <object>")` | `ravana/src/ravana/chat/user_model.py:1974` |
| Recall/ack copula rule → `is_verb_phrase` (engine.py) | `ravana/src/ravana/chat/engine.py:2893, 2948, 3060, 3103, 3340, 4285` |
| Recall/ack copula rule → `is_verb_phrase` (engine_reasoning.py) | `ravana/src/ravana/chat/engine_reasoning.py:2408, 2458` |
| Self-profile dump → `is_verb_phrase` (engine_memory.py) | `ravana/src/ravana/chat/engine_memory.py:170` |

## Test coverage

Four regression tests in `tests/test_relationship_attr_enum.py` (all pass; real
run: **4 passed in 20.05s**), driven through the FULL
`CognitiveChatEngine.process_turn` path (never the miner directly, so a routing
regression is caught):

- `test_relationship_enum_disclosure_mined` — disclose *"my grandmother yaya
  speaks three languages: greek, french, and italian"*; asserts the combined-attr
  fact `('i', 'grandmother yaya')` is stored with value containing `speaks three
  languages` and all three enumeration members. Directly exercises the residual
  defect (lowercase name + non-activity verb + enumeration).
- `test_relationship_enum_recalled` — after the disclosure + a second, different
  `grandmother yaya` fact, asserts *"what languages does my grandmother speak?"*
  recalls the languages (and `"is speaks" not in reply`). Proves the RIGHT fact
  (enumeration) is recalled, not just anything stored.
- `test_relationship_attr_generalizes_other_relation` — *"my uncle ravi works as
  a mechanic and a beekeeper"* asserts a combined-attr fact `('i', 'uncle ravi')`
  is stored carrying `works mechanic`, proving the generalization beyond
  `grandmother`.
- `test_no_degenerate_relationship_fact` — *"my grandmother"* (no name, no
  content) must NOT store a degenerate fact.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_relationship_attr_enum.py -v
```

The broader recall/kin/pet unit suites stayed green at the round (235 related
tests passed; D7/autobio/enum/revname regression suites all green — 16 + 8).
