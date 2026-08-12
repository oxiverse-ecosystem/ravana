# Reverse Pet Lookup by Name

How RAVANA answers a question about a companion *by the name you actually use*
— e.g. *"my dog's a retriever called wren"* → *"who is wren to me?"* →
*"your dog is wren."*

This capability was added in round **2026-08-12T0613Z** (residual limitation
**T40**) and is verified against the live engine (`dim=64`, offline). No LLM, no
retrain, no authored reply pools — the answer is read from the durable
`PersonalFactStore` and rendered with a single short templated frame.

## What it does

1. **Stores a pet under its species slot, keyed by the name you said.** A
   disclosure like *"my dog's a retriever called wren"* or *"my cat is ember"*
   lands in the `PersonalFactStore` as `('i', '<species>_slot', '<name>')` —
   the name is the *value*, the species is the *attribute* (see
   `pet_slots.slot_for` / `base_species`).
2. **Answers a name-keyed relationship query.** *"who is wren to me?"* or
   *"what is ember to me?"* reverse-indexes the pet store by **value** (the
   name) and replies *"your dog is wren."* / *"your cat is ember."* — the
   relationship it surfaces is the species, the inverse of the existing
   species-keyed recall (*"what is my dog's name?"*).
3. **Honors a renamed pet.** After *"no, my dog is actually called briar
   now"*, the old name is **superseded**; *"who is wren to me?"* no longer
   asserts *"your dog is wren"* (it falls through honestly), and *"who is briar
   to me?"* returns *"your dog is briar."*
4. **Preserves the self/other boundary.** A third-party pet
   (*"my sister's cat is mochi"*) is stored under a non-user subject and is
   **out of scope** for *"to me"* — the query falls through instead of claiming
   someone else's pet as the user's.
5. **Generalizes to runtime-learned species.** A pet of a species RAVANA has
   never seen (*"i have an axolotl named nyx"*) is recallable by name
   (*"your axolotl is nyx."*) because the lookup reads `pet_slots` at query
   time, not a per-animal table.

## How it grew from the conversation (source citations)

### Mining — `ravana/chat/user_model.py`, `UserModel.mine_personal_facts`

The possession/name miner writes a pet through the **same `pet_slots` resolver**
the recall uses, so the key agrees by construction. Names land under subject
`"i"` (the user's own companion) with the species as the attribute:

```python
# ravana/src/ravana/chat/user_model.py  (~L1109)
_i = 1
while _pet_slots.slot_for(_species, _i) in self.personal_facts.facts:
    _i += 1
_put_fact(_pet_slots.slot_for(_species, _i), _nm, 0.6)
```

The species word is normalized to a canonical singular via `pet_slots`:
`cat_2` / `dog` keep the species in the key, multiplicity in a numeric suffix
(so a correction path can find and supersede the prior value). The seed
vocabulary (`_SPECIES_SEED`) is **extended at runtime** by `learn_species`:

```python
# ravana/src/ravana/chat/pet_slots.py  (L52 / L89 / L84)
def learn_species(word: str) -> str:   # register an unseen animal, return canon
    ...
def base_species(attr: str) -> str:     # strip the _N suffix -> "cat"
    return re.sub(r"_\d+$", "", str(attr or "").strip().lower())
def is_pet_attribute(attr: str) -> bool:  # True when an attribute is a pet slot
    return species_of(base_species(attr)) is not None
```

### Surfacing — `ravana/chat/engine.py`, `CognitiveChatEngine._structured_recall`

A new branch **`(2c) Reverse pet lookup by NAME`** (commit `f5829aa`, after the
`(2c)` block comment at `engine.py` L2844) matches a name-query and resolves it
against the live `PersonalFactStore` by **value**:

```python
# ravana/src/ravana/chat/engine.py  (~L2861)
_NAMEQ = re.search(
    r"\b(?:who|what)\s+(?:is|was|are|were)\s+([a-z][a-z'\-]{1,20})\s*"
    r"(?:to|with|for)\s+(?:me|you|us|myself)\b", q)
if _NAMEQ and pf is not None:
    _qnm = _NAMEQ.group(1).strip().lower().strip(".,!?")
    if len(_qnm) >= 2:
        from . import pet_slots as _psl
        _matched = None
        for _k, _f in pf.facts.items():
            if getattr(_f, "superseded", False):
                continue                       # (L2873) renamed -> skip stale
            if _k[0] != "i":
                continue                       # (L2881) self/other boundary
            if not _psl.is_pet_attribute(_k[1]):
                continue                       # (L2883) only pet slots
            if getattr(_f, "value", "").strip().lower().strip(".,!?") == _qnm:
                _matched = (_k[1], getattr(_f, "value", _f))
                break
        if _matched is not None:
            _sp = _psl.base_species(_matched[0])
            return f"your {_sp} is {_matched[1]}."   # (L2889-2890)
```

The reply `f"your {_sp} is {_matched[1]}."` is **state-driven**: `_sp` comes
from `pet_slots.base_species` (the live slot name), `_matched[1]` from the
stored fact value. There is no per-animal answer table and no authored reply
string — only this one grounded render. Unknown names return `None` and fall
through to the honest "who am I" self-blurb, which the caller does not mistake
for a pet relationship.

This is the inverse of the existing species-keyed recall (`(1c)` /
`engine_memory` entity scan); both directions go through `pet_slots`, so the
keys agree by construction.

## Verified behavior (live probe, dim=64, offline)

Each line below was produced by a fresh `CognitiveChatEngine(dim=64, seed=42,
baby_mode=True)` taken straight from source on this branch:

```
>> my dog's a nova scotia duck tolling retriever called wren
>> who is wren to me?              -> 'your dog is wren.'
>> my cat is a maine coon called ember
>> what is ember to me?            -> 'your cat is ember.'
>> i have a dog called wren
>> no, my dog is actually called briar now
>> who is wren to me?              -> (honest fall-through — NOT 'your dog is wren')
>> who is briar to me?             -> 'your dog is briar.'
>> i have an axolotl named nyx
>> who is nyx to me?               -> 'your axolotl is nyx.'
>> my sister's cat is mochi
>> who is mochi to me?             -> (honest fall-through — mochi is not 'to me')
```

The two fall-through cases are the desired behavior: a superseded name and a
third-party pet must **not** be claimed as the user's companion. They return
the generic self-blurb rather than a fabricated relationship.

## Tests

`tests/unit/test_round_2026_08_12T0613_pet_name_recall.py` (4 tests, all
passing; verified `4 passed in 17.76s` with
`RAVANA_OFFLINE=1 pytest tests/unit/test_round_2026_08_12T0613_pet_name_recall.py -q`):

- `test_reverse_pet_name_recall_who_is_to_me` — `"who is wren to me?"` after a
  forward contraction disclosure → `"your dog is wren"` (and not the generic
  self-blurb).
- `test_reverse_pet_name_recall_other_species` — `"what is ember to me?"` (cat)
  → `"your cat is ember"`.
- `test_reverse_pet_name_recall_tracks_rename` — after *"no, my dog is actually
  called briar"*, `who is wren to me?` must NOT say `"your dog is wren"`;
  `who is briar to me?` → `"your dog is briar"`.
- `test_reverse_pet_name_recall_runtime_learned_species` — `"i have an axolotl
  named nyx"` → `"who is nyx to me?"` → `"your axolotl is nyx"`.

**Red-capability verified:** with the `(2c)` regex assignment disabled, all 4
tests FAIL (the engine falls through to the generic self-blurb / honest miss);
with the branch present, all 4 PASS.

## Design properties

- **Seed-driven, learnable.** The capability lives in the stores + miners + a
  species vocabulary that grows at runtime (`learn_species`), not in reply
  tables. A user can correct a pet name at runtime and recall reflects it — no
  retrain, no LLM.
- **Self/other boundary enforced at the source.** Only facts stored under
  subject `"i"` are in scope for *"to me"* (L2881), so a sister's cat is never
  reported as the user's.
- **Honors active memory.** Superseded facts are skipped (L2873), so a renamed
  pet resolves to the corrected name, never the retired one.
- **Generic, not enumerated.** Any species known to `pet_slots` (seed or
  runtime-learned) resolves; there is no per-animal answer pool (hardcoding
  audit: clean — the only literal reply is the grounded
  `f"your {_sp} is {_matched[1]}."`).
