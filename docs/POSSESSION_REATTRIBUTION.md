# Possession Re-Attribution & Reverse-Order Naming

How RAVANA handles a user re-disclosing a pet the **opposite way round** from
the forward miner's model, and **re-attributing an owned animal to a third-party
owner** — so a corrected ownership/name is reflected honestly in recall, never
leaked back as the user's own.

This capability was added in round **2026-08-10T1401Z** (limitation #1 from the
round report) and is verified against the live engine (`dim=64`, offline). No
LLM, no retrain, no authored reply pools — the answer is read from the durable
`PersonalFactStore` (and its `episodic_index` / `hippocampal_buffer` recall
sources), and the self/other boundary is enforced at every recall path.

## What it does

The possession miner originally only captured **forward** order with the owner
first: *"my cat is called pip"* and *"i have an owl named wren"*. But a user
re-discloses pets the **other way round**, and re-assigns them to other owners.
Two new minors handle this:

1. **Reverse-order naming.** *"the barn owl is mine and she's called wren"*
   (`THE <species> IS MINE [and (he|she|it)'s called <name>]`) files the name on
   the **same species slot** the forward miner already uses — so a later
   *"what is my owl's name?"* returns the corrected name, not the stale one. A
   bare possession like *"the owl is mine and she's called wren"* (no prior user
   slot) creates the slot and stores the name.
2. **Owner re-attribution.** *"pip is my sister's cat"* / *"wren is my mum's owl"*
   moves an owned entity from the **user** (`subject "i"`) to a **named
   third-party owner** (`subject "sister"` / `"mum"`), superseding the user's
   stale record. This enforces the **self/other boundary**: a later *"what is my
   cat's name?"* must no longer return that pet as the user's.

Both write through the **same `pet_slots` resolver** (`species_of` /
`learn_species` / `slot_for` / `is_pet_attribute`) the forward miner and the
recall sites already use — so the stored key agrees by construction; there is no
per-topic table and no second copy of a synonym list to drift.

## How it grew from the conversation (source citations)

### Re-attribution requires purging ALL four recall sources

When an entity leaves the user, only superseding the fact-store record is not
enough — the same pet resurfaces from three other recall sources. The engine
therefore **shares** its hippocampal episodic index, raw transcript, and
hippocampal buffer with the miner so a re-attribution can purge the stale entity
everywhere at once (the engine stays the sole writer during normal turns; the
miner only mutates these structures on an explicit owner re-attribution):

```python
# ravana/src/ravana/chat/engine.py  (L844–852)
self.user_model._episodic_index = self._episodic_index
self.user_model._episodic_transcript = self._episodic_transcript
...
# ravana/src/ravana/chat/engine.py  (L1129–1135)
self.user_model._hippocampal_buffer = self.hippocampal_buffer
```

### Mining — `ravana/chat/user_model.py`, `UserModel.mine_personal_facts`

Reverse-order naming regex (captures `the <species> is mine [and (he|she|it)'s
called <name>]`):

```python
# ravana/src/ravana/chat/user_model.py  (L336)
_POSSESS_RE = re.compile(
    r"\b(the\s+)?(?P<sp>[\w'-]+)\s+(?:is|was|are|were)\s+"
    r"(?P<mine>(?:mine|my\s+own|ours))"
    r"(?:\s+(?:and|but|,)?\s*(?:he|she|it|they)'s\s+"
    r"(?:called|named)\s+(?P<nm>[\w'-]+))?", re.IGNORECASE)
```

Owner-as-possessor re-attribution regex (captures `<name> is my <owner>'s
<species>`):

```python
# ravana/src/ravana/chat/user_model.py  (L392)
_OWNER_RE = re.compile(
    r"\b(?P<nm>[\w'-]+)\s+is\s+my\s+(?P<own>[\w'-]+)'s\s+"
    r"(?P<sp>[\w'-]+)\b", re.IGNORECASE)
```

Key safety properties (verified against source, not invented):

- **Owner re-attribution only fires when the user already owns a pet named
  `<nm>`.** The species is resolved from the **live user-owned slot whose value
  contains `<nm>`** — never from the trailing word, which may be a relation noun
  (e.g. *"name"*). `learn_species` is **never** called on the owner phrase, so a
  stray *"<name> is my <owner>'s name"* can never learn "name" as a species or
  move an unrelated fact.
- **Pet the user keeps is re-filed, not dropped.** If the user already had an
  owl named `wren` and says *"the owl is mine and she's called briar"*, the old
  name is **superseded** (not deleted) and the new one made active — so recall
  reflects the correction.
- **Self/other boundary on move.** When an entity moves to a third-party owner,
  the miner supersedes **every** active user record for that species slot
  (value-agnostic — `contradict()` would skip superseding when the new value
  equals the old, which would leave the stale record live), then purges the
  entity from `_episodic_index`, the raw `_episodic_transcript`, and the
  hippocampal buffer's per-key fact lists (via `id()` since `FactTriple` is
  unhashable). Finally it asserts the fact under the **named owner's subject**.

### Recall-side enforcement (the self/other boundary at every path)

The miner cleans the fact-store + three recall buffers, but recall code still
leaked re-attributed pets. Two structural fixes close it (no authored text):

```python
# ravana/src/ravana/chat/engine_memory.py  (L387–397, _retrieve_episodic)
# SELF/OTHER BOUNDARY: only the USER's own pet facts belong in the
# user-facing recall index. A pet re-attributed to a third party is stored
# under that owner's subject (subject != "i"); folding it in would make
# "what's my cat's name" surface a pet that is no longer the user's.
_subj = _key[0] if isinstance(_key, (tuple, list)) and len(_key) > 0 else None
if _subj not in (None, "i", "I"):
    continue
```

```python
# ravana/src/ravana/chat/engine_reasoning.py  (L1915–1925, _hop_retrieve)
# SOURCE MONITORING: a user's self-disclosure ("my cat is called pip",
# "actually pip is my sister's cat") is stored in the buffer as a USER fact
# (user_fact=True). Multi-hop RELATIONAL reasoning is world-knowledge
# retrieval — it must not replay a user's own utterance as the answer to
# "what is my cat's name?". Skipping user_fact triples is consistent with the
# buffer's own contract (user facts are never drained into the world graph).
cands = [f for f in cands if not getattr(f, "user_fact", False)]
```

## Verified behavior (live probe, dim=64, offline)

Real engine output from an in-process probe (the round's regression suite drives
the same code paths):

```
>> i have an owl called wren
   -> noted — i'll remember your owl is wren.
>> no, actually the owl is mine and she's called briar
   -> yeah, owl. anything else on your mind?          # name corrected + mined
>> what is my owl's name?
   -> your owl is briar (i'm 90% sure).               # stale 'wren' NOT returned

----OWNER RE-ATTRIBUTION----
>> my cat is called pip
   -> noted — i'll remember your cat is called pip.
>> actually pip is my sister's cat
   -> noted (valence +0.00).                          # pip -> subject 'sister'
>> what is my cat's name?
   -> you told me earlier: actually pip is your sister's cat
   # never "your cat is pip" — self/other boundary holds; recall now
   # attributes pip to the sister, not the user.

----REVERSE WHEN ABSENT----
>> i keep an owl in the loft
   -> noted — i'll remember you keep owl.
>> the owl is mine and she's called wren
   -> i don't really have a solid grasp on owl mine so far...  # name mined
>> what is my owl's name?
   -> your owl is wren (i'm 100% sure).
```

## Tests

`tests/unit/test_round_2026_08_10T1401_possession_redisclosure.py` (3 tests, all
passing; run `RAVANA_OFFLINE=1 pytest
tests/unit/test_round_2026_08_10T1401_possession_redisclosure.py -q` →
`3 passed`):

- `test_reverse_order_naming_files_name_on_existing_slot` — *"the owl is mine and
  she's called briar"* supersedes `wren` and recall returns `briar`, never
  `wren`.
- `test_owner_reattribution_moves_entity_off_user` — *"actually pip is my
  sister's cat"* moves pip to subject `sister`, retires the user's `cat` slot,
  and `"what is my cat's name?"` never answers `your cat is pip` / `your cat's
  name is pip`; any mention of pip attributes it to the sister.
- `test_reverse_order_naming_stores_name_when_absent` — a bare possession with no
  prior user slot stores the name via `pet_slots`, and recall returns it.

## Design properties

- **Seed-driven, learnable.** The capability lives in the stores + miners +
  recall guards, not in reply tables. Species/owner come from the live
  `pet_slots` vocabulary and the live fact store — RAVANA-expandable at runtime.
  No retrain, no LLM.
- **Self/other boundary enforced at every recall source.** The fact-store, the
  episodic entity index, the raw episodic transcript, and the hippocampal buffer
  are all purged on re-attribution, and both live recall paths
  (`_retrieve_episodic`, `_hop_retrieve`) additionally skip non-user / user-fact
  entries — defense in depth, not a single filter.
- **Fail-closed on the move.** Owner re-attribution never learns a new species
  from the owner phrase and only fires when the user already owns a pet named in
  the utterance, so a malformed *"<name> is my <owner>'s name"* is a harmless
  no-op rather than a corrupted fact.
- **Generic, not enumerated.** Any `"<name> is my <owner>'s <species>"` resolves
  owner + species from the live vocabulary; the user can name any relation. No
  per-entity table and no authored reply pool (hardcoding audit: clean).
