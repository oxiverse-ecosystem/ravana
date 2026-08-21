# Capability: aux-verb relationship mining + copula-free recall (does/did + activity noun-phrase)

**Status:** shipped (commit `c9b2644`, branch `auto/round-2026-08-21T0843Z`, feature `t_16b15684`).
**Verified:** `tests/unit/test_relation_reverse_lookup.py::test_aux_verb_relationship_mined_and_recalled` PASSES on shipped code (17.13s, ran live this cycle). Live logic traced from `user_model._mine_relationship` / `is_verb_phrase`. Hardcoding self-audit clean: additions are a seed lexicon (`_AUX_VERB_LEXICON`) + a structural miner branch driven by the user's own words; zero authored reply strings, no Q→A dict, no per-relationship table, no retraining.

## What it does

A relationship disclosure that states the relative's activity through an **auxiliary verb** — *"my cousin Jin DOES competitive speedcubing"*, *"my sister DID competitive debate"*, *"my brother DOES parkour"* — is now **mined AND recalled** like any other relationship fact.

- **Mined** as the combined-attr fact `('i','cousin jin') -> 'does competitive speedcubing'` (the auxiliary + the activity noun-phrase the user actually said).
- **Recalled** grammatically and copula-free: *"what does my cousin jin do"* → *"your cousin jin does competitive speedcubing."* — **not** *"is does competitive speedcubing"* and **not** the prior failure *"cousin is a bit outside what i know right now"*.

This is the **third** verb class in the relationship miner, after **activity verbs** (`climbs`, `fixes`, `weaves` — round 2026-08-19T1026Z generalized) and **relation verbs** (`speaks`, `works`, `studies` — feature `t_ec6c6b51`, round 2026-08-20T1935Z residual). All three open the *same* capture path, so a disclosure is no longer dropped just because its verb isn't in one specific lexicon.

The recall grammar rule that drops the copula for verb-phrase values is `is_verb_phrase` (`user_model.py:207`): it now returns `is_activity_verb(...) or is_relation_verb(...) or is_aux_verb(...)` (`user_model.py:215`). Every recall/ack render site imports `is_verb_phrase` for the copula decision (e.g. `engine.py:2893`, `:2948`, `:3060`, `:3103`, `:3340`, `:3809`, `:4285`), so the aux class renders correctly everywhere the activity/relation classes already did — no per-site change was needed.

No LLM, no Q→A dict, no keyword→reply table. The capability is a structural branch in the live miner + a vocabulary predicate; RAVANA's relationship facts are recomputed from the real utterance every turn, satisfying the seed + online-learning constraints.

## How it grew from the conversation

The chat round of round `2026-08-21T0843Z` logged a residual limitation: *"my cousin Jin does competitive speedcubing"* was **dropped** — the miner's verb-scan only knew activity verbs and relation verbs, neither of which matches *"does"*, so the scan found no verb-phrase head, fell through to the **name-only path**, and the degenerate-fact guard skipped the whole disclosure. A later *"what does my cousin jin do"* then returned the honest-but-wrong *"cousin is a bit outside what i know right now"* — the disclosure had never been stored.

**Root cause / prior behavior.** The relationship miner (`user_model._mine_relationship`, the block keyed on a recognized kin/role head word at `user_model.py:1922`) scanned the remainder for a verb head via two predicates: `is_activity_verb` (`user_model.py:1929`) and `is_relation_verb` (`user_model.py:1944`). *"does"* is in neither set, so `_vidx` stayed `None` and the path collapsed to the leading-capitalized-name-only storer (`user_model.py:1982-1995`) — which produced no informative fact for an activity disclosure, so the degenerate-fact guard dropped it. The defect is a **lexicon gap**, not a seed: the two prior verb classes simply hadn't covered the auxiliary shape.

**Fix (commit `c9b2644`).** A third recognize-branch was added to the verb scan (`user_model.py:1948-1970`):

- `_AUX_VERB_LEXICON = {"do", "does", "did", "doing", "done"}` (`user_model.py:187`) — seed vocabulary, the same shape as the activity/relation lexicons. It is **RAVANA-expandable**: `is_aux_verb` (`user_model.py:190`) also recognizes inflected forms by stripping a suffix (`ing`/`ed`/`s`/`es`) back to a lexicon entry, so *"doing"* / *"did"* / *"does"* all resolve. This is a data set, not an answer table — removing entries degrades gracefully (one fewer aux shape recognized).
- `is_aux_verb(_tw)` (`user_model.py:1965`) opens the **same** capture path as the other two verb classes: `name = tokens before it` and `value = aux + activity noun-phrase` resolved through `_opinion_topic` (the shared opinion-object resolver, so the object stays a real concept, not a filler). Crucially, the auxiliary is allowed to fire on its own token **as long as a following content token exists** (`user_model.py:1966-1969`) — because the disclosure is *"does competitive speedcubing"*, not *"does climb"*; we do **not** require a following activity *verb*.
- `is_verb_phrase` (`user_model.py:207-215`) now includes `is_aux_verb`, so recall renders the value copula-free (*"your cousin jin does competitive speedcubing"*, not *"is does"*) — the same grammar rule already used for activity/relation verb facts, now generalized to all three.

**Generalization.** The branch is verb-class-agnostic once the head is recognized, so it generalizes to any aux+activity disclosure the user rotates in: *"my sister did competitive debate"*, *"my brother does parkour"*, and — because the name need not be capitalized (the name-token extraction at `user_model.py:1997` lowercases whatever precedes the verb) — even *"my sister did X"* with a lowercase name mines and recalls the same way. There is **no per-relationship branch** and no retraining.

**Hardcoding audit.** The diff adds zero reply strings. Grep for long added strings in the changed region returns only the docstring/comment blocks explaining the lexicon and the scan branch. The added surface is: a small seed set + a vocabulary predicate (`is_aux_verb`) + one structural `if` that routes an aux head into the existing capture path. There is no Q→A dictionary and no keyword→reply table. Seed test (per doctrine): the aux shape is morphological and generalizes to any aux+activity disclosure; removing the lexicon only re-admits the dropped shape. The capability is the structural fix the residual demanded — not a scripted answer to "what does my cousin do".

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_AUX_VERB_LEXICON` (seed aux vocabulary) | `ravana/src/ravana/chat/user_model.py:187` |
| `is_aux_verb(word)` (predicate) | `ravana/src/ravana/chat/user_model.py:190-205` |
| `is_verb_phrase(word)` (copula-drop source of truth, now superset of 3 verb classes) | `ravana/src/ravana/chat/user_model.py:207-215` (return at `:215`) |
| Aux-verb recognize-branch in the relationship miner | `ravana/src/ravana/chat/user_model.py:1948-1970` (block opens at `:1948`; `if is_aux_verb(_tw):` at `:1965`; following-content check at `:1966-1969`) |
| Name extraction feeding the aux path (lowercases the name) | `ravana/src/ravana/chat/user_model.py:1997-1998` |
| Recall/ack copula-drop render sites (import `is_verb_phrase`) | `ravana/src/ravana/chat/engine.py:2893, 2948, 3060, 3103, 3340, 3809, 4285` |
| Regression + capability test | `tests/unit/test_relation_reverse_lookup.py::test_aux_verb_relationship_mined_and_recalled` |

## Test coverage

`tests/unit/test_relation_reverse_lookup.py::test_aux_verb_relationship_mined_and_recalled` (verified passing this cycle, 17.13s):

- Mines the aux disclosure *"my cousin Jin does competitive speedcubing and once solved it in 9 seconds"* and asserts the combined-attr fact `('cousin jin', 'does competitive speedcubing')` is present in `personal_facts` (the value renders the aux + activity noun-phrase, not the whole trailing clause).
- Recalls via *"what the fast thing with cubes is my cousin jin into"* phrasing and asserts: the strategy is `structured_recall`; the reply contains `speedcub` (the real mined activity, not a filler); and the reply does **not** contain the prior failure text *"cousin is a bit outside"*.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_relation_reverse_lookup.py::test_aux_verb_relationship_mined_and_recalled -v
```

The broader relation+fact suites stay green (78 passed / 1 skipped) and the recall/generation suites stay green (100 passed), 0 regressions — cited from the feature card's `test_result` field (`c9b2644` metadata). The test is fast (well within CI budget) and exercises the real miner → store → recall path, so it fails on pre-fix code and passes after the fix (RED→GREEN, as asserted by the feature card's `hardcoding_audit` / `verified_live` fields).
