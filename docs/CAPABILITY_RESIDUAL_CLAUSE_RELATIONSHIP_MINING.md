# CAPABILITY: Clause-object relationship mining & recall

How a relationship disclosure whose verb **object is a non-noun-phrase
clause** — *"my mentor taught me how to read the hive's mood"* — is now **mined
and recalled** instead of being silently dropped. Verified against the live
engine (offline, `dim=64`) via
`tests/unit/test_round_2026_08_22T0058_residual_clause.py` (3 tests, green).

## The gap

RAVANA's relationship miner (`UserModel.mine_personal_facts`,
`ravana/src/ravana/chat/user_model.py`) handles disclosures like
*"my brother theo fixes bicycles"* (activity verb) and *"my grandmother yaya
speaks three languages"* (relation verb, enumerated). But a disclosure whose
object is a **clause** used to vanish entirely:

> *"my old beekeeping mentor, Dr. Osei, taught me how to read the hive's mood"*

The noun-phrase object path runs the opinion-topic resolver
(`self._opinion_topic`) on the object. That resolver is built to collapse a
clause to a **content head** (`"read"` / `"where"` / `"why"`) and then **reject**
that head as verb-residue (it lives in the `_OBJ_NONCONTENT` set). So `_obj`
came back **empty**, the degenerate-fact guard dropped the WHOLE disclosure, and
a later *"what did my mentor teach me?"* had nothing to recall.

This was the residual defect (DEFECT B-variant) left at the end of round
2026-08-22T0058Z.

## The fix

When the topic-resolver yields nothing, fall back to the user's **own clause
words** as the value — instead of dropping the disclosure. The raw clause is
real, informative content the user said, so it is kept verbatim
(`user_model.py:2257-2310`, inside the `else:` that follows the noun-phrase
object path).

Concrete behavior:

- *"my old beekeeping mentor, Dr. Osei, taught me how to read the hive's mood"*
  → mines `('i','mentor dr. osei') -> 'taught how to read the hive's mood'`.
- A later *"what did my mentor teach me?"* recalls it:
  *"your mentor dr. osei taught how to read the hive's mood."*

The fallback is **bounded and fail-closed**, mirroring the relation-verb path's
*"keep the user's own phrase"* philosophy:

- **Clause-boundary splitter** — the same `re.split(r"\s*(?:[.!?]+|where|that|which|when|but)\b", ...)`
  used by the relation-verb path cleanly ends the value at a trailing
  `.!?` / `where|that|which|when|but`, so a trailing relative clause does not
  swallow the sentence (`user_model.py:2289-2291`).
- **Leading possessive framer stripped** — a clause opener like `"me"` / `"us"`
  carries no content (*"my mentor taught me HOW..."*); it is dropped before the
  value is formed (`user_model.py:2295-2299`).
- **Trailing closed-class framer / preposition stripped** — same `_FALLBACK_PREP`
  list as the relation-verb path (`up|down|from|at|in|on|with|to|of|by|for|…`),
  so *"… taught me how to read the hive's mood in"* → *"… mood"* (`user_model.py:2301-2307`).
- **Length bound** — value must be `≤ 12` tokens, so a runaway clause cannot
  swallow the sentence (`user_model.py:2309`).

## Why this is not hardcoding

- Genuine **noun-phrase** objects still go through the resolver and keep the
  existing *"real concept head"* shape — the fallback only fires when the
  resolver produced **nothing** (`user_model.py:2253-2256`). Regression test
  `test_noun_phrase_object_still_resolved` guards this (e.g. *"… taught me
  astronomy"* → `astronomy`, not the raw clause).
- **Pure verb-residue** clauses (*"my mentor taught me."*) are honestly **not
  stored**: the leading framer `"me"` is stripped and nothing real remains, so
  the disclosure is skipped rather than fabricated. Regression test
  `test_pure_verb_residue_clause_not_stored` guards this.
- Content is the user's **own words**; there is **no authored reply prose, no
  per-relationship table, no LLM, and no retraining**. The fact lands in the
  **same** `PersonalFactStore` the user can later correct, so it is
  RAVANA-expandable (the seed relation/activity lexicons grow online via
  `learn_relation` / `learn_material`, no code change).

## Verification

Run against the live engine (offline, `dim=64`):

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_round_2026_08_22T0058_residual_clause.py -v
```

- `test_clause_object_relationship_mined_and_recalled` — *"… taught me how to
  read the hive's mood"* is mined under `('i','mentor dr. osei')` and recalled by
  both *"who is dr. osei to me?"* and *"what did my mentor teach me?"* (asserts
  `hive` and `taught` in the recalled answer). **RED→GREEN**: without the
  fallback the fact is never mined.
- `test_noun_phrase_object_still_resolved` — regression: a real noun-phrase
  object still resolves to its concept head (`astronomy`).
- `test_pure_verb_residue_clause_not_stored` — honesty guard: *"my mentor taught
  me."* is not stored as junk.

All 3 pass (34.3s, .venv-real / Python 3.11).

## Source pointers

| What | Where |
|------|-------|
| Relationship miner (noun-phrase object path) | `ravana/src/ravana/chat/user_model.py:2250-2256` |
| Raw-clause fallback (the fix) | `ravana/src/ravana/chat/user_model.py:2257-2310` |
| Clause-boundary splitter | `ravana/src/ravana/chat/user_model.py:2289` |
| Leading possessive-framer strip | `ravana/src/ravana/chat/user_model.py:2295-2299` |
| Fail-closed length bound (`≤ 12`) | `ravana/src/ravana/chat/user_model.py:2309` |
| Regression tests | `tests/unit/test_round_2026_08_22T0058_residual_clause.py` |
