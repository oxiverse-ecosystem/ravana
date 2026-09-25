# Capability: recall routing guards + score-based fact matching

**Status:** shipped (commit `28258eff`, branch `auto/round-2026-09-24T0559Z`).
**Feature card:** `t_f3248dc4` (round `2026-09-24T0559Z`).
**Verified:** the 3 fix-specific tests in `tests/test_round_2026_09_24_recall_routing.py`
pass (6/6, offline). Logic-only tests (regex matching, scoring math, guard predicates)
are deterministic and GloVe-independent — safe to assert on this box. Hardcoding
self-audit clean (no authored reply prose, no per-topic answer table — the fixes are
to matching/scoring/guard logic only, no new reply strings).

## What it does

When the user asks RAVANA to recall a *prior disclosure* — *"what did i tell you
i'm planning for next spring"*, *"did i tell you i liked japan"* — three bugs
in the recall routing path could return the WRONG fact or misroute the query
entirely:

| Bug | Symptom | Fix |
|-----|---------|-----|
| **1. _match_fact index confusion** | Single-token ties returned the wrong fact (the partner/moving fact beat planning/trip because `_overlap > _best[2]` compared confidence instead of overlap). | Score = `overlap + conf*0.1 + attr_match*0.5`; compare `_score > _best[3]`. |
| **2. _B confirmation regex stole recall queries** | "what did i tell you i liked X" fired the confirmation path → returned a yes/no echo instead of the recalled fact. | Guard: skip `_B` when query starts with `what`. |
| **3. _TOLD regex too restrictive** | Required "about" after tell-verbs; rejected "just" adverb and apostrophes. "what did i tell you i'm planning" missed entirely. | Make `about` optional; allow `(?:just\s+)?`; add apostrophe to capture group. |

All three fixes are **logic-only** — they change how RAVANA *finds* a fact, never
what it *writes* about one. The recalled content still comes entirely from the live
`PersonalFactStore` / `UserStanceStore`.

Real engine output (fresh engine, seeded with *"i am planning a trip to japan next
spring"* + *"my partner and i are thinking about moving in together next year"*, then
queried):

```
Q: what did i tell you i am planning for next spring
A (BEFORE): yes — i remember: your partner and i is thinking about moving in together next year.  [WRONG]
A (AFTER):  yes — i remember: you told me about your trip.                         [CORRECT]
```

## Fail-closed

- `_match_fact` scoring: ties still return *a* fact (the highest-scored one);
  only the tiebreaker changed. When no fact overlaps, returns `None`.
- `_B` guard: a `what`-prefixed query falls through to the normal recall pipeline
  (`_TOLD`), so no query is dropped — it just reaches the right handler.
- `_TOLD` regex: the generalized pattern is a superset of the old one, so every
  query the old regex matched still matches (verified in test cases).

## How it grew from the conversation

The chat round of this cycle (`t_2734fc75`, report
`tmp/reports/ravana-2026-09-24T0559Z.md`) surfaced three recall-routing failures
when RAVANA was probed with natural recall phrasings:

- **Bug 1** — "what did i tell you i am planning for next spring" returned the
  *partner/moving* fact instead of the *planning/trip* fact. Root cause: the
  overlap comparator read `_best[2]` (confidence) when it should have read
  overlap. The fix adds a composite score (overlap + confidence weight + attribute
  bonus) and compares against `_best[3]`.

- **Bug 2** — "what did i tell you i liked japan" fired the (B) confirmation
  path. The `_B` regex `\\b(did|have|had)\\s+(i|you)\\s+(tell|told...)\\b`
  matched inside `what did i tell you...`. Fix: gate `_B` behind `not
  re.match(r'^what\b', q)` so content-request recall queries skip confirmation.

- **Bug 3** — "what did i tell you i'm planning" missed entirely because the
  old regex required `about` after the tell-verb. Fix: make `about` optional,
  allow a `just` adverb, add apostrophe to the character class.

### Root cause — index confusion in a tuple comparator

`_match_fact` (`engine.py:4907`) tracks the best candidate as a 4-tuple
`(_attr, _val, _conf, _overlap)`. The comparator read `_best[2]` (which is
_confidence_) and compared it against the raw overlap count. On a single-token
tie, a lower-overlap fact with higher confidence won — even though the
higher-overlap fact was the one the user was clearly asking about.

The fix replaces the raw-overlap comparator with a composite score:

```python
_score = _overlap + _conf * 0.1
if _attr_l and any(t in _attr_l for t in _ptoks):
    _score += 0.5
if _best is None or _score > _best[3]:
    _best = (_attr, _val, _conf, _score)
```

`overlap` dominates; `confidence` is a 0.1-weight tiebreaker; `attr_match`
adds 0.5 when the query topic appears in the stored attribute itself (so
"planning trip" matches attr=`planning` even when the value is "planning trip"
and the overlap is already 2).

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_match_fact` score-based selection | `ravana/src/ravana/chat/engine.py:4922-4928` |
| `_B` confirmation guard (what-prefix skip) | `ravana/src/ravana/chat/engine.py:5064-5067` |
| `_TOLD` regex generalization (optional about, just, apostrophe) | `ravana/src/ravana/chat/engine.py:3118-3131` |
| Parent dispatcher `_structured_recall` (calls both) | `ravana/src/ravana/chat/engine.py:~3140+` |

## Hardcoding audit (summary)

Every change is matching/scoring/guard logic — **no authored reply prose, no
`random.choice` reply pools, no keyword→response tables, no Q→A dict**:

- `_match_fact` scoring: the score is computed from real `_overlap`, `_conf`,
  and `_attr_l` values — no hand-written threshold, no per-topic table.
- `_B` guard: a single `re.match(r'^what\b')` prefix check — structural,
  not content-specific.
- `_TOLD` regex: grammatical generalizations (optional word, added adverb,
  character class addition) — not per-phrase branches.

**Seed-vs-hardcoding:** no seed vocabulary here at all — the answers are still
entirely derived from runtime-grown stores. **No retraining:** all changes are
online/incremental.

## Test coverage

Six round tests in `tests/test_round_2026_09_24_recall_routing.py` (all pass;
real run: `6 passed in ~0.01s`, offline). They test the FIX LOGIC directly
(regex patterns, scoring math, guard predicates) — deterministic and
GloVe-independent, so they are safe to assert on this box (no routing ambiguity):

- `test_match_fact_score_selects_higher_overlap` — score-based selection picks
  `planning/trip` (score 2.58) over `thinking/moving` (score 0.0).
- `test_match_fact_attr_match_bonus` — attribute-match bonus (0.5) applied when
  the query topic appears in the stored attribute.
- `test_B_guard_skips_what_prefixed` — `_B` stays `None` for `what did i tell
  you...` queries.
- `test_B_guard_fires_on_confirmation` — `_B` fires for genuine confirmation
  (`did i tell you...`).
- `test_TOLD_optional_about_and_just` — the generalized regex matches phrasings
  the old one missed (no "about", "just" adverb, apostrophe).
- `test_TOLD_regression_still_matches_about` — queries with "about" still match.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_round_2026_09_24_recall_routing.py -v
```
