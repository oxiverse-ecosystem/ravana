# CAPABILITY: Degenerate-head skip in `_opinion_topic` (D5 residual)

How the opinion-topic resolver (`UserModel._opinion_topic`,
`ravana/src/ravana/chat/user_model.py`) now skips a stop-word boundary when
the head collected so far is **entirely non-content** — letting disclosures
like *"i used to sneak out at night just to watch the stars"* resolve to a
real activity head (`"night watch"`) instead of being silently dropped.
Verified against the live engine (offline, `dim=64`) via
`tests/test_d5_degenerate_head_skip.py` (5 tests, green).

## The gap

A disclosure whose opinion-object phrase opens with a **degenerate head** —
a token that is a bare timeframe (`"night"`) or generic noun and lives in
`_OBJ_NONCONTENT` — was vanishing:

> *"i used to sneak out at night just to watch the stars"*

The resolver strips leading particles (`"out"`) and leading stop words
(`"at"`) first, leaving `["night", "just", "to", "watch", "the", "stars"]`.
It then collects a head: `"night"` is appended. The next token `"just"` is
a stop word → the loop **breaks**. The head is `["night"]`, which is
entirely in `_OBJ_NONCONTENT` → the content-adequacy gate rejects the whole
phrase → `_opinion_topic` returns `None` → the degenerate-fact guard drops
the disclosure → *"what did i sneak out to do?"* had nothing to recall.

This was the residual **DEFECT D5** from round 2026-09-23T0952Z (the earlier
fix had closed the `_gen_verb_pat` "$" anchor and added the "used to" skip,
but the head-collection path still broke on the first stop word).

## The fix

When a stop-word token would break the loop, check whether the head
collected so far is **entirely non-content**. If it is, **skip the stop
word and keep collecting** until a content token anchors the head:

```python
# ravana/src/ravana/chat/user_model.py:5684-5685
if head and all(h in _OBJ_NONCONTENT for h in head):
    continue
break
```

A **contentful** head (e.g. `"small talk"` — `"small"` is not in
`_OBJ_NONCONTENT`) still breaks at the stop word exactly as before. The
skip is a single structural conditional reading from the existing
`_OBJ_NONCONTENT` vocabulary — no authored reply, no per-topic rule, no
new store, no retraining.

Concrete verified behavior:

| Input | Before (broken) | After (fixed) |
|-------|-----------------|---------------|
| `"out at night just to watch the stars"` | `None` (degenerate head rejected) | `"night watch"` |
| `"small talk at the village market"` | `"small talk"` | `"small talk"` (unchanged) |
| `"just to"` | `None` | `None` (all non-content, correctly rejected) |

End-to-end, the canonical disclosure now mines an activity fact:

```
mine_personal_facts("i used to sneak out at night just to watch the stars")
→ ('does:sneak', 'sneak night watch')
```

Before the fix this produced **zero** facts; after, a later
*"what did i sneak out to do?"* resolves to the mined `does:sneak` value.

## Why this is not hardcoding

- The skip **only** fires when every token in the current head is in
  `_OBJ_NONCONTENT` (`user_model.py:5684`). Contentful heads break at stop
  words on the very next line (`user_model.py:5686`) — existing behavior is
  preserved exactly.
- `_OBJ_NONCONTENT` is **seed vocabulary** the miners already share
  (`user_model.py:903-917`); the fix reads it, it does not extend it. No new
  word list was added.
- The content-adequacy gate below (`user_model.py:5748`) still rejects a
  fully-degenerate phrase (`"just to"` → `None`), so junk is not admitted.
- No authored reply string, no per-topic branch, no retraining — the change
  is a **two-line structural conditional** inside an existing loop.

## Verification

Run against the live engine (offline, `dim=64`):

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_d5_degenerate_head_skip.py -v
```

- `test_degenerate_head_skip_collects_content_after_stopword` —
  `"out at night just to watch the stars"` → `"night watch"` (not `None`).
  **RED→GREEN**: pre-fix returns `None`.
- `test_contentful_head_still_breaks_at_stopword` — `"small talk at the
  village market"` → `"small talk"` (skip does NOT extend contentful heads).
- `test_leading_particles_still_stripped` — `"out in the stars"` → `"stars"`
  (leading particles still stripped first).
- `test_all_noncontent_phrase_returns_none` — `"just to"` → `None`
  (content-adequacy gate still rejects fully-degenerate phrases).
- `test_d5_end_to_end_mines_activity_fact` — the canonical disclosure mines
  a `does:sneak` activity fact containing `"sneak night watch"`.

All 5 pass (15.6s, `.venv-real` / Python 3.11).

Live in-process (the source of the verified table above):

```
_opinion_topic("out at night just to watch the stars")  → 'night watch'
_opinion_topic("small talk at the village market")      → 'small talk'
_opinion_topic("just to")                               → None
mine_personal_facts("i used to sneak out at night ...")  → [('does:sneak', 'sneak night watch')]
```

## Source pointers

| What | Where |
|------|-------|
| Opinion-topic resolver (head-collection loop) | `ravana/src/ravana/chat/user_model.py:5670-5686` |
| Degenerate-head skip (the fix) | `ravana/src/ravana/chat/user_model.py:5684-5685` |
| `_OBJ_NONCONTENT` seed vocabulary | `ravana/src/ravana/chat/user_model.py:903-917` |
| Content-adequacy gate | `ravana/src/ravana/chat/user_model.py:5748-5749` |
| Leading particle strip | `ravana/src/ravana/chat/user_model.py:5656-5657` |
| Leading stop-word strip | `ravana/src/ravana/chat/user_model.py:5660-5664` |
| Regression tests | `tests/test_d5_degenerate_head_skip.py` |
