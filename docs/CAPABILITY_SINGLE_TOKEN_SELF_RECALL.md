# Capability: single-topic self-reference recall (single-token + 'remind' stopword)

**Status:** shipped (commit `10e31fb9`, branch `auto/round-2026-09-08T1738Z`).
**Verified:** `tests/test_round_2026_09_09_selfrecall_singleton.py` PASSES (3/3, ran live this cycle, ~12s). Hardcoding self-audit clean: zero authored reply strings, no Q→A dict, no retraining; both changes are structural (the recall gate's stopword set and overlap threshold).

## What it does

A self-recall query with only **one** content token — *"remind me what you said about silence"*, *"what did i tell you about art"* — now returns the stored reply instead of honest uncertainty.

- Before the fix, `_route_agent_own_recall` required `_best_overlap >= 2` between query tokens and stored reply `src_tokens`. A query like *"remind me what you said about silence"* has only **one** content token (`"silence"`) after stopword removal — it could **never** reach `>= 2` overlap, so the recall always failed and RAVANA replied *"honestly, I don't have that stored"* even when it **had** a stored reply about silence.
- After the fix, when the query yields `<= 1` content tokens, the minimum overlap drops to 1. Multi-token queries still require `>= 2` overlap to prevent confabulation on incidental word matches.

Two structural changes in `_route_agent_own_recall` (`engine.py`):
1. Added `"remind"` / `"reminded"` to the stopword set so the cue verb doesn't pollute candidate tokens.
2. Lowered the minimum overlap threshold to 1 when `len(_q_set) <= 1`.

The behavior is **fail-closed and consistent**: a query with zero content tokens (all stopwords) still returns `None`; a multi-token query with only one incidental shared token still returns `None` (no confabulation); a single-topic recall now succeeds. No LLM, no per-topic reply table.

## How it grew from the conversation

The chat round noted that *"remind me what you said about silence"* — a query RAVANA **had** answered and recorded — was met with honest uncertainty. The defect is structural, not a one-off: any self-recall query whose content collapses to a single token after stopword removal (common with short-topic queries + a reminder cue) is unrecoverable under a `>= 2` threshold.

**Root cause.** The recall gate scored every stored `_own_replies` entry by `len(_q_set & _src)` (shared content tokens) and rejected matches below 2. For a single-token `_q_set`, the maximum possible overlap is 1, so the gate **always** returned `None`. The word `"remind"` itself wasn't in the stopword set, so it counted as a candidate token — but it never appeared in `src_tokens` (which are the *topic* words, not the cue), so it contributed 0 overlap. Net effect: single-topic recall was structurally impossible.

**Fix.** Two lines in `engine.py`:
1. `"remind", "reminded"` added to the `_stop` set (`engine.py:5819-5820`) so the cue verb doesn't pollute candidates.
2. `_min_overlap = 1 if len(_q_set) <= 1 else 2` (`engine.py:5859`) so a single content token is sufficient when there's nothing else to overlap on. Multi-token queries still need `>= 2`.

**Generalization.** The gate is topic-agnostic — it keys on the *count* of content tokens, not their identity — so it generalizes to any single-topic phrasing (*"remind me what you said about winter"*, *"what was it i told you about my grandmother"*, *"do you remember my thing about telescopes"*) with no per-topic code. There is no retraining and no new reply authoring.

**Hardcoding audit.** The diff adds zero reply strings. Grep for long added strings in the changed region returns only the stopword entries and a threshold variable. The added surface is: two stopword tokens + one ternary threshold. There is no Q→A dictionary and no keyword→reply table. Replies render REAL stored state from `_own_replies`.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_stop` (stopword set, now incl. `remind`/`reminded`) | `ravana/src/ravana/chat/engine.py:5810-5820` |
| `_route_agent_own_recall(user_input)` (the recall gate) | `ravana/src/ravana/chat/engine.py:5706` |
| `_min_overlap = 1 if len(_q_set) <= 1 else 2` (dynamic threshold) | `ravana/src/ravana/chat/engine.py:5859` |
| `_best_overlap < _min_overlap → return None` (confabulation guard) | `ravana/src/ravana/chat/engine.py:5860` |
| `_record_own_reply(query, text, topic)` (writes `src_tokens`) | `ravana/src/ravana/chat/engine.py:~5692` |
| Regression + capability test | `tests/test_round_2026_09_09_selfrecall_singleton.py` (3 checks) |

## Test coverage

`tests/test_round_2026_09_09_selfrecall_singleton.py` (verified **3 passed, ~12s**, ran live this cycle). The 3 checks:

- `test_single_token_self_recall_succeeds` — stores a reply about `"silence"`, recalls with *"remind me what you said about silence"*; asserts the stored reply is returned (single content token + remind cue).
- `test_single_token_self_recall_with_scaffold` — stores a reply about `"art"`, recalls with *"earlier you said something about art"* (scaffold-heavy); asserts the stored reply is returned.
- `test_multi_token_recall_still_requires_overlap` — stores a reply about `"music"`, queries *"earlier you said something about board games"`; asserts `None` (different topic, no confabulation).

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_round_2026_09_09_selfrecall_singleton.py -q
```
