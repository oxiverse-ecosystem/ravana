# Capability: agent-own-recall disambiguates tied topic overlap (fail closed)

**Status:** shipped (commit `4b50767a`, branch `auto/round-2026-10-27T0858Z`, feature `t_37946f9d`).
**Verified:** import of `CognitiveChatEngine` verified live this cycle; the change is a 9-line, single-file edit with no new reply strings.

## What it does

When RAVANA recalls something it told the user before (via `_route_agent_own_recall`), it scores each stored reply by how many content tokens the query shares with that reply's `src_tokens` and echoes the best-matching reply. Before this fix, a **tie** between two stored replies on overlap would be silently broken by dict iteration order — RAVANA could echo the reply about a topic the user did *not* ask about.

After the fix, the score loop also tracks the **second-best** overlap. When the best and second-best overlap are equal, the method **fails closed** (returns `None`, falling through to the utterance-log ring buffer and then honest uncertainty) instead of risking a wrong-topic echo.

- No LLM, no Q→A dict, no keyword→reply table. The change is a structural comparison in the live recall path.
- Genuine single-topic recalls (one clear winner) still recall correctly; only the ambiguous tie case fails closed.

## How it grew from the conversation

The chat round `t_93d0df77` (31 in-process turns, `RAVANA_OFFLINE=1`) surfaced a residual limitation: RAVANA occasionally **echoed the wrong stored reply** when the query's content tokens overlapped two different stored topics equally (e.g. a query sharing 2 tokens with two distinct stored replies). The defect is structural rather than a one-off typo: the score loop only kept the single best match and had no way to detect ambiguity.

**Root cause / prior behavior.** `_route_agent_own_recall` (`engine.py:6540`) iterates `_store` (`self._own_replies`), computes `_ov = len(_q_set & _src)` for each entry, and keeps only `_best` / `_best_overlap`. There was no record of the runner-up, so a tie between two entries on overlap was indistinguishable from a clear win, and dict iteration order decided which reply got echoed. That is the wrong-topic echo the limitation round flagged.

**Fix (commit `4b50767a`).** Three small pieces, all in `engine.py:6672-6690`:

1. Initialize a runner-up accumulator alongside the best: `_second_best_overlap = 0`.
2. In the scoring loop, when a new best is found, demote the previous best to runner-up (`_second_best_overlap = _best_overlap`); otherwise, update the runner-up when `_ov > _second_best_overlap`.
3. Extend the fail-closed guard to include the tie case:
   ```python
   if _best is None or _best_overlap < _min_overlap or _best_overlap == _second_best_overlap:
   ```
   On a tie the method falls through to the existing utterance-log ring-buffer fallback (RFIX-01) and, if that also misses, to honest uncertainty — never a wrong-topic echo.

The `_min_overlap` threshold (`1 if len(_cands) <= 1 else 2`) is unchanged, so single-token queries keep their existing behavior and only genuine ties fail closed.

**Generalization.** The detector is topic-agnostic: it keys on the *shape* of the overlap distribution (best == second-best), not on any specific topic words, so it generalizes to any pair of stored replies the user might ask about. There is no retraining and no new reply authoring.

**Hardcoding audit.** The diff adds zero reply strings. The only added surface is a runner-up accumulator and one extra term in an existing boolean guard. Grep for long added strings in the changed region returns nothing. Replies still render REAL stored state via the existing echo path; on a tie the method falls through to the pre-existing ring-buffer / uncertainty path. No Q→A dictionary, no keyword→reply table.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_route_agent_own_recall` (method head) | `ravana/src/ravana/chat/engine.py:6540` |
| `_store` load + empty-store fallback to `_route_agent_own_recall_stances` | `ravana/src/ravana/chat/engine.py:6656-6670` |
| Score loop + runner-up tracking (`_second_best_overlap`) | `ravana/src/ravana/chat/engine.py:6672-6689` |
| Fail-closed guard including the tie case | `ravana/src/ravana/chat/engine.py:6690-6692` |
| `_min_overlap` threshold (unchanged) | `ravana/src/ravana/chat/engine.py:6689` |
| Utterance-log ring-buffer fallback on miss | `ravana/src/ravana/chat/engine.py:6692-...` |
| Existing regression tests exercising this method | `tests/unit/test_round_2026_08_17_ownrecall.py`, `tests/test_rfix01_utterance_log.py`, `tests/test_d5_self_ref_recall.py`, `tests/test_round_2026_09_09_selfrecall_singleton.py`, `tests/unit/test_d3_first_person_attr_guard.py` |

## Test coverage

No new test was added this cycle; the change is a one-line extension to an existing fail-closed guard and is exercised by the existing `_route_agent_own_recall` test suite listed above. The disambiguation case (two stored replies with equal overlap → `None`) is the gap a future test could close within CI budget; the method's public contract (returns `Optional[str]`, falls through to the ring buffer on `None`) is unchanged.
