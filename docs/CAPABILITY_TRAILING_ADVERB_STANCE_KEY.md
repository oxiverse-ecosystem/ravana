# Capability: trailing-discourse-adverb stance-key trim (no temporal residue in a stance key)

**Status:** shipped (round `2026-10-12T0729Z`, feature `t_407c0ff4`).
**Verified:** regression suite `tests/unit/test_trailing_adverb_stance_key.py` passes
5/5 on the shipped code. The tests exercise real `CognitiveChatEngine` probes
(`dim=64, seed=42, baby_mode=True`, offline) — not mocks. Hardcoding self-audit
clean: no authored reply prose, no per-topic table, no retraining — only a
closed-class stop-word addition to the existing `_OPINION_STOP` set.

## What it does

When a disclosure ends with a **temporal or discourse adverb** — *"actually i
love board games now"*, *"i love board games today"*, *"i love board games
tomorrow"*, *"i love board games yesterday"* — the opinion-object head used to
keep the trailing adverb, producing a stance key like `board games now`
alongside the clean `board games`. The next recall about board games then read
the **weaker** (duplicate) stance instead of the real one.

The fix adds a set of **closed-class temporal/discourse adverbs** to
`_OPINION_STOP` (`user_model.py:6483-6492`). The existing
`_opinion_topic` method already trims trailing `_OPINION_STOP` words from the
head phrase as a final safety (`user_model.py:6683-6684`), so the new entries
are picked up by that same trim loop — no new code path, no per-topic rule.

Concretely:

- *"i love board games now"* → stance key **`board games`** (not `board games now`).
- *"i love board games today"* → stance key **`board games`**.
- *"i love board games tomorrow"* → stance key **`board games`**.
- *"i love board games yesterday"* → stance key **`board games`**.

The adverb set (`now`, `today`, `tonight`, `yesterday`, `tomorrow`, `soon`,
`later`, `earlier`, `then`, `here`, `there`, `still`, `yet`, `already`,
`anymore`, `lately`, `recently`, `currently`) is the same closed-class
stop-word pattern as the existing entries (`really`, `very`, `just`, etc.).
It generalizes to **any** trailing temporal adverb the user rotates in — no
per-topic rule, no retraining.

No LLM, no Q→A dict, no keyword→reply table. The capability is a structural
addition to the existing stop-word set; RAVANA's stance keys are recomputed
from the real utterance every turn, satisfying the seed + online-learning
constraints.

## How it grew from the conversation

The round report `tmp/reports/ravana-2026-10-04T0827Z-t_f30f89ba.md` logged a
new defect spotted during that cycle (item 1 in "Residual risk and open
items"):

> *"a trailing discourse adverb leaks into a stance key — `actually i love
> board games now` created the stance `board games now` alongside `board games`,
> so the next recall about board games reads the weaker one."*

The feature card (`t_407c0ff4`) picked this as the last limitation to turn into
a real capability.

**Root cause / prior behavior.** `_opinion_topic` builds the object head by
cutting at the first closed-class word (determiner/preposition/connector), then
trims trailing `_OPINION_STOP` words as a final safety. But temporal/discourse
adverbs (`now`, `today`, `tomorrow`, etc.) were **not** in `_OPINION_STOP`, so
they survived the trim and became part of the stance key. The stance store
then held both `board games` and `board games now`, and recall read the weaker
duplicate.

**Fix (round `2026-10-12T0729Z`).** Add the temporal/discourse adverb set to
`_OPINION_STOP` (`user_model.py:6483-6492`). The existing trailing-trim loop
(`user_model.py:6683-6684`) picks them up automatically — no new code path.

**Hardcoding audit.** The diff adds zero reply strings. Grep for long added
strings in the changed region returns only the comment block explaining the
addition. The stop-word set is a closed-class grammatical category, not an
authored answer; there is no Q→A dictionary and no keyword→reply table.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_OPINION_STOP` set (the shared stop-word set) | `ravana/src/ravana/chat/user_model.py:6438` |
| Trailing temporal/discourse adverb entries (the addition) | `ravana/src/ravana/chat/user_model.py:6483-6492` |
| Trailing-trim loop that applies `_OPINION_STOP` | `ravana/src/ravana/chat/user_model.py:6683-6684` |
| `_opinion_topic` (the shared opinion-object resolver) | `ravana/src/ravana/chat/user_model.py:3582` |
| Regression tests | `tests/unit/test_trailing_adverb_stance_key.py` |

## Test coverage

`tests/unit/test_trailing_adverb_stance_key.py` (5 tests, all pass on shipped
code):

- `test_trailing_now_not_in_stance_key` — *"i love board games now"* must NOT
  create a stance keyed on `board games now`; the key must be `board games`.
- `test_trailing_now_stance_recall_works` — after *"i love board games now"*,
  recalling *"what do i think about board games"* must find the stance (not
  fall through to *"still forming a view"*).
- `test_trailing_today_not_in_stance_key` — *"i love board games today"* must
  NOT create `board games today`.
- `test_trailing_tomorrow_not_in_stance_key` — *"i love board games
  tomorrow"* must NOT create `board games tomorrow`.
- `test_trailing_yesterday_not_in_stance_key` — *"i love board games
  yesterday"* must NOT create `board games yesterday`.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_trailing_adverb_stance_key.py -v
```
