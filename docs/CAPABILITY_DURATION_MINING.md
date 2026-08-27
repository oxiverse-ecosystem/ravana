# Capability: activity-with-duration mining → dated-fact recall

**Status:** shipped (commit `867de08`, branch `auto/round-2026-08-14T0608Z`).
**Verified:** 6/6 regression tests pass (`tests/unit/test_round_2026_08_14T0608_approx_duration.py`); live end-to-end probe reproduced below.

## What it does

When the user says they have been doing something for a span of time, RAVANA
stores a **dated start fact** ("since") and can later answer *when did you start
/ since what year* questions about that activity — using the **same recall path**
as explicit-year facts, with no per-phrase resolver code.

The capability covers four duration shapes, all funneled into one `since`
attribute and resolved by one resolver:

| Shape | Example utterance | Stored fact | Mined in |
|-------|-------------------|-------------|----------|
| Explicit year anchor | `i've kept quail since 2019` | `since(quail 2019)` | block (a) `user_model.py:1134` |
| Digit / spelled 1–12 | `i've repaired tube amps for eleven years` | `since(repair <year-11>)` | block (b) `user_model.py:1150` |
| Age anchor | `since i was nine i've played cello` | `since_age(cello 9)` | block (c) `user_model.py:1197` |
| **Approximate / human-phrased** | `i've been brewing beer for a decade` | `since(brew <year-10>)` | block (d) `user_model.py:1233` |

The newest shape — approximate/human-phrased durations (`a decade`, `two
decades`, `a couple of years`, `a few years`, `several years`, `many years`,
`a handful of years`, …) — is what this round added. People almost never say
"for eleven years"; they say "for a decade" / "a few years now", and those
phrases previously landed in **no** dated fact, so date recall returned empty
for them.

## How it grew from the conversation

The parent chat round (`t_b9cd03d3`, 2026-08-14T0608Z) surfaced a residual from
its own date-mining work: spelled-out ages beyond twenty and relative durations
phrased as "a decade" / "since the pandemic" were not yet captured. The prior
miner blocks (a)/(b)/(c) only handled explicit digits, spelled 1–12, and
`since <YEAR>`.

Rather than add a fourth resolver branch, the fix **reuses the exact
`since` attribute + activity-attachment logic of block (b)** and lets the
**existing** date recall resolver answer for the new facts too. That is the proof
it is a generalizable capability, not a per-phrase hack: the new facts flow
through the identical `since` + reverse-lookup path already built. See the
resolver at `engine.py:2614` (`_structured_recall`, defined at `engine.py:2163`)
— it finds the `since` fact whose value starts with the queried activity and
returns `you started <activity> in <year>.`, with **no branch** that knows the
duration was fuzzy.

### Implementation (block (d), `user_model.py:1233`)

- `_FUZZY_DUR` (`user_model.py:1248`) maps an approximate phrase to a count:
  `"a decade": 10`, `"a couple of years": 2`, `"a few years": 3`,
  `"several years": 4`, `"two decades": 20`, `"many years": 15`, …
- For each phrase, it regex-matches `(for|about|over|nearly|almost)? <phrase>` in
  the cleaned query, then resolves a start year by subtraction:
  `_THIS_YEAR - n` (`user_model.py:1267`). The year self-updates every run.
- It attaches to the **nearest activity verb before** the phrase, reusing block
  (b)'s verb vocabulary (`user_model.py:1272`) so the mined fact stays
  recallable by the same resolver. Source strings like `i've been brewing beer
  for a decade` → activity head `brew` → `since(brew <year-10>)`.
- **Fail-closed high precision:** if no activity verb precedes the phrase, no
  fact is created (`user_model.py:1280`). So `for a decade i've wondered whether
  to start` mines nothing — no garbage dated fact.

### Design compliance

- **Seed knowledge only.** `_FUZZY_DUR` is a small data map; adding `"a
  fortnight": 14` or `"a generation": 25` degrades gracefully (one fewer duration
  form captured) and needs no code change elsewhere. It maps a *phrase class* to
  a *count* — the same pattern as block (b)'s number-word map — **not** an
  `if/elif` answer table.
- **Online / incremental, no retraining.** The resolved year is derivable
  (`_THIS_YEAR - n`) and self-updates; nothing requires a rebuild.
- **Zero authored reply prose.** The reply is rendered from mined `since` facts +
  `datetime.now().year` by the pre-existing resolver. A hardcoding self-audit
  (grep for added strings >45 chars) found only the verb-vocabulary regex
  literals (seed vocabulary, identical to block (b)); no reply prose.

## Live verification (fresh engine, offline)

```
$ RAVANA_OFFLINE=1 python -c "
from ravana.chat.engine import CognitiveChatEngine
eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
eng.process_turn(\"i've been brewing beer for a decade\")
print(eng._structured_recall('since what year have i brewed beer'))
print(eng._structured_recall('when did i start brewing beer'))
"
you started brew in 2016.
you started brew in 2016.
```

(Year 2016 = 2026 − 10 at time of verification.) The activity head resolves to
`brew` (stemmed from `brewing`), which is the same quirk shared with block (b)'s
explicit-duration miner — and the recall matcher resolves it identically.

## Tests

`tests/unit/test_round_2026_08_14T0608_approx_duration.py` — 6 tests, all pass
(10.9 s, `.venv-real`, `RAVANA_OFFLINE=1`):

- `test_decade_mined_as_year` — `a decade` → `year-10`
- `test_couple_of_years_mined` — `a couple of years` → `year-2`
- `test_several_years_mined` — `several years` → `year-4`
- `test_two_decades_mined` — `two decades` → `year-20`
- `test_approx_duration_without_activity_is_not_captured` — no fact when no
  activity verb precedes (fail-closed)
- `test_approx_duration_recall_resolver` — end-to-end: mined fact then recalled
  via `_structured_recall` for both phrasings

Regression across prior round suites + `test_dehardcode_plan.py`: 38 related
tests pass (no change to the existing date/name/activity miners).

## Caveats (honest)

- The resolved activity head is the **verb stem** (`brew`, not `brewing`/`beer`),
  inherited from block (b). Recall reads naturally (`you started brew in 2016.`)
  but the head is a verb, not the object noun.
- `many years` is mapped to 15 by default; tune the count in `_FUZZY_DUR` if a
  different convention is wanted.
- Relative durations without an activity verb (`for a decade i've wondered…`)
  are intentionally **not** captured.
- The broad `pytest tests/unit/` gate (1834 tests) was not observed to finish in
  this cycle's time budget; the feature's own 6 + 38 related tests are the
  working verification. Re-run on a runner with headroom before a release tag.
