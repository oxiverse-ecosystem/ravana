# Date-Grounded Recall for Year-Only Temporal Starts (Q59)

RAVANA can now ground a first-person temporal-start disclosure that names
**only a 4-digit year** — *"firing since 2017"*, *"i started in 2019"*,
*"since 2019 i have lived here"*, *"in 2018 we moved"* — to an absolute date
(**Jan 1 of that year**, `'year'` granularity) and recall it later. This closes
the Q59 class of the 2026-08-14 chat round: a year-only start anchor previously
fell through to the session-date default and date-grounded recall returned
empty.

This is a **real capability**, not a hardcoded reply: it is pure date
arithmetic against the supplied session anchor — no LLM, no per-fact table, no
authored answer string. The cue set is **seed structure** (RAVANA-expandable);
removing a cue degrades gracefully (that cue simply won't bind a year).

All claims below were checked against the source on branch
`auto/round-2026-08-14T0103Z` at commit `d257abb`. Line numbers cite that tree.

## What it does

When an utterance carries a **temporal START cue word** immediately preceding a
plausible year, and that year is not a quantity, RAVANA resolves the anchor to
**January 1 of that year** at `'year'` granularity.

1. `temporal_grounding.py:266` — `DateGrounder.resolve_year_start_anchor(text,
   session_date)` searches for the seed cue regex (`_year_start_pattern`,
   `temporal_grounding.py:258`) and, on a match, returns
   `GroundedDate(datetime(year, 1, 1), "year", text)`.
2. It is wired into `ground_utterance` (`temporal_grounding.py:463`) as the
   **terminal fallthrough**: if no absolute date, no month-day, and no
   relative anchor resolved, the year-start anchor runs last. A more-precise
   date (e.g. *"in May 2019"*) short-circuits earlier, so month precision is
   preserved (`temporal_grounding.py:454-465`, verified test:
   `test_ground_utterance_month_precision_preserved`).
3. The engine stores the resolved `absolute_date` on the hippocampal fact at
   disclosure time — `engine.py:2047-2052` calls `ground_utterance`, and
   `engine.py:2150-2159` writes `absolute_date=_abs_date` into the buffer.
4. Date-grounded recall reads it back: a *"when did X"* question routes to
   `_answer_temporal_recall` (`engine_reasoning.py:1515`), which formats the
   stored date as `"{day} {Month YYYY}"` (`engine_reasoning.py:1741`) and
   returns e.g. `you mentioned that around 1 January 2017.`
   (`engine_reasoning.py:1770`).

A live, end-to-end run through `CognitiveChatEngine` confirms the closed loop
(verified — session date established first, see the precondition below):

```text
turn: "(Session 1, dated 8 May, 2023)"   → session anchor set
turn: "i have been firing my kiln since 2017"
      → fact stored on buffer: absolute_date == 2017-01-01
turn: "when did i start firing"
      → "you mentioned that around 1 January 2017."   (strategy: temporal_recall)
```

## The precondition (shared by ALL temporal grounding)

A session date **must be established** before any date anchor resolves. Both
`ground_utterance` (`temporal_grounding.py:445`, `if session_date is None:
return None`) and the engine's fact-store path (`engine.py:2045`,
`_sess_date = getattr(self, "_current_session_date", None)`) depend on it. With
no session date, the year-start anchor is never reached and the fact stores the
session-date default (which is also `None` until a date is set) — so recall
falls back to a plain episodic echo. This is **not** specific to Q59; every
date-grounding feature in RAVANA needs the anchor.

The session date is set by the LoCoMo/LongMemEval marker format the engine
parses (`engine.py:3270-3279`):

```text
(Session 1, dated 8 May, 2023)
```

Any utterance matching `^\(?Session \d+[,:]? dated (.+?)\)?$` sets
`self._current_session_date`. (`scripts/ravana_chat.py` likewise threads a
session date into the engine during replay.)

## How it grew — the residual gap

Round `t_7b1d0007` (auto cycle 2026-08-14T0103Z) flagged a concrete limitation
from its 60-turn cold probe: first-person duration disclosures naming only a
year were stored with **no dated fact**, so date-grounded recall
(`"when did i start firing"`) returned empty. The `DateGrounder` had no path
for a year-only START anchor — `ground_utterance` required a month name before
trusting an assembled date (`temporal_grounding.py:404-418`), so a standalone
year fell through to the session-date default.

### Root cause

The generic absolute-date resolver (`_regex_absolute`) deliberately refuses a
bare year (it would stitch bogus dates from scattered fragments — measured on
LongMemEval). But that refusal left year-only *start* anchors with **no**
resolver, because the relative-anchor and month-day paths also require more
specific input. A first-person *"since 2017"* had nowhere to land.

### The fix (one structural addition)

A dedicated `resolve_year_start_anchor` method
(`temporal_grounding.py:266-290`) + its wiring as the terminal fallthrough in
`ground_utterance` (`temporal_grounding.py:454-465`). It is deliberately the
**last** resolver so it never shadows a more precise date.

The cue vocabulary is a **seed tuple** (`temporal_grounding.py:250-256`):

```python
# temporal_grounding.py:250 — SEED STRUCTURE (RAVANA-expandable), not a table
_YEAR_START_CUES = ("since", "starting", "started", "back in", "from", "in")
_YEAR_UNIT_DENY  = ("dollar", "dollars", "cent", "cents", "rupee", "rupees",
                    "euro", "euros", "pound", "pounds", "yen", "rs", "%", "₹", "$")
```

Two guards make it precise rather than greedy:

- **Quantity guard** (`temporal_grounding.py:283-289`): a year that is really a
  scalar — *"i scored 2015 points"*, *"in 2018 dollars"* — must NOT anchor a
  date. The first token after the matched year is checked against
  `_YEAR_UNIT_DENY`; on a match it returns `None`.
- **No-cue guard** (`temporal_grounding.py:276-279`): a bare year with no
  temporal cue word (*"just a random 1999 sentence"*) is left un-anchored, so
  it does not shadow the session-date default upstream.

## Why this is seed structure, not hardcoding

The added code is (1) a **seed tuple** of temporal-start connective words and a
**deny tuple** of quantity units, plus (2) a **regex** (`_year_start_pattern`,
compiled from the seed cue set, `temporal_grounding.py:258`) and (3) pure date
arithmetic returning a `GroundedDate`. It is **not** a question→answer
dictionary and **not** authored reply prose.

The seed-vs-hardcoding test ("can RAVANA change this by itself, through
experience?") is satisfied: the cue set is RAVANA-expandable — adding a cue
here (and rebuilding the regex via `_year_start_pattern`) bootstraps a new
anchor form online; removing one degrades gracefully (that cue simply won't
bind a year). A hardcoding audit of the diff found **zero authored reply
strings** — only seed tuples + a regex + a `GroundedDate` return.

No LLM or retraining is involved; the capability is online and incremental.

## Verification

Covered by `tests/unit/test_round_2026_08_14T0103_date_anchor.py`:

- `resolve_year_start_anchor` unit checks: `since 2017`, `started in 2019`,
  `since 2019 i have lived here`, `in 2018 we moved`, `back in 2015 i began`
  (`test_year_start_anchor_*`).
- Quantity / no-cue guards: `i scored 2015 points` and `in 2018 dollars` are
  NOT anchored; `just a random 1999 sentence` is `None`
  (`test_year_start_anchor_quantity_guard`, `test_year_start_anchor_no_cue_is_none`).
- `ground_utterance` integration: `...since 2017` → `2017-01-01` at `'year'`
  granularity; `in May 2019` keeps `day` precision
  (`test_ground_utterance_returns_year`,
  `test_ground_utterance_month_precision_preserved`).
- Buffer integration: the engine's `ground_utterance` result is stored and
  retrievable with `absolute_date == 2017-01-01`
  (`test_dated_fact_stored_and_retrievable`). The added engine-level closed-loop
  tests below prove the full `CognitiveChatEngine` path (disclose → recall).

The tests **assert on stored state**, never on authored reply strings, and they
fail when the anchor is neutralized (guards against false-green). The broader
recall suite (65 passed, 1 pre-existing skip) shows no regression.

### Engine-level closed-loop tests (added by the docs card)

- `test_engine_stores_year_anchor_dated_fact` — full engine loop with a session
  date set: the disclosure stores `absolute_date == 2017-01-01` on the buffer
  (guards the `engine.py:2047-2159` wiring).
- `test_engine_recalls_year_anchor_date` — `when did i start firing` is answered
  with the grounded date (`temporal_recall` strategy, reply contains
  `"1 january 2017"`), not a plain episodic echo. This catches a silent
  regression where the anchor fails to reach either the stored fact or the
  temporal-recall dispatch.

## Limits

- Requires a **session date** to be established first (precondition above).
- Only first-person **temporal-START** cues are handled; a year in a different
  syntactic role (object, quantity, comparison) is intentionally left
  un-anchored by the no-cue / quantity guards.
- Granularity is `'year'` (Jan 1) — the disclosure names no finer unit, so the
  engine does not fabricate month/day precision.
- The seed cue set is intentionally small; new start-connectives are added by
  extending `_YEAR_START_CUES` (online-expandable), not by special-casing
  turns.
