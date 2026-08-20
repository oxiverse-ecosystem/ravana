# Capability: durable RAVANA-own stance store + revisit queries (limitation #2)

**Status:** shipped (commits `7d113ff`, `b331c49`, branch
`auto/round-2026-08-19T0625Z`). **Verified:** regression test
`tests/unit/test_own_stance_persistence.py` passes (1 collected test, green);
live in-process probe reproduced below (real engine output, `dim=64, seed=42,
baby_mode=True`, offline). Hardcoding self-audit clean — every added reply string
interpolates live recorded state (`{_word}` = stored polarity word, `{target}` =
user topic, `{_reason}` = stored reason); no authored per-topic prose.

## What it does

When a user asks RAVANA for **its own opinion** ("what do you think about open
source?"), RAVANA computes a stance and — for the first time — **records that
stance durably** into a persisted `_agent_own_stances` store, keyed by the
canonical topic. A later **revisit query** ("do you still feel that way about
open source?", "do you still think that about open source?", "do you feel the
same about open source?") is then answered **from that recorded stance** — naming
the topic and affirming continuity — instead of recomputing a fresh provisional
line or echoing.

Two prior limitations are closed here:

1. **No durable record.** Before this round, opinion questions were answered but
   the stance was only cached in-memory in `_agent_preferences` (which is purged
   of `stance:` keys on load). A reload "forgot" what RAVANA had said. The new
   `_agent_own_stances` store is written by `_agent_stance_on`, persisted in
   `save()`, and restored in `load()` exactly like the other durable signals.
2. **No revisit path.** A "do you still feel that way about X?" could only be
   answered from the echo store (C/D), never from a recorded stance. The new
   `_route_own_stance_revisit` consults the durable record and reports what
   RAVANA actually said before.

Replies are built entirely from the **live recorded tuple** `(polarity_word,
confidence, reason, turn)` — the topic is the user's real query target; the
orientation word is RAVANA's own stored polarity word, not an authored sentence.

Real engine output (fresh persona, asked its own view, then revisited):

```
Q: what do you think about open source
A: 'i strongly value open source. knowledge should be shared, not locked away.'
  strategy=self_model
  recorded_stance=('strongly value', 0.9, 'knowledge should be shared, not locked away', 0)

Q: do you still feel that way about open source?
A: "yeah, i still strongly value open source — that hasn't shifted for me.
    knowledge should be shared, not locked away"
  strategy=own_stance_revisit

Q: do you still think that about open source?
A: "yeah, i still strongly value open source — that hasn't shifted for me.
    knowledge should be shared, not locked away"
  strategy=own_stance_revisit        # verified via router isolation

Q: do you feel the same about open source?
A: "yeah, i still strongly value open source — that hasn't shifted for me.
    knowledge should be shared, not locked away"
  strategy=own_stance_revisit        # verified via router isolation
```

**Honest across confidence bands.** A topic RAVANA was only "still forming a
view" on is recorded (low confidence) too, so a revisit about it is answered from
the record honestly — not recomputed fresh:

```
Q: what do you think about quokkas                      # no seeded value -> provisional
A: "i'm still forming a view on quokkas. i don't have a fixed stance on quokkas
    yet — what's your take? i'd rather hear how you see it than guess."
  strategy=self_model

Q: do you still feel that way about quokkas?
A: "i hadn't settled on quokkas — last i said i was still forming a view.
    has your sense of it changed? i'm happy to land one."
  strategy=own_stance_revisit
```

**Fail-closed when nothing was recorded.** A revisit on a topic with no record is
answered honestly — it does not invent a stance:

```
Q: do you still feel that way about left-handed smokestacks?
A: "i don't actually have a recorded view on left-handed smokestacks from before
    — i'd be guessing. what made you bring it up again?"
  strategy=own_stance_revisit
```

No LLM, no per-topic answer table, no retraining. The store is runtime-expandable
(every stance `_agent_stance_on` computes is written here) and persisted, so a
reload that wiped it would make the agent "forget" its own prior opinions.

## Known rough edges (honest — logged for a future round)

- **"have you changed your mind about X?" does NOT currently reach the revisit
  router through the full pipeline.** The router itself handles this phrasing
  correctly in isolation (the regex at `engine_self_query.py:478-483` includes
  `changed your mind`, and a direct call returns the recorded stance). But in
  `process_turn`, an **earlier** self-model gate intercepts the query first and
  returns a generic identity answer instead of the recorded stance:

  ```
  Q: what do you think about open source        # records ('strongly value', 0.9, ...)
  Q: have you changed your mind about open source?
  A: "that's about me, not you — i'm still quite unsettled about who i am, and
      it's been growing as we talk. i don't always keep the exact words of what
      i said earlier, but the shape of it holds."
    strategy=self_model   # NOT own_stance_revisit — recorded stance ignored
  ```

  The revisit check at `engine.py:6752` sits *after* the broad self-opinion /
  self-experience gates (~`engine.py:5441-5503`), so a query that matches one of
  those earlier gates returns before the revisit check runs. The "still feel /
  still think / feel the same" family is not caught by those earlier gates and
  therefore works end-to-end; "changed your mind" is. This is a routing-order gap,
  not a defect in the revisit router. Tracked as a follow-up limitation.

- The provisional-stance write keys on the raw `target` string (the user's own
  words), while the high-confidence write keys on the canonical concept
  (`_canon`). A revisit that phrases the topic differently from how it was stored
  relies on the containment fallback in `_route_own_stance_revisit`
  (`engine_self_query.py:511-516`), which is best-effort, not exact.

## How it grew from the conversation

This cycle's chat round (round `2026-08-19T0625Z`) surfaced, among its residual
limitations, that RAVANA **answered opinion questions but never persisted them
as stances** — so "do you still feel that way about X?" could not be answered
from a recorded stance. The feature card (`t_29d42bfe`, limitation #2) picked it
as a concrete capability gap.

**Root cause / prior behavior.** `_agent_stance_on` (`engine_self_query.py:314`)
computed and returned a stance, but only cached it in the in-memory
`_agent_preferences` dict — which is purged of `stance:` keys on `load()`. There
was no durable `_agent_own_stances` store and no revisit router, so a later
revisit could only be answered from the echo store (C/D) or fall through to a
recomputed provisional line.

**Fix (commits `7d113ff`, `b331c49`).**
1. A new durable store `self._agent_own_stances`
   (`engine.py:1058`) — topic → `(polarity_word, confidence, reason, turn)`,
   distinct from the innate `_agent_values` constitution.
2. `_agent_stance_on` now **writes** the stance it computes into that store —
   both the real computed stance (`engine_self_query.py:418`) and the provisional
   "still forming a view" branch (`engine_self_query.py:464`) — so every expressed
   opinion is recorded.
3. A new `_route_own_stance_revisit` (`engine_self_query.py:471-532`) detects
   revisit cues ("still feel that way", "still think that", "changed your mind",
   "feel the same about", "still the same about", "do you still") and replies from
   the recorded `(word, conf, reason)`. High confidence → affirm continuity; low /
   provisional → honest that it was tentative; no record → honest abstention.
4. The revisit check is wired into `process_turn`
   (`engine.py:6752-6758`) and the store is persisted (`engine.py:8541`) and
   restored (`engine.py:8864-8876`).

**Hardcoding audit.** `_route_own_stance_revisit` contains no authored reply
prose and no per-topic answer table — only connective scaffolding around slots
read from the live record at call time: `{_word}` (the stored polarity word),
`{target}` (the user's real topic), `{_reason}` (the stored reason), and the
confidence band. The RED→GREEN test asserts: a stance is recorded, survives
save/load, a revisit reports the recorded orientation, and a revisit with no
record is honest.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Durable own-stance store (declaration) | `ravana/src/ravana/chat/engine.py:1058` |
| Real computed stance written to store | `ravana/src/ravana/chat/engine_self_query.py:418` (inside `_agent_stance_on`, `:314`) |
| Provisional "still forming" stance written to store | `ravana/src/ravana/chat/engine_self_query.py:464` |
| Revisit router (`_route_own_stance_revisit`) | `ravana/src/ravana/chat/engine_self_query.py:471-532` |
| Revisit cue regex | `ravana/src/ravana/chat/engine_self_query.py:478-483` |
| Containment fallback for clipped topic keys | `ravana/src/ravana/chat/engine_self_query.py:511-516` |
| Revisit check wired into `process_turn` (BEFORE opinion/identity gates) | `ravana/src/ravana/chat/engine.py:6752-6758` |
| Persisted in `save()` | `ravana/src/ravana/chat/engine.py:8541` |
| Restored in `load()` (guarded) | `ravana/src/ravana/chat/engine.py:8864-8876` |
| Regression test | `tests/unit/test_own_stance_persistence.py` |

## Test coverage

`tests/unit/test_own_stance_persistence.py` (1 collected test, green in ~16s):

- Records a real opinion question ("what do you think about open source") into
  `_agent_own_stances` with the seeded value word (`strongly value`).
- The record **survives a `save()` / `load()` roundtrip** (durability — the core
  of limitation #2).
- A revisit query ("do you still feel that way about open source?") answers from
  the **recorded** stance — it names the topic and affirms continuity, and must
  NOT fall through to a recomputed provisional "still forming a view" line.
- A revisit on a never-recorded topic ("do you still feel that way about
  quokkas?" in the no-record variant) answers honestly (no fabrication).

The test **fails** if the capability is stashed out (`_agent_own_stances` store
absent, `_route_own_stance_revisit` absent), proving a real RED→GREEN.

> Note: the test exercises the "still feel that way" phrasing and the no-record
> case end-to-end. The "still think that" / "feel the same" phrasings and the
> provisional-revisit branch are verified by the live probe above and by router
> isolation, but are not yet asserted in the regression suite. The "changed your
> mind" phrasing is currently NOT covered because, as noted above, it is
> intercepted upstream before the revisit router runs.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_own_stance_persistence.py -v
```
