# Capability: user-model aggregation (answers about *what I know about you*)

**Status:** shipped (commit `845478c`, branch `auto/round-2026-08-16T1241Z`).
**Verified:** regression tests in
`tests/unit/test_round_2026_08_16_user_model_agg.py` pass (3/3); live
end-to-end probe reproduced below (real engine output, `dim=64, seed=42,
baby_mode=True`, offline). Hardcoding self-audit clean.

## What it does

When the user asks RAVANA to **report the accumulated *content* of its model of
the user** — *"what have you picked up about me"*, *"what stands out about
me"*, *"tell me about myself"*, *"everything you know about me"*,
*"describe me"*, *"what do you remember me telling you"*, *"what's your read on
me"* — RAVANA renders the **REAL learned profile** from its LIVE durable
stores: the user's name, location, disclosed facts, stated beliefs, and stance
polarities.

This is distinct from the **meta-identity** capability
(`docs/CAPABILITY_META_IDENTITY.md`), which only reports *counts + topics* and
RAVANA's own self-coherence. Aggregation reports the *actual facts/stances*
themselves.

Every sentence is composed from runtime stores RAVANA grows autonomously
(`PersonalFactStore`, `UserStanceStore.stances`, `BeliefStore`); no authored
reply prose, no per-topic answer table, no retraining. Rendering picks ONE word
for a stance's polarity band (a single lexicon token — vocabulary, not a
script) and slots in the REAL topic/value from state.

Real engine output (fresh persona `corvin`, taught name + village + a niece
disclosure + a pet + two stances + a belief):

```
Q: what have you picked up about me
A: "here's what i've picked up about you so far: your name is corvin;
    you're from aldermoor in the hills; you grew village called aldermoor;
    you an astronomer who studies pulsars; on how you feel about things:
    you're strongly for sea; you're strongly against put."

Q: tell me about myself
A: (identical reply)

Q: what's your read on me
A: (identical reply)
```

The **same** answer renders for all aggregation phrasings because every word of
content is read from state, not authored.

**Fail-closed.** For a brand-new user with nothing stored, the aggregator
returns `None` so the honest-uncertainty path answers instead of fabricating
content:

```
Q(empty): what have you picked up about me
A(empty): None
```

**Fragment dedupe.** A single disclosure is frequently mined into several
OVERLAPPING facts (e.g. *"i grew up in a village called aldermoor in the
hills"* → `location:"aldermoor in the hills"` AND `grew:"grew village called
aldermoor"`, and the `grew` fact can be stored twice under the same attribute).
The aggregator deduplicates by **value content** — dropping exact-duplicate
values even across different attributes, and dropping a value that is a strict
substring of another retained value — so a line is never listed twice. This is
general (keyed by value content, no per-entity special-casing).

## Known rough edges (honest — logged for a future round)

The aggregation path renders whatever the **miner** stored, so it inherits the
miner's current quality. Observed in the probe above:

- A disclosure's mined fact can lack a leading verb, so it renders as a bare
  clause: `"you an astronomer who studies pulsars"` (from *"my niece priya is
  an astronomer…"*) and `"you grew village called aldermoor"`. The factual
  content is correct; the surface realization is not yet grammatical.
- A complex stance disclosure (*"i really dislike being put on the spot in
  meetings"*) yields a degraded topic token `"put"`, rendered as `"you're
  strongly against put."`

These are **miner/realizer** gaps upstream of this capability, not invented
content — the aggregator faithfully reflects what was stored. They are out of
scope for this round and tracked for a future fix. The aggregator itself is
correct, store-driven, and fail-closed.

## How it grew from the conversation

The chat round of this cycle (round `2026-08-16T1241Z`) surfaced, among its
residual limitations, that the **content-reporting** class of questions was
unhandled — distinct from the meta-identity count-reporting class. The feature
card (`t_3d147353`) picked it as a concrete capability gap.

**Root cause / prior behavior.** Aggregation queries were not recognised as a
distinct intent. They fell through to the **graceful-uncertainty fallback**,
which resolved the subject to a closed-class/garbage token and emitted
degenerate text — measured at turn T48 of the 1241Z round:
`"i don't really have a solid grasp on picked far so far"`. The engine in fact
held 9 facts + 4 stances but had **no path to render them**. (A second defect,
T60, returned an unrelated episodic memory for *"what stands out about me"*.)

**Fix (commit `845478c`).** A single regex in `_structured_recall`
(`engine.py:2437-2455`) detects the aggregation intent (7+ grammatical
phrasings, **no per-topic phrase list**) and delegates to a new
`_aggregate_user_model` (`engine.py:3206-3341`) that builds the reply from the
LIVE stores. The detection is fail-closed: it returns `None` for non-aggregation
input, leaving every other path untouched.

**Hardcoding audit.** `_aggregate_user_model` contains no authored reply prose
and no per-topic answer table — only connective scaffolding
(`here's what i've picked up about you so far`, `on how you feel about things`)
around slots read from state at call time (name, location, facts, beliefs,
stance topics + polarity words). The 3 regression tests assert the deleted
degenerate phrase ("solid grasp") does not leak and that an episodic echo
("you told me earlier") does not appear.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Aggregation intent detection (regex) | `ravana/src/ravana/chat/engine.py:2437-2455` |
| Branch entry in `_structured_recall` (returns the aggregate) | `ravana/src/ravana/chat/engine.py:2456` |
| State-driven reply builder | `ravana/src/ravana/chat/engine.py:3206-3341` |
| Fail-closed when nothing stored (returns `None`) | `ravana/src/ravana/chat/engine.py:3302-3304` |
| Fact collection + per-attribute longest-value dedupe | `ravana/src/ravana/chat/engine.py:3249-3265` |
| General value-content fragment dedupe | `ravana/src/ravana/chat/engine.py:3276-3287` |
| Stance polarity → single lexicon word | `ravana/src/ravana/chat/engine.py:3321-3336` |
| Regression tests | `tests/unit/test_round_2026_08_16_user_model_agg.py` |

## Test coverage

`tests/unit/test_round_2026_08_16_user_model_agg.py` (3 tests, all pass):

- `test_aggregation_renders_real_profile` — for each of the 7 aggregation
  phrasings, asserts the reply references the learned name (`corvin`), a learned
  fact (`aldermoor`/`priya`/`vesper`), and a learned stance (`sea`/`meetings`),
  does **not** contain the degenerate phrase ("solid grasp"), does **not**
  contain an episodic echo ("you told me earlier"), and contains no duplicated
  fragment (dedupe regression).
- `test_aggregation_fails_closed_on_empty_profile` — a brand-new user (nothing
  stored) yields `None` for aggregation queries.
- `test_non_aggregation_query_unaffected` — a plain biographical query
  (*"what's my name"*) still resolves from its own path, and the aggregation
  detector does **not** fire on a world-knowledge query
  (*"what is a pulsar, exactly?"*).

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_round_2026_08_16_user_model_agg.py -v
```

The broader recall/self/fact/meta suite stays green (179 passed, 3 skipped) with
the capability in place; the 3 new tests **fail** if the capability is stashed
out, proving a real RED→GREEN.
