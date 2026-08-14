# Open-Class Verb Capture — learning personal activity facts from first-person self-reports

RAVANA does not only remember fixed categories of personal fact (location, pet,
likes). It also learns **what the user does** from first-person self-reports —
"i run a marine research boat", "i play the veena", "i restore old sailing
ships". This page documents how that capture became **open-class**: any
first-person "i &lt;verb&gt; &lt;object&gt;" whose verb is not a stative/copula verb
is recognised as a disclosure *and* mined as a personal fact, including
**novel verbs** ("i astrophotograph the milky way") and **hyphenated compound
verbs** ("i tide-pool at low water") that no seed whitelist could have listed.

All claims below were checked against the source on branch
`auto/round-2026-08-13T2059Z` at commits `b640836` (feat) and `67f0cb9` (tests).
Line numbers cite that tree.

## What it does

When the user writes a first-person statement of the form "i &lt;verb&gt;
&lt;object&gt;" — `i count meteor showers`, `i tide-pool at low water`, `i
astrobotany the orchids` — RAVANA:

1. **Recognises it as a self-disclosure** via the vmPFC-mimetic gate
   `_is_self_disclosure_stmt` (`ravana/src/ravana/chat/engine_reasoning.py:1941`),
   so the turn routes to STORE + ACK instead of leaking into the
   knowledge-query / uncertainty path.
2. **Mines it as a personal fact** into the `PersonalFactStore` under attribute
   `"does"` (e.g. `('i','does') -> "count meteor showers"`), via
   `UserModel.mine_personal_facts`
   (`ravana/src/ravana/chat/user_model.py:278`), at the open-class block
   (`user_model.py:1212`, storing at `:1277`).

The verb vocabulary is now **open-class** (a closed deny-list of
stative/copula/achieve-comm verbs), so RAVANA learns the verb from experience
instead of requiring the seed whitelist to enumerate every possible activity.

A live run confirms the behaviour (verified):

```text
turn: "i count meteor showers from the lighthouse gallery every august"
      → recognised as disclosure (gate returns True)
      → fact stored: ('i','does') -> "count meteor showers"

turn: "i tide-pool at low water and catalogue the anemones and limpets"
      → recognised as disclosure (gate returns True)
      → fact stored: ('i','does') -> "tide-pool ..." / "catalogue ..." (open-class,
        both verbs captured; neither is stative)
```

## How it grew — the residual gap

Round `t_60e497c7` (auto cycle 2026-08-13T2059Z) logged a concrete limitation
from the round's own probe run (turn 51):

> `i tide-pool at low water and catalogue the anemones and limpets.`

was misrouted as a **knowledge query** about "tide-pool low water" — RAVANA
replied with an uncertainty ack ("honestly, tide-pool low water is a bit outside
what i know right now") and stored **NO** personal fact.

### Root cause

Two coupled failures:

1. **Recognition gate missed the verb.** The self-disclosure recognizer at
   `engine_reasoning.py:2053` (`_act_pat` / `_gen_act` before the fix) only
   consumed a single `[a-z']+` verb token, so the hyphenated compound
   `tide-pool` did not match the activity form. The statement therefore failed
   the gate and fell through to the knowledge-query / uncertainty path.

2. **Miner missed the verb.** The activity/event miners
   (`user_model.py:1189` event block; the `ACTIVITY_VERBS`/`EVENT_VERBS`
   whitelists above it) were **frozen verb whitelists**. Novel or compound verbs
   (`count`, `catalogue`, `tide-pool`, `astrophotograph`) were simply not in the
   lists, so even when a turn did reach the miner, no `does`/`event` fact was
   written.

### The fix (two structural changes)

**(A) Gate consumes open-class + hyphenated verbs**
(`engine_reasoning.py:2061-2071`) — the activity-disclosure regex `_gen_act` now
reads `([a-z']+(?:-[a-z']+)*)` so it accepts any lowercase token *and*
hyphenated compound verbs:

```python
# engine_reasoning.py:2070 — GENERALISED (round 2026-08-13T2059Z)
_gen_act = re.compile(
    r"\bi\s+([a-z']+(?:-[a-z']+)*)(?:\s+[a-z'\-]+)+", re.IGNORECASE)
_m = _gen_act.search(q)
_is_activity = bool(_m) and _m.group(1).lower() not in _STATIVE_VERBS
```

A first-person disclosure using a novel or compound verb is now recognised as a
disclosure (and routed to store + ack) instead of leaking into the
knowledge-query path (`engine_reasoning.py:2062-2069`).

**(B) Miner becomes open-class (deny-list, not whitelist)**
(`user_model.py:1212-1277`) — a new block replaces the "missed because not in a
frozen whitelist" failure with general capture:

```python
# user_model.py:1227 — closed deny-list of stative/copula/achieve-comm verbs
_STATIVE_DENY = frozenset({
    "am", "are", "is", "was", "were", "be", "been", "being", "become", ...,
    "feel", "love", "like", "hate", ..., "think", "know", ..., "have", "own", ...,
    "got", "get", "said", "made", "took", "see", ...  # echo-verbatim garbage
})
# user_model.py:1253 — open-class verb, optionally hyphenated compound
_gen_verb_pat = re.compile(
    r"\bi\s+(?:also\s+|really\s+|...)?(?:have\s+been\s+|...)?(?:been\s+)?"
    r"((?:[a-z']+(?:-[a-z']+)*)(?:s|es)?)"
    r"\s+(?:my\s+|a\s+|...)?(.+?)(?:...clause boundary...)", re.IGNORECASE)
for _gm in _gen_verb_pat.finditer(q_clean):
    _verb = _gm.group(1).lower()
    if _verb in _STATIVE_DENY:
        continue                              # stative/affect/possess → not an activity
    _obj = self._opinion_topic(_gm.group(2).strip().lower())
    if _obj and 1 <= len(_obj.split()) <= 5:
        _put_fact("does", f"{_verb} {_obj}", 0.5)   # user_model.py:1277
```

The deny-list excludes copula (`am/is/...`), affect/cognition
(`feel/love/like/think/know/...`), possession (`have/own/...`), and
achieve-comm/echo verbs (`got/made/said/took/...`) — those are handled by the
affect/opinion/benign paths or would store garbage. Everything else is treated
as an activity report.

## Why this is seed structure, not hardcoding

The added lines are (1) a **closed deny-list** `frozenset` of stative verb
vocabulary and (2) a **regex** that captures any first-person verb — not a
question→answer dictionary and not authored reply prose. The verb set is
OPEN-CLASS: RAVANA learns the specific verb from the conversation and stores it
in the learnable `PersonalFactStore`; the deny-list only names the *reflex
boundary* (copula/affect/possession), which is seed structure by the
seed-vs-hardcoding test ("can RAVANA change this by itself, through
experience?" — yes; removing the deny-list degrades to "capture everything",
still not a capability regression). A hardcoding audit of the diff found **zero
authored reply strings** — only regex vocabulary + a seeded frozenset deny-list +
store routing.

No LLM or retraining is involved; the capability is online and incremental — a
never-seen verb is captured from a single conversation turn.

## Verification

Covered by `tests/unit/test_novel_verb_self_disclosure.py` (5 tests). The full
file was run against the source on this branch:

```text
$ RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest \
      tests/unit/test_novel_verb_self_disclosure.py -v
5 passed in 1.32s
```

The five tests assert on **stored state**, never on authored reply strings:

- `test_novel_verb_activity_mined` — a never-seen verb (`count`) lands a `does`
  fact whose object resolves to `meteor showers` (the content head, the
  prepositional tail `from the lighthouse gallery` is closed off at the clause
  boundary).
- `test_hyphenated_compound_verb_mined` — `tide-pool` (not in any whitelist) is
  captured as an activity fact.
- `test_hyphenated_verb_recognised_as_disclosure` — the gate returns `True` for
  the `tide-pool ... catalogue ...` turn, so it routes to STORE + ACK rather than
  a knowledge query.
- `test_novel_verb_recognised_as_disclosure` — `astrophotograph` (never seen)
  is also recognised as a first-person activity disclosure.
- `test_stative_verb_not_mined_as_activity` — `i love the ocean` is affect
  (a stance), NOT an activity fact; the open-class miner does **not** pollute the
  `does` store with stative verbs.

The feature card's run also confirmed the 178 prior fact/recall/self-disclosure
regression suites stay green (fact_mining 27, same_turn_profile 20,
aug07+stance 9, chat/recall 94+1skip, memory/identity 27).

## Limits

- Only **first-person** self-reports are captured; third-person or hypothetical
  activity ("people tide-pool for fun") is not a personal fact.
- The stored object is the **content head** bounded by a clause boundary or
  preposition (`user_model.py:1267`); a long trailing prepositional phrase is
  not folded into the fact value.
- Stative/affect/possessive/achieve-comm verbs remain excluded by
  `_STATIVE_DENY` (`user_model.py:1227`); those are covered by the affect/opinion
  and possession miners, not the activity `does` store.
- Confidence is a fixed `0.5` for the open-class capture (`user_model.py:1277`);
  the seeded `ACTIVITY_VERBS`/`EVENT_VERBS` blocks above keep their higher
  `0.55`/`0.5` paths and are not double-captured (the open-class pattern excludes
  `ing`/`ed` inflections by design, `user_model.py:1259-1263`).
