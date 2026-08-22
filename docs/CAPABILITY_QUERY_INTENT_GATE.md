# Capability: query-intent disambiguation gate (knowledge questions don't echo memory)

**Status:** shipped (commit `69c3622`, branch `auto/round-2026-08-16T0738Z`).
**Verified:** regression tests in `tests/test_round_2026_08g_lim3.py` pass (3/3,
real `CognitiveChatEngine.process_turn` drive, offline). Hardcoding self-audit
clean: zero new reply strings were added — the change is purely a routing gate.

## What it does

RAVANA answers **two very different kinds of question**, and the gate keeps them
from being confused:

1. **Autobiographical-recall queries** — questions about the *user's own
   disclosed life* ("what did you say about cooking?", "do you remember when I
   went to berlin?", "what is wrong with my car?").
2. **General world-knowledge questions** — definitional / encyclopedic / how-does
   questions that happen to share a word with something the user once said
   ("what is cooking oil made of?", "what is a decorator in python?", "how does a
   black hole form?").

Before this gate, a general knowledge question whose a token coincidentally
stem-matched a stored autobiographical fact could be answered by **echoing the
unrelated fact** — e.g. *"what is cooking oil made of?"* answered with *"you told
me earlier: you enjoy cooking pasta on weekends"*. That is a confident
confabulation that violates the RAVANA honest-uncertainty bar.

The fix adds a **query-intent disambiguation gate** so the episodic echo only
fires for questions genuinely about the user's disclosed life. General
knowledge questions fall through to internal-knowledge / web / honest-uncertainty
and never echo a stored life fact.

Real engine behaviour (verified by `tests/test_round_2026_08g_lim3.py`,
`dim=64, seed=42, baby_mode=True, RAVANA_OFFLINE=1`):

- `i enjoy cooking pasta on weekends` → (stored)
  then `what is cooking oil made of?` → **no** `pasta` / `weekends` echoed; the
  turn does **not** take an episodic-echo strategy (it reaches honest
  uncertainty offline).
- `my car's gps is broken and it keeps rebooting` → (stored)
  then `what is wrong with my car?` → **does** recall the stored fact (`gps` /
  `reboot` present; strategy is an episodic-echo one). This is the
  non-regression control — personal-possessive recall is preserved.
- The gate itself: True for `"what did you say about cooking?"`,
  `"do you remember when i went to berlin?"`, `"what is wrong with my car?"`;
  False for `"what is cooking oil made of?"`, `"what is a decorator in python?"`,
  `"how does a black hole form?"`.

**Distribution-driven, not a frozen topic list.** The gate is a small intent
classifier built from (a) explicit recall markers and (b) a personal-possessive
reference to the user's own entity — not a hand-maintained list of subjects — so
it generalizes across every topic and never needs retraining. No LLM, no
per-topic reply table.

**Fail-open by construction.** Any malformed / empty / non-interrogative input
is classified as NON-recall (`False`), so it can never be answered by an
autobiographical echo — it stays on the honest pipeline. And when a general
knowledge question reaches the gate, the code raises a control-flow signal
(`_SkipEpisodicEcho`) that is swallowed by the enclosing `except`, so the turn
**falls through to honest uncertainty** instead of a memory echo. Confident
confabulation is structurally prevented; honest uncertainty is the default
fallback.

## How it grew from the conversation

The chat round that fed this cycle (`t_68216d25`) surfaced, among its residual
limitations, a class the prior rounds had not yet closed: **limitation #3 —
"offline web query echoes memory."** The feature card (`t_a83a7170`) picked it
as the concrete gap to close.

**Root cause / prior behaviour (documented in the commit + regression test).**
The episodic-recall block in `process_turn` called
`_try_hippocampal_retrieval`, whose broad stem-matching pooled any stored fact
whose buffer key stem-matched a question token. So a knowledge question about
`cooking` (oil) matched a stored autobiographical fact keyed under `cooking`
(pasta), and `_phrase_recalled_fact` surfaced that unrelated fact as the answer —
a confident confabulation.

**Fix (commit `69c3622`).**
- A new method `_is_autobiographical_recall_query` (`engine.py:3703-3753`)
  classifies the intent. It returns `False` (fail-open) for anything that is not
  a well-formed interrogative (`engine.py:3725-3729`); `True` on explicit recall
  markers such as *"what did you say / do you remember / tell me about / remember
  when / anything i told you"* (`engine.py:3732-3736`); `True` on a
  personal-possessive reference to the user's own entity combined with a
  life-attribute word (`engine.py:3741-3749`); and `False` for everything else —
  bare world knowledge (`engine.py:3750-3753`).
- The gate is invoked once before the episodic echo (`engine.py:5700`), and
  immediately before the hippocampal retrieval the code raises `_SkipEpisodicEcho`
  when the query is **not** a recall intent (`engine.py:5757-5758`). The enclosing
  `except` swallows it, so the turn falls through.
- A `_SkipEpisodicEcho` control-flow exception class (`engine.py:424-430`)
  documents the signal and its fail-open semantics.

**Hardcoding audit.** Zero new authored reply strings. The change only routes
turns; it adds no sentence, no keyword→reply table, no probe-tuned prose. A
mid-round grep for long added strings would return nothing in the diff for this
capability.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_SkipEpisodicEcho` control-flow signal (fail-open semantics) | `ravana/src/ravana/chat/engine.py:424-430` |
| Intent gate `_is_autobiographical_recall_query` (method) | `ravana/src/ravana/chat/engine.py:3703-3753` |
| Fail-open early return (non-interrogative / empty) | `ravana/src/ravana/chat/engine.py:3725-3729` |
| Explicit recall-marker branch → `True` | `ravana/src/ravana/chat/engine.py:3732-3736` |
| Personal-possessive branch → `True` | `ravana/src/ravana/chat/engine.py:3741-3749` |
| Default (bare world knowledge) → `False` | `ravana/src/ravana/chat/engine.py:3750-3753` |
| Gate invocation before the episodic echo | `ravana/src/ravana/chat/engine.py:5700` |
| Raise `_SkipEpisodicEcho` (skip echo for non-recall) | `ravana/src/ravana/chat/engine.py:5757-5758` |
| Regression tests | `tests/test_round_2026_08g_lim3.py` |

## Test coverage

`tests/test_round_2026_08g_lim3.py` (3 tests, all pass, real engine drive):

- `test_knowledge_question_does_not_echo_unrelated_memory` — seeds an
  autobiographical fact keyed under `cooking`, then asks a `cooking`-worded
  world-knowledge question; asserts the stored fact (`pasta` / `weekends`) is
  **not** surfaced and the strategy is **not** an episodic-echo one.
- `test_personal_possessive_question_still_recalls` — the non-regression
  control: a `my`-possessive question about a disclosed entity still recalls the
  stored fact (strategy is an episodic-echo one, `gps`/`reboot` present).
- `test_gate_function_detects_recall_vs_knowledge` — drives the gate directly on
  six phrasings, asserting True for the three recall intents and False for the
  three general-knowledge questions.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_round_2026_08g_lim3.py -v
```

No additional test was needed: the three tests above cover both the fixed
behaviour (knowledge question → no echo) and the preserved behaviour
(personal-possessive → still recalls), and they assert on real engine strategy /
reply signals rather than on any hardcoded reply string.
