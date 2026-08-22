# Capability: meta-identity reflection (answers about *who you are* to RAVANA)

**Status:** shipped (commits `3e9652f`, `f4682bd`, branch
`auto/round-2026-08-15T1537Z`).
**Verified:** regression tests in
`tests/unit/test_round_2026_08_15T1537_meta_identity.py` pass (2/2); live
end-to-end probes reproduced below (real engine output, `dim=64, seed=42,
baby_mode=True`, offline). Hardcoding self-audit clean.

## What it does

When the user asks RAVANA to **reflect on its accumulated model of the user** —
*“do i seem like a real person to you”*, *“what am i to you”*, *“tell me
something true about who i am”*, *“what have you learned about me”* — RAVANA
answers from its **LIVE durable state**, not from a biographical fact lookup
(name/location) and not from an episodic echo of a stored turn.

The reply is composed entirely from runtime stores RAVANA grew autonomously:

- the user's **real name** (`user_model.user_name`), or an honest “i'm still
  learning who you are” when none is known (`engine.py:2997`, `:3014-3017`);
- the **stance count + topics** it has learned (`UserStanceStore.stances`,
  `engine.py:3001-3002`, `:3029-3033`);
- the **fact count** (`PersonalFactStore.facts`, `engine.py:2999`, `:3020-3027`);
- RAVANA's **own identity strength + trend** (`IdentityEngine.state.strength` /
  `get_trend()`, `engine.py:3003-3004`, `:3006-3011`, `:3035-3037`).

Real engine output (fresh persona `corvin`, taught “i love oysters” + “i think
surveillance is wrong”):

```
Q: do i seem like a real person to you
A: i know you as Corvin. and from what you've told me i've picked up 2 stances
   you've shared and 1 facts about your life. you've let me see where you stand
   on things like oysters, surveillance. my own sense of self is still forming —
   my self-coherence sits around 0.25 and is holding steady.
```

The same answer renders for *“what am i to you”*, *“tell me something true about
who i am”*, and *“what have you learned about me”* — because every word of content
(except light connective scaffolding) is read from state, not authored.

**Fail-closed.** The meta detector is the *first* branch in
`_structured_recall` (`engine.py:2389-2398`); it returns `None` when no
meta signal is present, so:

- a plain biographical query — *“what's my name”* — is **not** intercepted; it
  still resolves on its own structured path (`your name is corvin.`);
- a query that matches no meta pattern stays on the normal honest pipeline.

No LLM, no per-topic reply table, no retraining. The user can correct any
fact/stance and the stores merge on correction, so the reflected profile
updates live.

## How it grew from the conversation

The chat round that fed this cycle surfaced, among its residual limitations, a
class of questions that were neither factual (name/location) nor episodic
(repeat what I said) but **metacognitive about the user themselves**. The feature
card (`t_6fca4160`, “Bug 5”) picked it as a concrete capability gap.

**Root cause / prior behavior.** Meta-identity queries were not recognised as a
distinct intent. They fell through to one of two wrong paths:

1. the **episodic cued-recall** path, which echoed a stored turn verbatim
   (e.g. “you told me earlier …”), or
2. an **authored “feeling-real” frame** in `response_gen.py` keyed on the word
   “real” (`{subj} is fuzzy for me...`) — probe-tuned prose that was not derived
   from any runtime store.

**Fix (commit `3e9652f`).** A single regex in `_structured_recall`
(`engine.py:2389-2398`) detects the meta-identity intent and delegates to a new
`_meta_identity_reply` (`engine.py:2985-3039`) that renders the reply from the
LIVE stores listed above. The detection is fail-closed: it returns `None` for
non-meta input, leaving every other path untouched.

**Hardcoding audit (commit `f4682bd`).** The authored probe-tuned
“feeling-real” frame was **deleted** from `response_gen.py`. `_meta_identity_reply`
contains no authored reply prose and no per-topic answer table — only connective
scaffolding (`and`, `you've let me see where you stand on things like`) around
slots that are read from state at call time (name, stance topics, counts, identity
strength/trend). The 2 regression tests assert the deleted phrase (“fuzzy for me”)
does not leak and that an episodic echo (“you told me earlier”) does not appear.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Meta-identity intent detection (regex) | `ravana/src/ravana/chat/engine.py:2389-2398` |
| Fail-closed return for non-meta input | `ravana/src/ravana/chat/engine.py:2397-2398` (returns `None`) |
| State-driven reply builder | `ravana/src/ravana/chat/engine.py:2985-3039` |
| Name read (`user_model.user_name`) | `ravana/src/ravana/chat/engine.py:2997` |
| Fact count (`PersonalFactStore.facts`) | `ravana/src/ravana/chat/engine.py:2999` |
| Stance count + topics (`UserStanceStore.stances`) | `ravana/src/ravana/chat/engine.py:3001-3002`, `:3029-3033` |
| Identity strength / trend (`IdentityEngine`) | `ravana/src/ravana/chat/engine.py:3003-3004`, `:3006-3011`, `:3035-3037` |
| Deleted authored “feeling-real” frame | `response_gen.py` (removed in `f4682bd`; only the historical note remains at `engine.py:2382`) |
| Regression tests | `tests/unit/test_round_2026_08_15T1537_meta_identity.py` |

## Test coverage

`tests/unit/test_round_2026_08_15T1537_meta_identity.py` (2 tests, both pass):

- `test_meta_identity_reads_real_state` — for each of the four meta phrasings,
  asserts the reply references the learned name (`corvin`), references a learned
  stance topic (`oysters`/`surveillance`), does **not** contain the deleted
  “fuzzy for me” phrase, does **not** contain an episodic echo (“you told me
  earlier”), and reflects identity state.
- `test_non_meta_query_fails_closed_in_meta_branch` — a plain *“what's my name”*
  is not intercepted by the meta branch and still resolves from the structured
  name path.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_round_2026_08_15T1537_meta_identity.py -v
```
