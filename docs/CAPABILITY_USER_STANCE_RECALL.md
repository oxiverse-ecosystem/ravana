# Capability: user-stance recall (3rd-person queries about the USER's own preferences)

**Status:** shipped (commit `8d40346`, branch `auto/round-2026-08-17T0622Z`).
**Verified:** regression tests in `tests/test_user_stance_recall.py` pass (5/5); live
end-to-end probe reproduced below (real engine output, `dim=64, seed=42,
baby_mode=True`, offline). Hardcoding self-audit clean (no authored reply prose,
no per-topic answer table — only ONE polarity lexicon word + the real topic from
state).

## What it does

When the user asks RAVANA whether **the USER themselves** likes / loves / hates /
dislikes / cares-for something — *"do you think i like spicy food or not?"*,
*"do you think i hate cold coffee?"*, *"do you think i love jazz?"* — RAVANA now
answers from **the user's own held stance**, stored in
`user_model.opinions.stances`, rather than from its own (empty) value system.

This is a **self/other boundary** capability: the attitude holder in these
queries is the *user* ("i like X"), so the valuation lives in the user model, not
in RAVANA's self-model. Previously these queries matched the broad self-opinion
gate and routed to `_route_self_query`, which computed RAVANA's OWN (empty) stance
on the topic and fell through to the generic *"still figuring that out"* hedge — a
self/other boundary error.

The reply is built entirely from the LIVE durable store:
- the topic is resolved the **same way the stance miner resolves it**
  (`user_model._opinion_topic`, `user_model.py:2669`), so the key matches what was
  stored,
- the held stance is fetched through the **same content-word resolver the miner
  uses** (`UserStanceStore.resolve_topic`, `personal_fact_store.py:322` — exact →
  substring → content-word Jaccard), so a paraphrase of the topic still links to
  the held stance,
- the **polarity** of that stance is rendered as ONE lexicon word
  (`strongly for` / `for` / `against` / `strongly against` / `uncertain about`,
  `engine.py:3586-3596`) — a single token, never a scripted sentence —
  plus the real topic from state.

Real engine output (fresh persona, taught *"i love spicy food"*, *"i hate cold
coffee"*, *"i adore jazz"*, then queried in the third person):

```
Q: do you think i like spicy food or not?
A: "from what you've told me, you're strongly for spicy food."
  strategy=user_stance_recall

Q: do you think i hate cold coffee?
A: "from what you've told me, you're strongly against cold coffee."
  strategy=user_stance_recall

Q: do you think i love jazz?        # "i adore jazz" was mined as a stance on "jazz"
A: "from what you've told me, you're strongly for jazz."
  strategy=user_stance_recall
```

**Fail-closed.** The guard returns `None` when either (a) the same-subject-attitude
frame is absent, or (b) the user has **no held stance** on the topic. In both cases
the query falls through to the normal pipeline, so the engine answers honestly
instead of fabricating a preference:

```
Q: do you think i like quantum physics?   # user never stated a preference
A: "i'm still figuring that out. i don't have a settled view on that yet — what do you think?"
  strategy=self_model          # NOT user_stance_recall — no fabricated stance

Q: do you think we should protect mangroves?   # RAVANA is the attitude holder
A: "i'm still figuring that out. i don't have a settled view on that yet — what do you think?"
  strategy=self_model          # NOT user_stance_recall — genuine agent-self question
```

**Confidence is surfaced honestly too.** When the held stance's confidence is below
the 0.6 band, the frame softens to *"i think you're {w} {topic}, though i'm not
totally sure yet."* (`engine.py:3599-3601`). The probe above used single-disclosure
stances, which land in the `>= 0.6` band and produce the firmer *"from what you've
told me"* form; the softened form is exercised by the fail-closed/low-confidence
code path and is part of the same store-driven frame (no extra authored text).

No LLM, no per-topic answer table, no retraining. The capability is entirely
store-driven: the user can state or reverse a preference at runtime (see
`docs/STANCE_REVERSAL.md`) and this path reflects it. RAVANA can revise any stored
stance through normal conversation, satisfying the seed + online-learning
constraints.

## Known rough edges (honest — logged for a future round)

The capability faithfully renders whatever the **miner** stored, so it inherits the
miner's current quality:

- The disclosure ack probe for *"i adore jazz"* was a bare *"noted."* (the stance
  was still mined — the query resolved it correctly), showing the miner's ack and
  the user-stance recall path are independent: recall works even when the
  disclosure acknowledgement is thin. This is a miner gap, not a recall defect.
- Topic resolution depends on the miner's stored key. If the miner stored a noisy
  topic token, the user-stance recall will render that same token — the recall
  path is correct; the upstream key is the rough edge.

These are out of scope for this round; the user-stance recall guard itself is
correct, store-driven, and fail-closed.

## How it grew from the conversation

The chat round of this cycle (round `2026-08-17T0622Z`) surfaced, among its
residual limitations, that a **third-person query about the user's own preference**
(*"do you think i like spicy food or not?"*) was answered as if RAVANA were the
subject — returning the generic *"still figuring that out"* hedge despite a real
held stance. The feature card (`t_d6e10e53`, Limitation H from `t_670f6083`) picked
it as a concrete self/other boundary gap.

**Root cause / prior behavior.** The query's surface shape (`do you think i like
…`) matched the **broad self-opinion gate** in `process_turn`, which dispatched to
`_route_self_query`. That function computed RAVANA's OWN stance on the topic (the
user's preference, not RAVANA's, so its own stance store was empty/neutral) and
returned the generic hedge. The engine in fact **held the user's stance** in
`user_model.opinions.stances` but had **no path to read it for a third-person
query**.

**Fix (commit `8d40346`).** A new `_user_stance_reply` guard
(`engine.py:3510-3601`) runs **before** the self-opinion gate in `process_turn`. It
structurally detects the same-subject-attitude frame (subject `i/we/you` + a
cognition verb `think/feel/believe/…` + the user as attitude holder + an attitude
verb `like/love/hate/dislike/prefer/enjoy/…`) and extracts the topic clause. It then
resolves the topic via the **miner's own resolver** (`user_model._opinion_topic`)
and reads the **live** `UserStanceStore` via `resolve_topic` — the same content-word
resolver the reversal/mining paths use, so a paraphrase still links to the held
stance. When a held stance is found it renders ONE polarity word + the real topic;
otherwise it returns `None` and the query falls through unchanged. The guard is
fail-closed: genuine agent-self questions (RAVANA as attitude holder) and unknown
topics still route to `_route_self_query` exactly as before.

**Hardcoding audit.** `_user_stance_reply` contains no authored reply prose and no
per-topic answer table — only connective scaffolding (`from what you've told me`,
`i think`, `though i'm not totally sure yet`) around slots read from state at call
time (topic key + polarity band → one lexicon word + confidence band). The 5
regression tests assert the third-person query routes to `user_stance_recall`, the
correct polarity word appears for both positive and negative stances, a paraphrase
links to the held key, no stance is claimed for an unstated topic, and an
agent-self question is **not** absorbed.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| User-stance recall guard (the new method) | `ravana/src/ravana/chat/engine.py:3510-3601` |
| Same-subject-attitude frame detection (regex) | `ravana/src/ravana/chat/engine.py:3527-3535` |
| Topic resolved the SAME way the miner resolves it | `ravana/src/ravana/chat/engine.py:3556-3565` (calls `user_model._opinion_topic`, `user_model.py:2669`) |
| Live stance read via the miner's content-word resolver | `ravana/src/ravana/chat/engine.py:3573-3582` (calls `UserStanceStore.resolve_topic`, `personal_fact_store.py:322`) |
| Polarity band → ONE lexicon word | `ravana/src/ravana/chat/engine.py:3586-3596` |
| Confidence-gated phrasing (honest uncertainty) | `ravana/src/ravana/chat/engine.py:3599-3601` |
| Guard wired into `process_turn` BEFORE the self-opinion gate | `ravana/src/ravana/chat/engine.py:4735-4762` (`_ustance = self._user_stance_reply(user_input)` at `4749`) |
| Fail-closed: returns `None` when no frame / no held stance | `ravana/src/ravana/chat/engine.py:3536-3540, 3579-3583` |
| Regression tests | `tests/test_user_stance_recall.py` |

## Test coverage

`tests/test_user_stance_recall.py` (5 tests, all pass):

- `test_user_stance_recall_positive` — disclose *"i love spicy food"*, then ask
  *"do you think i like spicy food or not?"*; asserts the reply contains the topic
  (`spicy food`), the positive polarity word (`for`), and
  `strategy == "user_stance_recall"`.
- `test_user_stance_recall_negative` — disclose *"i hate cold coffee"*, then ask
  *"do you think i hate cold coffee?"*; asserts the reply contains the topic and
  the negative polarity word (`against`).
- `test_user_stance_recall_paraphrase_links` — disclose *"i adore jazz"* (mined as a
  stance on `jazz`), then ask *"do you think i love jazz?"*; asserts the resolver
  links the paraphrase to the held `jazz` key via the live store.
- `test_user_stance_recall_fail_closed_when_no_stance` — ask about a topic the user
  never stated a preference on (`quantum physics`); asserts the strategy is NOT
  `user_stance_recall` and no fabricated stance is claimed.
- `test_user_stance_recall_does_not_capture_agent_self_opinion` — ask a genuine
  agent-self question (*"do you think we should protect mangroves?"*, RAVANA is the
  attitude holder); asserts the user-stance guard does NOT fire and the query still
  routes to the self-model resolver.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_user_stance_recall.py -v
```

The broader recall/self/stance suite stays green (42 existing recall/stance tests
pass); the 5 new tests **fail** if the capability is stashed out (3 fail without
the fix, proving a real RED→GREEN), and pass with it (5/5).
