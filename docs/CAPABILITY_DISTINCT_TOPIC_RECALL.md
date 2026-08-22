# Capability: distinct-topic gist recall returns ONLY the asked episode (no sibling echo)

**Status:** shipped (commit `4e9aa23`, branch `auto/round-2026-08-21T2156Z`, feature `t_c336b8d6`).
**Verified:** `tests/unit/test_d7_d9_topic_recall.py` PASSES (5/5, ran live this cycle, 45.55s). Broader recall/confab/memory regression suite green (112 passed, 0 failed — ran live this cycle). Hardcoding self-audit clean: zero authored reply strings, no Q→A dict, no retraining; the routing is structural (a detector + ceding to the existing scoped retriever); lexicons (`_WHOLE_PROFILE_FRAMES`, the stop-word/attribute sets) are SEED structure RAVANA grows at runtime.

## What it does

A single-topic recollection — *"what did i tell you about sourdough?"*, *"remind me what i said about the telescope"*, *"do you remember what i mentioned about my sourdough?"* — now returns **only the one episode the user asked about**, and never bleeds in sibling disclosures.

- Before the fix, that same query shape was routed into the **generic whole-profile dump**, which reconstructed a gist from the aggregate of *every* disclosed fact. The asked gist came back **plus** unrelated siblings (*"... you took glassblowing last winter; you watch rings; you bought small telescope; you bake sourdough; ..."*) — the D7/D9 wrong/unrelated-turn echo.
- After the fix, the query cedes to the precise scoped semantic retriever over the user's own disclosure transcript, so **only the asked gist is reconstructed**. The generic aggregate summary is reserved for true whole-profile frames.

The behavior is **fail-closed and consistent**: a topic the user never disclosed (e.g. *"what did i tell you about origami?"*) returns no sibling standing in for the answer; a genuine whole-profile query (*"what do you remember about me?"*) still aggregates ≥ 2 disclosed topics; and a biographical-attribute recall (*"where do i keep the bees?"*) keeps its precise entity path.

No LLM, no Q→A dict, no keyword→reply table. The capability is a **structural routing decision** in the live recall path (detect the distinct-topic shape → cede to the existing scoped retriever that already reconstructs gists from real stored turns), satisfying the seed + online-learning constraints.

## How it grew from the conversation

The chat round `t_b5bae68f` logged a residual limitation: RAVANA occasionally returns a **sibling** episode when asked about one specific topic (the D7/D9 class — rare episodic echoes of wrong/unrelated turns on some recall paths). The defect is *generalizable* (not a one-off typo): a distinct-topic gist recall was indistinguishable from a true whole-profile recall inside the recall router.

**Root cause / prior behavior.** `_try_memory_query` (the self-recall branch of `engine_memory.py`) unified a **distinct-topic** gist recall (*"what did i tell you about sourdough?"*) with a **true whole-profile** recall (*"what do you know about me?"*). Both fell into the generic self-recall branch that builds the aggregate summary from the `_episodic_index` + the `PersonalFactStore`. For a single-topic query whose topic lived **only in the transcript gist** (not the entity index — it didn't match the `my X` / `i love X` mining shapes that populate `_episodic_index`), the aggregate dump returned the asked gist **plus** every sibling disclosure. A same-shape query (*"about the marathon"*) instead **failed closed** — so the behavior was inconsistent, which is itself the defect.

**Fix (commit `4e9aa23`).** Three pieces, all in `engine_memory.py`:

1. `_is_distinct_topic_recall(q)` (`engine_memory.py:1346-1377`) — a **structural detector**, not a per-topic list. It requires: (a) a recollection speech-act word (`remember`/`recall`/`remind`/`told`/`said`/`mentioned`/...), (b) that the query NOT be a canonical whole-profile frame (`_WHOLE_PROFILE_FRAMES`, `engine_memory.py:1338-1344`), and (c) that after stripping a stop-word/attribute lexicon a **genuine topic noun remains** (e.g. `sourdough`, `marathon`, `glassblowing`). A pure self-pronoun or biographical-attribute query has no remaining noun → it is NOT a distinct-topic recall, so it stays on its correct path.

2. `_scoped_topic_transcript(q)` (`engine_memory.py:1379-1404`) — returns the **user's own disclosure transcript** verbatim so the scoped retriever cannot cross-contaminate with a sibling. When the live `_episodic_transcript` is empty (post-load, the transcript is not persisted), it rebuilds from the durable hippocampal indexer the **same way `_retrieve_episodic` already does**, so recall stays correct after a reload. Returns `None` only when there is truly no disclosure history (caller then fails closed).

3. **Routing** (`engine_memory.py:1830-1836`) — inside the self-recall branch, a distinct-topic recall now cedes to the precise scoped retriever:
   ```python
   if self._is_distinct_topic_recall(user_input):
       _scope = self._scoped_topic_transcript(user_input)
       if _scope is not None:
           _topic_ep = self._retrieve_episodic(user_input, transcript=_scope)
           if _topic_ep is not None:
               return _topic_ep
   ```
   `_retrieve_episodic` already accepts a `transcript=` argument (`engine_memory.py:438-439`) to restrict the search set; the fix simply passes the scoped transcript so only the asked episode matches. If the scoped retriever misses, it **falls through** to the generic summary below (fail-closed, so a genuine whole-profile query still answers).

**Generalization.** The detector is topic-agnostic (it keys on the *shape* of a distinct-topic recollection, not on a list of topic words), so it generalizes to any rotated single-topic phrasing the user tries (*"remind me what i said about the telescope"*, *"what was it i told you about the marathon i ran?"*) with no per-topic code. There is no retraining and no new reply authoring.

**Hardcoding audit.** The diff adds zero reply strings. Grep for long added strings in the changed region returns only the docstring/comment blocks explaining the detector and routing. The added surface is: a structural boolean detector (`_is_distinct_topic_recall`) + a transcript-scope helper (`_scoped_topic_transcript`) + one `if` that cedes to the pre-existing scoped retriever. There is no Q→A dictionary and no keyword→reply table. Replies render REAL stored state via `_retrieve_episodic` / `_reconstruct_gist`. The stop-word/attribute lexicons are SEED structure RAVANA grows at runtime (the same shape the doctrine permits): removing entries degrades gracefully (one fewer shape recognized).

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_WHOLE_PROFILE_FRAMES` (canonical whole-profile frames, excluded from distinct-topic detection) | `ravana/src/ravana/chat/engine_memory.py:1338-1344` |
| `_is_distinct_topic_recall(q)` (structural detector) | `ravana/src/ravana/chat/engine_memory.py:1346-1377` |
| `_scoped_topic_transcript(q)` (returns user's own disclosure transcript, rebuilds post-load from the hippocampal indexer) | `ravana/src/ravana/chat/engine_memory.py:1379-1404` |
| `_retrieve_episodic(query, transcript=None)` (the scoped retriever the fix cedes to; `transcript=` restricts the search set) | `ravana/src/ravana/chat/engine_memory.py:438-439` (body reconstructs gist / fails closed) |
| Routing: distinct-topic recall cedes to the scoped retriever, fail-closed to the generic summary | `ravana/src/ravana/chat/engine_memory.py:1830-1836` |
| Regression + capability test | `tests/unit/test_d7_d9_topic_recall.py` (5 checks) |

## Test coverage

`tests/unit/test_d7_d9_topic_recall.py` (verified **5 passed, 45.55s**, ran live this cycle). The 5 checks:

- `test_distinct_topic_recall_returns_only_asked_episode` — feeds 4 semantically-separated disclosures (glassblowing / telescope / sourdough / marathon), then probes each topic; asserts the asked gist is present and **no sibling topic's distinctive word appears**.
- `test_whole_profile_recall_still_summarizes` — a generic self-recall (*"what do you remember about me?"*) still surfaces **≥ 2** distinct disclosed topics (proving the aggregate path is preserved, not collapsed to one episode).
- `test_distinct_topic_recall_beats_generic_summary` — a single-topic query resolves to the one asked gist and does **not** fall through to the generic summary shape (*"i've picked up"* / *"stands out"*).
- `test_topic_recall_without_entity_does_not_confabulate_unknown` — a never-disclosed topic (*origami*) fails closed; **no** disclosed sibling stands in as the answer.
- `test_biographical_attribute_recall_unaffected` — a location/name attribute recall keeps its precise entity path (*"six hives"* present; user name *not* leaked).

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_d7_d9_topic_recall.py -q
```

Broader regression check (ran live this cycle, **112 passed / 0 failed**, 173.62s) across the recall/confabulation/memory suite: `test_d7_d9_topic_recall.py`, `test_r1_entity_link_recall.py`, `test_recall_confabulation_2026g.py`, `test_round_2026_08_19_d1_recall.py`, `test_memory_architecture.py`, `test_memory_reconstructor.py`, `test_episode_injector.py`, `test_grace_memory_sleep_state.py`, `test_working_memory_v2.py`. The suite exercises the real miner → store → recall path, so the D7/D9 test fails on pre-fix code and passes after the fix (RED→GREEN).
