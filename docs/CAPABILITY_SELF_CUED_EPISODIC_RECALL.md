# Capability: self-cued episodic retrieval (retrieval-by-cue, not retrieval-by-act)

**Status:** shipped (commits `ff795982`, `e2fc348f`, `65a61555`, `a084c6d4`,
branch `auto/round-2026-09-25T0748Z`).
**Verified:** live end-to-end probe reproduced below (real engine output,
`dim=64, seed=42, baby_mode=True`, `RAVANA_OFFLINE=1`, isolated
`user_suffix="docprobe_selfcued"`); regression suite
`tests/unit/test_self_cued_episodic_recall.py` — **8 passed in 197.07s**.
Hardcoding audit: the capability adds **no reply prose** — the answer is the
existing `_reconstruct_gist` rendering of the stored record.

## What it does

RAVANA's hippocampal retrieval used to be gated on the **shape of a recall
request** — *"remember what I told you about X"*. A follow-up question about
content the user **already disclosed** carries no recall verb, so the gate never
fired: the turn fell through to `_consult_internal_knowledge` / the agentic
hands, web search ran, and RAVANA answered about a fact **it already held from
the user**. That is a source-monitoring failure (Mitchell & Johnson 2009):
reaching outside before checking its own record.

`_self_cued_episodic` is the hippocampal check that runs **before** the external
hands: retrieval **by cue** rather than by act.

Live probe output (fresh persona, nothing pre-seeded but the two turns below):

```
A(d1): noted — i'll remember your cousin meera restores antique clocks.
A(d2): you mentioned: "my cousin meera restores antique clocks in pune"
strategy: self_cued_episodic
web_evidence: None
```

`web_evidence: None` is the load-bearing part: RAVANA answered **from the user**,
and never reached for the world. A second disclosure is independently
retrievable — asked *"where does kiran repair bicycles"* after *"my brother
kiran repairs bicycles in jaipur"*, the probe returned
`'you mentioned: "my brother kiran repairs bicycles in jaipur"'`.

## The evidence bar (all structural — no topic table, no answer strings)

`_self_cued_episodic` returns `None` (fails closed) unless **every** one of these
holds (`ravana/src/ravana/chat/engine_memory.py:631-769`):

| Condition | Line | Why |
|---|---|---|
| the input is a question | `engine_memory.py:667` (via `_is_question`, `:554`) | a new disclosure is never a recall |
| it is not an ask about RAVANA's **own** stance | `engine_memory.py:684` | the retrieval target is the wrong source — see below |
| the store is non-empty (live transcript, else the durable episodic indexer) | `engine_memory.py:686-697` | post-load recovery, not a separate store |
| the winning record is a **user turn** — not a question, not a recall query, not the query itself | `engine_memory.py:700-711` | a question is not shareable content |
| a content cue survives scaffold stripping | `engine_memory.py:714-719` | `where/does/you/tell` carry no content |
| the cue is **rare** in the live store (df ≤ half the eligible set) | `engine_memory.py:755`, `:761-762` | a word common to every disclosure cannot anchor a recall |
| the record covers a **majority** of the query's cues, stem-matched | `engine_memory.py:756`, `:759-764` | coverage, not vocabulary |

The thresholds are **structural, not tuned constants**: `_need` is a strict
majority of the query's own cues (`len(cues) // 2 + 1`) and `_max_df` is
half the live store, both recomputed per call — so they scale with the store and
with the query.

### Why majority coverage, not a single cue (commit `e2fc348f`)

The first cut anchored on **one** shared word, and the round probe caught the
consequence immediately: after *"i enjoy cooking pasta on weekends"*, the world
question *"what is cooking oil made of?"* matched on the lone word *cooking* and
was answered out of the autobiographical record — confabulation dressed as recall.
Under the majority rule that query is 1/3 covered, and so is *"what tide tables
does meera use"* (meera is covered; the question is about tide tables). Both now
return `None` in the live probe above.

### Why a self-opinion ask is not a recall (commit `a084c6d4`)

*"do you think i hate cold coffee?"* shares **every** content cue with the stored
disclosure *"i hate cold coffee"*, so cue coverage alone matched it and the
capability echoed the user's own words back instead of letting the stance
machinery answer. The retrieval target is RAVANA's belief, not the user's
record. The gate consults the **single shared** `_SELF_OPINION_SHAPE`
(`engine_memory.py:90-102`) — the same constant `_route_self_query._agent_opinion`
matches on (`engine_self_query.py:161`, `:1269`) — so the two routers cannot
drift apart and one steals the other's turns. The same pattern also covers the
user confirming **their own** stance (*"do i like cold coffee"*, matched by the
`do\s+i\s+(like|love|hate|…)` alternative at `engine_memory.py:100`). Both
return `None` in the live probe.

## How it grew from the conversation

The feature card for this cycle (`t_9530dece`) took the residual limitation the
chat round surfaced: **a fact the user had just given us was being re-looked-up
from the world.** The round probe *"where does meera restore clocks"* (after
*"my cousin meera restores antique clocks in pune"*) fired IntentForge web search
and returned marketplace listings for a clock the user had described to RAVANA
directly.

**Root cause.** The only paths into the episodic store were *act-gated*
(explicit recall verbs, entity-index pattern completion). A cued follow-up
question is a memory question by content, not by grammar, so it had no door.

**Fix — placement, then the bar, then the abstentions.** The capability is
inserted in `process_turn` **before** the agentic pre-check and before
`_consult_internal_knowledge` (`engine.py:6648-6674`), so a user-given fact is
answered from the user. On a hit it records the episode and its own reply
(`engine.py:6671-6672`) and sets `_last_strategy = "self_cued_episodic"`. The
three follow-up commits tightened it to what the probe could actually defend:
majority coverage, `None` instead of a quote when the cue cannot be resolved,
and the self-opinion source-monitoring gate.

**Online and incremental.** The store is read live, so a disclosure made this
turn is retrievable on the next one; nothing is retrained and no new subsystem
is added. The answer text is rendered by the pre-existing `_reconstruct_gist`
(`engine_memory.py:1692`) from the stored record — **no reply string is authored
here**. RAVANA can learn a new cue tonight, from one conversation, with no
rebuild.

## Where it lives (with line cites)

| Concern | Location |
|---|---|
| Capability docstring + the whole evidence bar | `ravana/src/ravana/chat/engine_memory.py:631-769` |
| Source-monitoring gate (self-opinion ask) | `engine_memory.py:670-685` |
| Shared self-opinion constant (one definition, two routers) | `engine_memory.py:90-102`; imported at `engine_self_query.py:161`, used at `:1269` |
| Closed-class recall-scaffold cue vocabulary | `engine_memory.py:110-121` |
| Post-load store recovery (durable episodic indexer) | `engine_memory.py:686-697` |
| User-turn eligibility filter | `engine_memory.py:700-711` |
| Majority coverage + rarity (idf-like) computation | `engine_memory.py:743-765` |
| Answer rendering (pre-existing, not authored here) | `_reconstruct_gist`, `engine_memory.py:1692` |
| Call site — runs BEFORE the agentic hands / internal KB | `ravana/src/ravana/chat/engine.py:6648-6674` (call at `:6660`) |
| Regression tests | `tests/unit/test_self_cued_episodic_recall.py` |

## Test coverage

`tests/unit/test_self_cued_episodic_recall.py` — **8 passed in 197.07s**
(`RAVANA_OFFLINE=1`, `.venv-real/Scripts/python.exe -m pytest … -q
-p no:cacheprovider`). Each test gets a fresh engine with a cleared
`user_suffix` pickle, so no result depends on ordering or a prior worker's
leftover state.

- `test_question_about_disclosed_entity_is_answered_from_own_record` — the
  capability fires and the answer carries the user's stored content.
- `test_full_turn_routes_to_self_cued_strategy_and_skips_web` — end-to-end:
  `_last_strategy == "self_cued_episodic"` and `_pending_web_evidence is None`,
  i.e. the web was **not** reached.
- `test_world_query_with_no_stored_cue_fails_closed` — *"what is the capital of
  france"* → `None`.
- `test_second_disclosure_is_independently_recallable` — generalization beyond
  the first disclosure, and a scaffolding-only query → `None`.
- `test_world_question_sharing_one_word_is_not_recalled` — the exact
  regression the single-cue first cut caused (*"cooking"*).
- `test_partial_overlap_world_question_is_not_recalled` — coverage, not
  vocabulary (*"what tide tables does meera use"*).
- `test_self_opinion_question_is_not_answered_from_the_record` — the
  self-opinion source-monitoring gate.
- `test_user_stance_confirmation_is_not_answered_from_the_record` — the
  `do i <stance-verb>` alternative.

Reproduce:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_self_cued_episodic_recall.py -q
```

## Honest limitations

- The cue vocabulary is a **closed-class** English set (`_RECALL_SCAFFOLD_CUES`,
  `engine_memory.py:110-121`). It is a scaffolding class, not a topic table, but
  it is **not** extended at runtime — an unusual content word that doubles as
  scaffolding would be over-stripped.
- Stemming uses NLTK's `PorterStemmer` when importable and degrades to exact
  match when not (`engine_memory.py:720-724`); morphology variation coverage
  therefore depends on that optional import.
- The answer is a **quotation of the stored turn**, not a paraphrase. That is
  honest and source-preserving, but it is not a composed reply.
- The capability answers a question **about** the disclosure. It does not update
  a fact when the user's follow-up *corrects* the disclosure — that stays with
  the fact-mining and supersession paths.
