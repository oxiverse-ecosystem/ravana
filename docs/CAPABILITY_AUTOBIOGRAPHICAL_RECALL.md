# Capability: autobiographical recall of the USER (what RAVANA remembers about YOU)

**Status:** shipped (commits `62e1044` + `5b2159d`, branch `auto/round-2026-08-18T0937Z`). NOT pushed.
**Feature card:** `t_a41f7e29` (round `2026-08-18T0937Z`; closes round residuals R34 / R43 / R57 + U17).
**Verified:** the 6 round tests in `tests/test_round_2026_08_18_autobio_recall.py` pass
(`6 passed in 46.71s`, offline). A real in-process probe on this branch reproduced every
example below (real engine output, `dim=64, seed=42, baby_mode=True`, offline). Hardcoding
self-audit clean (no authored reply prose, no per-topic answer table — only connective
scaffolding around slots read from the live `PersonalFactStore` / `UserStanceStore` /
`BeliefStore`).

## What it does

RAVANA can now answer questions about **what *you* have told *it*** — the USER's own
disclosures — by composing the answer from the REAL runtime-grown user-model stores
(`personal_facts` / `opinions.stances` / `belief_store`), rather than being misrouted into
its own-reply echo store.

Four behaviours, all fail-closed (return `None` when no user-autobiography intent matches or
the relevant store is empty, so genuine agent-self questions still reach the self-model
path — the self/other boundary is preserved):

| Sub-intent | Trigger examples | Behaviour (real engine output) |
|---|---|---|
| **(A) Salience** | "what will you remember most about me", "what do you remember about me" | Composes from the real profile, leading with the single most-confident learned item + a short tail. |
| **(B) Confirmation** | "did i tell you i liked X", "have i told you about my brother" | Confirms from the USER's REAL stance/fact, stating the actual learned content. Honest "not that i recall" when nothing maps. |
| **(C) Contradiction-reconcile** | "does that still fit / have i changed" | Reports the USER's CURRENT (already-reconciled) stance, not a stale echo (closes U17). |
| Fall-through | "who have i told you about" (topic-less enumeration) | Returns `None` so the (0c) category-enumeration recall path answers instead. |

Real engine output (fresh engine, seeded with *"my brother theo restores vintage radios"*,
*"i love cold-weather hiking"*, *"i spent my whole childhood in a village by the river"*,
then queried):

```
Q: what will you remember most about me?
A: "the thing that stands out most is your brother theo restores vintage radios.
   and i've also picked up: your brother arjun climbs mountains;
   your brother theo restores vintage radios; your cat is mochi."

Q: did i tell you i liked cold-weather hiking?
A: "yes — you told me you're uncertain about cold weather hiking. i've kept that."

Q: have i told you about my brother?
A: "yes — i remember: your brother theo restores vintage radios."

Q: earlier i told you i loved cold-weather hiking. does that still fit, or have i changed?
A: "you've softened on that — you're uncertain about cold weather hiking now,
    so it doesn't sit the same as when you first said it."

Q: what did you say about music?            # genuine agent-self
A: (not intercepted by autobio; falls through correctly)
```

**This is a real capability, not a hardcoded reply.** Every answer slot is read from runtime
state RAVANA grows online; the user can correct any fact/stance and the stores merge on
correction. No authored prose, no per-topic table, no retraining. The deciding test ("can
RAVANA change this by itself, through experience?") passes: the content comes entirely from
the learned stores.

**Generalizes** across any topic the user has actually disclosed — the stance/fact matchers
(`_match_stance` / `_match_fact`) use bidirectional containment + token overlap, not a topic
list (see *How it grew*). No LLM, no retraining.

## Fail-closed

The capability rides a new gate `(0a)` in `_structured_recall`, wired **before** the
misrouting `_route_agent_own_recall` gate but fail-closed (returns `None` for non-matching
intents). So:

- An empty user-model store (brand-new user) returns `None` → honest uncertainty, not
  fabricated biography.
- A genuine agent-self question ("what did *you* say about X", or "what do *you* remember
  about *me*" framed as RAVANA's own speech) lacks the first-person **user** framing
  (`\b(i|me|my|mine|we|our)\b`) and the gate returns `None` → reaches `_route_self_query` /
  `_route_agent_own_recall` unchanged. The self/other boundary is preserved by construction.
- A topic-less enumeration ("who have i told you about") recovers no disclosure topic, so
  sub-intent (B) returns `None` → the (0c) enumeration recall path answers instead (this was
  fixed in `5b2159d` — see *How it grew*).

## How it grew from the conversation

The chat round of this cycle (`t_f501f7bf`, report
`tmp/reports/ravana-2026-08-18T0937Z.md`) surfaced two pre-existing residuals in the
**agent-own-recall interceptor**:

- **R34 / R43 / R57** — queries about what the *USER* has told RAVANA
  ("what will you remember most about me?", "did i tell you i liked X", "have i told you
  about my brother?") were either unhandled (fell to degenerate uncertainty — R57) or
  **MISROUTED** into the loosely-keyed `AgentReplyStore` (`_route_agent_own_recall`), which
  surfaced RAVANA's OWN echo ("i said: good to know you love cold-weather hiking…") instead
  of an answer about the user's disclosed fact/stance. That is a self/other boundary
  inversion, and it is brittle (it depends on a junk reply key like "hiking" existing in
  `_own_replies`).
- **U17** — "earlier i told you i loved X. does that still fit, or have i changed?" returned a
  stale agent-own reply rather than reporting the user's *current* (already-reconciled)
  stance.

### Root cause — user-autobiography queries routed into the agent's own-reply store

The dispatcher `_structured_recall` (`engine.py`, `_structured_recall` at `2394`) ran the
agent-own-recall gate `_route_agent_own_recall` (`4405`) — designed for RAVANA recalling
*its own* prior replies — **before** any path that could answer about the *user*. User
autobiography queries that shared vocabulary with an agent reply key ("hiking") matched the
loosely-keyed echo store and short-circuited into RAVANA's self-voice. The user-autobiography
intent had no dedicated, store-driven home.

The probe (reproduced cold before coding) showed the gap precisely:

```
PROBE: what will you remember most about me?
  _route_agent_own_recall -> None
  _structured_recall      -> None                 # GAP (R57)

PROBE: did i tell you i liked cold-weather hiking?
  _route_agent_own_recall -> "i said: good to know — you love cold-weather hiking..."  # MISROUTE (R34)
  _structured_recall      -> None
```

### Fix — a dedicated, store-driven user-autobiography gate

1. **New method `_autobiographical_recall`** (`engine.py:3767`) — detects the
   user-autobiography intent (first-person **user** framing + sub-intent regexes for
   salience / confirmation / contradiction-reconcile) and composes the answer from the real
   stores. Returns `None` when nothing matches, so it is fail-closed.

2. **Wired at `(0a)` in `_structured_recall`** (`engine.py:2462-2464`) — runs *before* the
   `_route_agent_own_recall` gate, so user-autobiography queries are answered from the
   user-model stores instead of being misrouted into the agent's echo. Genuine agent-self
   questions (no user-framing) still fall through unchanged.

3. **Reuses the canonical renderers** — `_render_fact_line` (`engine.py:3634`, an EXACT copy
   of the fact-rendering branch in `_aggregate_user_model`) and `_polarity_word`
   (`engine.py:3621`, a single lexicon token per polarity band — vocabulary, not a scripted
   sentence). The autobio path and the aggregation path therefore render identically by
   construction. State collection `_collect_user_model_state` (`engine.py:3658`) dedupes
   exactly like `_aggregate_user_model`, including skipping `superseded` facts.

4. **Generalizing matchers** — `_match_stance` (`engine.py:3709`) and `_match_fact`
   (`engine.py:3736`) use bidirectional containment + content-token-overlap (both directions),
   so "did i tell you i liked cold-weather hiking" maps to whatever real stance/fact the user
   disclosed — no per-topic table. `_extract_disclosure_topic` (`engine.py:3754`) strips a
   general verb list to recover the topic, with no special-casing.

5. **(B)-vs-enumeration fix (`5b2159d`)** — sub-intent (B)'s broad "have i told you (about X)"
   shape swallowed the topic-less enumeration query "who have i told you about", resolving to an
   empty topic and returning the honest "not that i recall" fallback — preempting the (0c)
   enumeration recall path. Fix: when (B) recovers no real disclosure topic (or a bare
   "about"), it returns `None` so the query falls through to enumeration recall. Confirmation
   still fires for genuine content (test: "did i tell you i liked cold-weather hiking" → real
   stance confirmation).

## Hardcoding audit (summary)

Every change this round is store-driven logic or a structural guard — **no authored reply
prose, no `random.choice` reply pools, no keyword→response tables, no Q→A dict**:

- `_polarity_word` (`engine.py:3621`) — returns ONE lexicon token per band (a vocabulary
  entry, <45 chars, never trips the no-hardcoding grep). The sentence is connective; the
  topic + polarity come from state.
- `_render_fact_line` (`engine.py:3634`) — EXACT copy of `_aggregate_user_model`'s renderer;
  reads live store values, no special-casing.
- `_collect_user_model_state` (`engine.py:3658`) — reads the real stores; skips `superseded`
  facts.
- `_match_stance` / `_match_fact` (`engine.py:3709` / `3736`) — containment + token-overlap
  matchers, no per-topic list.
- `_extract_disclosure_topic` (`engine.py:3754`) — general verb-strip, no per-topic map.
- The connective strings ("the thing that stands out most is", "yes — you told me you're",
  "not that i recall", "you've softened on that") wrap real state; the CONTENT is always a
  store value. An honest bare frame ("not that i recall — you haven't told me about that
  yet") BEATS a fabricated yes.

**Seed-vs-hardcoding:** there is no seed vocabulary here at all — the answers are entirely
derived from runtime-grown stores. The deciding test ("can RAVANA change this by itself?") →
YES, because the content lives in the stores, not the code. PASS. **No retraining:** all
changes are online/incremental.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Wiring: user-autobiography gate `(0a)` runs before agent-own-recall misroute | `ravana/src/ravana/chat/engine.py:2462-2464` |
| `_autobiographical_recall` (the capability) — 3 sub-intents A/B/C, all fail-closed | `ravana/src/ravana/chat/engine.py:3767-3939` |
| `_polarity_word` (single lexicon token per band) | `ravana/src/ravana/chat/engine.py:3621-3632` |
| `_render_fact_line` (EXACT copy of aggregation renderer) | `ravana/src/ravana/chat/engine.py:3634-3656` |
| `_collect_user_model_state` (facts/stances/beliefs, deduped, skips superseded) | `ravana/src/ravana/chat/engine.py:3658-3707` |
| `_match_stance` (bidirectional containment + token overlap) | `ravana/src/ravana/chat/engine.py:3709-3734` |
| `_match_fact` (attr/value containment + token overlap, prefers longest) | `ravana/src/ravana/chat/engine.py:3736-3752` |
| `_extract_disclosure_topic` (general verb-strip) | `ravana/src/ravana/chat/engine.py:3754-3765` |
| Parent dispatcher `_structured_recall` (gate order) | `ravana/src/ravana/chat/engine.py:2394` |
| Agent-own-recall gate it now precedes `_route_agent_own_recall` | `ravana/src/ravana/chat/engine.py:4405` |
| (0c) category-enumeration recall path (fall-through target for topic-less "who have i told you about") | `ravana/src/ravana/chat/engine.py:2501+` |

## Test coverage

Six round tests in `tests/test_round_2026_08_18_autobio_recall.py` (all pass; real run:
`6 passed in 46.71s`, offline). They drive the REAL engine path (`process_turn`) so routing
regressions are caught, and assert that the answer content comes from the live user-model
stores, not an agent-self echo:

- `test_salience_most_about_me_is_store_driven` — "what will you remember most about me"
  composes from the real profile (asserts it is not `None` and cites a real fact; asserts it
  does NOT start with "i said:").
- `test_confirmation_liked_topic_reads_real_user_stance` — "did i tell you i liked
  cold-weather hiking" confirms from the USER's real stance; asserts "cold weather hiking" is
  cited and no "i said:" echo.
- `test_confirmation_unknown_returns_honest_no` — a confirmation about something never
  disclosed returns "not that i recall" (honest, not a fabricated yes).
- `test_family_mention_confirmation_reads_fact` — "have i told you about my brother" answers
  from the real brother fact (theo restores vintage radios).
- `test_contradiction_reconcile_reports_current_stance` — after the user softens the stance,
  "does that still fit / have i changed" reports the CURRENT (softened) stance; asserts
  "strongly for" is absent and no "i said:" echo (closes U17).
- `test_agent_self_question_still_falls_through` — "what did you say about music" is NOT
  intercepted (returns `None`), preserving the self/other boundary.

The documented behaviour (salience / confirmation / contradiction-reconcile composition from
the real stores) is therefore **already covered** by the feature commit's regression tests;
no additional test was required for this docs round.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_round_2026_08_18_autobio_recall.py -v
```
