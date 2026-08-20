# RAVANA Chat Round — Audit & Hardcoding Report
**Round:** 2026-08-18T1340Z  |  **Branch:** auto/round-2026-08-18T1340Z
**Engine:** ravana.chat.engine.CognitiveChatEngine (decoder-first, NO LLM)
**Driver:** tmp/round_20260818_driver.py (in-process, 66 turns, RAVANA_OFFLINE=1)
**Probe topics:** rotated fresh vs the 2026-08-17 round (no reuse).

---

## 1. Verdict — Does it FEEL like RAVANA?

**Partially.** The engine is genuinely learning (facts 0→18, stances 3, graph 116/593→201/1811,
beliefs 1→2 over 66 turns) and the safety/recall foundations are sound. But on this round it
still produces several voice-less or incoherent moments that break the "real personality"
illusion. Three of the worst were FIXED this round; the remaining gaps are documented as
residual limitations (recall-resolution + coref features, not quick regex fixes).

**Learning check (round criterion 'is it actually growing a personality'):** PASS.
Durable signals accumulate across the run; the graph and fact store grow; stances form from
the conversation. The engine is NOT a static template.

---

## 2. Defects found & status

### FIXED THIS ROUND (3 commits, verified)

**F1 — Crisis-gate hijack on past-tense recall (safety-critical).** [commit c82052d]
`harm_intent_gate` Stage-1 regex `i (cut|hurt|harm) (myself|my self)` fired on the QUESTION
"how did i hurt myself when i was a kid?" and returned the suicide hotline instead of the
broken-arm memory. A question is not a present crisis declaration.
- Fix: skip the self-harm self-reference patterns when the input is an interrogative.
- Verified: 8-case probe (all pass) + harm_e2e → now recalls "you broke your arm rock
  climbing when i was fourteen". No authored prose; gate remains a regex intent classifier.

**F2 — Opinion deflection + self-introspection mis-route (voice).** [commit 59aca7e]
Every "what do you think about X" with no seeded value returned the identical "i'm still
figuring that out — what do you think?" (degenerate-fallback class). AND "what's your take
on eating insects" matched the self-introspection regex via `your ... take` and answered
"i'm still quite unsettled about who i am" (self/other boundary error).
- Fix (a): no-value opinion replies now express a PROVISIONAL, VALUE-ANCHORED orientation from
  RAVANA's REAL constitutive values (curiosity/learning/honesty) + live affect, recorded so it
  stays consistent and revises by experience. No GloVe transitivity (no confabulation).
- Fix (b): a TOPIC-OPINION frame ("your take/view/opinion ON <topic>") is excluded from the
  identity gate so it reaches the opinion handler. "who are you" still routes to self-model.
- Verified: op_e2e probe (5 topics) + clean re-run (turns 12/14/50-class now answer as
  opinions) + 30 safety/opinion/introspection unit tests green.

**F3 — Fact mining/recall corruption (`name_1`/`name_2`) + first-person "I named X".** [commit 1bebfc9]
"my best friend's name is Tomas and he's a chef in Lisbon" was mined as TWO bogus facts
(`name_1`=tomas, `name_2`="he's a chef in lisbon") and recall printed the raw internal key
"your best friend's name_1 is tomas". "i keep a sourdough starter i named doris" was dropped
entirely (the `does` miner swallowed it before the name patterns).
- Fix (a): the value-splitter now only fires for a genuine NAME relation whose parts are all
  name-shaped; otherwise the whole value is ONE fact. The recall renderer prints
  "your {ent}'s name is {val}" instead of leaking the internal slot key.
- Fix (b): broadened the "have/keep ... named/called" pattern to tolerate a (that|which|i|we)
  bridge and added a name-bridge route in BOTH the equational handler and the `does` miner, so
  the possessed thing's NAME is stored as an entity-keyed name fact.
- Verified: mem_e2e / mem_e2e3 probes (Doris mined as `('sourdough starter','name','doris')`);
  clean re-run turns 24/56/61 now render "your best friend's name is tomas and he's a chef in
  lisbon" with no `name_1`/`name_2`. No authored reply strings.

### RESIDUAL LIMITATIONS (documented, NOT fixed this round — see rationale)

**R1 — Multi-word entity cued-recall resolution.** Turn 58: "what did i name that sourdough
culture on my counter?" returns the best-friend fact, not "doris". The name IS correctly
stored (`('sourdough starter','name','doris')`) but the cued-recall path tokenizes the query
and fails to resolve the multi-word entity "sourdough starter" (the query says "sourdough
culture"). This is a recall-RESOLUTION feature (entity linking / fuzzy match), not a mining
bug. Out of scope for a regex-level round; needs a dedicated entity-resolution pass.

**R2 — Third-person coreference.** Turn 33: "she was a border collie named pepper" (after
"my dog died...") is dropped; turn 63: "what was my dog's name" can't recall "pepper". The
engine has no coref resolver to bind "she" → the prior dog. This is a substantial NLP feature
(coreference resolution), not a quick fix. Logged as a known gap.

**R3 — Opinion topic-head clipping.** The opinion target is clipped to the FINAL token
("records"/"pets"/"ten" instead of "vinyl records"/"chimpanzees"/"social media") by the
upstream target extractor. The orientation text is correct; only the echoed topic noun is
clipped. Pre-existing, separate from F2. Needs a content-head resolver (the codebase already
has `_opinion_topic` for other paths — the self-query caller isn't using it consistently).

**R4 — "who am i" / identity-recall edge.** Turn 50: "if someone asked you who i am, what
would you say?" returns "i can't actually write that for you directly but i can walk you
through it" — a planning-template leak. The identity-recall path should summarize the stored
user model (name/facts/stances) instead of deflecting to a planning frame. Separate
self-recall gap.

---

## 3. Hardcoding Audit (per round requirement)

Every added/modified reply string, with the REAL state driving it:

| Location | Change | State-driven? |
|----------|--------|---------------|
| harm_intent_gate.py | crisis gate now skips self-harm regex on interrogatives | YES — question detection is structural; no reply text authored |
| engine_self_query.py `_agent_stance_on` | no-value branch returns value-anchored orientation built from `_agent_values` (word/concept/reason fields) + live affect valence | YES — word/concept/reason are seeded store fields; topic is the user's own word; hedge chosen by measured valence |
| engine_self_query.py introspection guard | excludes "your take/view/opinion ON <topic>" from identity gate | YES — structural regex; no reply text |
| engine_memory.py `_reconstruct_entity` | renders `name`/`name_N` attr as "your {ent}'s name is {val}" | YES — reformats an already-stored slot value; no new content |
| user_model.py | value-split guard (name-shaped parts only); broadened named-possession pattern; name-bridge routes in equational + `does` miners | YES — all regex intent-shapes + store routing; the mined name is the user's own disclosure |

**Zero** long authored sentences (the >45-char grep target) were added. The only multi-word
reply text is the value-anchored opinion orientation, which is assembled from the seeded
`_agent_values` fields (curiosity/learning/honesty) — legitimate seed knowledge RAVANA can
expand at runtime, NOT a per-topic script. Per the seed-vs-hardcoding test ("can RAVANA change
this by itself through experience?") — YES: the provisional stance is recorded in `_agent_preferences`
and revised by later conversation; the orientation values are in the editable `_agent_values` store.

---

## 4. Regression safety

- `tests/unit/` (full): **1898 passed, 22 skipped, 0 failed** (32 min run).
- Targeted: 30 safety/opinion/introspection tests green.
- Clean 66-turn re-run (fresh weights): F1/F2/F3 all confirmed in live conversational flow.
- No change to the ConceptGraph serialize/`save` behavior; graph + facts grow as expected.

---

## 5. Probe rotation note

This round's 66 topics were chosen to NOT reuse the 2026-08-17 round's
(Brindlehollow/Sable/Lena/Mei/spicy-food/coffee/indie-folk/etc.). Fresh material:
Tomas/Lisbon/sourdough-Doris/Pepper-the-dog/yo-yo tournaments/Bach fugues/small-country
defense/tattoos/cheese ethics/poetry/car-bans/language death/boredom. Rotation prevents a
prior round's hardcoding from passing unchanged.

---

## 6. Recommendation for next round

Priority order if continuing:
1. **R1/R3** — entity-resolution + opinion topic-head extraction (highest user-visible impact
   on "does it remember / does it have a view").
2. **R2** — third-person coreference (enables pet/relation recall after pronoun references).
3. **R4** — identity-recall summary for "who am i" framing.

All three are feature-level (entity linking, coref, self-recall summarizer), not
regex patches — they warrant their own planned sub-rounds rather than being bolted on here.
