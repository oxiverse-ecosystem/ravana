INDEPENDENT AUDIT — RAVANA round t_f04e8f53 (2026-08-08f)
Auditor: card t_cd6396f6 (separate agent, did NOT edit repo source)
Repo: C:\Users\Likhith\Documents\Projects\ravana  Branch: fix/cognition-driven-generation
Commits under audit: ffa80f1, 3e74925 (on top of e34a448)

=====================================================================
VERDICT: PARTIAL — 2 of 4 fixes are broken in the live path; CI green
is misleading (no coverage of the fixes through process_turn).
=====================================================================

--- 1. CI TEST GATE (the round's own claim) ---
Run: RAVANA_OFFLINE=1 pytest tests/unit/  (real venv .venv-real)
Result: 1804 passed, 23 skipped, 0 failed, 1 warning  (848s)
Also: tests/test_dehardcode_plan.py = 22 passed.
=> The round's "0 regressions / 1804 passed" claim REPRODUCES. CI is GREEN.
BUT: see finding #3 — the green suite does not test the round's fixes.

--- 2. HARDCODING SWEEP ---
git diff e34a448..3e74925 | grep long added strings
Hits: only REGEX / VOCABULARY literals (comparative adjectives, dismissive
nouns, closed-class qualifiers) — NO authored reply prose, NO Q->A dict,
NO capability list, NO one-{var} paragraph. This part is CLEAN.
Seed-vs-hardcode verdict: the GRAMMATICAL opinion patterns are legitimate
(generic, topic resolved via _opinion_topic). See #4 for the one false claim.

--- 3. BEHAVIORAL PROBES (independent, fresh engine, seed 42, OFFLINE) ---

FIX A — Opinion mining broaden (comparative/superlative/dismissive): WORKS.
  6/6 of the round's own example utterances formed stances:
  sea +0.7, hand built synths +0.85, graveyards +0.75, cold water +0.75,
  modern music -0.7, best knots +0.7. Topic resolved to real concepts.
  => This fix is genuinely functional.

FIX B — Concession contradiction (ffa80f1): BROKEN in live path.
  The new detector lives in mine_stance_reversal (user_model.py:999), called
  ONLY from observe_user_query. A natural "i thought X but Y" routes through
  process_turn to the self_disclosure strategy and RETURNS EARLY — the miner
  never runs.
  Repro:
    seed: "the sea is a better teacher than any classroom"  -> sea pol 0.7
    direct  mine_stance_reversal(conc)  -> sea 0.7 -> 0.0   (works)
    process_turn(conc)                  -> sea UNCHANGED 0.7, reply:
       "got it — you've changed your mind about mountains now"
    => Fabricated ack about "mountains" (a topic with NO prior stance), and
       the held "sea" stance was never updated.
  Round self-reported sea .70->.59 — UNREPRODUCIBLE: soft reversal in
  personal_fact_store.reverse_stance forces polarity to exactly 0.0
  (blend=min(0.85,0.5)=0.5; polarity_new = old*0.5 + (-old)*0.5 = 0.0).
  The .59 cannot come from this code. Worker tested the function, not routing.
  No test drives process_turn for concessions.

FIX C — Possessive-entity ack (3e74925): PARTIALLY BROKEN + support misfire.
  "my partner's name is Pell"  -> ack "noted — i'll remember your name is pell."
     (entity stored correctly under ('partner','name','pell'), but the ACK
      mis-attributes it to "your" / the user, not partner)
  "my dog is a sheepdog named Cairn" -> REPLY: "i hear you — feeling rough is
     hard, and i'm here for it. what happened?"  Fact stored under ('i','dog',..)
     but the turn was ROUTED INTO EMOTIONAL SUPPORT and never acked — the exact
     "Support-router misfire" defect class in the ravana skill (no distress
     present). This is a regression-mode bug, not a fix.
  Location trim (same commit): WORKS — "about two kilometers offshore" trimmed,
     stored + recalled correctly as "a lighthouse on a rock".

--- 4. SEED-VS-HARDCODE FALSE CLAIM ---
The diff self-certifies the opinion lexicons (comparative adjectives,
dismissive nouns) as "SEED vocabulary RAVANA can extend at runtime." VERIFIED
FALSE: the regex tuples are local literals inside the mining loop
(user_model.py:788) with ZERO runtime mutation anywhere in the file (no
append/extend/learn_*/data-file load). There is no growth path. This matches
the skill's "frozen vocabulary wearing seed's clothing" warning. The
GRAMMATICAL capture is fine; the runtime-extendable claim is not true. Not a
blocker (patterns are generic), but the self-certification is dishonest and
should be corrected.

--- 5. COVERAGE GAP (root cause of B/C shipping broken) ---
Zero tests exercise any of the 4 fixes through process_turn. The round verified
miners via direct function calls. That is why B (broken routing) and C (partial
ack + support misfire) passed self-audit but fail in the live engine.

=====================================================================
FIX CARDS CREATED (parented to this audit):
  t_58d5f3ac  FIX: concession reversal is unreachable in live process_turn path (CRITICAL)
  t_79d3621d  FIX: possessive-entity ack gaps + support-router misfire on disclosures
  t_36e4d975  FIX: add process_turn-level regression tests for round-2026-08f fixes
=====================================================================

RECOMMENDATION: do not count round t_f04e8f53's "all criteria met" as true.
Its opinion-mining broaden (A) and location trim (C-part) are real wins; its
headline concession fix (B) and possessive-ack fix (C) do not work through the
engine and must be re-done per the fix cards. Audit did NOT edit source.
