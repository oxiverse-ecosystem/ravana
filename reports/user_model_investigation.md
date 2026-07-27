# Investigation: personal-facts extraction robustness / world-knowledge vs user-model solidity

Audit of the Plan A/B/C implementation against the user's three open questions.
Findings are evidence-based (grep + live e2e). Two real gaps remain; extraction
robustness is now acceptable.

## Q1. Is personal-facts extraction hardcoded / not improved over time / only simple sentences?
PARTIALLY FIXED.
- The miners are STILL fixed regex (user_model.mine_personal_facts: name / location /
  "my X is Y" / "I have a X named Y"; opinions: like/hate/favorite/think-good/bad/
  believes-beats). No LLM, no learning of NEW patterns. So "only simple sentences"
  is still true — a paraphrase like "the cat I own is called Pixel" is missed.
- BUT the STORE is now learned (PersonalFactStore): confidence, reinforce, confirm,
  contradict(supersede), reconcile, decay. So the *storage* improves over time; the
  *extraction* does not learn new shapes. Verdict: extraction is still a frozen regex
  bucket, but it now feeds a learned store instead of dead fields. Acceptable per the
  project's "seed high-precision patterns, learn the rest from behavior" philosophy —
  the missing piece is wiring confirmation/correction (see Q-gap-1).

## Q2. What about opinions?
RESOLVED by Plan C. UserStanceStore captures like/hate/favorite/think-±/believes-beats
with VAD folding; recalled separately from facts; drained to opinion edges. Verified.

## Q3. Is world knowledge solid, unlike the user-model thing?
NO — the user-model is now MORE solid (learned, persisted, same-turn), but world
knowledge is being POLLUTED by user disclosures. This is the real gap.

### Gap 1 — confirmation/correction pipeline is DEAD (B4 incomplete)
- user_model._detect_correction() computes detected_correction / detected_correction_fact
  every turn. grep of engine.py for those flags: 0 readers. Nothing consumes them.
- PersonalFactStore HAS confirm()/contradict() but NO code calls them.
- Evidence: stating "my cat is pixel" then "no, it's Milo" sets detected_correction=True
  but leaves pixel in the store (verified earlier: contradict must be called manually).
- So "the store learns from user behavior" is NOT wired. The learning loop exists but
  is disconnected.

### Gap 2 — user facts LEAK into the world-knowledge graph (source-monitoring failure)
- _ingest_episodic() keys the hippocampal buffer by the ENTITY, not the user: for
  "my cat is pixel" subj = "cat" (first content word after skipping pronouns). Buffer
  stores predicate="is_about", object=<full utterance text>.
- Sleep consolidation drains the buffer -> graph via _ensure_relation(subject, value,
  attribute). Result: a WORLD-graph edge about "cat" from a USER utterance.
- LIVE E2E PROOF: after "my cat is pixel" + "i live in berlin" + sleep:
    buffer_facts_graduated: 2
    personal_facts_graduated: 2   (correct — user store)
    edges touching leaked nodes (pixel/berlin/cat): 11   <-- world graph polluted
  So "the user's cat is Pixel" becomes a (messy) semantic edge about cats, conflating
  "a specific user's pet" with "cats in general". The earlier worry ("world knowledge is
  solid, user-model isn't") is INVERTED: the user-model is now solid, but world knowledge
  is contaminated by user-specific facts.

## Recommended fixes (not yet implemented — awaiting go)
FIX 1 (Gap 1, B4): in process_turn's feedback path, when user_model.detected_correction
  is set, route detected_correction_fact to personal_facts.contradict() (or confirm() on
  a "yes" cue). Reuse the existing _detect_correction output — just close the loop. Low
  risk, high value (makes the learned store actually learn).

FIX 2 (Gap 2, source monitoring): tag user-pronoun disclosures so they do NOT become
  world-graph edges keyed by the entity. Two options:
  (a) LIGHT: in _ingest_episodic, if the utterance is a possessive/self-disclosure
      ("my X is Y" / "i am X"), key the buffer by a dedicated USER node ("i" /
      user_name) instead of the entity, and set source_metadata['is_user_statement']=True.
      The graph drain then skips edges where is_user_statement (or re-keys them to the
      user node), so "my cat is pixel" -> edge (user, has_pet, cat) not (cat, is, pixel).
  (b) FULL: a dedicated user-subgraph in the graph (node "user"/user_name) holding all
      personal facts + opinions, queried separately from world knowledge. Larger change.
  Recommend (a): minimal, reuses source_metadata which already has is_user_statement,
  and the drain can filter on it. Keep personal_facts as the authoritative user store;
  the graph copy (if any) should be user-node-anchored, not entity-anchored.

## Verification status
- Plan A/B/C: tests/test_same_turn_profile.py PASSES; dehardcode suite 21 passed / 1
  pre-existing unrelated failure.
- Gaps 1 & 2: reproduced with live e2e (above), then FIXED and re-verified (below).

## Fixes implemented (same session)
FIX 1 (Gap 1) — correction/confirmation loop wired where it cannot be skipped:
- user_model.mine_personal_facts(): correction cue detector ("no,", "actually",
  "that's wrong", "i said/told you"). When a mined fact's attribute already holds a
  DIFFERENT active value AND the turn is corrective -> personal_facts.contradict()
  (supersede). Plain restatements still reinforce via assert_fact(). Runs at the top
  of process_turn, so no early return can starve it (the lesson from the dead
  _detect_correction flags — which remain untouched for ToM use).
- engine.py: personal-fact recall answers now set _last_pf_recall; a bare
  affirmation next turn ("yes"/"that's right") calls personal_facts.confirm().
- Bonus fix: recall regex captured "cat's" for "what is my cat's name" — possessive
  now normalized to the bare attribute.

FIX 2 (Gap 2) — source monitoring at the hippocampal->neocortex boundary:
- FactTriple gains user_fact: bool (serialized in get_state/set_state, default
  False so old saves load fine).
- _ingest_episodic() flags first-person self-disclosures (my X.../i am/live/have...)
  as user_fact=True when storing to the buffer.
- _sleep_consolidate() drain: user_fact triples are marked consolidated but NEVER
  become world-graph edges. They still graduate through the personal_facts drain
  (tagged personal_fact), and episodic recall still works (buffer untouched).

LIVE E2E RESULT (after fixes):
    buffer_facts_graduated: 0      (was 2 — leak closed)
    user_facts_withheld:    2
    personal_facts_graduated: 2    (correct channel unaffected)
    world graduation:       1      (genuine world fact still graduates)
    "no, my cat is milo" -> "what is my cat's name?" -> "your cat is milo (80%)"
    "where do i live?" -> "you told me you live in berlin."
