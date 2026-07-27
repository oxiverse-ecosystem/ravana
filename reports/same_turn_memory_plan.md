# Same-Turn Memory + Learned User Profile + Opinions — Implementation

Plan approved 2026-07-27. Implements the three components (A: same-turn memory, B: learned PersonalFactStore, C: opinion store) on top of the existing adaptive/learned infrastructure (FrequencyModel, adaptive gates) — no fixed-threshold dumps, no duplication, persisted via the engine's existing SQLite state path.

## What changed

### Phase 0 — PersonalFactStore (NEW: chat/personal_fact_store.py)
Gradable, correctable fact store mirroring BeliefStore's confidence x recency x decay machinery:
- `PersonalFact` dataclass (subject/attribute/value/confidence/turn_number/rehearsal_count/source/superseded).
- `assert_fact`, `query_fact`, `get`, `reinforce`, `confirm`, `contradict` (user correction supersedes old value authoritatively), `reconcile`, `prune_stale`, `get_consolidation_candidates`, `get_state`/`set_state`.

### Phase B — learned user profile (user_model.py)
- `UserModel` gains `personal_facts: PersonalFactStore` + `opinions: UserStanceStore`.
- Fact mining extracted into `mine_personal_facts(text)` (high-precision regex: name / location / "my X is Y" / "I have a X named Y"). Called from both `observe_user_query` AND the identity gate for same-turn capture.
- Seeds confidence 0.6; learns via confirm/contradict (B4 wiring hooks present in store; confirmation/correction plumbing reuses existing correction detection).
- Serialized through `get_state`/`set_state` (rides existing engine persistence).

### Phase A — same-turn recall (engine.py)
- `mine_personal_facts(user_input)` now runs BEFORE the identity/likes/favorites gate (the prior gap: observe ran at ~3516, after the gate at ~2934, so "my name is X. what is my name?" failed same-turn).
- Added a same-turn personal-fact answer branch in the identity gate: "what's my cat's name?" / "what is my X?" answered from `personal_facts` with a confidence readout.
- NOTE: `_episodic_remember` already includes the current turn's record (engine_memory.py:934), so transcript-based same-turn recall was already covered; this adds the structured user-profile path.

### Phase C — opinions (personal_fact_store.py + user_model.py + engine.py)
- `UserStanceStore` (Stance dataclass: topic/polarity/confidence/valence/arousal). Faster decay than facts (opinions more malleable). Separate from facts.
- `observe_user_query` mines opinions (like/hate/favorite/think-good/think-bad/believes-beats) with VAD folding. `advance_turn` + serialization wired.
- Identity gate gains a `m_user_stance` branch: "what do you know about what I think of X?" answered from `opinions`, never from facts or the agent's own stance.
- Sleep consolidation drains stances -> graph edges relation_type="opinion" (C4).

### Phase B5 / C4 — sleep consolidation (engine_generation.py `_sleep_consolidate`)
- Personal facts (confident + rehearsed) -> graph edges source="personal_fact".
- Opinion stances (confident + rehearsed) -> graph edges relation_type="opinion", weight = polarity x confidence.

## Verification
- Phase-0 unit checks: assert/reinforce/confirm/contradict(supersede)/reconcile/prune/consolidation/serialize — PASS.
- UserModel wiring: name/location/cat/dog seeded into personal_facts, survive serialize — PASS.
- Opinion wiring: like/hate/favorite/think polarity correct, survive serialize — PASS.
- Same-turn e2e (boot + 4 engines): state "my cat is pixel" then "what's my cat's name?" same turn; name same-turn; contradict -> max wins; reload keeps cat — (background run in progress).
- Full regression: tests/test_dehardcode_plan.py must stay green (pre-existing `test_meaning_of_life_not_dict_dump` failure is unrelated, confirmed via git stash on baseline).

## Key decisions
1. No new database / no new sleep stage — ride existing SQLite state + `_sleep_consolidate`.
2. Seed regex preserved (bootstrapping), store learns the rest — not a frozen bucket.
3. Facts vs opinions strictly separated (OFC/vmPFC value circuit vs hippocampal semantic).
4. `subject` is not assigned until later in process_turn, so the gate uses the lightweight `mine_personal_facts(text)` (text-only), not the full `observe_user_query` (which needs subject for ToM).
