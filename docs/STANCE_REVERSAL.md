# Stance Reversal — recoding a held attitude when the user changes their mind

RAVANA does not only *form* stances; it also *re-codes* them when the user
retracts or reverses a position it already holds. This page documents that
capability: what it does, how it is implemented, how it grew out of a real
conversational gap, and how it is verified.

All claims below were checked against the source on branch
`auto/round-2026-08-09T1953Z` at commit `0628ef1`. Line numbers cite that tree.

## What it does

When the user expresses a **first-person reversal** of an attitude RAVANA already
holds a stance on — "i flipped, the reef tank is more work than joy", "i recant,
veganism was a phase", "i've had a change of heart, the transit plan is a
mistake" — the stored stance is **recalibrated toward the opposite pole** rather
than leaving a stale positive stance behind or spawning a second, contradictory
one.

Concretely, for a seed stance `reef tank` at polarity +0.95:

```text
turn 1: "i love my reef tank, watching the corals is what i live for"
        → stance 'reef tank' recorded, polarity +0.95, confidence 0.60
turn 2: "i flipped, the reef tank is more work than joy"
        → stance 'reef tank' recoded, polarity -0.665, confidence 0.294
        → opinions.last_reversal == ('reef tank', 0.95, -0.665)
```

These numbers are from a live run of `UserModel` (not a hand-written example):
`mine_personal_facts("i love my reef tank …")` then
`mine_personal_facts("i flipped, the reef tank is more work than joy")` produced
`before: 0.95`, `after: -0.665`,
`last_reversal: ('reef tank', 0.95, -0.665)`.

A reversal is a **valuation recode linked to the prior stance**, not a fresh
opinion merge. The link is preserved so the acknowledgment can reference what was
reversed (see `PersonalFactStore.last_reversal`
`ravana/src/ravana/chat/personal_fact_store.py:261`, set at `:411`).

## How it grew — the residual gap

Round `t_6c023144` (auto cycle 2026-08-09T1953Z) logged a residual limitation: a
free-text reversal like *"i flipped, the reef tank is more work than joy"* formed
a **fresh FOR stance** on `reef tank` instead of recoding the held one.

Root cause: the word `flipped` (and sibling first-person change-of-mind verbs
were absent from the retraction-cue seed set
`_RETRACTION_CUES` (`ravana/src/ravana/chat/user_model.py:93`). With no cue
matching, `mine_stance_reversal` (`user_model.py:1108`) never entered its
reversal path, so the utterance fell through to ordinary opinion mining and
accumulated a second, contradicting stance.

The fix extended `_RETRACTION_CUES` with a SEED set of first-person reversal
speech acts (`user_model.py:116-131`):

```python
# Round t_6c023144 (2026-08-09T1953Z residual): first-person reversal
# speech acts that the round worker saw slip through to a fresh FOR stance.
r"\bi\s+(?:flipped|flip-?flopped|have\s+flipped|'ve\s+flipped)\b",
r"\bi\s+(?:recant|recanted|renounce|renounced|revoked|reversed|reneged)\b",
r"\bi\s+(?:backtracked|went\s+back\s+on|backed\s+off\s+from)\b",
r"\bi\s*'?ve\s+had\s+a\s+change\s+of\s+heart\b",
r"\bi\s+(?:had|have)\s+a\s+change\s+of\s+heart\b",
```

## Mechanism (real code paths)

Reversal detection runs inside `UserModel.mine_personal_facts`
(called at `user_model.py:1069`, which invokes `mine_stance_reversal`). The
resolver:

1. Scans the utterance against `_RETRACTION_CUES` (hard recants) and
   `_SOFTENING_CUES` (`user_model.py:141`, a subset of the former — *relax
   toward neutral*, never invert). A softening idiom anywhere in the utterance
   governs the whole speech act (`user_model.py:1137-1138`).
2. Extracts the **topic** from the clause after the cue
   (`user_model.py:1199-1207`), with fallbacks that resolve a held stance by
   token containment when the tail is empty or non-content-led
   (`user_model.py:1219-1256`).
3. **Bounds false positives** with a scope guard: a recant whose content is a
   *strict subset* of a broader held topic is treated as a narrowing, not a
   reversal, and is rejected (`user_model.py:1257-1343`). This prevents flipping
   "acoustic music" when the user only walked back "acoustic-*only*".
4. Calls `PersonalFactStore.reverse_stance(topic, utterance=text)`
   (`personal_fact_store.py:359`), which:
   - returns `None` (no-op) if no stance is held on the topic
     (`personal_fact_store.py:385-387`) — a flip on something the user never
     stated an attitude about corrupts nothing;
   - blends the polarity toward the opposite pole with strength
     `0.85` (hard) or `min(0.85, 0.5)=0.5` (softening)
     (`personal_fact_store.py:403-405`);
   - drops confidence toward the pivot (attitude change injects uncertainty,
     `personal_fact_store.py:407`);
   - records `(topic, old_polarity, new_polarity)` in `last_reversal`
     (`personal_fact_store.py:411`);
   - is idempotent within a turn via `_reversed_utterance`
     (`personal_fact_store.py:394-410`).

### Why this is seed structure, not hardcoding

The added lines are a small closed set of retraction **verbs/idioms** in a module
level tuple — the same status as the correction/opinion cue lists already in the
file (`user_model.py:88-91`). They name no topic, carry no authored reply, and
RAVANA can extend the set at runtime; the resolver still reads the live stance
store to decide *what* to reverse. This passes the seed-vs-hardcoding test
("can RAVANA change this by itself, through experience?") — it is a reflex, not a
per-topic answer table. A hardcoding audit of the diff found **zero authored
reply strings**.

No LLM or retraining is involved; the capability is online and incremental — a
reversal can be learned from a single conversation turn.

## Verification

Covered by `tests/unit/test_round_2026_08_09T1953_stance_reversal.py`
(4 tests, all passing — `4 passed in 1.70s`):

- `test_flipped_reverses_held_stance` — "i flipped, X…" recodes a held stance on
  X to the opposite pole and records `last_reversal`.
- `test_flipped_without_held_stance_is_noop` — a flip with no held stance creates
  **no** bogus stance (the false-positive bound).
- `test_softening_flip_relaxes_toward_neutral` — "i kind of flipped … it's not
  that bad" relaxes rather than hard-inverts.
- `test_change_of_heart_cue_reverses` — the "change of heart" idiom also reverses
  a held stance.

The surrounding stance/opinion suites stay green (77 tests passed in the feature
card's run).

## Limits

- A flip resolves against the **held** stance store; if the topic can't be linked
  to a held stance by tight token overlap, the recant is ignored rather than
  guessing (scope guard at `user_model.py:1283-1343`).
- Third-person or hypothetical reversals ("people flip on diets") are not
  attitude changes about the user and are not reversed.
- Reversal strength is fixed per cue class (hard `0.85` / soft `0.5`); it is not
  yet calibrated per-user confidence. That is a deliberate simplification, not a
  gap in the mechanism.
