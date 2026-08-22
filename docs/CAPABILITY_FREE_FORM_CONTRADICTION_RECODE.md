# Capability: free-form contradiction recodes a held stance (no retraction keyword)

**Status:** shipped (commit `cf85597`, branch `auto/round-2026-08-20T1229Z`).
**Verified:** `tests/unit/test_freeform_contradiction_recode.py` passes (1 test, 5
checks, RED→GREEN — fails on the pre-fix stash, passes with the change); plus a
live end-to-end in-process probe (real engine output, `dim=64, seed=42,
baby_mode=True`, offline), reproduced below. Hardcoding self-audit clean — the
diff adds **zero authored reply strings** (grep for quoted `>=45`-char literals on
added lines = 0 hits); the recode is store-driven + seed-lexicon, no per-topic
answer table.

## What it does

A contradiction is not always signalled by a **retraction keyword** (*"i take it
all back"*), a **"but" concession frame**, or a **"can't / anymore" limitation
cue**. The user very often just **re-states an attitude that OPPOSES a stance they
already hold**, with none of those shapes:

- *"actually i've gone off winter, the cold gets to me now"*
- *"not all street art is good"*
- *"they wear me out these days"*

Previously none of the reversal branches matched, the opinion miner wrote **no**
fresh stance, and the stale held stance persisted **un-reversed** (the
`2026-08-20T0701Z` contradiction-mining gap — provenance was later bridged for
*keying*, but the reversal itself was still never *detected* for these). Now, when
such a free-form reassessment opposes a held stance, the held stance is
**recalibrated toward the new value** via `recode_stance_toward()`, instead of
leaving a stale positive (or negative) stance behind.

Real engine output (fresh persona, offline probe):

```text
turn 1: "i really love street art, especially big murals on warehouse walls"
        → stance keyed 'street art', polarity +0.95, confidence 0.60

turn 2: "i saw a mural downtown that was just tagged over a local business's
         sign — changed my mind, not all street art is good"
        → stance 'street art' RECODED, polarity -0.275, confidence 0.495
        # held attitude recalibrated toward the opposed restatement

turn 1: "i love the silence of deep winter, it's the only quiet i get"
        → stance keyed 'silence', polarity +0.95, confidence 0.65

turn 2: "actually i've gone off winter, the cold gets to me now"
        → stance 'silence' RECODED, polarity -0.275, confidence 0.528
        # 'winter' bridges to held 'silence' via the provenance resolver
```

(The `winter → silence` case reuses the provenance bridge from
`docs/CAPABILITY_STANCE_PROVENANCE.md`: `resolve_topic("winter")` returns
`"silence"`, so the free-form contradiction is recoded onto the *correct* held
stance even though the user never named `silence` again. One resolver, two
capabilities.)

No LLM, no retraining. The recode is **online and incremental** — a contradiction
can be learned from a single conversation turn. The pre-existing *"you've changed
your mind about X"* acknowledgment renders **from the recoded stance** (its
`last_reversal` field); no authored reply string was added for this feature.

## How it grew from the conversation

The chat round of this cycle (feature `t_ac517b1a`, residual limitation #1, carried
from `2026-08-20T0701Z`) surfaced that a view re-stated with **opposed polarity
but no retraction keyword** left the held stance stale. The provenance bridge
(fixed separately) solved *keying* a reversal onto a differently-keyed held
stance, but the *detection* of the contradiction itself was still missing for
free-form restatements — so the user could contradict themselves and RAVANA would
neither reverse nor even stack a new stance, just stay silent.

**Root cause / prior behavior.** `mine_stance_reversal` only entered its reversal
path on an explicit retraction cue (`_RETRACTION_CUES`), a *but*/belief concession
frame, or a *can't/anymore* limitation cue (the branches above the new one at
`user_model.py:2817`). A bare opposed restatement — *"not all street art is good"*
— matched **none** of those, so the function returned early and the opinion miner
wrote nothing. The held positive stance survived un-reversed.

**Fix (commit `cf85597`).** Two cooperating additions:

1. *Detect the reassessment affect* — a SEED sentiment lexicon of
   reassessment-affect terms and their valence, `_REASSESS_NEG` /
   `_REASSESS_POS` (`user_model.py:619`, `:630`), surfaced through
   `_assess_reversal_polarity(text)` (`user_model.py:641`). It returns a polarity
   in `[-1, 1]` from the strongest matching term, or `None` when the utterance
   carries **no** reassessment signal — so a contradiction is **never guessed from
   neutral wording** (honest no-op). The lexicon names **no topic**; the topic is
   resolved separately (below). It is seed (like the VAD affect lexicon / the R3
   limitation lexicon) and RAVANA can **grow it online** as new reassessment
   phrasings are observed — not an answer table.
2. *Recode the held stance* — a new `recode_stance_toward(topic, new_polarity,
   blend=0.7, utterance=None)` (`personal_fact_store.py:477`). Unlike
   `reverse_stance` (which flips toward the *opposite pole* of the prior value,
   for explicit retractions), this is the **delta-rule update for a free-form
   contradiction**: the stance is moved toward the user's *newly-stated value* with
   a decisive `0.7` blend (a weighted merge alone is too weak for an opposed
   restatement), and confidence relaxes toward the new read (the contradiction
   injects uncertainty about the prior value). It sets `last_reversal` so the
   existing ack composer can render the linked *"you've changed your mind about X"*
   acknowledgment — content from the recoded stance, no authored prose.

The free-form branch itself lives in `mine_stance_reversal`
(`user_model.py:2817` comment, detection+recode at `:2837`–`2857`):

```python
_new_pol = _assess_reversal_polarity(text)
if _new_pol is not None:
    _target = self.opinions.resolve_topic(text)        # live store (provenance bridge)
    if _target is not None:
        _held = self.opinions.stances.get(_target)
        if (_held is not None
                and _held.turn_number < self.opinions.turn_num   # held in a PRIOR turn
                and (_new_pol * _held.polarity) < 0.0):          # new attitude OPPOSES held
            self.opinions.recode_stance_toward(_target, new_polarity=_new_pol,
                                               blend=0.7, utterance=text)
            return
```

**Guards (honest — no false reversal):**

- *Held in a PRIOR turn only.* `_held.turn_number < self.opinions.turn_num` — a
  stance mined *this* turn conveys the user's current view and is **not** walked
  back by a coincidental reassessment term in the same turn.
- *Opposed polarity only.* `(_new_pol * _held.polarity) < 0.0` — a same-sign
  reassessment (*"i still love winter, it's the best"*) is already handled by the
  upstream weighted merge and is left alone (no double-write).
- *No signal ⇒ no-op.* If `_assess_reversal_polarity` returns `None` (no
  reassessment term), the branch is skipped entirely — neutral wording never
  triggers a guess.
- *Per-utterance idempotency.* Inherited from `reverse_stance` via the
  `_reversed_utterance` guard, so repeating the same contradiction does not
  double-flip.

**Hardcoding audit.** The added code is: a seed reassessment-affect lexicon
(module-level tuples, no topic, runtime-extendable), a polarity estimator that
reads the strongest term, a 17-line delta-rule `recode_stance_toward`, and a
branch in `mine_stance_reversal` gated on those. **Zero authored reply strings.**
The topic comes from the live stance store (`resolve_topic`), the new value from
the user's real words — the same seed-vs-hardcoding test ("can RAVANA change this
by itself, through experience?") passes: the lexicon is extendable, the resolver is
store-driven, the recode is a generic reflex.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `recode_stance_toward(topic, new_polarity, blend=0.7, utterance=None)` | `ravana/src/ravana/chat/personal_fact_store.py:477` |
| Seed reassessment-affect lexicon `_REASSESS_NEG` | `ravana/src/ravana/chat/user_model.py:619` |
| Seed reassessment-affect lexicon `_REASSESS_POS` | `ravana/src/ravana/chat/user_model.py:630` |
| Frozen term sets `_REASSESS_NEG_SET` / `_REASSESS_POS_SET` | `ravana/src/ravana/chat/user_model.py:637-638` |
| `_assess_reversal_polarity(text)` — new-value polarity estimator | `ravana/src/ravana/chat/user_model.py:641` |
| Free-form contradiction branch (comment + detection) | `ravana/src/ravana/chat/user_model.py:2817`, `:2837` |
| Topic resolution bridge (provenance) | `ravana/src/ravana/chat/personal_fact_store.py:350` |
| Prior `reverse_stance` (explicit retractions) | `ravana/src/ravana/chat/personal_fact_store.py:422` |
| Test (RED→GREEN) | `tests/unit/test_freeform_contradiction_recode.py` |

## Test coverage

One test module, `tests/unit/test_freeform_contradiction_recode.py`, with 5
checks (all pass, `1 passed in ~20s`):

- `test_freeform_contradiction_recode` (the `run()` entrypoint) asserts:
  1. **Free-form contradiction recodes a held stance** — *"not all street art is
     good"* after *"i love street art"* → `street art` polarity goes `+0.95 →
     -0.275` (was `>= +0.9` before the fix).
  2. **Broad co-mention bridging** — *"gone off winter"* recodes the `silence`
     stance (`+0.95 → -0.275`) via `resolve_topic`.
  3. **Same-sign reassessment is NOT recoded** — *"i still love winter, it's the
     best"* leaves the stance untouched (no false reversal).
  4. **No reassessment term ⇒ honest no-op** — *"street art is interesting to
     think about"* leaves the stance untouched (no guessed reversal).
  5. **Idempotent** — repeating the same contradiction does not double-flip.

The test **fails on the pre-fix stash** (held polarity stays positive) and
**passes with the change**, proving a real RED→GREEN. The surrounding stance /
opinion / reversal suites stay green (150 unit + 29 non-unit stance/opinion/
reversal tests in the feature card's run).

## Limits (honest — logged for a future round)

- The recode strength is a **fixed per-class blend (`0.7`)**, not yet calibrated
  per-user confidence. (Same deliberate simplification as `reverse_stance`'s fixed
  `0.85`/`0.5`.)
- The reassessment-affect lexicon is **seed**; phrasings outside it (e.g. a
  culture-specific idiom) are not yet detected as a contradiction. It is
  runtime-extendable, so the gap shrinks as RAVANA sees new terms.
- The contradiction must **oppose** a held polarity to fire; a neutral restatement
  with no reassessment term is intentionally ignored (no fabricated reversal).
- As with all reversal paths, third-person or hypothetical contradictions
  (*"people go off winter"*) are not attitude changes about the user and are not
  recoded.
