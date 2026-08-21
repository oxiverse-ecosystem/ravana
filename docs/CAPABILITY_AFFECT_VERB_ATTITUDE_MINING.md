# Capability: affect-verb attitude mining (`<subject> <affect-verb> me`)

**Status:** shipped (commit `ac36f81`, branch `auto/round-2026-08-21T1653Z`).
**Verified:** the feature test suite
`tests/unit/test_round_2026_08_21T1653_affect_verb_stance.py` passes —
**6 passed in 7.68s** (real run, `RAVANA_OFFLINE=1`, `.venv-real`, `dim=64`,
`seed=42`). Hardcoding self-audit clean — the only long quoted fragments in the
commit are the regex pattern and the interrogative guard (structural matchers,
the allowed seed class), not authored prose. A live in-process probe (real
`UserModel` output, reproduced below) was used to back every claim in this doc.

## What it does

RAVANA now mines the **grammatical** attitude construction
*"\<subject\> \<affect-verb\> me"* — *"lab-grown meat creeps me out"*,
*"that flickering light freaks me out"*, *"his constant humming gets to me"* — as
a **stance keyed on the subject**, with polarity **derived from the shared VAD
affect lexicon** (the same `UserEmotionDetector` matrix the empathy / support
gate already grows online via Hebbian learning). It is a *pattern*, not a
per-topic reply list: any subject the user rotates in lands as a stance.

Before this capability, the stance miner only caught explicit like/love/hate
verbs and comparative/dismissive shapes — so *"lab-grown meat creeps me out"* was
silently dropped, and a later *"i changed my mind about lab-grown meat"* had
nothing to recode. That was the round-`2026-08-21T1653Z` residual limitation #1.

Real engine output (fresh `UserModel`, offline probe — these lines are actual
`opinions.stances` state, not paraphrased):

```text
turn: "lab-grown meat creeps me out, i can't stand the texture"
  → stance key 'lab grown meat', polarity -0.700, conf 0.60
  → (also a separate 'texture' attitude, a different existing miner)

turn: "that flickering light freaks me out every time it blinks"
  → stance key 'flickering light', polarity -0.750, conf 0.60

turn: "does lab-grown meat creep you out? tell me"   (interrogative)
  → resolve_topic('lab-grown meat') is None
  → no stance mined: a QUESTION is not a self-report

turn: "the tax form wibbles me something fierce"      (unknown verb)
  → resolve_topic('tax form') is None
  → no stance mined: fail-closed, no confabulated polarity

turn: "lab-grown meat creeps me out" → then →
      "actually i changed my mind about lab-grown meat, it's fine now"
  → before reversal: polarity -0.700
  → after  reversal: polarity  0.000   (reversal found the mined stance)
```

Design properties (seed-vs-hardcoding + no-retraining, per doctrine):

- **Polarity comes from the live VAD matrix, never an authored reply.** There is
  no second affect-word list. The construction's verb is validated against
  `UserEmotionDetector._lookup_word` (which applies the SAME
  `_morphological_normalize` the rest of the system uses, so *"creeps"*→*"creep"*
  resolves without a parallel stemmer). A verb absent from the matrix scores
  `0.0` and is skipped — **fail-closed, no confabulation** (probe lines 4–5
  above).
- **Coverage compounds with use.** Every affect verb that is observed is
  registered into the shared VAD matrix via `learn_association`
  (`_learn_affect_verb`), so the next mention — in *any* construction — is scored
  even without a seed entry. A verb in the seed affect-verb class but with no VAD
  entry yet bootstraps a **structural default negative valence** (the
  "<subject> \<affect-verb\> me" construction encodes an aversive reaction, like
  "X is better than Y" is structurally positive) and is registered for online
  growth.
- **Subject resolution reuses the shared `_opinion_topic` chokepoint**, so the
  stance lands on the real content head (*"lab-grown meat"*), never a
  closed-class word.
- **Reversals remain fully operable.** The stance is stored in the same
  `opinions` store `mine_stance_reversal` already reads (it runs *next* in the
  `mine_personal_facts` pipeline), so a later *"i changed my mind about X"*
  recodes it (probe lines 6–8). Nothing is frozen.
- **Affect-verb lemma set is SEED vocabulary** — the same allowed class as the
  dismissive-metaphor noun set and the sentiment adjectives. It is
  RAVANA-extendable; removing an entry only loses one verb shape. NOT a per-topic
  answer table.

No LLM, no retraining, store-driven. Satisfies the seed + online-learning
constraints: a new verb appears in the matrix the first time it is uttered,
without any rebuild.

## Known rough edges (honest — logged for a future round)

- **Interrogative discrimination is shape-based.** The guard rejects any
  utterance ending in `?` or beginning with a closed-class question word
  (what/who/does/can/…). A rhetorical or tag-question that nevertheless states an
  attitude ("that creepy doll, it freaks me out, doesn't it?") could be skipped.
  This is the same first-person-self-report discipline the rest of the miner uses.
- **Fail-closed depends on the VAD matrix.** A genuine aversive verb RAVANA has
  not yet learned will score `0.0` and be skipped until it is registered — the
  default-negative bootstrap only fires for verbs *already in the seed class*, so
  a novel verb outside that class is withheld, not guessed. The bootstrap shrinks
  the gap over time as the matrix grows, but a first-time unusual verb is
  genuinely un-mined.
- **Sign reinforcement is conservative.** The turn-affect buffer only *reinforces*
  the lexical VAD polarity (never reverses it), so an affect-verb stance is never
  flipped to the opposite pole by a mismatched turn-affect reading. This is by
  design (the lexical VAD valence is the ground-truth attitude signal); it means
  an affect-verb construction will not produce a positive stance.

These are within the capability's intended behavior; the miner is
store-driven and fail-closed.

## How it grew from the conversation

This cycle's chat round (round `2026-08-21T1653Z`, card `t_9cffe17d`) surfaced,
among its residual limitations, that *"lab-grown meat creeps me out"* was **not**
mined as a stance, so the user's later reversal had nothing to act on. The
feature card (`t_9dceaa84`, residual limitation #1) picked it as a concrete
grammar gap: a first-person affect-verb attitude the existing miner missed
despite the affect verb already living in the shared VAD lexicon.

**Root cause / prior behavior.** `UserModel.mine_stance` only caught explicit
like/love/hate verbs and comparative/dismissive shapes. The *"<subject>
\<affect-verb\> me"* construction — where the verb (creeps/grosses/freaks/…)
already lives in the shared VAD affect lexicon — fell through entirely. The result
was a silent drop: the engine held **no** stance on the subject, so
`mine_stance_reversal` (which reads the same store) found no target to recode,
and the reversal produced only a harmless no-op instead of reflecting the change.

**Fix (commit `ac36f81`).**

1. *Call the new miner in the pipeline.* `UserModel.mine_personal_facts`
   (`user_model.py:2876`) now calls `self._mine_affect_verb_stance(text)` before
   `mine_stance_reversal` (which runs last so it can see and reverse any stance
   just mined).
2. *New miner `_mine_affect_verb_stance`* (`user_model.py:2885`). It:
   - returns early on interrogatives (ends with `?` or starts with a
     closed-class question word — a question is not a self-report);
   - cheap-pre-filters for a first-person object (`me`/`us`/`myself`) so the
     regex only runs when relevant;
   - iterates the affect-verb regex (seed lemma set:
     creeps/grosses/freaks/weirds/unnerves/disgusts/repulses/scares/terrifies/
     bothers/annoys/rattles/spooks/unsettles/gets to + stem forms) over the
     subject phrase, resolving the salient content head via the shared
     `_opinion_topic` chokepoint (`user_model.py:3860`);
   - validates each verb against the shared VAD matrix via
     `_vad_for_affect_verb` (`user_model.py:2990`) — if `None` and the verb is
     outside the seed class, scores `0.0` and is **skipped** (fail-closed); if in
     the seed class but no VAD entry, bootstraps `(-0.6, 0.55, -0.35)` and
     registers it;
   - registers the observed verb into the shared matrix via
     `_learn_affect_verb` (`user_model.py:3013`) so coverage compounds;
   - applies a sign-preserving affect blend and calls
     `opinions.express_stance` (`personal_fact_store.py:333`) into the **same**
     store `mine_stance_reversal` reads.
3. *No second lexicon.* `_vad_for_affect_verb` reuses
   `UserEmotionDetector._lookup_word` and falls back to the shared `_VAD_SEED`
   (`ravana/core/mirror.py`) for stem forms, so the VAD matrix remains the single
   source of truth — exactly the seed-vs-hardcoding discipline (no per-topic
   answer table, polarity rendered through `express_stance`, never an authored
   prose line).

**Hardcoding audit.** The commit adds **zero** authored reply strings. A grep for
quoted `>=45`-char literals on added lines over the feature range
(`git show ac36f81 -- ravana/src/ | grep "^+" | grep -oE '"[a-z][^"]{45,}"'`)
returns nothing prose-shaped — the only added quoted fragments are the regex
pattern and the interrogative guard (structural matchers). The capability is
generic and derived from live state: every utterance registers new verbs and the
stance merges on new input, so RAVANA can change this by experience (the
deciding test). No retraining.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `mine_personal_facts` calls the new miner | `ravana/src/ravana/chat/user_model.py:2876` |
| `_mine_affect_verb_stance(text)` — the miner | `ravana/src/ravana/chat/user_model.py:2885` |
| Interrogative + first-person pre-filter | `user_model.py:2897-2914` (inside the miner) |
| Affect-verb regex (seed lemma class) | `user_model.py:2917-2925` (inside the miner) |
| `_opinion_topic` shared content-head chokepoint | `ravana/src/ravana/chat/user_model.py:3860` |
| `_vad_for_affect_verb` — shared-matrix lookup | `ravana/src/ravana/chat/user_model.py:2990` |
| `_learn_affect_verb` — register into shared VAD matrix | `ravana/src/ravana/chat/user_model.py:3013` |
| `express_stance(topic, polarity, …)` — same store reversal reads | `ravana/src/ravana/chat/personal_fact_store.py:333` |
| `mine_stance_reversal` — runs after, reads same store | `ravana/src/ravana/chat/user_model.py:3145` |
| Shared VAD seed (`_VAD_SEED`) fallback source | `ravana/src/ravana/core/mirror.py` (`_VAD_SEED`) |
| Feature test suite (6 tests) | `tests/unit/test_round_2026_08_21T1653_affect_verb_stance.py` |

## Test coverage

`tests/unit/test_round_2026_08_21T1653_affect_verb_stance.py` — **6 tests, all
GREEN (7.68s)**. The first 5 fail on the pre-fix baseline (`077c1a3`) and pass
after the fix; the 6th (fail-closed unknown verb) documents a branch that was
only implicitly covered before:

- `test_affect_verb_mines_negative_stance` — *"lab-grown meat creeps me out"*
  must form a negative stance on the subject (was silently dropped before).
- `test_affect_verb_variant_freaks` — *"that flickering light freaks me out"* →
  negative stance on the light (variant verb + multi-word subject).
- `test_affect_verb_no_second_list_registered` — the observed verb is registered
  into the shared VAD matrix (online growth); a seed-class verb with no VAD entry
  still mines.
- `test_affect_verb_question_not_mined` — a QUESTION (*"does lab-grown meat creep
  you out?"*) must NOT mine an affect-verb stance (interrogative guard).
- `test_affect_verb_reversal_later_operable` — once mined, a later reversal
  recodes the stance (the original defect: reversal had nothing to act on).
- `test_affect_verb_fail_closed_unknown_verb_outside_seed_class` — a verb
  neither in the VAD matrix nor in the seed class (*"wibbles"*) scores `0.0` and
  is skipped (no confabulated polarity). Added in this docs pass to cover the
  documented fail-closed branch explicitly.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_round_2026_08_21T1653_affect_verb_stance.py -v
```

The surrounding stance/affect/opinion/recall suite stays green (the feature
card's related run reported **94 passed / 0 failed**; the 6 new tests fail RED if
the capability is stashed, proving a real fix, then pass with it).
