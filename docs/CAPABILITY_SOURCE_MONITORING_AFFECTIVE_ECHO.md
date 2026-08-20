# Capability: source-monitoring guard on affective self-reports (D3 verbatim-echo limitation)

**Status:** shipped (commits `dbd38ad`, `27f7486`, branch
`auto/round-2026-08-19T1628Z`). **Verified:** regression suite
`tests/unit/test_d3_affective_echo.py` passes — **6 passed in 12.25s** (real
run, `RAVANA_OFFLINE=1`, `.venv-real`, `dim=64`, `seed=42`). Hardcoding
self-audit clean — the fix adds **zero** authored reply strings; the only added
logic is a structural first-person + VAD-affect test (seed vocabulary, not
content), exactly like the D4 guard it parallels.

## What it does

The in-prompt causal reasoner (`ravana/src/ravana/core/in_prompt_reasoner.py`)
ephemerally binds facts asserted in the *current* user message into answers,
without graph retrieval — it is the "physics engine" for same-turn causal
conditionals ("if/when X, Y" multi-hop chains). It is correct for world-state
transitions like *"when you turn on the lamp, it lights up."*

This capability closes the **D3 limitation** logged in the round: a combined
"statement + question" turn like

> *"that parking lot plan makes my blood boil. do you get why i'm furious?"*

was intercepted by the combined-fact reasoning handler and fell through to
`answer_in_prompt_causal`. That function mined the user's **affective
statement** (*"X makes my blood boil"*) as a causal premise (cause=`that parking
lot plan`, effect=`my blood boil`), and because the query contained a `?` its
loose question-gate admitted it; the query seeds then overlapped the user's own
words, so the reasoner replayed the effect clause *"my blood boil"* **verbatim**
as RAVANA's reply — a **source-monitoring failure**: the agent parroted the
user's utterance as its own reasoning.

After the fix:

- A premise whose **EFFECT is a first-person affective self-report** — *"the
  parking lot plan makes my blood boil"*, *"that makes me furious"*, *"waiting
  makes me seethe"*, *"the news makes my heart race"* — is a felt state, **not**
  a world-state transition the causal simulator should bind and replay. The edge
  is **refused** and the turn falls through to the genuine affective-response
  path (honest uncertainty / empathy), which is the correct behavior.
- A **legitimate world-state conditional** still binds and answers. The control
  arm of the suite asserts `"when you turn on the lamp, it lights up. what
  happens if you turn on the lamp?"` still produces the chained edge
  `("you turn on the lamp", "it lights up")` — no over-blocking.
- **Detection is seed-driven and grows.** The classifier reads "affective load"
  from RAVANA's **own learnable VAD lexicon** (`_VAD_SEED` in
  `ravana/src/ravana/core/mirror.py`, the same lexicon the intent router uses to
  classify user affect), not a frozen keyword table. The seeded lexicon grows
  online via Hebbian learning, so the capability expands as the system
  encounters new feeling words.

This mirrors the brain's **source-monitoring** (Johnson, 1993: a claim built
from the speaker's own felt state is not a manipulable cause→effect the
"physics engine" should simulate; it is attributed, not replayed) and the
Situation-Model path's own **D4 guard** (`CAPABILITY_SM_UNKNOWN_SUBJECT_GROUNDING.md`),
so the two free-generation monitors share one notion of "what is real
knowledge vs. what is a confabulation." No LLM, no per-topic reply table, no
retraining.

## How it grew from the conversation

This cycle's chat round (round `2026-08-19T1628Z`, card `t_8c6f434b`) surfaced
among its residual limitations a **garbled verbatim echo** — the agent
reproducing the user's own clause *"My blood boil."* as its reply. The feature
card (`t_f7144e57`, limitation D3) picked the concrete *source-monitoring*
sub-capability as the gap: the reasoner was treating an affective self-report as
a premise it owned.

**Root cause / prior behavior.** In `parse_causal_edges`
(`in_prompt_reasoner.py:77`), the `"X causes/leads to/results in/makes Y"`
pattern bound the user's statement `"that parking lot plan makes my blood boil"`
into the edge `("that parking lot plan", "my blood boil")`. The effect clause
`"my blood boil"` is the speaker's own felt state, not a world transition — but
nothing in the old binder distinguished "felt state" from "event." Downstream,
`answer_in_prompt_causal` (`in_prompt_reasoner.py:310`, which calls
`parse_causal_edges` at line 316) admitted the turn because it carried a `?`,
built a causal graph from the single edge, and its query seeds overlapped the
user's own words — so the replayed effect became RAVANA's "answer."

**Fix (commit `27f7486`).** `parse_causal_edges` now refuses to bind an edge
whose effect is a first-person affective self-report:

1. After a `"X makes/causes Y"` match, the guard (`in_prompt_reasoner.py:115`,
   active at `133-134`) calls `_is_first_person_affect(effect)`.
2. `_is_first_person_affect` (`in_prompt_reasoner.py:148`) returns `True` only
   when the effect contains **both** a first-person pronoun (closed-class grammar
   set `_FP_PRON`, `143-145` — by definition not learnable content, so a fixed
   set is correct) **and** at least one content word carrying affective load,
   where "affective load" is read from RAVANA's VAD lexicon via
   `UserEmotionDetector._lookup_word` (`179`) using the detector's own "felt"
   band: `abs(v) >= 0.4 or a >= 0.5` (`185`).
3. If `True` → `continue` (the edge is dropped). The effect is a felt state, not
   a simulate-able transition, so the causal reasoner never owns it and the turn
   falls through to the real affective path.

**Bootstrap (commit `dbd38ad`).** The VAD lexicon did not yet recognize the
somatic/affective idioms the D3 class needs (`boil`, `seethe`, `race/racing`,
`sink/sinking`, `pound/pounding`). Seven high-arousal interoceptive idioms were
seeded into `_VAD_SEED` (`mirror.py:127-140`, block under `_VAD_SEED` declared at
`57`). These are **bootstrapping lexicon, not authored prose**: the matrix still
grows online via Hebbian learning, so the capability expands as the system
learns new feeling words. Seed values mirror human-rated VAD norms (Warriner
2013 / NRC-VAD): high arousal + negative valence for anger idioms, moderate for
`sink`.

**Hardcoding audit.** The diff adds **zero** authored reply strings. A grep for
long added reply literals over the feature range
(`git diff dbd38ad 27f7486 -- ravana/src/ | grep "^+" | grep -oE '"[a-z][^"]{45,}"'`)
returns nothing prose-shaped — the only added strings are (a) code comments and
(b) the VAD numeric tuples (a vocabulary table, not a reply). The control tests
prove legit causal edges still bind and the engine still reasons, so the change
is subtraction-of-a-leak, not addition-of-a-script.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| In-prompt causal edge extractor (declaration + docstring) | `ravana/src/ravana/core/in_prompt_reasoner.py:77` |
| D3 source-monitoring guard comment + refuse-edge branch | `ravana/src/ravana/core/in_prompt_reasoner.py:115` (guard active at `133-134`) |
| First-person pronoun set (closed-class, grammar-correct fixed set) | `ravana/src/ravana/core/in_prompt_reasoner.py:143-145` (`_FP_PRON`) |
| `_is_first_person_affect` detector (docstring) | `ravana/src/ravana/core/in_prompt_reasoner.py:148` |
| Affective-load test (VAD "felt" band) | `ravana/src/ravana/core/in_prompt_reasoner.py:185` (`abs(v) >= 0.4 or a >= 0.5`) |
| VAD lexicon seeded with interoceptive idioms | `ravana/src/ravana/core/mirror.py:127-140` (under `_VAD_SEED` at `57`) |
| `answer_in_prompt_causal` (the function that used to echo) | `ravana/src/ravana/core/in_prompt_reasoner.py:310`, calls `parse_causal_edges` at `316` |
| Combined-fact reasoning wiring that routes here | `ravana/src/ravana/chat/engine_memory.py:990-1001` (calls `answer_in_prompt_causal` at `999`) |
| Regression suite (D3 class) | `tests/unit/test_d3_affective_echo.py` (6 tests, green) |

## Test coverage

`tests/unit/test_d3_affective_echo.py` — **6 tests, all GREEN (12.25s)**:

- `test_affective_statement_not_mined_as_causal_edge` — the literal D3 class:
  `"that parking lot plan makes my blood boil. do you get why i'm furious?"`
  must yield **no** causal edges (`[]`).
- `test_affective_alias_variants_not_mined` — other first-person affect idioms
  (`furious`, `blood boil`, `seethe`, `heart race`) are also refused as effects.
- `test_legit_causal_edge_still_mined` — control: an external world-state
  conditional (`"when you turn on the lamp, it lights up"`) still binds
  `("you turn on the lamp", "it lights up")` — no over-blocking.
- `test_answer_in_prompt_causal_does_not_echo_affect` — the reasoner returns
  `None` for the affective turn, and `"blood boil"` is **not** in the reply.
- `test_engine_does_not_echo_affective_clause` — **engine-level**: an end-to-end
  `process_turn` of the D3 turn must not parrot `"blood boil"` back.
- `test_engine_still_reasons_legit_causal` — **engine-level**: a genuine causal
  premise+query (`"when you turn on the lamp, an explosion occurs. what happens
  if you turn on the lamp?"`) must neither echo the user's words nor parrot the
  premise; the reply is a real chained answer or an honest "i don't know."

The suite **fails RED** on the pre-fix code (4/6 assertions fail — the guard
absence + both echo assertions), proving a real fix. Verified by this docs pass:
`6 passed in 12.25s`.

> Note: the feature card's "add a test if a documented behaviour lacks coverage"
> is already satisfied — the 6-test D3 suite covers both the affective-withholding
> arm and the legit-causal control arm at reasoner **and** engine level. No further
> test was added in this docs pass; this card's scope is documentation only.

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_d3_affective_echo.py -v
```

## Known rough edges (honest — logged for a future round)

- **Detection depends on shape.** The guard keys off (a) a first-person pronoun
  and (b) an affect word present in the VAD lexicon. A third-person affective
  report (*"that makes him furious"*) or a first-person self-report whose feeling
  word the VAD lexicon has not yet learned will not be caught by *this* path and
  may still be mined as a causal edge — though the online Hebbian growth of the
  lexicon shrinks the second gap over time.
- **Guard is scoped to the in-prompt causal reasoner.** It covers the
  `answer_in_prompt_causal` path (wired at `engine_memory.py:999`). If an affective
  statement+question were to be routed into a *different* reasoning/response path
  entirely, this guard would not apply — that would be a separate source-monitoring
  surface to audit.
- The `abs(v) >= 0.4 or a >= 0.5` thresholds for "affective load" are the VAD
  detector's own "felt" band, reused for consistency; they are data-derived
  vocabulary, not a hardcoded reply, but a future round may tune them per-form.
