# Capability: paraphrase-tolerant date-grounded recall + natural gerund replies

**Status:** shipped (commits `bbcca74`, `1a7c671`, branch `auto/round-2026-08-14T1110Z`).
**Verified:** 5/5 regression tests pass (`tests/unit/test_round_2026_08_14T1110_temporal_feature.py`); live end-to-end probe reproduced below. Hardcoding self-audit clean.

## What it does

When the user has told RAVANA *when* they started an activity (a mined `since` /
`since_age` fact), **date recall now survives paraphrased and rotated queries**,
and the reply is **grammatical English** instead of a broken bare verb.

Two concrete gaps closed in the round's chat probe (`t_bc372ced`):

- **Fix A — semantic-ish activity matching (stem linkage).** A rotated query that
  shares *no literal token* with the stored activity but *does* describe it
  elsewhere in the user's own mined facts now recalls the right dated fact,
  instead of falling through to a verbatim episodic echo. Example from the probe:
  `"what year did i start all this volcano stuff again"` →
  `"you started studying volcanoes back in 2015."` — even though the stored
  `since` fact only says `study 2015` and the word *volcano* lives in a *separate*
  `does` fact (`start studying volcanoes back`).
- **Fix B — natural gerund display.** The reply frame previously emitted the bare
  stored verb ("you started **study** basaltic eruptions"), broken English. It now
  realizes the activity as a morphological gerund ("you started **studying**
  basaltic eruptions"), and drops a redundant inceptive verb that would otherwise
  double up ("started **starting** studying" → "studying …").

No LLM, no per-topic reply table, no retraining. Every answer slot is read live
from the `PersonalFactStore` and morphologically generated.

## How it grew from the conversation

The parent feature card (`t_dc953ac2`, round 2026-08-14T1110Z) took a residual
from the chat round's own rotated-probe run: the date-recall resolver would match
a query against a `since` activity only when the user said the *same leading
verb* ("study …" → "study …"). A rotated paraphrase — *"all this volcano stuff"* —
contains none of the frozen activity verbs, so the resolver failed closed and the
turn fell back to an episodic echo. The fix generalizes the matcher so that an
activity described under a *different* leading verb (a `does`/`event` fact) still
contributes its distinctive words to the match context for the dated `since` fact.

### Fix A — stem-linked activity context (`engine.py:2840-2846`)

The resolver builds a per-activity context map `_verb_ctx`. For every
`does`/`event` fact it now **links that fact's value to every `since` activity it
shares a salient stem with** (or whose bare verb matches the fact's leading word):

```python
# engine.py:2840-2846
_stems = {_stem(t) for t in re.findall(r"[a-z']+", _val)}
for _act in _since_acts:
    if _stem(_act) in _stems or _act == _val.split()[0]:
        _verb_ctx.setdefault(_act, []).append(_val)
```

`_stem` (`engine.py:77`) is a crude morphological normalizer (strips
`-s/-es/-ies/-ing/-ed/-er`, min length 3) — so `"studying"` == `"study"` and
`"volcanoes"` == `"volcano"`. In the probe, `"start studying volcanoes back"`
contains the stem `volcano`, which links it to the `study` dated fact; the
scorer `_activity_query_overlap` (`engine.py:90`, also stem-based) then sees
`volcano` in the query and scores a match. Fail-closed: **no shared stem ⇒ no
link ⇒ zero overlap ⇒ the resolver returns `None`** (honest fallback), it never
latches onto an unrelated fact.

> **No GloVe here.** An earlier draft of the test file's docstring claimed Fix A
> uses "GloVe cosine" semantic matching. That is incorrect — the resolver matches
> purely on morphological stems over the live fact store (see the corrected
> `tests/unit/test_round_2026_08_14T1110_temporal_feature.py` header). The only
> cosine reference in `engine.py` (line 673, `_RECALL_DETECTION_THRESHOLD`) is an
> unrelated general recall gate.

### Fix B — morphological gerund realization (`engine.py:2897-2898`)

After picking the best dated fact, the display phrase is taken from the richer
`does`/`event` value (e.g. `"studying volcanoes"`) when present, else the bare
`since` verb, then passed through `_verb_phrase_to_gerund`:

```python
# engine.py:2897-2898
_qact = (_verb_ctx.get(_best_act) or [_best_act])[0] or _best_act
_qact = _verb_phrase_to_gerund(_qact)
```

Three small pure functions do the morphology (no LLM, no lookup table of phrases):

- `_IRREGULAR_GERUND` (`engine.py:126`) — a tiny **seed** map for closed-class
  irregular verbs (`go→going`, `die→dying`, `see→seeing`, …) — the irregular
  verb table a child is born with, extendable online, **never** an answer.
- `_gerund_of(verb)` (`engine.py:142`) — rule order: irregular seed →
  C/V/e consonant-doubling (`run→running`) → silent-e drop (`make→making`) →
  default `-ing` append (`paint→painting`).
- `_verb_phrase_to_gerund(phrase)` (`engine.py:164`) — converts the leading verb
  and drops a **redundant inceptive** leading verb (`start`/`begin`/…) that sits
  in front of an already-gerund verb (`_INCEPTIVE`, `engine.py:193`), because the
  reply frame already supplies "started". So `"start studying volcanoes"` →
  `"studying volcanoes"`, avoiding "started starting studying".

The realized phrase then drops into the pre-existing reply frames
(`engine.py:2902-2905`):

```python
return f"you started {_qact} in {_best_year}."
# "you started studying volcanoes back in 2015."
```

## Design compliance

- **Seed knowledge only.** `_IRREGULAR_GERUND` is a small closed-class
  morphological seed, not a per-topic reply table. RAVANA still learns the user's
  *own* phrasing at runtime; the seed only governs how a stored bare verb is
  realized. It can be extended online; removing an entry degrades gracefully.
- **Online / incremental, no retraining.** Every link is computed from the live
  `PersonalFactStore` at query time. Nothing requires a rebuild.
- **Zero authored reply prose.** The reply is a template (`f"you started {x} in
  {y}."`) with `x` and `y` both read from cognition. A hardcoding self-audit
  (grep for added strings >45 chars) found only the short structural frames and
  the `_IRREGULAR_GERUND` seed vocabulary — no authored sentences.

## Live verification (fresh engine, offline)

Real output, engine `dim=64, seed=42, baby_mode=True`, taught
`i started studying volcanoes back in 2015` / `i study basaltic eruptions` /
`i also keep three tarantulas`:

```
'what year did i start all this volcano stuff again' -> 'you started studying volcanoes back in 2015.'
'when did i start studying volcanoes'                -> 'you started studying volcanoes back in 2015.'
'how long have i been studying volcanoes'            -> "you've been studying volcanoes back since 2015 — about 11 years."
'when did i start the moon landing'                 -> None
```

The first line is the rotated-paraphrase case (Fix A): *volcano stuff* shares no
token with `study basaltic eruptions` but recalls the `study 2015` fact via the
`does` fact `start studying volcanoes back`. The last line shows fail-closed:
*"the moon landing"* shares no stem with either stored activity, so the resolver
returns `None`. (Year 2015 is the taught value; the `about 11 years` is
`datetime.now().year − 2015` at verification time, 2026.)

## Tests

`tests/unit/test_round_2026_08_14T1110_temporal_feature.py` — 5 tests, all pass
(33 s, `.venv-real`, `RAVANA_OFFLINE=1`):

- `test_semantic_date_recall_paraphrase` — "volcano stuff" recalls 2015 and
  contains "studying" (Fix A + Fix B).
- `test_explicit_date_recall_still_grammatical` — literal query recalls 2015,
  grammatical gerund.
- `test_how_long_gerund` — `how long have i been …` returns the duration frame
  with a gerund.
- `test_unrelated_when_query_fails_closed` — "when did i start the moon landing"
  returns `None` (honest fallback).
- `test_gerund_morphology` — unit-level: `_gerund_of` / `_verb_phrase_to_gerund`
  for regular, silent-e, doubling, irregular, and inceptive-drop cases.

Affected suites (regression sweep from the feature card): 94 passed / 1 skipped
(the 1 skip is the pre-existing ConceptNet skip), hardcoding audit clean.

## Caveats (honest)

- The activity head resolved for display is the **richer `does`/`event` phrase**
  when present (e.g. "studying volcanoes") rather than the bare `since` verb. If
  no `does`/`event` fact shares a stem with the `since` activity, the reply falls
  back to the bare verb realized as a gerund (e.g. "started **studying**" from
  `study 2015`).
- The redundant-inceptive drop (`engine.py:193`) only fires when the leading verb
  is in `_INCEPTIVE` **and** the next token is already a gerund. "started to
  study volcanoes" (infinitive, not gerund) is *not* collapsed — that is a
  different syntactic shape and is intentionally left unhandled rather than
  guess-corrected.
- Linkage is stem-based, not semantic. Two activities whose distinct meanings
  collapse to the same stem (rare for content words) could co-contribute words to
  a match context; the `_best_score` tie-break then prefers the higher-overlap
  fact. This is the documented, data-driven behavior.
