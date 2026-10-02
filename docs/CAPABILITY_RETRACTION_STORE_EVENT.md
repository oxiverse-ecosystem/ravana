# Retraction — a user correction is a store-level event

**Capability:** when the user takes something back, RAVANA records that as an
event against the **record** and stops serving the retired value from *every*
memory store — not just the fact store — and the correction survives
save/load.

This closes **defect 2** ("fact correction does not stick") from the
`2026-09-29T1239Z` chat round. Source: `ravana/src/ravana/chat/retirement.py`
(the module), wired in at `ravana/src/ravana/chat/engine_memory.py:296`
(`_observe_retirement`) and `engine_memory.py:374` (`_propagate_retirement`).

Every number and reply quoted below is copied from real executed output of
`scratch/_retire_e2e_postfix.py` (post-fix) and
`scratch/_retire_marker_growth_probe.py` (pre-fix), run at
`RAVANA_OFFLINE=1` on the `.venv-real` interpreter. Nothing here is inferred.

---

## 1. The defect, as measured

The capability was written because a correction was **acknowledged and then
ignored**. The transcript, from `scratch/_retire_probe.py`:

```
USER:    my brother nikhil plays the shehnai, he is in his thirties
RAVANA:  noted — i'll remember your brother nikhil plays shehnai.
USER:    no, nikhil plays the surbahari, not the shehnai
RAVANA:  thanks — i'll be more careful there. what should i have said?
USER:    which instrument does my brother play now?
RAVANA:  your brother nikhil is the surbahari.     <- the retracted value
```

Two independent causes, both in `retirement.py:12-27`:

1. **The correction was never written.** The natural contrastive tail shape —
   `"<subject> <rel> <NEW>, not <OLD>"` — is not one of the closed set of
   correction shapes `UserModel._extract_correction_fact` recognises, so
   `detected_correction_fact` stayed `None` and *nothing* was stored: no new
   value, no retirement.
2. **Even a recognised correction only half-applied.**
   `PersonalFactStore.contradict` retired the old value *in the fact store*.
   The episodic transcript kept the original utterance verbatim, and
   `_self_cued_episodic` / `_retrieve_episodic` answered from it — so
   *"what did i tell you about my brother?"* re-surfaced the retracted claim.

Cause 2 is the structural one: RAVANA had no notion that *"the user told me
this and then took it back"* is a property of the **record**, independent of
which store happens to hold the string.

## 2. The design

### 2.1 `RetirementLedger` — a store, not a table (`retirement.py:183`)

Four fields, all empty at birth:

| field | meaning | line |
|---|---|---|
| `retired` | slot key → retired values. The live fact for a slot is whatever is in the stores and **not** in here. | `retirement.py:193` |
| `retired_at` | turn index per retirement, so "what did you get wrong recently" is index math. | `retirement.py:196` |
| `log` | every grounded retraction with its **evidence** utterance, newest last, capped at 200. | `retirement.py:198` |
| `learned_markers` | contrast markers RAVANA confirmed **online**. | `retirement.py:200` |

`to_state` / `from_state` (`retirement.py:304`, `:323`) make it ride the
engine's existing pickle persistence, so no new persistence mechanism was
added.

### 2.2 Direction is decided by the marker's GRAMMATICAL CLASS, not a per-shape table

The two marker classes (`retirement.py:76`, `:80`) decide *which side* of the
marker is rejected:

- **NEGATION** (`not`, `n't`, `never`, `neither`, `nor`) takes a complement.
  What FOLLOWS it is denied:
  `"nikhil plays the surbahari, not the shehnai"` → rejects `shehnai`.
- **REVISION / interjection** (`actually`, `sorry`, `correction`, `mistake`,
  `no`, `instead`, …) marks the material BEFORE it as needing repair:
  `"he moved to berlin, actually he moved to lisbon"` → rejects `berlin`.

Collapsing the two classes into one list is exactly what produces inverted
answers on interjections, so they are kept apart in the source, in the marker
sets, and in the class query `RetirementLedger.interjective_markers()`
(`retirement.py:245`).

`parse_contrast` (`retirement.py:377`) returns `(asserted, rejected, marker)`.
`_object_of` (`retirement.py:347`) then reduces `"<subject> <relation> <value>"`
to the value itself by walking back to the last relation head — a closed-class
grammar vocabulary (`_RELATION_HEADS`, `retirement.py:123`) that says where a
predicate *ends*, never what a value *is*.

### 2.3 GROUNDING — this is what keeps it honest (`engine_memory.py:296`)

> **A retraction only ever fires when the retired value is FOUND in a live
> store. Nothing to retire → no-op.**

`parse_contrast` is cheap and easy to get wrong; grounding it is not. The
engine searches `personal_facts` first, then the episodic transcript, requiring
that the rejected value's stems be a **subset** of a live record's stems
(`engine_memory.py:337-352`). If nothing holds it, `_observe_retirement`
returns `None` at `engine_memory.py:355-357`.

Measured (`_retire_e2e_postfix.py` case 3) — a correction about something
RAVANA was never told changes nothing:

```
USER: no, my cousin rohit plays the sarangi, not the bansuri
  ledger unchanged -> True | ledger = {}
```

Measured (case 4) — an ordinary disclosure is never a retraction:

```
USER: my sister meera collects vinyl records
  ledger.log length -> 0
  any transcript record flagged retracted -> False
```

This fail-closed property is what stops the capability from becoming a
hallucination source: RAVANA can never manufacture a "correction" out of
ordinary conversation.

### 2.4 Propagation — the structural half (`engine_memory.py:374`)

Retirement is applied in each store **where that store already filters**,
never by inventing a second convention:

1. **PersonalFactStore** — via its own `contradict()`, so the existing
   `superseded` filter does the work; the replacement is asserted as an active
   fact with `source="correction"`.
2. **Episodic transcript** — the record is flagged `retracted` / `retracted_by`,
   **kept, not deleted**. It is real conversational history and
   `_reconstruct_gist` may legitimately need it to answer *"what did you use
   to think"*. Retirement hides it from ORDINARY recall; it does not erase the
   trace.

Recall paths read it back through `_record_is_retracted`
(`engine_memory.py:420`), consulted at `engine_memory.py:949` and
`engine_memory.py:1523`. It accepts a record if it was flagged at write time
**or** if the value it uniquely carries is now in the ledger — the second clause
covers records written before the ledger existed and keeps the two paths from
drifting.

### 2.5 Placement — a correction cannot be skipped by a short-circuit

`_observe_retirement` is called from `_record_episode` (`engine_memory.py:443`,
call at `:470`), the one function **every** user turn passes through — both the
early-return and the main pipeline. It runs *before* the turn is stored, so the
rejection is learned from the state as it stood when the user spoke (the
rejected value is still live) and the corrected utterance is recorded as the
new truth.

The whole thing is wrapped in `try/except` (`engine_memory.py:469-475`): a
failure means the capability is inert — which the tests catch — and must never
become a silent crash that breaks conversation.

### 2.6 Persistence

- Created at boot: `engine.py:1072`.
- Saved: `engine.py:10555` (`'retirement_ledger_state'`).
- Restored: `engine.py:11106`.

The restore sits **early** in `load()`, next to the RNG, with the reason in
code: `load()` throws `[Load error] 'str' object has no attribute
'edge_reactivations'` further down, which silently skips every store restored
after that point. That pre-existing bug is **not fixed** by this capability —
the round moved the ledger restore above it so the feature actually works. See
[Known limits](#5-known-limits).

---

## 3. What it does — measured, end to end

`scratch/_retire_e2e_postfix.py`, `RAVANA_OFFLINE=1`.

### Case 1 — negation (rejects what follows the marker)

```
USER:    my brother nikhil plays the shehnai, he is in his thirties
RAVANA:  noted — i'll remember your brother nikhil plays shehnai.
USER:    no, nikhil plays the surbahari, not the shehnai
RAVANA:  thanks — i'll be more careful there. what should i have said?
USER:    which instrument does my brother play now?
RAVANA:  your brother nikhil is the surbahari.
USER:    what did i tell you about my brother?
RAVANA:  yes — i remember: your brother nikhil is the surbahari.

  ledger.retired   -> {'i|brother nikhil|plays shehnai': ['the shehnai']}
  is 'shehnai' retired? -> True
  is 'surbahari' retired? -> False
  FACT ('i', 'brother nikhil', 'plays shehnai') -> 'plays shehnai' superseded=True  src=seed_regex
  FACT ('i', 'brother nikhil', 'the surbahari') -> 'the surbahari' superseded=False src=correction
  TRANSCRIPT idx=0 retracted=True  text='my brother nikhil plays the shehnai, he is in his thirties'
  TRANSCRIPT idx=1 retracted=None  text='no, nikhil plays the surbahari, not the shehnai'
```

Both stores moved, and the retracted value is gone from **both** the direct
query and the self-recall.

### Case 2 — authored interjection (rejects what precedes the marker)

```
USER:    my uncle ravi lives in pune
RAVANA:  noted — i'll remember your uncle ravi lives is in pune.
USER:    pune, actually he lives in patna
RAVANA:  yeah, lives. tell me more about it?
USER:    where does my uncle ravi live?
RAVANA:  your uncle ravi lives is patna.

  ledger.retired   -> {'i|uncle ravi lives|in pune': ['pune']}
  is 'pune' retired? -> True
  is 'patna' retired? -> False
```

(The clumsy `lives is patna` phrasing is the *relationship miner's* value
rendering, not this capability's — the retired/asserted values are correct.)

### Case 3 / 4 — fail-closed, and inert on ordinary conversation

See §2.3 above.

### Matching is morphological, not literal

`RetirementLedger._same_value` (`retirement.py:262`) compares stem-wise on the
whole phrase **and** on the head noun, with determiners stripped. So
`"the shehnai"` matches a fact stored as `"shehnai"`, and `"shehnais"` matches
the retirement recorded for `"shehnai"`. Without this the retirement silently
misses its own target and the stale value keeps being served — the exact
failure it exists to fix. A single-word stem match is enough evidence because a
retirement only ever **suppresses** a value; it never asserts one.

---

## 4. A defect this documentation round found and fixed

The module docstring claimed the interjection class worked. **It did not**, and
no test covered it.

`_observe_retirement` called
`parse_contrast(text, led.all_markers(), led.learned_markers)` — the
`learned_markers` argument is the bug, and the comment quoting
`engine_memory.py:313` in the transcript below is the pre-fix line number.
`parse_contrast` **replaces** its `REVISION_MARKERS` default whenever an
explicit interjective set is passed, and `learned_markers` is empty until
something is learned — so every authored revision marker (`actually`, `sorry`,
`correction`, `mistake`, `no`) was classified as a **negation** and the parse
ran backwards. Measured pre-fix (`scratch/_retire_marker_growth_probe.py`):

```
TEXT: pune, actually he lives in patna

A) DEFAULT interjective set (module default = REVISION_MARKERS):
   parse_contrast(TEXT, CONTRAST_MARKERS)          -> ('patna', 'pune', 'actually')

B) AS THE ENGINE CALLED IT (engine_memory.py:313):
   led.learned_markers = set()
   parse_contrast(TEXT, led.all_markers(), led.learned_markers)
                                                     -> ('pune', 'he lives in patna', 'actually')
                                                      ^^^^^ the ASSERTED replacement marked rejected

C) the fix's shape — union of the authored class and the learned one:
   parse_contrast(TEXT, led.all_markers(), merged)  -> ('patna', 'pune', 'actually')

   is_interjective('actually') reports: True — the LEDGER knew the class;
   the CALL did not pass it.
```

The inversion then failed grounding, so the ledger stayed **empty** and the
correction silently did not stick — the round's headline defect, still open for
every interjection shape. `RetirementLedger.is_interjective` answered `True`
all along; the knowledge existed in the store and the call site simply didn't
read it.

**Fix** (commit `af32504d`): `RetirementLedger.interjective_markers()`
(`retirement.py:245`) returns the authored revision class **union** the learned
one, and the engine passes that. This also keeps `learned_markers` meaning one
thing — online growth — instead of doubling as the class definition.

A test-isolation bug surfaced alongside it: every engine test shared
`user_suffix="retiretest"`, and `stop_background_learning()` persists, so one
test's grounded retraction loaded into the next engine's ledger and failed
`test_ordinary_disclosure_is_never_retracted`. Each engine test now has its own
tag. The suite is order-independent: **18 passed**, twice, the second run with
the previous run's pickles left on disk.

Gates were demonstrated **RED before GREEN**. Reverting only the engine call
site (the new method left in place) fails at `assert led_.log` — the ledger was
empty end to end. Reverting both files fails earlier, on the missing method.

---

## 5. Known limits (honest)

1. **`_RELATION_HEADS` is a curated predicate list** (`retirement.py:123`).
   The most arguable constant in the round. A broader morphological rule would
   be cleaner but measurably less reliable on this material. It says where a
   predicate ends, never what a value is, and removing it degrades value
   slicing — legitimate grammar vocabulary, not a topic table.
2. **The marker growth path is not reachable through the engine.**
   `learn_contrast_marker` (`retirement.py:217`) is real and unit-tested, and
   `learned_markers` persists, but as wired no *unseen* marker can ever reach
   it: `parse_contrast` only scans tokens already in `markers`, and
   `retire()`'s marker always comes from that scan, so
   `learn_contrast_marker` returns `False` at its `m in CONTRAST_MARKERS` guard
   every time. Measured: `parse_contrast("he moved to berlin, nah he moved to
   lisbon", led.all_markers(), led.learned_markers)` → `None`, and
   `led.learned_markers` is `[]` after a grounded-looking turn. The API is the
   documented growth path and it is unit-tested, but **nothing calls it with a
   new marker end to end.** Fixing this needs a design decision (what makes a
   token a *candidate* marker before it is known?) and is out of scope for a
   docs card. Flagged rather than quietly documented as working.
3. **The prefilter blocks one documented shape.** `looks_like_retraction`
   (`retirement.py:462`) scans the first 4 tokens for a `RETRACTION_OPENER`
   (or matches a pure-retraction phrase). The docstring's
   `"he moved to berlin, actually he moved to lisbon"` does **not** pass it
   (measured: `False`), because `actually` is the 5th token. The working
   interjection shapes are those that open with a retraction marker or lead
   with the rejected value (`"pune, actually he lives in patna"` → `True`).
   Shape matrix measured in `scratch/_retire_shape_matrix.py`.
4. **Pre-existing `load()` bug, not fixed.** `load()` throws
   `[Load error] 'str' object has no attribute 'edge_reactivations'` (the
   offending attribute is read in `engine.py`, past the ledger restore at
   `engine.py:11106`), silently skipping every store restored after that point.
   It bit this capability — the first ledger restore sat behind it and did
   nothing. Moved the restore early so the feature works; the root cause is
   untouched and still silently drops other stores.
5. **The full `tests/unit` suite was not run** by this round — it exceeds
   12 min on this host. The retirement file (18 tests) and the related
   recall/confabulation tests were run green. Coverage evidence is incomplete
   and a reviewer should treat it as such.
6. **Grounding is stem-subset, so a superset record can be retired.** A record
   holding both the old and new value is explicitly skipped as a disjunction
   (`engine_memory.py:342`), but a record that merely *contains* the rejected
   token among others is eligible. The user-visible risk is bounded — an
   ordinary recall is filtered, and the record is retained for
   "what did you use to think" — but it is a conservative-approximation seam,
   not a proof of correctness.

---

## 6. Tests

`tests/unit/test_retirement_propagation.py` — **18 passed**, split
deliberately:

- **Parser / ledger (13)** — no engine boot, pure Python, fast. Contrastive
  tail split, inverted-shape negation, no false contrast without a marker,
  identical-both-sides rejection, learned-marker honouring, opposite
  directions for negation vs interjection, empty ledger at birth,
  retire/query, marker growth, state round-trip, opener detection, and the
  authored-interjection direction (added this round).
- **Engine (5)** — the user-visible symptom. Retirement in every store,
  self-recall not re-surfacing the claim, an ordinary disclosure never
  retracted, the authored interjection end to end (added this round), and
  save/load survival.

No authored reply strings are asserted. The gate is on **which value** comes
back, never on the prose that carries it.

## 7. No hardcoding

Zero reply strings were added by this capability. It renders nothing
user-facing — it only mutates stores and records events; the reply is composed
by the existing strategies from live state.

The string sets in the module are **grammar vocabularies**:
`NEGATION_MARKERS`, `REVISION_MARKERS`, `_RELATION_HEADS`,
`RETRACTION_OPENERS`. They say how English marks a contrast and where a
predicate ends — they contain no answer to any question and are not keyed to
any topic. (`_PURE_RETRACTION` and the docstring examples quote the measured
transcript; the two comment copies of that transcript live in
`engine_memory.py:284` and `engine.py:1068`.)
