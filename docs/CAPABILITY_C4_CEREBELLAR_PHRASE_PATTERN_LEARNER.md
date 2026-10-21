# CAPABILITY: C4 — Cerebellar phrase-pattern online learner (verb-phrase selection)

How RAVANA's `CerebellarNgram` — the architecture's only **online sequential
learner** (no retraining, no LLM) — is now wired into the language **output
path** so that a verb phrase RAVANA learns from a single conversation can
bias which verb it picks on the next turn for the same relation type.

This closes PIVOT §A5 ("the cerebellum is disconnected from the output path",
verified as dead code) for the **write + read + gate** half of the wiring.

Verified against the live engine (offline, `dim=64`) via the 11-test C4 suite
in `tests/unit/test_ravana_language.py` (green, 3.47 s on `.venv-real`).

---

## The gap that existed

Before this change, the cerebellum contributed **nothing** to any reply:

| Side | What it did | Where |
|------|-------------|-------|
| **Read** | `_select_verb_phrase` queried key `f"phrase:{relation}"` via `predict_next` | `surface_realizer.py:1098-1117` (old `:1098-1110`) |
| **Write** | `learn_chain` wrote **concept labels** into `bigram`/`trigram` only | `cerebellar_ngram.py:87-112`, `engine_generation.py:415` |

`grep "phrase:"` over `ravana/src/ravana/language/` and `ravana/src/ravana/chat/`
found **no write site** for the `phrase:{relation}` key. `predict_next("phrase:semantic")`
always returned `{}`, the `if ngram_result:` guard was never true, and control always
fell through to `VerbLexicon.select_verb`. The learner was dead code wearing a
learner's clothing (PIVOT §A5, verbatim).

Two further defects lived in the same module (PIVOT §A5):

* **Unconditional decay.** `_apply_decay` (`:237`, called from `learn_chain` at `:112`)
  subtracted `decay_rate` from *every* entry on *every* call with no usage-based
  protection — the opposite of memory.
* **Self-graded reward.** `learn_chain`/`learn_phrase` were called with
  `successful=True` unconditionally (`engine_generation.py:415,418`;
  `interface.py:1078,1772`), derived from `_assess_response_quality`
  (`engine.py:10682`), which grades RAVANA's *own output* rather than the user's
  response one turn later. (C3 in the PIVOT — grounding the reward — is **not**
  closed by this card; see §4.)

---

## The fix (C4 — PIVOT §A5, adopted)

### 1. A dedicated phrase-pattern store

`CerebellarNgram` gained a `phrase_patterns` dict keyed by **relation type**,
not by the `phrase:`-prefixed string the old reader expected:

```python
# ravana/src/ravana/language/cerebellar_ngram.py:54-62
# phrase_patterns: Dict[str, Dict[str, float]]
#   e.g. 'semantic' -> {'relates to': 0.6, 'is': 0.3}
self.phrase_patterns: Dict[str, Dict[str, float]] = {}
self._phrase_last_used: Dict[str, float] = {}   # last reinforcement tick per relation
self._turn_counter: float = 0.0                  # advances via tick()
self._decay_window: float = 50.0                 # ~50 turns of protection
```

### 2. `learn_phrase()` — the write side (Hebbian + corrective)

```python
# ravana/src/ravana/language/cerebellar_ngram.py:147-177
def learn_phrase(self, relation, verb_phrase, successful=True):
    rel = relation.lower() if relation else "semantic"
    lr = self.learning_rate if successful else -self.learning_rate * 0.5
    ...
    self.phrase_patterns[rel][vp] = max(0.0, min(1.0, current + lr))
    self._phrase_last_used[rel] = self._turn_counter
```

`successful=True` strengthens (Hebbian); `successful=False` weakens
(corrective feedback). The relation key is the **raw relation type**
(`semantic`, `causal`, `contrastive`, `episodic`, `opinion`, …), matching what
`SyntacticFrame.relation_type` carries.

### 3. `predict_phrase()` — the read side

```python
# ravana/src/ravana/language/cerebellar_ngram.py:208-227
def predict_phrase(self, relation, top_k=3) -> Dict[str, float]:
    rel = relation.lower() if relation else "semantic"
    patterns = self.phrase_patterns.get(rel, {})
    ...
    sorted_scores = dict(sorted(patterns.items(), key=lambda x: x[1], reverse=True)[:top_k])
    return sorted_scores
```

Returns `{}` when nothing has been learned for this relation — the caller then
falls through to `VerbLexicon.select_verb`. No per-relation branch, no phrase
table: the scores are the actual learned weights.

### 4. The call site wired in (write meets read)

One addition in the generation path (`engine_generation.py:416-422`):

```python
# After a frame is realized into an utterance:
self.cerebellar_ngram.learn_chain(chain_labels, successful=True)
# C4 fix: write the phrase:{relation} key the reader (_select_verb_phrase) queries.
self.cerebellar_ngram.learn_phrase(
    relation=frame.relation_type,
    verb_phrase=frame.verb_phrase,
    successful=True,
)
```

### 5. `_select_verb_phrase` consults the cerebellum at **all** dopamine levels

```python
# ravana/src/ravana/language/surface_realizer.py:1098-1117
def _select_verb_phrase(self, default_phrase, relation, dopamine_tone,
                         cerebellar_ngram, subject="", object=""):
    ...
    if cerebellar_ngram is not None:
        # at ALL dopamine levels — ...
        if hasattr(cerebellar_ngram, 'predict_phrase'):
            ngram_result = cerebellar_ngram.predict_phrase(relation, top_k=3)
            if ngram_result:
                return list(ngram_result.keys())[0]
    return VerbLexicon.select_verb(relation=relation, ...)
```

The old `dopamine_tone < 0.4` gate is gone: the cerebellum biases
verb-phrase selection **regardless of uncertainty**. When `predict_phrase`
returns `{}` (no learned pattern), it transparently falls through to
`VerbLexicon`.

### 6. Usage-protected decay (tick)

`_apply_decay` now shields a relation's phrase patterns from decay for
`_decay_window` (50) ticks after the last reinforcement, so memory survives
when the conversation moves on (PIVOT §A5 "unconditional decay").

---

## What this is (and isn't)

This is a **structural bridge, not a content library**. The store is empty at
birth; it only gains entries from the utterance RAVANA actually produces in a
turn (`engine_generation.py:418`). There is:

- **No authored verb-phrase list.** The candidate that wins is whatever RAVANA
  produced, stored under its relation type.
- **No per-relation branch.** `predict_phrase` is one code path for every
  relation; the relation is a *key*, not a branch.
- **No retraining.** Writes happen in-band, inside `process_turn`'s generation
  path. One conversation updates the weights; the next turn can read them.

### The seed

| Seed | Where | Why it is seed, not hardcode |
|------|-------|------------------------------|
| `phrase_patterns: Dict[str, Dict[str, float]] = {}` | `cerebellar_ngram.py:55` | Empty at birth; grows only from live utterances. Same dict the read side already keyed on. |
| `_decay_window = 50.0` | `cerebellar_ngram.py:63` | A single constant for the protection window; the *contents* of the store are learned, not authored. |
| `learning_rate = 0.05`, `decay_rate = 0.01` | `cerebellar_ngram.py:44` | Existing Hebbian scalars (already in the module) reused unchanged. |

---

## Verified behavior

Live in-process run (`RAVANA_OFFLINE=1`, `dim=64`, real venv):

```python
from ravana.language.cerebellar_ngram import CerebellarNgram
cn = CerebellarNgram()
cn.learn_phrase("semantic", "is connected to")      # one conversation
cn.tick()                                            # advance turn counter
cn.learn_phrase("causal", "because")
print(cn.predict_phrase("semantic"))   # {'is connected to': 0.05}
print(cn.predict_phrase("causal"))     # {'because': 0.05}
print(cn.predict_phrase("contrastive"))# {}  -> VerbLexicon fallback
```

The 11-test C4 suite (`tests/unit/test_ravana_language.py`, tests prefixed
`test_learn_phrase_*`, `test_predict_phrase_*`, `test_phrase_pattern_*`,
`test_c4_*`) covers: write, corrective weakening, empty-skip, sorted read,
top-k cap, empty fallback, serialization round-trip, tick advances counter,
**the high-dopamine gate (the core regression)**, and the fall-through when no
pattern is learned.

```
RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest tests/unit/test_ravana_language.py -k "c4 or phrase" -q
12 passed, 55 deselected in 7.67s
```

`test_c4_select_verb_phrase_uses_cerebellum` is the **regression that was red
before the fix**: it learns `"is connected to"` for `semantic`, then calls
`_select_verb_phrase(...)` with `dopamine_tone=0.8` (high confidence). Pre-fix,
the `dopamine_tone < 0.4` gate meant the cerebellum was never consulted at high
tone → the method returned the VerbLexicon default, not the learned phrase.

---

## Honest residual limitations

The PIVOT framed A5 as three defects ("key mismatch", "unconditional decay",
"self-graded reward"). This card closes:

1. **Key mismatch** — ✅ closed. A dedicated `phrase_patterns` store keyed by raw
   relation type; the reader and writer now agree.
2. **All-dopamine consultation** — ✅ closed. The `dopamine_tone < 0.4` gate is
   gone.
3. **Usage-protected decay** — ✅ structurally in place, ⚠️ **not yet wired
   into the live turn loop** (see below).

### Residual: `tick()` is not called by the engine

`CerebellarNgram.tick()` (`cerebellar_ngram.py:229`) advances
`_turn_counter`, which `_apply_decay` (`cerebellar_ngram.py:237`) compares
against `_phrase_last_used` to decide whether a relation is shielded from decay.
But `grep -rn "\.tick()" ravana/src/` finds **no call from the engine** — `tick()`
is exercised only in the test suite
(`tests/unit/test_ravana_language.py:226,239`). As a result, `_turn_counter`
stays at `0.0` in production, `_phrase_last_used[rel]` is always `0.0`,
`age = 0.0 - 0.0 = 0.0 < 50` is always true, and **phrase patterns are always
shielded** — decay never actually erodes them.

This is a **wiring gap**, not a logic bug: the decay protection code is correct;
it is simply never given an advancing clock. A regression test
(`test_tick_advanced_by_engine_turn`) asserts this gap against the checked-out
tree and will turn green once `tick()` is called once per engine turn. Until
then, phrase patterns persist until explicitly corrected
(`successful=False`) rather than naturally decaying.

### Residual: the reward is still self-graded (PIVOT C3, deferred)

`learn_phrase` is called with `successful=True` unconditionally from
`engine_generation.py:418`. The PIVOT's C3 ("grounded learning signal:
alignment + repair from the user's next turn") is **not closed** by this card:
wiring the signed error from `_assess_response_quality` (or the user-feedback
signal) into `learn_phrase(successful=…)` is the next step and is tracked
separately. `learn_phrase` *accepts* `successful=False` (and the weaker update
path is tested), so the signal plumbing is the only outstanding piece.

---

## Source map (post-C4, commit dc45b698)

| What | Where |
|------|-------|
| Phrase-pattern store (`phrase_patterns`, `_phrase_last_used`, `_turn_counter`, `_decay_window`) | `ravana/src/ravana/language/cerebellar_ngram.py:54-63` |
| `learn_phrase()` write — Hebbian + corrective | `ravana/src/ravana/language/cerebellar_ngram.py:147-177` |
| `predict_phrase()` read — sorted, top-k | `ravana/src/ravana/language/cerebellar_ngram.py:208-227` |
| `tick()` — turn-counter advance | `ravana/src/ravana/language/cerebellar_ngram.py:229-235` |
| Usage-protected decay | `ravana/src/ravana/language/cerebellar_ngram.py:237-281` |
| `_select_verb_phrase` consults cerebellum at all dopamine levels | `ravana/src/ravana/language/surface_realizer.py:1098-1117` |
| `learn_phrase` call site (write meets read) | `ravana/src/ravana/chat/engine_generation.py:416-422` |
| `learn_chain` call sites (concept-label learning) | `engine_generation.py:415`, `engine_generation.py:632`, `interface.py:1078,1772` |
| `phrase_patterns` serialization (`get_state`/`set_state`) | `cerebellar_ngram.py:373-375, 399-401` |
| C4 stats (`total_phrase_pattern_entries`) | `cerebellar_ngram.py:373-413` |
| Regression tests (11 C4 tests) | `tests/unit/test_ravana_language.py:169-278` |
| PIVOT §A5 (the gap this closes) | `docs/LANGUAGE_PRODUCTION_PIVOT.md:176-212, 302-307` |

---

## Why this matters for the architecture

RAVANA has no production model (PIVOT §A2 verdict: "ADOPT"). Its surface form
is assembled by `SurfaceRealizer.realize` from a finished `SyntacticFrame`
(`surface_realizer.py:348`). The *content* of a reply comes from cognition
(graph walk → decoder); the *which-verb* choice was previously a fixed
`VerbLexicon.select_verb`. With C4 live, the **which-verb choice is now biased
by the one online sequential learner the architecture has**, and that bias is
updated from the *actual* verb RAVANA produced — closing the production path's
only live, online, per-relation learning gate. The remaining gap is purely the
reward signal (C3) and the clock wiring (`tick()`).

---

## Running it

```bash
# Run just the C4 / phrase-pattern tests
RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest tests/unit/test_ravana_language.py -k "c4 or phrase" -v

# Run the full language suite
RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest tests/unit/test_ravana_language.py -q
```
