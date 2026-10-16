# Evaluative-Predicate Polarity — reading a judgment from where a word *sits*

**Capability shipped in** `7bb4ae23` (round `2026-09-30T1031Z`, PR #98), feature card
`t_159df91e`. Module: `ravana/src/ravana/chat/evaluative_polarity.py` (347 lines).
Tests: `tests/unit/test_evaluative_polarity.py` (30 tests).

Every number below was produced by running RAVANA, not by reading the source and
hoping. The probe scripts are in `scratch/` and every one is reproducible with
`RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe <script>`.

---

## 1. What problem this solves

RAVANA's opinion miner could only recognise a value judgment when the predicate
word appeared in a hand-written alternation list. That is a **frozen
vocabulary**. The consequence was not a slightly-wrong answer — it was *no
answer at all*:

```
"i think handmade mugs are overpriced"
```

matched no pattern, so **no stance was created**. And because there was no
stance, there was nothing for the retraction machinery to recode, so a
contradiction ("i was wrong about handmade mugs") left the store untouched. The
user's opinion was simply not in the model.

The fix is deliberately **not a longer word list** — that is the same
frozen-vocabulary bug with a bigger list. It is a second, *independent* route to
the same judgment: read the predicate's position in concept space.

Two projections of one GloVe vector answer both questions the alternation list
was being asked (`evaluative_polarity.py:246-254`):

| Axis | Formula | Question it answers |
|------|---------|---------------------|
| **EVALUATIVE** | `0.5 * (abs(cos_pos) + abs(cos_neg)) - cos_neu` | *is this word a judgment at all?* |
| **VALENCE** | `cos_pos - cos_neg` | *which pole — good or bad?* |

A word that carries an evaluation (*"overpriced"*, *"sturdy"*, *"dreadful"*) sits
in a different region of embedding space than a word that merely names a thing
(*"table"*, *"kiln"*, *"bicycle"*), and evaluative words split along a
positive/negative direction. The model reads that geometry instead of matching a
string.

### Why the evaluative axis uses ABSOLUTE pole similarities

This is a real design constraint, recorded in the source at
`evaluative_polarity.py:190-207`. An earlier version pooled the two pole
centroids into a single "evaluative direction". That is wrong twice over:

1. A word **is** evaluative whether it leans good or bad — pooling discards the
   distinction that doesn't matter here.
2. Summing two near-antipodal pole centroids **cancels to near-zero**, which is
   exactly the case for a well-formed positive/negative anchor pair. The
   pool-and-normalise silently produced a degenerate axis.

Taking the mean of the two **absolute** similarities makes the axis
pole-independent and avoids the cancellation. `abs` is what makes it correct.

---

## 2. The capability, measured

### 2.1 Unlisted predicates are judged from geometry

Six held-out words the seed lexicons have **never** judged, so the answer can
only come from geometry. Session 1, engine at `dim=64, seed=42`:

```
from scratch/probe_stale_binding.py
```

```
dreadful     -> polarity -1.000  eval 0.471  src=geometry
sturdy       -> polarity +0.635  eval 0.075  src=geometry
wretched     -> polarity -1.000  eval 0.359  src=geometry
elegant      -> None (abstained)
clumsy       -> polarity -0.869  eval 0.349  src=geometry
mediocre     -> polarity -0.333  eval 0.457  src=geometry

geometry answered 5/6 unseen words
```

Read honestly: `dreadful` and `wretched` are correct and confident; `sturdy` and
`clumsy` are correct; `mediocre` is only weakly negative; and **`elegant`
abstained** — a false negative. One miss in six on this small sample. The earlier
62-word measurement quoted in the module docstring
(`evaluative_polarity.py:39-44`) is a *different, larger* held-out set and is not
reproduced here; this page reports only what was actually run today.

### 2.2 It fails closed on neutral nouns

19 neutral nouns, none of which may become a judgment:

```
table, chair, window, morning, kiln, pottery, mug, street, teacher,
harbour, bicycle, letter, hammer, river, notebook, ladder, envelope,
bucket, curtain

neutral nouns leaked into a judgment: 0/19
```

"the mug is blue" must leave the store untouched, and it does. This is the
property that makes the capability safe to run on every copular frame.

Two further fail-closed paths, both verified:

- a model with **no** `vector_fn` returns `None` for 8/8 probe words;
- a `vector_fn` that **raises** also returns `None` rather than propagating.

### 2.3 The decision boundaries are read off the measured distribution

```
EVALUATIVE_MIN_MARGIN = 0.02    VALENCE_MIN_MARGIN = 0.02
VALENCE_SCALE        = 0.25    BASE_CONFIDENCE    = 0.45
CONFIDENCE_STEP      = 0.15    MAX_CONFIDENCE     = 0.8
```

These are not tuned to one test case. Per the source comments
(`evaluative_polarity.py:88-112`), the lowest-scoring evaluative word measured
was `+0.044` and the highest-scoring neutral word was `-0.016`, so `0.02` sits
in the **empty band between the two populations** with margin on both sides.
The three measured sign errors all had `|margin| <= 0.012`; the smallest correct
sign was `+0.021`. Below either margin the model **abstains** rather than
guessing.

---

## 3. How it grew online — no retraining, no authored reply

This is the part that matters most for the architecture, and it is genuinely
two-directional.

### 3.1 `observe()` — the model remembers what it judged

Every judgment is recorded in a persisted `judged` store with a running
confidence (`evaluative_polarity.py:261-300`). Five observations of the same
predicate:

```
observe #1: source=remembered polarity=-0.700 conf=0.600
observe #2: source=remembered polarity=-0.700 conf=0.750
observe #3: source=remembered polarity=-0.700 conf=0.800
observe #4: source=remembered polarity=-0.700 conf=0.800
observe #5: source=remembered polarity=-0.700 conf=0.800
capped at MAX_CONFIDENCE (0.8): True
```

Note `source=remembered` on the **first** line: the first observation already
counts as evidence, not as a bare prior. Recording at exactly `BASE_CONFIDENCE`
would make the first and the zeroth observation indistinguishable.

A confidently-read predicate (`|polarity| >= 0.5`) is **promoted to an anchor**,
so its morphological neighbours ("overpriced" → "pricey") can then be judged on
their own. Signed sets keep the valence axis honest.

### 3.2 `relearn()` — the user is ground truth

When a later **stance reversal** recodes a topic that was keyed from a geometric
read, the user's new position is written **back** into the model
(`evaluative_polarity.py:302-317`). It *overwrites*; it is not averaged against
the geometric read:

```
before relearn: -0.700 conf=0.600
after  relearn: +0.600 conf=0.800
sign flipped: True
flip landed at MAX_CONFIDENCE (not averaged): True
```

So a predicate RAVANA mis-signed is corrected **by the user talking**, not by a
retrain. This is the call site at `user_model.py:6110-6115`.

### 3.3 Persistence

`get_state` / `set_state` round trip preserves the whole memory:

```
clumsy      original=-0.600/remembered  restored=-0.600/remembered  same=True
brilliant   original=+0.700/remembered  restored=+0.700/remembered  same=True
wretched    original=-0.900/remembered  restored=-0.900/remembered  same=True
stats after restore: {'judged': 3, 'learned_pos': 1, 'learned_neg': 2,
                      'seed_pos': 10, 'seed_neg': 10, 'seed_neutral': 12}
```

The seed anchors stay *seeds* (10/10/12) while `learned_*` grows — which is the
observable signature of a seed that is genuinely being extended at runtime
rather than a frozen table wearing a seed's clothing.

### 3.4 End-to-end through the engine

```
> i think handmade mugs are overpriced
  stances: {'handmade mugs': -1.0}
  evaluative stance predicates learned: {'handmade mugs': 'overpriced'}

> i think my roommate's keyboard is flimsy
  stances: {'handmade mugs': -1.0, "roommate's keyboard": -0.9884263982346191}
  evaluative stance predicates learned: {'handmade mugs': 'overpriced',
                                         "roommate's keyboard": 'flimsy'}
```

Both predicates are in **no** source list. The stance is minted from geometry
and the predicate is remembered so a later reversal can address it by name.

---

## 4. Seed vocabulary vs hardcoding — the doctrine check

The three anchor sets (`SEED_POSITIVE`, `SEED_NEGATIVE`, `SEED_NEUTRAL`,
`evaluative_polarity.py:75-89`) are **seed vocabulary**, the same category as the
VAD affect lexicon and the relation-verb lexicon already in the repo. The
deciding test — *can RAVANA change this by itself, through experience?* — is
answered **yes**, and section 3 is the proof: anchors are added at runtime by
`observe()`/`relearn()`, and `stats()` shows `learned_pos`/`learned_neg` growing
from 0.

What the module is **not**:

```
from scratch/probe_evalpolarity.py
```

```
module lines: 347
long authored reply strings (>60 chars): 0 []
its whole public surface (11 methods): ['_all_centroids', '_centroid', '_unit',
  'get_state', 'observe', 'relearn', 'remembered', 'score', 'set_state',
  'set_vector_fn', 'stats']
```

It emits **no text at all** — it returns dicts of floats or `None`. Nothing here
decides what RAVANA *says*; it only decides the polarity of a judgment the user
actually made. There is no Q→A table and no per-topic list. Nothing is fitted,
nothing is rebuilt: a predicate learned tonight is usable tonight.

---

## 5. FIXED DEFECT — the capability used to die after any reload

**Historical record, now closed.** The capability was geometry-**alive** in a
first session and geometry-**dead** in every subsequent one, and the
failure was silent. Root cause, both failure shapes, and the fix that
closed it are below — kept in full because the *shape* of the bug is the
lesson: a naive "is it wired?" check reported healthy throughout.

### Measured evidence

Identical engines, identical seed, only difference is whether a save file
existed (so `__init__` took the auto-load branch at `engine.py:1972`):

```
SESSION 1 (fresh suffix)                SESSION 2 (auto-load path)
_glove_vector_fn present : True          _glove_vector_fn present : True
bound __self__ is engine : True          bound __self__ is engine : False
fn('overpriced') -> vec  : True          fn('overpriced') RAISED   :
                                          TypeError: 'str' object is not callable
geometry on 6 unseen    : 5/6 answered   geometry on 6 unseen    : 0/6 answered
```

And end-to-end, in session 2:

```
> i think the harbour ferry is dreadful
> i think my old bike is sturdy
  stances after mining: {}
  eval predicates learned: {}
```

Nothing is minted. **Zero.** Every one of those three capabilities was verified
working minutes earlier on a fresh engine.

### Root cause

`engine.py:1087` creates `self.user_model = UserModel()`, then injects four live
engine bindings into that specific instance:

```
engine.py:1113   self.user_model._episodic_index      = self._episodic_index
engine.py:1114   self.user_model._episodic_transcript = self._episodic_transcript
engine.py:1120   self.user_model._concept_vocab       = self._concept_keywords
engine.py:1129   self.user_model._glove_vector_fn     = self._glove_vector
```

But `__init__` **later** auto-loads when a save file exists:

```
engine.py:1971   if os.path.exists(self._save_path):
engine.py:1972       loaded = self._load()
```

and `_load()` **replaces the whole object**:

```
engine.py:11382   self.user_model = loaded_user_model
engine.py:11410   self.user_model = _separate_um
```

Nothing re-applies the four bindings afterwards. A grep across the entire
`_load()` body (`engine.py:10958-11687`) finds **zero** re-assignments of
`_glove_vector_fn`, `_concept_vocab`, `_episodic_index` or `_episodic_transcript`.

Two distinct failure shapes were observed, and **both fail closed**:

1. The attribute is simply absent → `_ensure_evaluative_polarity` builds the
   model with `vector_fn=None` → every lookup abstains.
2. The attribute survives the pickle as a **string** (a bound method cannot be
   pickled; something stringified it) → `model._vector_fn` is truthy, so a naive
   "is it wired?" check **passes**, but calling it raises
   `TypeError: 'str' object is not callable`. `score()` swallows the exception
   at `evaluative_polarity.py:_unit` and returns `None`.

Shape 2 is why this survived: the obvious presence check reports healthy. Any
regression test must therefore probe an **unseen** word (which can only be
answered from geometry), not check attribute presence.

### Scope of the bug

It is **not** specific to this capability. The same four bindings are lost, so
anything else depending on `_glove_vector_fn`, `_concept_vocab`,
`_episodic_index` or `_episodic_transcript` degrades identically after a reload —
**silently**, because the whole family of these lookups is written to fail
closed rather than raise.

### The fix

The injection block was extracted into ONE method,
`CognitiveChatEngine._bind_user_model_dependencies()`, and called from both
places that (re)assign `self.user_model`:

* `__init__`, where the bindings were originally injected;
* the end of `_load()`, after every restore — so a reload is
  indistinguishable from a fresh boot for any capability that reads engine
  state through the user model.

One method, one call site each, so the two cannot drift again. It re-binds the
same four structures plus `_hippocampal_buffer` (bound separately later in
`__init__`, so the method binds it whenever the buffer exists), which closes
the same class of staleness for the owner re-attribution boundary.

### Verification

`tests/unit/test_reload_bindings_survive.py` was landed with the five defect
tests marked `xfail(strict=False)` so the day the bug was fixed pytest would
report XPASS and make the stale marker visible. After the fix the markers were
removed and the tests assert unconditionally:

```
$ RAVANA_OFFLINE=1 OMP_NUM_THREADS=1 .venv-real/Scripts/python.exe -m pytest \
      tests/unit/test_reload_bindings_survive.py -q
9 passed
```

The tests deliberately probe **unseen** words and the *callability* of the
injected bound method, never attribute presence — shape 2 above passes a
presence check while being completely broken.

---

## 6. Honest scope summary

| Behaviour | Status |
|-----------|--------|
| Unlisted predicate read from geometry (5/6 on a 6-word sample) | verified working, session 1 **and** after a reload |
| Neutral nouns produce no judgment (0/19 leak) | verified working |
| Abstains with no / raising `vector_fn` | verified working |
| `observe()` memory + confidence ramp + cap | verified working |
| `relearn()` overwrites on a user flip | verified working |
| `get_state`/`set_state` round trip | verified working |
| End-to-end stance minting through the engine | verified working, session 1 **and** after a reload |
| Test suite `tests/unit/test_evaluative_polarity.py` | **30 passed** in 590.21s |
| Same capability after any `load()` | **FIXED** — re-bound by `_bind_user_model_dependencies()`; 9 passed in `test_reload_bindings_survive.py` (§5) |

Reproduce everything:

```bash
cd C:\Users\Likhith\Documents\Projects\ravana
export RAVANA_OFFLINE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv-real/Scripts/python.exe scratch/probe_stale_binding.py   # the §5 defect
.venv-real/Scripts/python.exe scratch/probe_evalpolarity.py    # §2, §3, §4
.venv-real/Scripts/python.exe scratch/probe_reload_death.py    # save/load survival
```

Note for future rounds: these scripts each boot a cold engine (~10 min each on
this host). Run them in the background, write output to a file, and never pipe
through `tail` — buffering hides the head of the run.