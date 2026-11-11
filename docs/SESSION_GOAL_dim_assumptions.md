# Session goal: finish removing the hard-coded GloVe-64 assumptions

## Context

`dim` is now free in principle -- #309 fixed four sites and #310 fixed the
intent router. But three more places in `ravana/src/ravana/chat/response_gen.py`
still hard-code the GloVe width at 64. They are harmless at the default
`dim=64` and wrong at any other dim.

This is the same class of bug as #309. Finish it.

## The sites

**1 & 2. `_vsa_event_narrative` -- `response_gen.py:141` and `:155`**

```python
vec = np.concatenate([vec[:64], sm])   # :141
v   = np.concatenate([v[:64], sm])     # :155
```

`sm` is the 11-D Lancaster sensorimotor tail, so the dual-code width is
`glove_width + 11`. The `[:64]` slice assumes `glove_width == 64`. At any other
dim this either truncates real signal (dim > 64) or produces a vector of the
wrong width (dim < 64).

Fix: derive the split from the actual vector width. `sm.size` is 11, so the
GloVe part is `vec[:vec.shape[0] - 11]` -- or better, do not slice at all and
concatenate the full `vec`, since `vec` is already the projected GloVe vector.
Verify which is correct by reading how `vec` is built above line 141.

**3. `_build_conditioned_bos` -- `response_gen.py:289` and `:293`**

```python
tail = bos[64:75].copy()
...
bos[64:75] = (0.7 * tail + 0.3 * sm).astype(np.float32)
```

`bos` is `self._decoder_dim()` wide (= projected GloVe width + 11). The
hard-coded `[64:75]` only lands on the sensorimotor tail when the GloVe width
is exactly 64. Derive the split: `gdim = bos.shape[0] - 11`, then
`bos[gdim:]`.

**4. Stale comment at `response_gen.py:666`**

```
# NOTE: the table dimension MUST be _DECODER_DIM (75), the same
# value passed as embed_dim at __init__ -- self.dim (64) is the
# distributional backbone ...
```

The code below it already calls `self._decoder_dim()` (fixed in #309). Only the
comment is wrong. Update it to describe `_decoder_dim()`.

## Rules

* **Do not change behaviour at dim=64.** Every fix must be a no-op at the
  default. Verify by comparing output before and after at dim=64.
* **Use `_decoder_dim()`** as the source of truth (added in #309). Do not
  introduce a second way of computing the width.
* **Do not weaken any test.** If a test fails, the fix is wrong.
* Run the targeted tests, not the whole suite:
  `uv run pytest tests/unit/test_neural_decoder.py tests/unit/test_response_gen.py -q`
  (add any test file that covers these paths).

## Acceptance

`goal_cmd` must exit 0:

```
uv run python -c "import tempfile,os; os.environ.pop('RAVANA_OFFLINE',None); from ravana.chat.engine import CognitiveChatEngine; d=tempfile.mkdtemp(); e=CognitiveChatEngine(dim=32, seed=42, baby_mode=True, data_dir=d, user_suffix='_g'); print('dim=32 OK'); d2=tempfile.mkdtemp(); e2=CognitiveChatEngine(dim=96, seed=42, baby_mode=True, data_dir=d2, user_suffix='_g2'); print('dim=96 OK')"
```

Both `dim=32` and `dim=96` must construct. `dim=96` is included because a
`[:64]` truncation bug only shows up when the true width exceeds 64.

Also run `tests/integration/test_ravana_chat_pipeline.py` and
`tests/integration/test_unified_semantic_loop.py` and confirm 17 passed.

## Out of scope

* `DualCodeSpace` (`core/dual_code_space.py`) hard-codes 64-D **by design** and
  says so in its docstring -- it is a VSA space, not the graph. Leave it.
* `data/intent_router.json` is fitted at dim=64; #310 made the router tolerant.
  Re-fitting the artefact is a separate decision.
* The 29 early returns starving the PFC gate -- that is the next piece of work,
  not this one.
