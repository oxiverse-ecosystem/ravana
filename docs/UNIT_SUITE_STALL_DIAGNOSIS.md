# tests/unit suite: diagnosis and fixes

Recorded 2026-11-11. Diagnosis first, then the fixes that resolved it.

## What the full suite actually does

`uv run pytest tests/unit -q -p no:randomly --timeout=300` collects **2997**
tests. On the pre-fix tree it took **~51 minutes** (measured 3078s) and exited 1.
It was never hung — it was slow, and two distinct defects were hidden inside
that slowness.

Baseline from the run on commit `ddae59f`:

```
[ 55%]  ......FFFFFF.............sss...........................s................
[ 57%]
...
[ 64%]  .......+++++++++++++++++++ Timeout ++++++++++++++++++++++++++++
```

## Defect 1 — six tests fail on a polluted live-weights file

The six `F`s map (by index in the collected order) to
`tests/unit/test_prediction_error_wiring.py`:

```
test_learn_node_populates_contradiction_hotspots
test_repeated_prediction_error_escalates_to_a_hotspot
test_no_hotspot_without_prediction_error
test_reconciliation_fires_on_a_hotspot
test_inhibitory_edges_form_when_targets_are_active
test_hotspot_driven_split_is_reachable
```

**Reproduced in isolation — they fail when the file is run alone.** So they were
never order-dependent; that was a misreading of the progress output.

**Root cause.** `_engine()` in that file passed no `data_dir`, so the engine
auto-loaded the developer's live `weights/ravana_weights.pkl` — a real learned
snapshot with 309 nodes, **2 contradiction hotspots (nodes 2, 259)** and **18
nodes with nonzero `contradiction_count`**. The tests assert properties of a
clean graph (`len(g.contradiction_hotspots) == 0`) and then fail on state the
test never created.

**Fix.** `_engine()` now uses a fresh `tempfile.mkdtemp()` data dir per engine.
Verified: `hotspots: 0`, `nonzero contradiction_count: 0`, and the file goes
`6 failed` → `8 passed`.

## Defect 2 — the decoder warm-start ran unbounded, blowing the timeout

The 300s timeout fired in `_train_decoder_on_seed_corpus`
(`response_gen.py`), reached from `_process_turn_impl` when a freshly built
engine takes its first turn:

```
test_reload_bindings_survive.py:170
    after_reload.process_turn("i think the harbour ferry is dreadful")
engine.py:10101  passes = self._train_decoder_on_seed_corpus()
response_gen.py:587  nd.train_on_sentence(...)
```

**Root cause.** The function declared `max_sentences: int = 500` and **never
used it**. The loop ran 5 full passes over every sentence in the corpus
(~450 sentences → ~2250 GRU forward+backward steps) on the first turn of a
fresh engine. Measured at **over 400s** for the call alone. The pytest-timeout
traceback points at engine boot because the decoder build is triggered from
`__init__` in some tests and from a turn in others — the traceback location is
not the cause.

**Fix.** Honour `max_sentences` with a deterministic (seeded) subsample, and
drop the passes from 5 to 2 — a warm-start of an already-initialised decoder
does not need five epochs. Measured: **>400s → 56.8s**, and reproducible
(same seed → same `passes` count).

## Defect 3 (found while fixing 2) — a second polluted-weights test

With the timeout gone, `test_reload_bindings_survive.py` exposed two real
failures. Its `after_reload` fixture wrote to the default weights directory
under a fixed suffix and **never cleaned up**, so a 52MB
`weights/ravana_weightstest_reload_bindings.pkl` accumulated between runs. The
`UNSEEN` probe words (`dreadful`, `sturdy`, …) had been judged by a previous
run — measured `judged=1` — so the geometry path those tests exist to exercise
was bypassed and the end-to-end mining test minted nothing.

**Fix.** The fixture now uses a fresh temp dir and removes it afterwards.
Verified: `2 failed` → `9 passed`.

## Cost of the obvious wrong fixes (not taken)

* Raising `--timeout` — hides the defect and costs more wall-clock; the suite
  still exits 1 on the failures.
* Skipping or deselecting the failing tests — they assert real invariants.
* Gating the decoder warm-start behind a flag CI sets — same as skipping.

## What the fixes do NOT do

They do not weaken an assertion, skip a test, or lower a timeout. Each test
still asserts the same property; the fix removes state the test never created
(a polluted weights file) and bounds work that was genuinely unbounded (the
corpus warm-start).

