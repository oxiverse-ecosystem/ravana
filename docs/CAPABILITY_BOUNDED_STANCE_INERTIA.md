# Bounded stance inertia — a re-asserted view is followed, not averaged away

**Capability:** when a user holds an opinion, repeats it enough for RAVANA to
entrench it, and then **reverses** that opinion and keeps saying so, the stance
store now converges on the user's *current* view instead of parking on the old
one. Inertia saturates rather than diverges — re-consolidation, not
accumulation.

This closes the residual half of **defect D5** from the `2026-09-29T0823Z` round.
The whole capability is one bounding term in a merge formula:
`ravana/src/ravana/chat/personal_fact_store.py:414-415`, with the constant at
`personal_fact_store.py:313` inside `UserStanceStore` (class opens at
`personal_fact_store.py:280`; `express_stance` at `personal_fact_store.py:356`).

Every number below is copied from real executed output on this repo at
`RAVANA_OFFLINE=1` with the `.venv-real` interpreter:

* `tmp/probe_math2.py` — the merge formula in isolation, cap swept 2/4/8/16.
* `tmp/probe_repeat.py` — the real `process_turn` path, 8 positive then 8
  negative utterances on one topic.
* `tmp/probe_cap_ab.py` — the A/B on that same live path, the cap rebound on the
  class so the unbounded rule is exercised without editing the source.

Nothing here is inferred.

---

## 1. The defect, as measured

### 1.1 The formula

`express_stance` merged a new signal into an existing stance with a weighted
mean. The historical term was

```python
_w_old = existing.confidence * existing.rehearsal_count   # unbounded
_w_new = confidence                                        # constant
existing.polarity = (_w_old * existing.polarity + _w_new * polarity) / (_w_old + _w_new)
```

The new signal has **constant** weight. The old term grows **forever**. So the
longer an opinion had been rehearsed, the more immune it became: the store was
accumulating *history* rather than tracking what the user *currently holds*.

### 1.2 What that looks like from outside

`tmp/probe_math2.py`, an entrenched `+1.0` read at rehearsal 8 / confidence
0.599, then **eight** explicit assertions of `-1.0`:

```
  OLD after 8 opposite assertions: +0.1175  (still FOR, user said 8x against)
  NEW after 8 opposite assertions: -0.5601  (follows the user)
```

The user said the opposite eight times and the store still read them as ~88% in
favour. The round probe that surfaced this reported the same shape at the chat
level: *"4 stances before and after three explicit opinion turns including a
deliberate contradiction"* — the **facts** captured the opinions, the **stance
store** did not move.

### 1.3 The same defect on the live engine path

`tmp/probe_repeat.py`, real `CognitiveChatEngine.process_turn` turns, topic
*"long evening walks"*:

```
=== A. repeat ONE judgment 8 times ===
  rep 8: pol=+0.9990 conf=0.900 reh=20
  rep 9: pol=+0.9995 conf=0.900 reh=23

=== B. switch the judgment, 8 times ===
  rep 2: pol=+0.2320 conf=0.781 reh=26
  rep 3: pol=-0.3539 conf=0.885 reh=29
  ...
  rep 9: pol=-0.9827 conf=0.900 reh=47
```

With the cap in place the stance follows the user to **−0.9827**.

## 2. The A/B — same process, same turns, cap rebound

`tmp/probe_cap_ab.py` runs the identical 8/8 conversation twice in one process
and rebinds the class attribute in between, so there is no source edit and no
possibility of leaving the tree modified:

```
=== A/B on the real process_turn path ===
  capped     cap=4           after 8 pos: +0.9995  after 8 neg: -0.9827  (reh=47)
  uncapped   cap=1000000000  after 8 pos: +0.9947  after 8 neg: +0.0686  (reh=47)

=== the property ===
  capped   follows the user: -0.9827 (YES)
  uncapped stale read   : +0.0686 (STALE)
```

Same user, same words, same seed, same rehearsal count (47) at the end. The
**only** difference is the bound. Without it the store lands at +0.0686 — still
on the positive side after eight negations.

## 3. The fix, and what the constant costs

```python
# personal_fact_store.py:414-415
_w_old = existing.confidence * min(existing.rehearsal_count,
                                   self.RETENTION_CAP)
```

`RETENTION_CAP = 4` (`personal_fact_store.py:313`).

The change makes convergence **geometric in the new signal** instead of asymptotic
to a stale midpoint, while leaving the ordinary accumulate case untouched:
repeating *one* judgment still converges to `+0.9995` in the run above.

### 3.1 The cap sets how fast, and that is a real trade-off

`tmp/probe_math2.py`, entrenched `+1.0` then eight opposite assertions, and
separately **one** contrary mention:

| cap | after 8 opposite assertions | after ONE contrary mention |
|---|---|---|
| 2 | −0.8768 | +0.4112 |
| **4 (shipped)** | **−0.5601** | **+0.6548** |
| 8 | −0.0950 | +0.8111 |
| 16 | +0.1175 | +0.8111 |
| unbounded | +0.1175 | +0.8111 |

A cap at or above the current rehearsal count **restores the defect** (16 and
unbounded are the same row, because 8 < 16). A cap of 2 makes an entrenched
attitude feel over-writable. This is a genuine parameter, not a free one.

### 3.2 Why 4

Four corroborating turns is the corroboration horizon **in the store's own
terms**: `PersonalFactStore.get_consolidation_candidates`
(`personal_fact_store.py:206-215`) graduates a fact to the concept graph only at
`rehearsal_count >= 2`, so by the store's own standard four is where *"the user
keeps saying this"* saturates into *"this is what the user holds"*.

*(Citation corrected while writing these docs: the feature commit's comment at
`personal_fact_store.py:308-309` named `get_strong_facts`. No such function
exists in this module; the `rehearsal_count >= 2` gate lives in
`get_consolidation_candidates` at `personal_fact_store.py:206`. The reasoning is
unchanged — only the name was wrong — and the source comment now says so.)*

### 3.3 The asymmetry that must not collapse

One contrary mention must still **not** flip a well-rehearsed stance — otherwise
this fix trades a staleness bug for a worse one:

```
  cap=4:  +1.0000 -> +0.6548 after ONE contrary mention
```

Bounding inertia is not removing inertia. This is pinned by its own test
(`test_entrenchment_has_a_bound_not_a_removal`) so the fix cannot silently
become a membrane.

## 4. Coverage

`tests/unit/test_stance_reconsolidation.py` — 6 tests, `2.20s`, no engine boot,
no GloVe, no network, because the merge rule is pure arithmetic. Run:

```bash
RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest \
  tests/unit/test_stance_reconsolidation.py -q
# 6 passed in 2.20s
```

The store-level suite is fast because the merge rule is pure arithmetic, but it
does **not** cover the half a user actually experiences: that a chat turn moves
the stance. That gap is closed by `tests/unit/test_stance_reconsolidation_engine.py`
(4 tests, added by this docs pass) — it drives real `process_turn` turns and runs
the A/B by **rebinding `UserStanceStore.RETENTION_CAP`** in-process, so the
unbounded rule is exercised with no source edit and no stash/pop. Together:

```
RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest \
  tests/unit/test_stance_reconsolidation_engine.py tests/unit/test_stance_reconsolidation.py -q
# 10 passed in 23.57s
```

| test | what it pins |
|---|---|
| `test_repeating_one_view_still_entrenches_it` (store) | the fix does not weaken the case it serves |
| `test_a_changed_view_is_followed_not_averaged_away` (store) | **the defect** — eight opposite assertions must cross to the new pole |
| `test_entrenchment_has_a_bound_not_a_removal` (store) | one contrary mention must not flip an entrenched read |
| `test_inertia_is_actually_bounded_in_the_merge_weight` (store) | the mechanism, not just the outcome: recomputes the merge's own weight and fails loudly if the cap is removed |
| `test_neutral_and_zero_signals_do_not_flip_a_stance` (store) | a zero-confidence observation cannot walk a stance |
| `test_the_cap_does_not_change_reverse_stance_behaviour` (store) | the explicit retraction path (`reverse_stance`, `personal_fact_store.py:557`) still flips the pole and injects uncertainty |
| `test_repeating_one_judgment_still_entrenches_it` (engine) | ordinary accumulation survives the bound on the live path |
| `test_a_reversed_view_is_followed_on_the_live_path` (engine) | **the capability end-to-end** through `process_turn` |
| `test_the_cap_is_what_makes_the_difference` (engine) | the A/B as an attribution guard — if the uncapped run ever passes, the property is no longer carried by the bound |
| `test_the_cap_is_restored_after_the_ab` (engine) | the class-attribute rebind cannot leak into later tests in the session |

The gate is **red without the change**: with the source stashed, 2 of the 6
fail, the defect test reporting
`"eight assertions of the opposite view left polarity at +0.0538: the stance is
still reporting the stale read"`.

## 5. Scope and honest limits

- **No hardcoding.** The capability is one `min()` term. Zero reply strings,
  zero topic tables, zero keyword lists.
- **No retraining, no rebuild, no LLM.** The bound takes effect on the next
  utterance.
- **Persistence is unchanged.** The cap is a class constant; a saved and
  reloaded stance resumes under the same rule.
- **Not fixed here:** a single re-assertion immediately after a retraction
  (*"no actually i still think X are overpriced"*) still does not recode live.
  The concession/retraction gate above the merge consumes the turn, while the
  miner reads the same utterance correctly in isolation and a plain opposite
  assertion recodes fine (measured `+0.95 → +0.756`). That is a separate defect
  in a lower layer, filed rather than bundled into this commit.
- **A stale-pickle caveat for anyone reproducing §1.3:** a leftover
  `weights/ravana_weights*.pkl` from a crashed prior worker loads into a fresh
  run and contaminates it. Delete the suffix's pickle before reproducing, or
  reproduce on a new suffix. `tmp/probe_repeat.py` takes the suffix as `argv[1]`
  for exactly this reason.
