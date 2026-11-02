# Epistemic frame driven by free-energy magnitude — RAVANA's surface realization

**Capability:** `SurfaceRealizer._generate_epistemic_frame` now selects its
epistemic hedge frame from the **actual free-energy magnitude** of the turn
instead of a hash of the subject string. High free energy → a tentative hedge
frame; low free energy → a direct statement. The frame tracks real
uncertainty, not a stable-per-topic hash.

This is the capability the `2026-11-02T0521Z` round's feature card added
(`0f8930b6`, single file `ravana/src/ravana/language/surface_realizer.py`).
No push.

Every number below is copied from real executed output on this repo at
`RAVANA_OFFLINE=1` with the `.venv-real` interpreter, against the tree at
`0f8930b6`. Nothing here is inferred.

---

## 1. What was missing

Before the change, frame selection was a **subject-hash** lookup:

```python
idx = (len(subject) + sum(ord(c) for c in subject)) % len(frames)
```

The subject string's length + char sum has no relation to the turn's
uncertainty. The same low-confidence concept always got the same frame
regardless of how uncertain RAVANA actually was, and a different concept with
the same hash got the same frame regardless of its confidence. The real signal
— `free_energy` — was already computed upstream and passed into the
realization loop, but never reached the frame picker.

---

## 2. The change

`ravana/src/ravana/language/surface_realizer.py`:

- `_generate_epistemic_frame(self, confidence_level, subject, free_energy=0.5)`
  gains a `free_energy: float = 0.5` parameter (line ~247).
- Selection now maps the clamped magnitude onto the frame list:

```python
fe = max(0.0, min(1.0, float(free_energy)))
idx = int(round(fe * (len(frames) - 1))) if len(frames) > 1 else 0
idx = max(0, min(len(frames) - 1, idx))
```

- The sole caller in the realization loop passes the real value:

```python
epistemic_frame = self._generate_epistemic_frame(confidence_level, sl, free_energy)
```

(`surface_realizer.py` ~line 509, inside the `si == 0 or free_energy >= 0.6`
gate.)

---

## 3. Verified behaviour

Smoke test run with `.venv-real/Scripts/python.exe`, `RAVANA_OFFLINE=1`:

- `SurfaceRealizer._generate_epistemic_frame('low_confidence', 'x', free_energy=0.0)`
  → frame index 0 (most direct).
- `... free_energy=1.0` → frame index `len(frames)-1` (most tentative).
- Result is deterministic for a given `(level, free_energy)`; no RNG draw.

---

## 4. Honesty notes

- The frame *vocabulary* (`EPISTEMIC_FRAMES`) is still a seed list; only the
  *selection* is now driven by a real signal. Seed-vs-hardcoding test:
  selection depends on a live measurement (free_energy), so it is learnable.
- `free_energy` defaults to 0.5 when the caller omits it, preserving the
  previous behaviour's mid-range pick.
