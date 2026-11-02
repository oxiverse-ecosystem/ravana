# Capability: superseded-fact filtering in whole-profile recall

**Status:** shipped (commit `811ab756`, branch `auto/round-2026-11-02T0628Z`). NOT pushed.
**Feature card:** `t_682ee056` (round `2026-11-02T0628Z`).
**Verified:** real in-process probe on this branch (engine built with
`dim=64, seed=42, baby_mode=True`, `RAVANA_OFFLINE=1`). Hardcoding self-audit clean
(no authored reply prose; the only change is a filter over a real store).

## What it does

When RAVANA surfaces a whole-profile recall ("you told me …") for a cued entity,
it now **drops any fact whose value is marked `superseded` in the durable
`PersonalFactStore`**. So a corrected value (e.g. a pet name that was later
replaced) no longer echoes the retired one alongside the current one.

Source: `ravana/src/ravana/chat/engine_memory.py` lines 2038–2057 (the generic
"surface every stored attribute" branch of `_route_entity_recall` / the entity
recall path).

## How it grew from the conversation

The defect class is "retired values leaking into recall" — a superseded fact
must be filtered where the recall index is *built*, not at render time, else a
corrected value races the active one. The `PersonalFactStore` already carried a
`superseded` flag on each `PersonalFact` (see `ravana/src/ravana/user_model.py`,
`PersonalFact` dataclass), but the whole-profile dump in `engine_memory.py`
iterated `facts.items()` without consulting that flag. The fix consults the
durable store at dump time and skips any value present in the superseded set.

## Mechanism

```python
_superseded_vals = set()
try:
    _pf = getattr(self, "user_model", None)
    _pf = getattr(_pf, "personal_facts", None) if _pf is not None else None
    if _pf is not None and hasattr(_pf, "facts"):
        for _f in _pf.facts.values():
            if getattr(_f, "superseded", False):
                _superseded_vals.add(str(getattr(_f, "value", "")).lower().strip())
except Exception:
    _superseded_vals = set()
# ... then in the dump loop:
for attr, val in facts.items():
    if str(val).lower().strip() in _superseded_vals:
        continue
    ...
```

- Fail-open: if `user_model` / `personal_facts` is unavailable, `_superseded_vals`
  stays empty and behaviour is unchanged (no crash, no dropped data).
- Match is on the **value string** (lowercased + stripped), so it works regardless
  of which attribute key the retired fact lived under.
- Only the *generic dump* path is affected; single-attribute hits (`attr_hit in facts`)
  return earlier and are untouched.

## Verified claim

A real in-process probe confirms: after seeding a fact and then marking it
superseded in `PersonalFactStore`, the whole-profile dump no longer contains the
retired value. (Verified by reading `eng.user_model.personal_facts.facts` and
calling the recall path; the superseded value is absent from the returned string.)

## Related

- `PersonalFactStore` / `PersonalFact.superseded` — `ravana/src/ravana/user_model.py`
- Whole-profile dump — `ravana/src/ravana/chat/engine_memory.py` (commit `811ab756`).
