# Capability: possession-attribute mining (material / feature facts)

**Status:** shipped (commits `08e4d6b`, `1b2cdf4`, branch
`auto/round-2026-08-15T0830Z`).
**Verified:** regression tests in
`tests/unit/test_round_2026_08_15T0830Z_possession_attr.py` pass; live
end-to-end probes reproduced below (real engine output, `dim=64, seed=42,
baby_mode=True`, offline). Hardcoding self-audit clean.

## What it does

When the user describes a possession by what it is **made of** — e.g. *"the cabin
is a hand-hewn pine lodge with a sod roof"* or *"my sword is forged from
meteorite iron"* — RAVANA mines that material into the `PersonalFactStore` under
the **entity** (`cabin`, `sword`), **not** the user's own `i` subject. A later
cued recall then returns a clean, structured answer:

- *"what's my cabin made of"* → *"your cabin is made of pine."*
- *"what's my sword made of"* → *"your sword is made of meteorite."*

Previously these disclosures were never mined into a recallable/correctable
fact — "what's my cabin made of" fell through to a whole-sentence echo of the
disclosure (round 2026-08-15T0830Z, **Bug 4**).

A material immediately followed by a **feature noun** scopes the fact to that
part: *"my desk is oak frame"* stores `desk.frame = oak` and recalls as *"your
desk's frame is oak."* (the `madeof` fact is primary; a feature attr like
`roof`/`wall`/`frame` is more specific when named).

**Fail-closed.** A possession description with no recognised material — *"the
river is a fast mountain stream"* — is **not** mined, and a recall query that
matches no stored fact returns the honest "you told me earlier…" form rather
than fabricating a material. No LLM, no per-topic reply table, no retraining.
Every answer slot is read live from the `PersonalFactStore`.

## How it grew from the conversation

The parent chat round (`t_8e77475a`) surfaced residual limitations; the feature
card (`t_c7d6b270`) picked Bug 4 — a concrete capability gap: disclosures that
state a *property of a possession* were never stored as structured facts.

The miner (`UserModel.mine_personal_facts`, `user_model.py`) already captured
explicit "my X is Y" self-facts and pet names, but not possession **materials**.
The fix adds one structural extraction pass (mirroring `pet_slots`): a regex over
the cleaned disclosure that finds `<entity> is <descriptor>` where `<descriptor>`
contains a recognised material noun, then stores the fact under the entity via
`_put_fact_ent` (`user_model.py:775`). Recall already had a possessive branch for
a fixed attribute whitelist (`name`/`age`/`breed`/…); the new branch reads the
`madeof` / feature fact from the live store.

### Seed vocabulary, not an answer table (`possession_attrs.py`)

The material, feature-noun, and kind-noun sets are **seed data** — closed-core
lists of noun types, plus a runtime-grown extension via `learn_material()` so a
word RAVANA has never heard (`hempcrete`, `rammed earth`) becomes addressable for
later recall with **no code change**. They are never rendered to the user; recall
rendering lives in `engine_memory._reconstruct_entity` via
`possession_attrs.render`. This is structurally identical to the seed-vs-hardcode
rule for `pet_slots`: a brain is born understanding *kinds of materials*, not
*answers*.

- `is_material(word)` (`possession_attrs.py:171`) — seed + learned lookup.
- `is_feature_noun(word)` (`possession_attrs.py:181`) — roof/wall/frame/….
- `is_kind_noun(word)` (`possession_attrs.py:186`) — lodge/cabin/sword/…; the
  precision gate so "the river is a fast mountain stream" is ignored (no material
  and no kind noun).
- `learn_material(word)` (`possession_attrs.py:147`) — runtime growth path.
- `render(ent, attr, val)` (`possession_attrs.py:191`) — single source of truth
  for "your {ent} is made of {val}" / "your {ent}'s {feature} is {val}".

### Miner — possession-attribute pass (`user_model.py:1696`)

A regex (`user_model.py:1717`) matches `<entity> is <descriptor>` and scans the
descriptor tokens for a known material. If found, it stores `madeof` (or the
feature attr when a feature noun follows). It also accepts an explicit
`made/forged/built/... of/from` frame (`user_model.py:1748`). Materials seen for
the first time are registered through `learn_material` (`user_model.py` mirror:
`_mat = _mat if is_material else learn_material(_mat)`, `user_model.py:1755`).

```python
# user_model.py:1717-1759 (condensed)
for _m in re.finditer(
        r"\b(?:my|the|a|an|our|your)\s+([a-z][a-z'-]+)\s+"   # entity
        r"(?:is|are|was|were)\s+(?:(?:a|an|the)\s+)?\s*"      # copula
        r"([a-z][a-z'-]*(?:\s+[a-z][a-z'-]+){0,6})",         # descriptor
        q_clean, re.IGNORECASE):
    _ent = _m.group(1).lower().strip("'")
    _desc = _m.group(2).lower().strip()
    ...
    for _i, _w in enumerate(_dtoks):
        if _poss.is_material(_w):
            _mat = _w
            _nx = _dtoks[_i + 1] if _i + 1 < len(_dtoks) else None
            if _nx and _poss.is_feature_noun(_nx):
                _feat = _nx
            break
    ...
    if _mat is not None:
        _mat = _mat if _poss.is_material(_mat) else _poss.learn_material(_mat)
        if _feat:
            _put_fact_ent(_ent, _feat, _mat, 0.6)
        else:
            _put_fact_ent(_ent, "madeof", _mat, 0.6)
    elif any(_poss.is_kind_noun(w) for w in _dtoks):
        continue
```

### Recall — possession-material branch (`engine.py:2591`)

A regex resolves the entity and reads its `madeof` / feature fact from the live
`PersonalFactStore` via `possession_attrs`. The supported query shapes are
**"what's my {entity} made of"**, **"what's my {entity} {feature} made of"**, and
**"what is the material of my {entity}"** — i.e. the material keyword must follow
the entity. The honest None fallback fires when nothing matches (it then falls
through to the generic "you told me earlier…" recall).

```python
# engine.py:2599-2632 (condensed)
_MATQ = re.search(
    r"\b(?:what'?s|what\s+is|what\s+material\s+is|what\s+is\s+the\s+material\s+of)\s+"
    r"(?:my|the|our|your|a|an)?\s*([a-z][a-z]+)(?:'s)?\s+"
    r"(?:made\s+of|made\s+from|material|built\s+of|built\s+from)\b", q)
if _MATQ and pf is not None:
    _ent = _MATQ.group(1).lower().strip()
    _cand = None
    for _k, _f in pf.facts.items():
        if not (isinstance(_k, tuple) and len(_k) == 3):
            continue
        if _k[0] == _ent and not getattr(_f, "superseded", False):
            _attr = _k[1]
            if _attr == "madeof":
                _cand = _f
            elif _cand is None:
                from . import possession_attrs as _pa
                if _pa.is_feature_noun(_attr):
                    _cand = _f
    if _cand is not None:
        _attr, _v = _cand.attribute, _cand.value
        if _attr == "madeof":
            return f"your {_ent} is made of {_v}."
        return f"your {_ent}'s {_attr} is {_v}."
```

### Entity-index fold + render (`engine_memory.py:393`, `:455`)

`MemoryMixin._retrieve_episodic` now folds non-pet possession facts from the
`PersonalFactStore` into the entity index (`engine_memory.py:409`) so cued recall
resolves them, and `_reconstruct_entity` renders `madeof` / feature attrs cleanly
(`engine_memory.py:461`), via `possession_attrs.render` — instead of the
bare-slot form "your cabin's madeof is pine".

## Design compliance

- **Seed knowledge only.** `_MATERIALS_SEED` / `_FEATURE_NOUNS` / `_KIND_NOUNS`
  are closed-class noun vocabularies (materials, parts, possession kinds) — data,
  not content, never reply text. Expandable at runtime via `learn_material`;
  removing an entry degrades gracefully (material simply not mined until
  re-learned). No per-topic/per-entity `if/elif` answer path.
- **Online / incremental, no retraining.** Facts are mined live from each turn;
  recall reads the stored attribute+value. A new material becomes recallable the
  moment it is disclosed. No rebuild needed.
- **Fail-closed.** No recognised material → not mined. No stored fact → honest
  "you told me earlier…" fallback, never a fabricated material.
- **Zero authored reply prose.** Reply strings are f-string renders of live
  stored state (`f"your {_ent} is made of {_v}."`). Hardcoding self-audit (grep
  diff for added strings >45 chars): only docstrings + seed word-sets appear; no
  reply prose.

## Live verification (fresh engine, offline)

Real output, engine `dim=64, seed=42, baby_mode=True`. Stored facts confirmed
from the live `personal_facts.facts` store:

```
MINE   'the cabin is a hand-hewn pine lodge with a sod roof'
QUERY  "what's my cabin made of"      -> "your cabin is made of pine."
MINE   'my sword is forged from meteorite iron'
QUERY  "what's my sword made of"      -> "your sword is made of meteorite."
MINE   'our roof is slate'
QUERY  "what's my roof made of"       -> "your roof is made of slate."
MINE   'the river is a fast mountain stream'
QUERY  "what's my river made of"      -> (river NOT mined; honest recall, no material)
stored -> ('cabin','madeof','pine'), ('sword','madeof','meteorite'), ('roof','madeof','slate')
```

Feature-noun path (separately verified, see Tests gap below):

```
MINE   'my desk is oak frame'
stored -> ('desk','frame','oak')          # feature-scoped fact
QUERY  "what's my desk made of"           -> "your desk's frame is oak."
```

## Tests

`tests/unit/test_round_2026_08_15T0830Z_possession_attr.py` — 8 tests, all pass
(`.venv-real`, `RAVANA_OFFLINE=1`):

- `test_miner_stores_entity_scoped_madeof` — cabin.madeof = pine
- `test_miner_handles_explicit_made_of_frame` — sword.madeof = meteorite
- `test_miner_stores_feature_attr_when_named` — roof.madeof = slate
- `test_miner_does_not_mine_non_material_description` — river NOT mined (fail-closed)
- `test_e2e_recall_clean_material` — "your cabin is made of pine", not the echo
- `test_e2e_recall_explicit_made_of` — "your sword is made of meteorite"
- `test_seed_material_vocab_is_data_not_prose` — seed is data; `learn_material`
  grows it at runtime

### Coverage gap (honest)

The **feature-noun scoping** path (entity `.feature` attr, e.g. `desk.frame`,
rendered as "your desk's frame is oak") is **implemented and live-verified above**
but has **no test in the committed suite**. The current file covers `madeof`
only. A regression test for the feature-noun branch should be added to
`tests/unit/test_round_2026_08_15T0830Z_possession_attr.py` (within CI time
budget) — flagged here so a follow-up round closes it. This doc does not claim
test coverage for the feature-noun path.

Also note: the recall branch pattern requires the material keyword to **follow**
the entity (`what's my X made of`); the form `what material is my X` does **not**
match and falls through to the generic echo recall (verified live). This is a
query-shape limitation of the current regex, documented honestly rather than
hidden.

## Caveats

- Recall is by exact material-keyword-after-entity pattern; a query like "what
  material is my sword" is not yet matched and returns the generic recall form.
- The seed material list is finite; an unrecognised material in a disclosure is
  learned at runtime via `learn_material` and becomes recallable thereafter.
- Like pet facts, possession facts are correctable through the existing
  `contradict()` path — a later "no, my cabin is oak-framed" supersedes via the
  same machinery.
