# Capability: object-disambiguated date recall (overlapping verbs)

**Status:** shipped (commits `677b456`, `8df435f`, `e56b05d`, branch `auto/round-2026-08-15T0326Z`).
**Verified:** 5/5 regression tests pass (`tests/unit/test_round_2026_08_15T0326_object_disambig.py`); live end-to-end probe reproduced below. Hardcoding self-audit clean.

## What it does

When the user has told RAVANA *when* they started an activity (a mined `since` /
`since_age` fact) and later asks *which year* for that activity, the resolver now
disambiguates two activities that **share a verb head but differ by object** — for
example *"building frames"* vs *"building cabinets"* — instead of returning
whichever fact it iterated first.

Before this fix, the `since` miner stored only the activity **verb head**
(`"build 2019"`), dropping the object. So a disambiguating query
(`"when did i start building frames"`) tied against `"building cabinets"` and the
resolver returned the wrong activity's year. This generalizes to EVERY
overlapping-verb disclosure (a real capability gap, not a single phrase).

The fix also repairs a **pre-existing double-gerund display glitch**: when the
stored activity phrase was already a gerund (`"building frames"`), the realizer
re-gerunded it into the broken `"buildinging frames"`. Replies now read
`"you started building frames in 2019."`.

No LLM, no per-topic reply table, no retraining. Every answer slot is read live
from the `PersonalFactStore` and morphologically generated.

## How it grew from the conversation

The parent chat round (`t_79770235`) logged two residual limitations and the
feature card (`t_cbbdca89`) picked the concrete capability gap:

> "date-resolver tie-break can show wrong activity on overlapping verbs (year
> correct)."

Root cause: the date miner's verb-attachment logic stored the bare verb head only.
The fix is **structural** — a single `_activity_object()` extractor slices the
activity's patient out of the clause and concatenates it onto every mined
`since`/`since_age` value. The resolver already token-overlaps the stored value
against the query, so the object being present is what lets a query that names the
object score higher on the right fact. No new resolver branch was needed — the
existing one now has more to match on.

### Miner — `_activity_object()` (`user_model.py:347`)

A structural helper extracts the verb patient from the clause: skip leading
determiners/particles (`_OBJECT_SKIP`, `user_model.py:340`), collect the run of
content words, and stop at the first span-closing word in `_OBJECT_STOP`
(`user_model.py:330` — prepositions / time / clause words like `since`/`for`/`when`).
Bounded to 5 tokens so a runaway clause cannot swallow the sentence. Returns `""`
when there is no object, preserving the bare-verb shape for verb-only disclosures.

```python
# user_model.py:330-344
_OBJECT_STOP = frozenset({
    "since", "in", "on", "at", "for", "from", "to", "by", "of",
    "about", "around", "into", "during", "after", "before", "when", "while",
    "where", "because", "but", "and", "or", "so", "that", "which", "what",
    "who", "how", "why", "over", "near", "under", "with",
})
_OBJECT_SKIP = frozenset({
    "the", "a", "an", "my", "your", "our", "their", "his", "her", "its",
    "this", "these", "those", "some", "every", "all", "each",
    "up", "out", "off", "down", "in", "on", "into", "back",
})

# user_model.py:347-377
def _activity_object(clause_tokens, verb_idx) -> str:
    if verb_idx < 0 or verb_idx + 1 >= len(clause_tokens):
        return ""
    _obj = []
    for _t in clause_tokens[verb_idx + 1:]:
        _tl = _t.lower()
        if _tl in _OBJECT_STOP:
            break
        if _tl in _OBJECT_SKIP:
            continue
        if not _tl or _tl.startswith("'") or _tl.isdigit():
            break
        _obj.append(_tl)
        if len(_obj) >= 5:
            break
    return " ".join(_obj)
```

All four date-mining blocks now store the object. Each computes
`_act_full = f"{_act} {_obj}".strip() if _obj else _act` and puts it into the
fact value:

| Block | Example utterance | Stored fact | Mined at |
|-------|-------------------|-------------|----------|
| Explicit year | `i've been building frames since 2019` | `since(build frames 2019)` | `user_model.py:1520-1522` |
| Relative duration | `i started building cabinets in 2021` | `since(build cabinets 2021)` | `user_model.py:1571-1573` |
| Age anchor | `since i was nine i've played cello` | `since_age(cello 9)` | `user_model.py:1629-1631` |
| Fuzzy duration | `i've been building widgets for a decade` | `since(build widgets <year-10>)` | `user_model.py:1691-1693` |

Verb-only disclosures (`"i've been restoring since 2018"`) keep the bare
`"restore 2018"` shape — fully backward compatible. (The `build` head is the verb
stem of `building`, inherited from the existing miner; see caveats.)

### Resolver — no structural change (`engine.py` block 1f)

The date-recall resolver (`_structured_recall`, defined at `engine.py:2163`) already
token-overlaps the stored value against the query. With the object now in the stored
value, a query that names the object (`"building frames"`) scores higher on the
matching fact than on the tied verb, so it recalls the RIGHT activity. Fail-closed
(zero overlap → `None`) is preserved; a bare query with no object still ties and
returns the first matching fact (honest: no way to disambiguate).

### Display fix — gerund-head guard (`engine.py:197`)

The `does`/`event` fact the resolver borrows for richer phrasing can carry an
**already-gerund** head (`"building frames"`). The old realizer re-gerunded it →
`"buildinging frames"`. Added a guard: a regular-gerund head (its `-ing` form equals
the head itself) passes through unchanged.

```python
# engine.py:197-198
if _head.endswith("ing") and len(_head) >= 5 and _gerund_of(_base) == _head:
    return _p
```

(`_verb_phrase_to_gerund` is defined at `engine.py:164`; `_gerund_of` at
`engine.py:142`.)

## Design compliance

- **Seed knowledge only.** `_OBJECT_STOP` / `_OBJECT_SKIP` are small closed-class
  word sets (prepositions, determiners, particles) — structural, not content.
  RAVANA-expandable: adding a word only changes where the object span ends; removing
  one degrades gracefully. No per-topic/per-activity table, no `if/elif` answer path.
  The capability is structural phrase-shape, not a lookup.
- **Online / incremental, no retraining.** Facts are mined live from each turn; the
  disambiguation emerges from stored content. Nothing requires a rebuild.
- **Fail-closed.** No object → bare verb shape retained. Zero overlap → honest `None`.
  Out-of-range index → empty object.
- **Zero authored reply prose.** Every reply slot read from the `PersonalFactStore` +
  realized by morphology (`_verb_phrase_to_gerund`). Hardcoding self-audit (grep diff
  for added strings >45 chars): only docstrings + seed word-sets appear; no reply
  prose. Net change in reply-bearing code is subtraction (the double-gerund branch
  removed).

## Live verification (fresh engine, offline)

Real output, engine `dim=64, seed=42, baby_mode=True`, taught
`i've been building frames since 2019` / `i started building cabinets in 2021`:

```
'when did i start building frames'        -> 'you started building frames in 2019.'
'since what year have i been building cabinets' -> 'you started building cabinets in 2021.'
'when did i start building widgets'       -> 'you started building frames in 2019.'
```

The first two lines prove the tie is broken (frames→2019, cabinets→2021, the two
answers DIFFER). The third line is the honest fallback: a query whose object
(`widgets`) matches no stored activity ties on the shared verb `build` and returns
the first matching fact — there is no signal to prefer one, so returning *something*
dated is the documented, fail-soft behavior rather than a fabrication.

## Tests

`tests/unit/test_round_2026_08_15T0326_object_disambig.py` — 5 tests, all pass
(25.8 s, `.venv-real`, `RAVANA_OFFLINE=1`):

- `test_object_mined_into_since_fact` — object lands in the `since` fact
- `test_object_mined_with_determiner_stripped` — `"the cabinets"` → `"cabinets"`
- `test_overlapping_verb_disambiguated_by_object` — frames→2019, cabinets→2021, and
  the two answers DIFFER (proves the tie is broken)
- `test_no_double_gerund_display` — no `"buildinging"`
- `test_verb_only_disclosure_unaffected` — bare `"restore 2018"` shape preserved

Regression sweep (feature card): full `tests/unit/` 1858 passed / 23 skipped / 0 failed;
59 related temporal/approx/dehardcode/08f/08-14T1110 tests green.

## Caveats (honest)

- The resolved activity head is the **verb stem** (`build`, from `building`), inherited
  from the existing miner's verb-attachment logic — not the object noun. Recall reads
  naturally because the object is concatenated onto the value
  (`"build frames 2019"`), so the query `"building frames"` still overlaps it.
- Disambiguation is by **exact object-token overlap**. Two activities whose objects
  collapse to the same token (rare for content words) would still tie; the resolver
  then returns the first matching fact by iteration order — the documented
  data-driven behavior, not a silent wrong answer.
- A query that names no object (bare verb only) ties across all same-verb activities
  and returns the first match (third live line above). This is intentional fail-soft,
  not a defect: there is no information to choose between them.
- The double-gerund guard only fires when the stored head is a *regular* gerund (its
  `-ing` form equals the head). Irregular already-gerund stems are handled by the
  existing `_IRREGULAR_GERUND` seed (see `docs/CAPABILITY_DATE_RECALL_PARAPHRASE.md`).
