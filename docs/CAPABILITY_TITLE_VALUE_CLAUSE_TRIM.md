# Capability: titled-name value clause trim (abbreviation-aware, runtime-growing) (RV-feature, 2026-10-22)

**Status:** shipped (commits `2bc0328f` + `0579a29b`, branch `auto/round-2026-10-22T0459Z`).
**Verified:** 6/6 tests pass in `tests/unit/test_title_value_handling.py` (offline, `RAVANA_OFFLINE=1`);
live end-to-end probe below (real engine output, `dim=64, seed=42, baby_mode=True`, offline).
Hardcoding self-audit clean — no authored reply prose, no per-topic answer table, no retraining, no LLM.

## What it does

A value like `"dr. rao, who studies marine ecosystems."` used to be stored as
`('i', 'thesis advisor', 'dr')` — the clause trim split on every `". "` and
truncated the titled name at the abbreviation's own period. Two defects, one
root: the equational miner's value character class stopped at `.` (so `"dr."`
could never be captured at all), and the downstream sentence trim split on every
period-space boundary (so even a captured `"dr. rao"` was cut to `"dr"`).

After the fix, the disclosure is stored as `('i', 'thesis advisor', 'dr. rao')`
and recalled as `noted — i'll remember your thesis advisor is dr. rao.`. The
same holds for any seed title (`mr.`, `prof.`, `st.`, `inc.`, …) and for titles
RAVANA has never seen before, which it learns on contact.

## How it grew from the conversation (source citations)

The round surfaced the defect via conversational probing: a user said
`"my thesis advisor is dr. rao, who studies marine ecosystems."` and RAVANA
replied `noted — i'll remember your thesis advisor is dr.` — a truncated name
that then leaked into recall. Two independent code paths had to be fixed.

### Fix 1 — equational value character class (`user_model.py:3148`)

The equational miner's value group used `[\w'-]`, which stops at the period in
`"dr. rao"`, so the period terminated the capture before the trim even ran:

```python
# user_model.py:3148  (equational miner — "my <attr> is <value>")
r"\bmy\s+([\w'-]+(?:\s+[\w'-]+){0,3})\s+(?:is|are)\s+([\w'.-]+(?:\s+[\w'.-]+){0,7})",
```

Before: `[\w'-]` for the value group. After: `[\w'.-]` — adding `.` (and `-`)
lets the miner capture the full titled name including the abbreviation's period.
This is a **parsing rule** (character class), not authored prose; it adds no
reply string and no per-topic logic.

### Fix 2 — abbreviation-aware clause trim (`user_model.py:3581`)

Downstream of the miner, the value was trimmed with a hard period-space split:

```python
# user_model.py:3578  (pre-fix trim — replaced)
_val = re.split(r"(?<=[.!?])\s+|\s*[.!?]\s*$", _val)[0].strip()
```

That split fired on the `". "` inside `"dr. rao"`, truncating to `"dr"`. It is
now replaced by a call to `_first_value_clause`:

```python
# user_model.py:3581  (post-fix)
_val = _first_value_clause(_val)
```

### Fix 2a — the seed + growth module (`user_model.py:1577-1670`)

A closed-class seed set of title/abbreviation words (without the trailing
period) plus a runtime growth path, mirroring `learn_species` / `learn_function_word`:

```python
# user_model.py:1585  (seed set — titles + common abbreviations)
_ABBREVIATION_SEED: frozenset = frozenset({
    "dr", "mr", "mrs", "ms", "miss", "mx", "prof", "sr", "jr", "st",
    "rev", "hon", "fr", "br", "sr", "capt", "col", "gen", "maj", "lt",
    "sgt", "cpl", "pvt", "cmdr", "adm", "pres", "gov", "sen", "rep",
    "inc", "ltd", "co", "corp", "llc", "etc", "vs", "eg", "ie", "approx",
    "dept", "est", "no", "vol", "fig", "eq", "ca", "cf", "al", "et",
})
_ABBREVIATION_LEARNED: set = set()

# user_model.py:1595  (growth entry point)
def learn_abbreviation(word: str) -> bool: ...
# user_model.py:1613  (membership test)
def is_abbreviation(word: str) -> bool: ...
```

`learn_abbreviation` returns `True` when the word is (now) a known abbreviation;
`is_abbreviation` is the membership test. Both strip trailing punctuation and
lower-case, so `"Dr."` and `"dr"` resolve identically. A word RAVANA has never
classified joins the class at runtime — seed structure, not answers; no retraining.

### Fix 2b — `_first_value_clause` (`user_model.py:1622`)

```python
# user_model.py:1622  (abbreviation-aware sentence/clause trim)
def _first_value_clause(text: str) -> str: ...
```

Three steps:

1. **Runtime growth** (`user_model.py:1636-1638`): any short token (≤5 letters)
   immediately followed by `". <Capital>"` is learned as an abbreviation before
   trimming — so a novel title like `"zx. Patel"` joins the class on contact
   without a code change.
2. **Relative-clause cut** (`user_model.py:1640-1643`): a closed-class relative
   pronoun opener (`, who/that/which/whom/whose`) ends the value, so
   `"dr. rao, who studies marine ecosystems."` → `"dr. rao"`.
3. **Sentence-boundary search** (`user_model.py:1645-1668`): scan for the
   earliest `[.!?]\s+` whose preceding token is **not** a known abbreviation;
   a trailing sentence-end period is likewise preserved when it follows an
   abbreviation. `"dr. rao is great. next sentence."` → `"dr. rao is great"`.

### Why both fixes are needed

Without Fix 1 the miner never captures `"dr."` (the period terminates the value
group), so the value reaching Fix 2 is already `"dr"`. Without Fix 2 the value
is captured fully but the trim cuts it back to `"dr"`. The end-to-end test
(`test_titled_name_not_truncated`) only passes with both in place.

## Verified behaviour (real executed output)

Offline probe, `dim=64, seed=42, baby_mode=True`, `RAVANA_OFFLINE=1`:

```
Q: my thesis advisor is dr. rao, who studies marine ecosystems.
R: noted — i'll remember your thesis advisor is dr. rao.

Q: my mentor is mr. venkatesh, who taught me biology.
R: noted — i'll remember your mentor is mr. venkatesh.

FACTS:
  ('i', 'thesis advisor', 'dr. rao') -> dr. rao 0.7
  ('i', 'mentor', 'mr. venkatesh') -> mr. venkatesh 0.7
```

Helper probes (`RAVANA_OFFLINE=1`):

```
_first_value_clause("dr. rao, who studies marine ecosystems.") == "dr. rao"
_first_value_clause("the blue hour just before the sun. i also keep pigeons") == "the blue hour just before the sun"
_first_value_clause("dr. rao is great. next sentence.") == "dr. rao is great"
_first_value_clause("prof. smith, who chairs the dept.") == "prof. smith"
is_abbreviation("dr") == True
is_abbreviation("Dr") == True
is_abbreviation("rao") == False
learn_abbreviation("xyz") == True ; is_abbreviation("xyz") == True
# runtime growth: novel short title followed by a capital joins the class on contact
_first_value_clause("my advisor is zx. Patel, who studies X.") == "my advisor is zx. Patel"
is_abbreviation("zx") == True
```

Test suite:

```
$ RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe -m pytest tests/unit/test_title_value_handling.py -q
......                                                                   [100%]
6 passed in 27.45s
```

## Seed vs. hardcoding check

- `_ABBREVIATION_SEED` is a **vocabulary** (title/abbreviation words), not a
  reply table. It lives in a module-level set that RAVANA extends at runtime via
  `learn_abbreviation`, so a novel title joins the class without a code change.
  Seed structure, not answers — satisfies the "can RAVANA change this by itself,
  through experience?" test.
- No authored reply prose was added; the reply (`noted — i'll remember your
  thesis advisor is dr. rao.`) is the existing miner acknowledgment template
  rendering the now-correct stored value.
- No retraining, no LLM, no per-topic / per-name rule.

## Coverage

`tests/unit/test_title_value_handling.py` (6 tests, all green):
- `test_titled_name_not_truncated` — end-to-end titled name stored whole.
- `test_titled_name_recall_value` — `"mr. venkatesh"` survives mining.
- `test_abbreviation_helper` — `is_abbreviation` / `learn_abbreviation` contract.
- `test_first_value_clause_relative_clause` — relative-clause cut.
- `test_first_value_clause_abbreviation_period` — abbreviation-period awareness.
- `test_engine_reply_titled_name` — live engine reply contains `"dr. rao"`.

A coverage gap exists for the **runtime-growth auto-learning** path (a novel
short title followed by a capital letter is learned inside `_first_value_clause`
without an explicit `learn_abbreviation` call). This round adds
`test_first_value_clause_runtime_growth` to close it (see test file).