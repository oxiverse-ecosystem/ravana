# Capability: bare-copula name mining is case-insensitive

## What it does

When a user opens a conversation with a capitalized multi-clause intro such as
`"I am Likhith and I build Oxiverse"`, the bare-copula name miner now mines
the name (`Likhith`) **and** the activity fact (`build Oxiverse`) instead of
silently dropping them. Previously the `re.search` for the bare first-person
copula (`i'm` / `i am`) lacked `re.IGNORECASE`, so a capitalized `"I am …"`
never matched the pattern even though the same lowercase `"i am …"` did. The
fix adds `re.IGNORECASE` to that single `re.search` call, so the downstream
guards (determiner-led phrase rejection, non-name stoplist) run on the
capitalized form too, and the name + activity facts are stored and recalled
correctly.

**Location:** `ravana/src/ravana/chat/user_model.py:2592-2596`

```python
m_name = re.search(
    r"\b(?:i'?m|i\s+am)\s+"
    r"([A-Za-z][A-Za-z']*(?:\s+[A-Za-z][A-Za-z']*){0,2})"
    r"(?=[\s.,!?]|$)",
    q_clean, re.IGNORECASE)
```

## How it grew from the conversation

The 2026-11-07 cycle (task 105, feature t_0bb384a4) probed multi-clause
intros like `"I am Likhith and I build Oxiverse"`. The bare-copula miner
matched only lowercase `"i am …"` because the regex was compiled without
`re.IGNORECASE`; capitalized intros (the natural form of a real user's opening
line) therefore never reached the miner, and a later `"what's my name?"` /
activity recall returned `"i don't know your name yet"`. Root cause was the
missing flag on the single `re.search`, not a logic change — the downstream
guards already reject determiner-led phrases and common nouns, so enabling
case-insensitivity does not let junk through. Verified live: the multi-clause
intro now stores both the name fact and the activity fact.

## Design

One flag added to one existing `re.search` (commit `fc165421`). No new reply
strings, no per-name table, no retraining. The miner's guards remain the
single source of truth for what counts as a name.

## Verification

- Source change: `git show fc165421` — 1 insertion, 1 deletion in
  `ravana/src/ravana/chat/user_model.py`.
- Live engine run (`RAVANA_OFFLINE=1`, `.venv-real`) with a capitalized
  multi-clause intro stores the name + activity facts in
  `eng.user_model.personal_facts.facts`.
