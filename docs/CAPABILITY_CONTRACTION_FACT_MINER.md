# Capability: fact-miner contraction handling — don't strip "'t" from contractions

## What it does

When the user discloses a relationship fact using a contracted verb (e.g.
"i don't like coffee", "she can't drive"), the fact-miner extracts the verb
without corrupting the contraction. Previously the verb was passed through
`.replace("'t", "")`, which turned "don't" into "don" and stored a junk fact
(`("i", "don", ...)`). The fix removes that replacement so the verb is kept
verbatim (lowercased only).

**Location:** `ravana/src/ravana/chat/user_model.py:4936`
```python
_verb = _gm.group(1).lower()
```
(previously `_gm.group(1).lower().replace("'t", "")`).

## How it grew from the conversation

The round report (ravana-2026-10-27T0445Z) identified the T7 defect: a
contraction such as "don't" was being mangled into "don" and persisted as a
junk personal fact. Root cause was the `.replace("'t", "")` call applied to
the mined verb group, intended to normalize contractions but instead stripping
the meaningful remainder.

## Design

The verb group is lowercased only; no contraction-stripping is applied. The
`_STATIVE_DENY` / `_activity_verb_ok` gates downstream still filter
non-activity words, so removing the replacement does not let junk verbs through
— it only stops the corruption of legitimate contracted verbs.

## Verification

- Source change: `git show 3555cdcd` (1 insertion, 1 deletion in
  `ravana/src/ravana/chat/user_model.py`).
- Existing unit suite remains green.
