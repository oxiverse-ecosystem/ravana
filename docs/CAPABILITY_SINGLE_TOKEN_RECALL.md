# Capability: single-content-token recall + GloVe-synonym fact matching (round 2026-09-13T1526Z)

**Status:** shipped (commit `76cffec8`, branch `auto/round-2026-09-13T1526Z-feature`, feature `t_bb85a4d9`).
**Verified:** `tests/test_single_token_recall_20260913.py` PASSES (5/5: single-token recall, remind-not-in-src-tokens, multi-token still requires 2 overlaps, remind excluded from candidates, GloVe synonym fact match). Hardcoding self-audit clean: zero authored reply strings, no Q→A dict, no retraining; both fixes are structural (adaptive threshold + stopword set extension), and the synonym fallback reuses the engine's seed GloVe embeddings (no new model).

## What it does

Two recall limitations closed in one round:

### A. Single-content-token recall no longer fails

A query like *"remind me what you said about privacy earlier"* — where only **one** content word ("privacy") survives stopword removal — now returns RAVANA's actual stored reply about that topic instead of falling through to a canned greeting.

- **Before:** `_route_agent_own_recall` required `_best_overlap >= 2`, but a single content token can never reach 2 overlaps with stored `src_tokens`. The query silently returned `None`.
- **After:** the threshold is adaptive — `_min_overlap = 1` when only 1 content token remains after stopword removal, else `>= 2` (preserving the original DEFECT C/D incidental-word collision guard for multi-token queries).

A second bug amplified the first: the word **"remind"** was missing from the `_stop` set, so it stayed as a candidate token and polluted both query candidates AND stored `src_tokens`. Added to both stop lists (`engine.py:5693` and `engine.py:5856`).

### B. GloVe-synonym fact matching (Pass 3)

When you phrase a recall query with a **synonym** of the stored fact's value — *"what am I afraid of?"* after storing `('i','fear','terrified of deep water')` — the literal token-overlap matcher (Pass 1 containment, Pass 2 token overlap) finds no shared word and the fact is missed.

- **Before:** `_match_fact` had two passes (containment, then token overlap). A synonym like "afraid" vs "terrified" cleared neither, so the fact was silently not recalled.
- **After:** a **third pass** checks whether any query content token is GloVe-cosine `>= 0.65` to any value/attribute token. This links "afraid" → "terrified", "scared" → "fearful", etc. — synonyms the user might phrase differently from the stored fact. Uses the SAME seed embeddings the rest of the engine reasons over (no new model, no retraining). Fail-closed: only fires when GloVe vectors exist for BOTH tokens and cosine clears the bar.

## How it grew from the conversation

The round's chat probe surfaced a residual limitation: *"remind me what you said about privacy earlier"* returned RAVANA's boot greeting instead of recalling its actual prior statement. Root cause tracing found two bugs in `_route_agent_own_recall` (the missing "remind" stopword + the hardcoded `>= 2` overlap threshold), and a structural gap in `_match_fact` (no semantic fallback for synonym phrasing).

**Fix (commit `76cffec8`).** Three pieces, all in `ravana/src/ravana/chat/engine.py`:

1. **Adaptive overlap threshold** (`engine.py:5899`):
   ```python
   _min_overlap = 1 if len(_cands) <= 1 else 2
   if _best is None or _best_overlap < _min_overlap:
       return None
   ```

2. **"remind" added to both stop lists** (`engine.py:5693` and `engine.py:5856`):
   ```python
   # Both _stop sets now include "remind" between "recall" and "answer"
   ```

3. **GloVe semantic synonym fallback (Pass 3)** (`engine.py:4873-4906`):
   ```python
   _gv = getattr(self, "_glove_vector", None)
   if _gv is not None and _ptoks:
       _SEM_BAR = 0.65
       # High-frequency verbs excluded to prevent false synonym hits
       # (know/afraid cosine ~0.818; think/like/want bridge many fields)
       _PASS3_STOP = {"know", "think", "want", "like", "make", "take", "get",
                      "got", "come", "came", "say", "said", "tell", "told", ...}
       _best_sem = None
       for _attr, _val, _conf in facts:
           _vtoks = set(w for w in re.findall(r"[a-z']+", _val_l + " " + _attr_l)
                         if len(w) >= 3 and w not in _PASS3_STOP)
           # ... compute max cosine between query tokens and value/attribute tokens
           if _max_sim >= _SEM_BAR:
               if _best_sem is None or _max_sim > _best_sem[3]:
                   _best_sem = (_attr, _val, _conf, _max_sim)
       if _best_sem is not None:
           return (_best_sem[0], _best_sem[1], _best_sem[2])
   ```

**Generalization.** The adaptive threshold is content-agnostic (keys on the count of surviving tokens, not on specific words). The synonym fallback is vocabulary-agnostic (any pair of GloVe-vectorized tokens clearing cosine 0.65 matches — no synonym table to maintain). Neither needs retraining or new reply authoring.

**Hardcoding audit.** The diff adds zero reply strings. The added surface is: one adaptive-threshold line, two stop-word set extensions, and a GloVe-cosine loop over existing seed embeddings. There is no Q→A dictionary and no keyword→reply table. Replies render REAL stored state via the existing recall path.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Adaptive `_min_overlap` (single-content-token recall) | `ravana/src/ravana/chat/engine.py:5899` |
| "remind" added to query-candidate stop list | `ravana/src/ravana/chat/engine.py:5693` |
| "remind" added to stored src_tokens stop list | `ravana/src/ravana/chat/engine.py:5856` |
| Pass 3 GloVe semantic synonym fallback in `_match_fact` | `ravana/src/ravana/chat/engine.py:4873-4906` |
| `_match_fact` (the three-pass matcher the fallback extends) | `ravana/src/ravana/chat/engine.py:4827-4910` |
| `_route_agent_own_recall` (the recall path the threshold fix unblocks) | `ravana/src/ravana/chat/engine.py:5746` |
| Regression + capability test | `tests/test_single_token_recall_20260913.py` (5 checks) |

## Test coverage

`tests/test_single_token_recall_20260913.py` (5 checks):

- `test_single_content_token_recall_succeeds` — a recall query with only ONE content token ("privacy") matches a stored reply whose `src_tokens` contain that token.
- `test_remind_not_in_src_tokens` — the word "remind" does NOT appear in stored `src_tokens` (it's a recall verb, not content).
- `test_multi_token_recall_still_requires_two_overlaps` — multi-content-token queries still require `>= 2` overlap (preserves DEFECT C/D guard).
- `test_remind_verb_excluded_from_candidates` — after fixing `_stop`, "remind" is excluded from query candidates.
- `test_glove_synonym_fact_match` — stores `('i','fear','terrified of deep water')`, queries "afraid"; asserts the fact matches via GloVe Pass 3 (cosine >= 0.65 between the two synonyms in the seed embeddings).

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/test_single_token_recall_20260913.py -q
```
