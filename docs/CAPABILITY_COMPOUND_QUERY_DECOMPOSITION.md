# Capability: compound / multi-part query decomposition (both conjuncts resolve)

**Status:** shipped (commit `6dc876f`, branch `auto/round-2026-08-22T0703Z`, feature `t_bda7d40f`).
**Verified:** `tests/unit/test_round_2026_08_22T0703_defects.py::test_compound_query_decomposition` PASSES on shipped code. Live behavior traced from `engine._split_compound_query` / `engine._compound_recall` / call site in `engine._structured_recall`. Hardcoding self-audit clean: the only long added strings under the no-hardcoding grep are the interrogative-detection **regex vocabulary** (`_INTERR`, `engine.py:2559`) and docstrings/comments — zero authored reply strings, no Q→A dict, no per-topic table, no retraining.

## What it does

A genuine **multi-part (compound) interrogative** — a question that asks TWO things joined by a coordinating *"and"*, or two *"?"*-terminated questions — is now answered in **full**, resolving **every conjunct**, instead of answering the first and silently dropping the rest.

- **Before:** *"what's my ferret's name and what does he do with my keys?"* returned only `your ferret is pip.` — the second conjunct (`what does he do with my keys`) was dropped.
- **After:** the same query returns `your ferret is pip and your ferret pip hides car keys.` — both conjuncts answered, joined by a single coordinating *"and"*, no double period.

The capability is **general and store-driven**: it splits the compound into independent sub-queries and runs the **existing** durable-store-backed recall resolver (`_structured_recall`) on each clause, then combines the **distinct** answers with *"and"*. Because it reuses the same machinery, it inherits every downstream resolver the engine already has (entity-scoped names, pet activity, kin activity, stance, reverse-name, paraphrase linking, …) with **no per-topic specialization**. It is **not pet-specific** — the same split+resolve path handles any compound whose individual clauses each resolve, e.g. *"what's my ferret's name? what does he do with my keys?"* (the *"?"*-joined shape) and, in general, *"who is X and what do they do"* style compounds.

No LLM, no retraining, no hardcoded reply. The capability is a deterministic query-shaping + fan-out wrapper around the live stores RAVANA grows autonomously.

## How it grew from the conversation

The chat round of round `2026-08-22T0703Z` logged a **residual limitation**: RAVANA's recall resolvers are **single-shot** — a compound question matches the FIRST clause and `return`s, so the second conjunct never gets resolved. This was surfaced empirically (the ferret name+activity compound returned only the name) and confirmed as a structural gap, not a one-off pet quirk.

**Root cause / prior behavior.** `_structured_recall` (`engine.py:2614`) walks a sequence of single-shot resolvers; each `return`s as soon as one matches. A compound like *"what's my ferret's name and what does he do with my keys?"* hit the name resolver on the first conjunct and returned, so *"what does he do with my keys"* was never evaluated. The limitation is **general** — it would equally drop *"who is X and what do they do"* or *"what's my brother's name and where does he live."*

**Fix (commit `6dc876f`).** Two small, deterministic helpers plus a single call-site insertion, all fail-closed:

1. **`_split_compound_query(q)`** (`engine.py:2509`) — a deterministic splitter. A query is treated as a genuine compound **only when**:
   - it has **two** `"?"` and is split on `"?"+` into `"?`"-terminated clauses (the *"a? b?"* shape), **OR**
   - it contains a coordinating `" and "` AND **both** resulting clauses are independently interrogative (ends with `"?"` or begins with an interrogative word from the seed `_INTERR` set at `engine.py:2559` — `what/who/which/where/when/why/how/is/are/…/name/list/show/…`).
   - A single bare `"and"` inside a **non-question** (*"i live in berlin and i work in munich"*) is NOT a query; the whole string is returned (safe no-op — callers that run this on a declarative turn get the whole string back). Non-interrogative trailing fragments (*"… and then the cat knocked it over"*) are dropped. The split is on the **top-level** `" and "` (no nested-clause handling), which is robust for the observed compound shapes without an LLM or a parser.
   - Returns a list of 1+ sub-query strings; if no genuine split happened, the original `q` (so a simple query is passed through untouched).
2. **`_compound_recall(q)`** (`engine.py:2570`) — runs `_split_compound_query`; if ≥2 parts, calls the **existing** `_structured_recall` on each clause (so every clause reuses the full store-driven resolver), collects **distinct** answers, strips one trailing period from each, and joins with `" and "` + a single `"."`. **Fail-closed:** if fewer than two clauses resolve to distinct answers, it returns `None` — never a partial or fabricated answer. Wrapped in `try/except` so decomposition errors can never mask a real answer.
3. **Call site inside `_structured_recall`** (`engine.py:2641` label `(0z)`, `_cmp = self._compound_recall(q)` at `engine.py:2650`) — runs **FIRST**, before any single-shot resolver, so no single-shot branch can pre-empt the second conjunct. When the compound returns `None`, the single-shot resolvers below still own simple queries untouched.

**Generalization.** Because the compound path fans out into the same `_structured_recall` the engine already runs, any future resolver improvement (a new fact type, a new entity linker) is automatically available to every clause of a compound — no compound-specific code to maintain. The interrogative detector is a **seed vocabulary** (`_INTERR`); it is a closed-class grammar set, not an answer table, and removing entries only re-admits the dropped-compound shape.

**Hardcoding audit.** The diff adds zero reply strings. The only long added strings caught by the no-hardcoding grep are: (a) the `_INTERR` interrogative-detection **regex vocabulary** at `engine.py:2559-2563` — a grammar lexicon, not prose; and (b) docstrings/comments explaining the splitter and fan-out. There is no Q→A dictionary and no keyword→reply table. The capability is structural: split → fan-out through the existing store-driven resolver → combine distinct answers. Verified against the doctrine's "can RAVANA change this by itself" test: the interrogative set is seed data, the resolution is entirely from live stores the user can correct at runtime.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| `_split_compound_query(q)` (deterministic compound splitter) | `ravana/src/ravana/chat/engine.py:2509` |
| `_INTERR` interrogative-detection seed regex (used to validate each clause) | `ravana/src/ravana/chat/engine.py:2559-2563` |
| `_compound_recall(q)` (fan-out over existing resolver, distinct-join, fail-closed) | `ravana/src/ravana/chat/engine.py:2570` |
| distinct-answer combine (strip one period, join with `" and "`) | `ravana/src/ravana/chat/engine.py:2602-2607` |
| Call site inside `_structured_recall` (runs first, `(0z)`) | `ravana/src/ravana/chat/engine.py:2641` (resolver entry `:2614`; `_cmp = self._compound_recall(q)` at `:2650`) |
| Regression + capability test | `tests/unit/test_round_2026_08_22T0703_defects.py::test_compound_query_decomposition` |

## Verified behavior (run live, this cycle)

All outputs below were produced by running the shipped code (`6dc876f`) in-process (`RAVANA_OFFLINE=1`, fresh engine, `process_turn("my pet ferret Pip hides my car keys under the couch")`):

| Query | Result |
|-------|--------|
| `what's my ferret's name and what does he do with my keys?` | `your ferret is pip and your ferret pip hides car keys.` |
| `what's my ferret's name? what does he do with my keys?` (`?`-joined shape) | `your ferret is pip and your ferret pip hides car keys.` |
| `what's my ferret's name?` (single clause, regression) | `your ferret is pip.` |
| `what does he do with my keys?` (single clause, regression) | `your ferret pip hides car keys.` |
| `who is my brother and where does he live?` (neither disclosed) | `None` — fail-closed, no fabricated answer |
| `i live in berlin and i work in munich` (declarative `and`) | `None` from the compound path — safe no-op, not mis-split as a query |

The first two rows are the capability's core claim (both conjuncts resolve). The last two are the fail-closed / safe-no-op guarantees. Each single clause still resolves through the pre-existing single-shot resolver (regression rows 3–4), confirming the compound path did not disturb simple queries.

## Test coverage

`tests/unit/test_round_2026_08_22T0703_defects.py::test_compound_query_decomposition` (verified passing this cycle, full file `3 passed in 45.60s`):

- Stores a pet ferret disclosure, then asserts the **single-clause** queries still work (`what's my ferret's name?` contains `pip`; `what does he do with my keys?` contains `hides car keys`) — the compound path must not have disturbed simple recall.
- Asserts the **compound** query returns both conjuncts: `pip` present, `hides car keys` present, exactly **one** coordinating `" and "`, and **no double period** (`not r.endswith("..")`).
- The test is fast (well within CI budget) and exercises the real mine → store → recall path, so it fails on pre-fix code and passes after the fix (RED→GREEN).

Run with:

```bash
RAVANA_OFFLINE=1 python -m pytest tests/unit/test_round_2026_08_22T0703_defects.py::test_compound_query_decomposition -v
```

The 55 related recall cases + 3 round-2026-08-22T0703 tests (D1 pet activity, D2 headless-possessive name, compound) = 58 all green on this branch, 0 regressions — cited from the feature card's `test_result` field (`6dc876f` metadata). The capability is doc-complete: this page plus the README bullet below; no further test was needed because the documented behaviour is already covered by `test_compound_query_decomposition`.
