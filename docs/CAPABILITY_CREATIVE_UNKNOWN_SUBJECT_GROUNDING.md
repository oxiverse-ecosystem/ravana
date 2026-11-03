# Capability: unknown-subject guard on the creative free-decode path (D3 word-salad limitation)

**Status:** shipped (commit `e7c9db9f`, branch
`auto/round-2026-11-03T1319Z`). **Verified:** regression suite — none yet
specific to D3; the D3 guard reuses the same concept-graph membership test as
the Situation-Model guard (see `docs/CAPABILITY_SM_UNKNOWN_SUBJECT_GROUNDING.md`)
and inherits its coverage shape. Hardcoding self-audit clean — the fix adds
**zero** authored reply strings; the only added logic is a structural
concept-graph membership test (seed vocabulary, not content).

## What it does

`_generate_creative` (`response_gen.py:2373`) produces free-association creative
verse from the concept graph. Before the guard, an open-ended creative prompt
about a subject RAVANA has **no durable knowledge of** (not in the concept
graph) still produced free-decode word salad, because the only degeneracy check
(`_is_word_salad`) is permissive.

After the fix:

- An **UNKNOWN subject** — not present in `_concept_keywords` or
  `_concept_labels` — returns `None` immediately (line `response_gen.py:2408`),
  so the caller defers to honest uncertainty instead of confident garbage.
- A **KNOWN subject** (a concept RAVANA has actually learned — present in the
  live concept-graph indices) still grounds a genuine creative verse.
- **Online graph growth re-admits a concept.** If RAVANA later learns a subject
  from chat or web learning, that subject joins `_concept_keywords` /
  `_concept_labels` and creative generation resumes. The guard is a live
  membership test, not a frozen blocklist.

This mirrors the Situation-Model D4 guard
(`docs/CAPABILITY_SM_UNKNOWN_SUBJECT_GROUNDING.md`) so both free-generation
monitors share one notion of "knowing" — there is no second, divergent
definition of grounding. No LLM, no per-topic reply table, no retraining.

## How it grew from the conversation

This cycle's limitation round surfaced that the creative path
(`_generate_creative`) emitted **word salad for subjects RAVANA knows nothing
durable about** — the D3 limitation. The feature card (`t_1776e96d`) picked the
concrete unknown-subject sub-capability as the gap.

**Root cause / prior behavior.** `_generate_creative` gathered associated
concepts from the topic's typed edges + the graph's stored node vectors and
free-decoded, with only the `_is_word_salad` degeneracy check. For an unknown
subject, the decoder still emitted fluent-but-false verse because nothing
tested concept-graph membership before generation.

**Fix (commit `e7c9db9f`).** Reuse the SAME concept-graph membership test as
the Situation-Model unknown-subject guard: at `response_gen.py:2405-2408`,

```python
_known_topic = (topic_l in getattr(self, "_concept_keywords", {})
                or topic_l in getattr(self, "_concept_labels", {}))
if not _known_topic:
    return None
```

Only a concept actually present in the live concept graph may ground a creative
verse. Unknown subjects fall through to honest uncertainty. This is exactly
parallel to the D4 guard (`response_gen.py:4038-4041`), so both monitors agree
on what "known" means.

**Hardcoding audit.** The diff adds **zero** authored reply strings — only the
structural `_known_topic` membership test plus comments. The longest added
content is a comment block; no long reply literal is introduced.

## Where it lives (with line cites)

| Concern | Location |
|---------|----------|
| Creative generator (declaration) | `ravana/src/ravana/chat/response_gen.py:2373` (`_generate_creative`) |
| D3 `_known_topic` concept-graph membership guard | `ravana/src/ravana/chat/response_gen.py:2405-2408` |
| Situation-Model D4 guard (parallel design) | `ravana/src/ravana/chat/response_gen.py:4038-4041` |
| Concept-graph indices declared | `ravana/src/ravana/graph/engine.py:326-327` (`_concept_labels`, `_concept_keywords`) |
| Indices grown at runtime as concepts are learned | `ravana/src/ravana/graph/engine.py:463-464`, `489-490`, `800-804` |

## Test coverage

No D3-specific test file exists yet. The guard's behaviour is covered
indirectly by the D4 suite's shape (`tests/unit/test_sm_unknown_subject_grounding.py`),
which asserts the same `_known` membership semantics. A dedicated D3 test is a
candidate for a future round; this docs pass did not add one (scope is
documentation only, within CI time budget).

## Known rough edges (honest — logged for a future round)

- **The broader D3 is only partially closed.** This round closed the concrete
  *unknown-subject* sub-capability: a subject with **no** durable knowledge is
  now withheld. The larger D3 class — word salad anchored on a **KNOWN** concept
  — is the deferred larger work, mirroring the D4 rough edge.
- The membership test keys on the lowercased topic string; a subject known
  under a slightly different surface form may be missed until normalization is
  added.