# Authored causal scaffold — measured, with limits

**Status: scaffolding, not a learning claim.** This is an authored set of
common-knowledge causal relations used to give RAVANA's explanation mechanism
enough trusted substrate to be testable offline. It is **not** evidence that
RAVANA learned from conversation or the web, and it must never be reported as
such.

## Brain basis

Human causal reasoning depends on prior structure: a schema guides which
relations are plausible (Griffiths & Tenenbaum 2009, Psychological Review
116:661–716) and supports one-trial learning (Tse et al. 2007, Science
316:76–82), but schema-congruence is also a source of false memories
(Johnson & Raye 1981, Psychological Review 88:67–85). The rule adopted here:
**schemas propose; sources admit.** Authored edges are therefore stamped with
their own provenance (`curriculum`, `curriculum:authored:<subject>`) and never
relabel a web or conversation edge.

## What was added

`data/curriculum/causal_chains.json`: 30 authored chains, 53 distinct causal
edges, each with an explicit evidence sentence. Loaded in `CognitiveChatEngine`
on BOTH boot paths — the persisted-weights path returns early and previously
skipped all cold-start seeding, which is why the first attempt wrote zero
edges. Web-derived edges keep their own provenance and the two-distinct-source
promotion rule; the trust gate (`provenance_trusted`, `assertable_chain`,
`grade_chain`) is untouched.

## Measured on cold boot

```
CURRICULUM_EDGES=53
backward chains from subject node:
  rust            chains=4 assertable=2 verdict=assert
  tides           chains=4 assertable=3 verdict=assert
  photosynthesis  chains=4 assertable=2 verdict=assert
  rain            chains=4 assertable=0
  moon            chains=0 assertable=0
  cloud           chains=0 assertable=0
  battery         NO_NODE
```

Test: `tests/unit/test_curriculum_scaffold.py` — 2 passed (structural validity
and provenance acceptance, no engine boot).

## Honest limits

- User-visible answers did **not** improve in the spot probes: `why does it
  rain` still returned a definition, `why does iron rust` still declined.
  Having an assertable chain in the graph is necessary but not sufficient —
  the generation route still has to select it.
- The full 60-item harness timed out twice (900 s and 3000 s) before writing
  a summary, so **no front score is claimed** for this change. The thinking
  front baseline of 8/100 remains the last valid number.
- `rain`, `moon`, `cloud` have no assertable chain from their indexed node;
  `battery` has no node. The scaffold covers only part of the probe set.
- This scaffold is static knowledge. It does not demonstrate online learning.
  Phase 2 of the plan (full-page extraction instead of search snippets) and
  RAVANA-LEARN-5 learning curves remain the real tests.

## Do not

Do not cite this as "RAVANA learned causal structure". Do not count these 53
edges as web facts. Do not use them to claim the thinking front improved.
