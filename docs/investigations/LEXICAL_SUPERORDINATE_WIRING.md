# Lexical-superordinate primitive: default-OFF engine wiring

**Status: implemented and unit-tested, DEFAULT-OFF. This is not an RSI/TRA PASS.**
No `CognitiveChatEngine` was constructed, no `process_turn` ran on a live store,
no store was written, no benchmark or answer key was read, and no network fetch
was made in this worktree. The parent agent owns the engine and the live store;
the live A/B below is a plan for that owner, not a result claimed here.

Branch `feat/lexical-superordinate-wiring` from `origin/main` `3465377` (PR #556,
which added the default-off primitive and its pure unit tests).

## 1. What changed

Two production files, additively:

| File | Change |
|---|---|
| `ravana/src/ravana/chat/response_gen.py` | module-level `lexical_teaching_pairs_from_graph(graph)` (read-only adapter over already-provenance-trusted edges) and `ResponseGenMixin._lexical_superordinate_candidate(raw_input, subject=None, current_response=None)` |
| `ravana/src/ravana/chat/engine.py` | one emission-boundary call in `process_turn`, after the H3/H4/typed-slot hooks and before the source monitor |

Call chain when enabled:

```
process_turn(user_input)
  -> _process_turn_impl(...)                      # unchanged baseline reply
  -> getattr(self, "_lexical_superordinate_candidate")   # defensive; skip if absent
     -> lx.enabled()                              # single source of truth for the flag
     -> parse_forced_choice(raw_input)            # existing structural parser
     -> subject resolution (staged subject | head noun | last content word of question)
     -> defer if the response is already a bare candidate (H3/H4/typed-slot won)
     -> defer if a directly taught trusted edge decides (existing selector, unchanged)
     -> lexical_teaching_pairs_from_graph(self.graph)
        -> schema_trusted_edges_from_graph -> traces_from_edges (existing trust predicate)
     -> ravana.learn.lexical_superordinate.pick(subject, options, pairs)
  -> response = pick only when the hook returned a non-None candidate
```

**Nothing else changed.** No gate, threshold (`DELTA=0.10`, `MARGIN=0.05`),
item bank, option list, answer key, `world_model`, `type_sets`, domain tag,
context model, store, or persisted weight. No new environment flag was invented:
the only switch is the module's existing `RAVANA_LEXICAL_SUPERORDINATE=1`.
`RAVANA_WORDNET_DATA` remains the optional licensed-corpus path and is read only
by the already-shipped module.

## 2. Why this hook placement (brain-grounded design)

The brain does not treat a novel word's type as a vote over other teaching words;
it infers the type from the word's own identity inside a **pre-existing structured
hypothesis space**, and it binds a one-shot trace only when a **schema already
exists** — while schema congruence alone is also a source of false memories.

The hook therefore:

1. **Scores the query subject**, not a demonstration centroid — subject-specific
   evidence (fast mapping: Carey & Bartlett 1978; Carey 2010; Coutanche &
   Thompson-Schill 2014). The module's `subject_scores` is already the
   subject-specific primitive; the wiring must not replace it with a demo score.
2. **Calibrates the admission bar from the item's own taught demonstrations**,
   read from provenance-trusted graph edges (Xu & Tenenbaum 2007; Tse et al.
   2007; van Kesteren et al. 2012). The adapter refuses untrusted edges by
   reusing the shared `_is_trusted_edge` predicate through `traces_from_edges`;
   it adds no new trust rule and never reads a gold key.
3. **Lets a schema propose but a subject-specific source admit** (Johnson et al.
   1993; Webb et al. 2016). The lexical score must clear a data-derived absolute
   criterion and a strict margin, and the hook defers to directly taught trusted
   edges and to earlier emission hooks. It never writes an edge or asserts a
   relation.
4. **Runs at the outer emission boundary**, after the existing answer hooks, so
   chatter, definitional and other early-return paths cannot bypass it and so a
   hook that already decided can never be overwritten. This is the placement rule
   learned from the earlier H4/source-monitor wiring: a hook staged deep inside
   `process_turn` receives `None` on the shortest paths.

WordNet is an engineering **substrate** for the structured lexical space, not a
brain claim; `path_similarity` and the frozen `DELTA`/`MARGIN` remain
extrapolations (see limits).

## 3. Extended literature ledger (new citations beyond the scratch doc)

Grades follow `neuroscience-to-architecture` §4: **[S]** strong, **[M]** moderate,
**[W]** weak; evidence type stated separately. The scratch doc
`LEXICAL_SUPERORDINATE_TRANSFER.md` already carried N1–N7; the entries marked
**NEW** below are the further research requested for this wiring.

| # | Finding | Source / DOI (verified this session unless noted) | Grade / type | Consequence for this hook |
|---|---|---|---|---|
| F1 **NEW** | Fast mapping establishes only an initial partial meaning; extended mapping refines it over later encounters. | Carey S. (2010), *Lang Learn Dev* 6(3):184–205, doi:10.1080/15475441.2010.484379 | [S] human behavioural (review) | do not rewrite stored demos into a prototype; keep them as separate calibration traces |
| F2 **NEW** | A novel word is rapidly integrated into existing cortical memory networks, and retrieval of an **already-known related concept** is critical to the effect. | Coutanche MN & Thompson-Schill SL (2014), *J Exp Psychol Gen* 143(6):2296–2303, doi:10.1037/xge0000020 | [S] human behavioural + fMRI | the demonstration words must be real stored items; a bare word string is insufficient |
| F3 **NEW** | Novel word–picture associations can be acquired rapidly and hippocampal-independently by fast mapping; anterior-temporal damage abolishes the advantage. | Sharon T, Moscovitch M, Gilboa A (2011), *PNAS* 108(3):1146–1151, doi:10.1073/pnas.1005238108 | [S] human neuropsychological | fast mapping is a cortical, not hippocampal, route; consistent with using a lexical store rather than new episodic writes |
| F4 | A novel word's type is inferred over a structured, tree-like hypothesis space; three Dalmatians are informative because `dog ⊂ animal` already exists. | Xu F & Tenenbaum JB (2007), *Psychol Rev* 114(2):245–272, doi:10.1037/0033-295X.114.2.245 (Crossref verified earlier); **NEW companion:** Xu & Tenenbaum (2007), *Dev Sci* 10:288–297, doi:10.1111/j.1467-7687.2007.00590.x | [S] behavioural + [M] computational | use a pre-existing lexical hierarchy; score the subject itself |
| F5 **NEW** | Bayesian word learning is inference over a constrained hypothesis space, not associative averaging. | Gopnik A & Tenenbaum JB (2007), *Dev Sci* 10(4), doi:10.1111/j.1467-7687.2007.00584.x | [M] computational | an argmax-with-criterion decider is closer to the mechanism than a centroid similarity |
| F6 | One-trial traces become durable only when a neocortical schema already exists. | Tse D et al. (2007), *Science* 316(5821):76–82, doi:10.1126/science.1135935 | [S] rodent neurobiology | calibrate from demonstrations actually taught; do not invent a prior |
| F7 **NEW** | Schema-dependent gene activation and memory encoding in neocortex. | Tse D et al. (2011), *Science* 333:891–895, doi:10.1126/science.1205274 | [S] rodent neurobiology | the schema is a pre-existing cortical structure, not a per-item template |
| F8 **NEW** | Schema and novelty jointly augment memory formation; prior knowledge speeds assimilation but also biases toward gist. | van Kesteren MTR, Ruiter DJ, Fernández G, Henson RN (2012), *Trends Neurosci* 35(4):211–219, doi:10.1016/j.tins.2012.02.001 | [S] review (human + rodent) | keep the demos separate from the query evidence; never average them |
| F9 | Fast and slow systems are complementary (complementary learning systems). | McClelland JL, McNaughton BL, O'Reilly RC (1995), *Psychol Rev* 102(3):419–457, doi:10.1037/0033-295X.102.3.419 | [M] computational framework | demos remain calibration episodes, not a learned prototype |
| F10 | Item memory and source memory dissociate. | Johnson MK, Hashtroudi S, Lindsay DS (1993), *Psychol Bull* 114(1):3–28, doi:10.1037/0033-2909.114.1.3 | [S] human behavioural | a schema-congruent candidate is not thereby an observed one |
| F11 **NEW** | Schema gist is mediated by left temporal cortex for **both** true and false recollection; non-schematic recollection needs effortful monitoring. | Webb CE, Turney IC, Dennis NA (2016), *Neuropsychologia* 93(Pt A):61–75, doi:10.1016/j.neuropsychologia.2016.09.023 | [S] human fMRI | strongest reason the hook must only propose and must defer to provenance-admitted edges |
| F12 | WordNet organises lexical concepts in a semantic relation network usable by machine. | Miller GA (1995), *CACM* 38(11):39–41, doi:10.1145/219717.219748; Fellbaum C (ed.) (1998), MIT Press (previously verified) | [M] resource description | implementation substrate only; no brain claim |

**What the evidence does not license.** It does not license the claim that the
brain computes WordNet `path_similarity`, that `DELTA=0.10`/`MARGIN=0.05` are
neural constants, or that this hook repairs RSI/TRA. F2 and F3 are about
*integration of a newly learned item*; this hook only proposes an answer for an
unseen subject. F6–F8 are largely rodent; F11 is human fMRI. The defensible claim
is narrow: **subject-specific input + a structured hypothesis space +
teaching-calibrated admission + proposal-only authority**.

## 4. Tests

`tests/unit/test_lexical_superordinate_wiring.py` — 18 new tests, no engine boot,
no store, no WordNet corpus. They call the **real** shipped methods as unbound
functions on tiny fakes and drive the **real** `process_turn` wrapper on a
stand-in that stubs only the expensive internals. Coverage:

* read-only adapter: dedupes repeated demonstrations, excludes untrusted edges,
  has no answer-key parameter;
* hook: inert with flag off, emits from stored demonstrations when on, abstains
  with no teaching pairs or no lexical resource, defers to a bare candidate and
  to a directly taught trusted edge, ignores non-forced-choice and single-option
  input, fails closed on a corrupt graph, resolves the subject from the question
  and from a phrase head noun, and exposes no key parameter;
* boundary: call counter proves the hook is **never called** with the flag off
  and called exactly once with it on; flag-off/flag-on are byte-identical when
  there is no evidence; engines lacking the hook stay unchanged; chatter turns
  stay unchanged.

Command actually run (from this worktree, note `PYTHONPATH` verified by printing
the imported modules' `__file__`, which resolved to this worktree):

```
PYTHONPATH=<worktree>/ravana/src;. <main>/.venv/Scripts/python.exe -m pytest \
  tests/unit/test_lexical_superordinate_wiring.py \
  tests/unit/test_lexical_superordinate.py \
  tests/unit/test_schema_boundary.py \
  tests/unit/test_source_evaluation_wiring.py -q
# 39 passed
```

A wider 18-module regression batch (hook/wiring/boundary/engine/source/schema/
slot/trusted/turn tests) gave **191 passed, 4 failed**. The same four failures
reproduce on a clean `origin/main` worktree in this environment
(`test_prediction_error_wiring`, `test_stance_reconsolidation_engine`, and two
`test_web_snippet_source_quality` cases — offline GloVe / missing corpus), so
they are pre-existing and not attributable to this diff.

## 5. Live A/B plan for the parent agent (owner of engine and store)

**Do not enable the flag on the strength of this document.** Execute this plan
on the parent's own store and decide from the endpoints.

1. **Pre-flight, before any arm.** Confirm a licensed WordNet 3.0 corpus is
   present and `RAVANA_WORDNET_DATA` points at it; assert
   `lexical_superordinate.wordnet_available() is True`. Record the revision,
   the config line (including the all-flags-off arm) and the frozen input
   hashes. Do not run the A/B if the corpus is missing — the primitive fails
   closed and every arm will be byte-identical, which measures nothing.
2. **Isolate state.** Give every arm its **own fresh copy** of the weight store
   in its own `data_dir`; never point two arms at the same live store, and never
   at the parent's production store. Run two repeated OFF arms first and require
   byte-identical outputs before attributing any delta to the flag.
3. **Arms differ in exactly one flag.** OFF: `RAVANA_LEXICAL_SUPERORDINATE`
   unset. ON: `=1`. All other flags, harness gates, item banks, thresholds,
   k-levels, seeds and turn sequences identical. Print the config line for both.
4. **Reachability first.** Wrap `self._lexical_superordinate_candidate` with a
   counter. Require `calls == 0` on OFF and `calls > 0` **and** `emits > 0` on
   ON; log subject, options, number of teaching pairs, `theta`, scores, pick and
   reason per item (use `schema_criterion`/`decide` directly for the audit log —
   the production hook stays pure). Zero calls means the wiring is unreachable;
   positive calls with zero emits means the store has no usable lexical
   evidence, which is a finding, not a tuning problem.
5. **Positive endpoints** (unchanged shipped harness): RSI and TRA accuracy on
   the primary pool and on the sealed pool, reported separately, with emit
   coverage (n emitted / n gradeable) as the headline for a precision route.
   Also report paired delta on the items where OFF and ON both emitted.
6. **Negative controls that must stay flat**: foil / none-of-the-above
   rejection; negated teaching → 0 emits; unrelated subject → 0 emits; OOV
   options → 0 emits; candidate-order permutation → 0 changed picks; gold-key
   permutation **inside each displayed option set** → picks unchanged and
   accuracy collapses; non-forced-choice turns and CAR chain articulation
   byte-identical; WFA fronts unchanged. Instrument controls must be certified
   on the fake/oracle arm and the live arm must still be allowed to report FAIL.
7. **Compute parity**: equal `process_turn` call counts, equal store snapshots
   at arm start, warm the lexical cache in both arms before timing. WordNet
   lookups are extra work in the ON arm only — report the ratio, but do not
   relax any bound to make a gain survive.
8. **Decision rule**: keep the flag OFF unless ON improves primary **and**
   sealed accuracy with every negative control flat and all gates OK. If ON
   fires but the metric does not move, record it as a finding (the hypothesis
   was refuted) and do not enable. If only an oracle arm moves, the missing
   input is subject-specific type evidence in the live store, not a code fix.
   A win with this proxy key is necessary-direction evidence only; it is not an
   RSI/TRA gate PASS until the parent's own relation-based grader says so.

## 6. Falsifiers (pre-registered; each has a measurable prediction)

| ID | Test | Prediction if the mechanism/hook is sound | What a failure means |
|---|---|---|---|
| X1 | OFF vs ON with no corpus or no teaching pairs, identical store | byte-identical emitted text on every item | flag leakage or an ungated side effect |
| X2 | call counter on forced-choice items, flag ON | `calls > 0`; on the scratch-matched subset `emits > 0` | boundary wiring unreachable (early return above the call) |
| X3 | insert one untrusted edge with a high-looking label pair | pick unchanged | provenance filter broken — schema admitting without a source |
| X4 | permute the gold key within each displayed option set | picks byte-identical, accuracy collapses toward chance | answer-key leakage into the hook |
| X5 | permute the query subject across items | ≥80% of emitted picks change (scratch measured 0.95) | path degenerated into the subject-independent comparator |
| X6 | negated teaching, unrelated subject, OOV options | 0 emits in each | fail-closed contract broken |
| X7 | plant a directly taught trusted edge for one candidate while ON | hook defers; the trusted edge's candidate wins | safety hierarchy inverted |
| X8 | ON minus OFF on primary and sealed accuracy | both > 0 with gates OK before enabling | the capability claim is false live; keep OFF |
| X9 | equal `process_turn` counts and store snapshots; timing ratio reported | counts equal; no unexplained extra turns | compute confound, not a mechanism gain |

## 7. Limits

* The scratch precision (0.700, +0.315 over the subject-independent comparator)
  was measured on a **Wikidata-type-sharing proxy key**, not on RSI/TRA's
  original role relations. Coverage there was 32%; this is a precision route,
  never a saturated capability.
* The live graph's trusted edges are not guaranteed to be WordNet-known. If all
  teaching pairs are OOV, `schema_criterion` has no criterion and the hook
  abstains by design. That is an information deficit, not a threshold to lower.
* `path_similarity` is a lexical-relatedness proxy; an independent lexical
  resource with different structure could reverse the margin without touching
  any code (scratch falsifier retained).
* The hook reads the whole trusted edge set on each fired turn. With a very
  large store this may be slow; the corpus was bounded by the flag being
  default-off and by the emission boundary (one call per turn), not by an
  indexing optimisation. Measure before optimising.
* No engine was booted here, so no live endpoint is claimed in this document.

## 8. Files

* `ravana/src/ravana/chat/response_gen.py` — adapter + hook method
* `ravana/src/ravana/chat/engine.py` — one emission-boundary call
* `tests/unit/test_lexical_superordinate_wiring.py` — 18 wiring tests
* `scratch/apply_lexical_wiring.py`, `scratch/refine_lexical_subject.py` — one-shot
  newline-preserving edit scripts (scratch is gitignored; not committed)
* this document
