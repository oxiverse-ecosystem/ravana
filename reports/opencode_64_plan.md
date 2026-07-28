# Section 6.4 — Triplet Inference Operator Routing Plan

## 0. Executive Summary

**Recommendation: Option C (additive) with an exposure-first warmup phase.**

The triplet operator must NOT displace `_closure` (T2) or `answer_universal_syllogism` (T1).
Instead, it enters as an extra answer candidate in `_try_fact_reasoning`, gated by its own
Wilson bounds — fail-closed when profiles are cold. The T1 path (0 firings across the
9-benchmark suite) carries nothing to regress. The T2 path (195 firings, all LogiQA) carries
the entire reasoning score for that benchmark and must remain the default.

A three-phase rollout: (1) profile warmup via snapshot-time seed feeding, (2) additive
candidate wiring with A/B measurement, (3) promotion to primary only if gated contributions
improve the LogiQA score above 0.374 with zero regression on the 195-call protected set.

---

## 1. Measured Facts (code-level evidence, not plan docs)

### 1.1 Triplet Inference Operator Architecture

All 11 files read and understood at:
`ravana/src/ravana/core/triplet_inference/`

| File | Role |
|------|------|
| `core.py` | `Triple`, `RelationProfile`, `InferenceResult`, `wilson_lower` |
| `canonical.py` | `canonical_predicate`, `canonical_term` — copula collapse |
| `memory.py` | `TripletMemory` — relational index + profile store + persistence |
| `learning.py` | `ProfileLearner` — online stat update + gating predicates |
| `operators.py` | `TransitiveChain`, `SymmetricClosure`, `InversePredicate`, `Composition`, `HierarchicalInference` |
| `engine.py` | `TripletInferenceOperator` — query/ingest entry + 7 inference channels |
| `abstention.py` | `AbstentionGate` — adaptive confidence floor (mu - sigma EMA) |
| `curiosity.py` | Epistemic-value hook for ActiveInferenceController |
| `sleep.py` | `SleepSchemaExtractor` — NREM batch replay + REM sabotage |
| `seed.py` | 5 exemplar triples (stored, NOT observed — no evidence movement) |
| `__init__.py` | Public API |

**Key gating contract** (`learning.py:30-31`):
- `DECISION_BOUNDARY = 0.5` (Bernoulli more-likely-than-not)
- Gates use `wilson_lower(pos, n)` — 95% Wilson score lower bound
- 10/10 positive → wilson=0.72 > 0.5 → fires
- 2/2 positive → wilson=0.34 < 0.5 → stays closed
- No `MIN_OBSERVATIONS` constant — evidence volume priced into the bound

### 1.2 T2 Call Chain (the 195 protected calls)

```
process_turn (engine.py:~2428)
  → _try_fact_reasoning (engine.py:1929)
    → select_option (fact_reasoning.py:137, calls _closure at line 152)
    → conditional_answer (fact_reasoning.py:210, calls _closure at line 305)
```

The gate location for any T2 replacement or additive candidate:
**`engine.py:2014-2024`** — the `_frz.select_option(…) or … conditional_answer(…)` chain.

`_closure` (fact_reasoning.py:88) is bag-of-words greedy replay over raw fact texts.
It is NOT an SPO triple matcher — it operates on token sets from prose chunks.

### 1.3 T1 Call Chain (0 firings)

```
process_turn (engine.py)
  → _try_combined_fact_query (engine_memory.py:469)
    → answer_universal_syllogism (engine_memory.py:588)
```

Fired 0 times across the full suite (phase0_protected_manifest). No LogiQA case triggers
the "All X are Y" + instance pattern. Carries no regression surface.

### 1.4 Cold-Start Confirmation

Measured via `scripts/tmp_probe_profile_state.py`:

```
profiles BEFORE: 0
triples count: 5 (seeds only, unobserved)

After 1 LogiQA case:
  profiles AFTER: 2 ('is', 'when')
  trans_lower: 0.000 for both
  symmetry_n: 4 ('is'), 1 ('when') — noise triples, no chains
```

The eval snapshot (`data/ravana_eval_snapshot.pkl`) is created at
`evaluate_ravana.py:1041` right after Shakespeare training — zero `process_turn` calls,
zero triple ingestion. Every benchmark restores from this cold snapshot.

### 1.5 PropositionParser Triple Yield on LogiQA

Measured via `scripts/tmp_probe_triple_yield.py`:

| Metric | Value |
|--------|-------|
| Mean triples/case | 2.20 |
| Cases with >0 triples | 15/20 |
| Dominant predicates | `is` (18), `are` (12), `have` (3), `if` (3) |
| Triple quality | Subjects are entire text blobs; no clean SPO |

The PropositionParser uses naive regex — it captures the ENTIRE context as the subject
field (e.g. `subject="context: black americans are twice as likely..."`). These are
not useful for profile learning.

### 1.6 Persistence Confirmation

`engine.py:4760`: `triplet_inference` key in `to_dict()` save
`engine.py:5224-5229`: `from_dict()` restore in load
**Both paths work** — the snapshot DOES contain triplet profiles if they were populated
before snapshot creation. The problem is they're empty at snapshot time, not that
persistence is broken.

---

## 2. Web Research Summary

### 2.1 Constraint-Adherent Approaches Surveyed

| Source | Approach | Fits Constraints? |
|--------|----------|-------------------|
| Stanford OpenIE (Angeli 2015) | Clause splitting + natural logic over dependency parses | No — requires Java/CoreNLP |
| ReVerb (Fader 2011) | POS-constrained relation extraction | Partially — syntax-only, no new deps |
| C-lite OpenIE (in-repo) | Verb-lexicon + noun-chunk heuristics | **Already exists** at `web/openie.py` |
| PropSegm (ACL 2023) | Proposition-level segmentation + entailment | No — needs trained model |
| Rule-based atomic extraction (Kamana 2026) | Dependency pattern matching for clause splitting | Possible — could be implemented with in-repo POS |
| Clause segmentation via dep parse (IBM 2020) | Wh-handling, conjunction-split, insertion-split | Possible with spaCy but `bs4`-gated |

### 2.2 Key Insight

No lightweight rule-based system produces clean SPO triples from LogiQA-style argumentative
prose (Chinese civil service exam translated to English). The text is dense with logical
operators ("because", "however", "if...then"), counterfactuals, and set-membership relations
that resist SVO extraction. Even Stanford OpenIE produces fragmented, low-confidence
extractions from such text (Pei 2023 survey).

The `_closure` BOW mechanism works precisely BECAUSE it doesn't try to extract structure —
it treats the fact texts as bag-of-words associative patterns, which is closer to how
the hippocampus reinstates memories from partial cues (Foster & Wilson 2006 replay).

---

## 3. Recommended Plan: Option C + Exposure-First Warmup

### 3.1 Strategy

The triplet operator and `_closure` serve complementary roles:

| Aspect | `_closure` (BOW replay) | Triplet operator (learned inference) |
|--------|------------------------|--------------------------------------|
| Input | Raw fact text tokens | Clean SPO triples |
| Mechanism | Greedy lexical overlap | Wilson-gated transitivity/symmetry chains |
| Strength | Noisy prose, argumentative text | Clean relational data (taxonomy, web facts) |
| Weakness | No relational abstraction | No purchase on prose benchmarks |

**Keep both. Route to the triplet operator only when its profiles provide evidence.**
`_closure` remains the default fallback — no hard swap.

### 3.2 Phase 1: Profile Warmup

**Problem**: Empty profiles at snapshot time → all Wilson gates closed → delta=0.
**Solution**: Add a profile-warmup pass that feeds clean SPO triples INTO the snapshot.

**Warmup mechanism** (`scripts/tmp_warmup_profiles.py`, ~30 LOC):
1. After `train_engine()` and `create_snapshot()`, restore from snapshot
2. Feed a warmup corpus of clean triples through `triplet_op.ingest_triple()`:
   - ~200 curated taxonomic triples from the existing ConceptNet ontology
     (`self._cn_ontology.isa` has known is-a relations)
   - ~50 seed factual triples extracted from the graph's existing typed edges
   - 0 triples from PropositionParser (too noisy)
3. Call `learner.observe()` for each — this builds `transitivity_pos/neg` counts
4. Save updated snapshot

**Verification**: After warmup:
- `profiles` must contain `"is"` with `transitivity_lower > 0.5` (>10 positive transitivity examples from is-a chains)
- At least 3 predicates must have non-zero evidence
- `scripts/tmp_probe_profile_state.py` must show transitivity_lower > 0.5 for `"is"`

**Why this works**: The existing ConceptNet graph (`self._cn_ontology.isa`) is a dictionary
of `{word: {parent_words}}`. Each parent relation IS a clean SPO triple (e.g., "dog", "is_a", "mammal").
Chaining these (dog→mammal→animal) produces the exact transitive pattern the `ProfileLearner`
observes. After ~50 such chains, `"is"` has transitivity_pos ≈ 50, wilson_lower > 0.99.
The gate opens.

### 3.3 Phase 2: Additive Candidate Wiring

**Gate location**: `_try_fact_reasoning` at `engine.py:2014-2024`.

**Current code** (`engine.py:2014-2024`):
```python
_resp = (_frz.select_option(user_input, _texts)
         or self._graph_reasoner_answer(user_input)
         or _frz.missing_entity_abstention(user_input, _texts)
         or _frz.conditional_answer(user_input, _texts))
```

**Proposed change** (additive channel, ~15 LOC):
```python
_resp = (_frz.select_option(user_input, _texts)
         or self._graph_reasoner_answer(user_input)
         or _frz.missing_entity_abstention(user_input, _texts)
         or _frz.conditional_answer(user_input, _texts))
# Triplet operator: additive candidate, never displaces _closure.
# Fires only when Wilson gated profiles have evidence.
if _resp is None and self._use_triplet_candidate:
    _resp = self._triplet_mc_answer(user_input, _texts)
```

Plus a new method on the engine (`~20 LOC` in `engine.py`):
```python
def _triplet_mc_answer(self, user_input, fact_texts):
    """Additive MC answer from triplet operator. Fail-closed: None unless
    a Wilson gate opens."""
    if not getattr(self, 'triplet_op', None):
        return None
    # Only fires for multiple-choice input (LogiQA shape)
    if not re.search(r"\boptions?\s*:", user_input.lower()):
        return None
    # Extract options, query triplet_op for each, pick the one with
    # highest confirmation from learned inference
    _, opts = _frz._split_options(user_input)
    if len(opts) < 2:
        return None
    main = user_input.split("Options:")[0] if "Options:" in user_input else user_input
    scores = []
    for opt in opts:
        q = f"{main} {opt}"
        results = self.triplet_op.infer(main, "is", target=opt)
        if results:
            scores.append((results[0].confidence, opt))
    if scores:
        best_conf, best_opt = max(scores)
        if best_conf > 0.5:  # distribution-relative via AbstentionGate
            return best_opt
    return None
```

**ABI-compatible**: Default flag `_use_triplet_candidate = False`.
Enabled via `--triplet-candidate` flag in `evaluate_ravana.py` (mirrors `--source-trust` etc.).

### 3.4 Phase 3: Natural Profile Growth

The existing ingestion paths already feed the triplet operator:

| Path | Location | Triple Source | Quality |
|------|----------|---------------|---------|
| Conversation | `engine.py:2223-2226` | PropositionParser | **Noisy** — fixes in canonical.py help but still poor on prose |
| Web learning | `web_learning.py:992-993` | C-lite OpenIE | Moderate — clean SVO from web text |
| Sleep consolidation | `engine_generation.py:1186` | SleepSchemaExtractor | High — batch stats over stored triples |
| HRR cross-signal | `triplet_inference/learning.py:140` | HRRReasoner.query_chain | High — vector binding success/failure |

These paths accumulate evidence across sessions (persistence confirmed at 1.6).
After ~3-5 web learning sessions, `"is"` and `"has"` profiles should have enough evidence
to open their Wilson gates naturally. The phase-2 warmup accelerates this to time-zero.

### 3.5 A/B Measurement Protocol

**Baseline command** (already run, `reports/phase0_baseline.json`):
```
python scripts/phase0_reasoner_probe.py --skip-train --no-curiosity
    --semantic-grade --output reports/phase0_baseline.json
```

**A/B measurement** (after warmup + wiring):
```
# A: baseline (no triplet candidate)
python scripts/phase0_reasoner_probe.py --skip-train --no-curiosity
    --semantic-grade --output reports/phase0_ab_a.json

# B: triplet candidate ON
python scripts/phase0_reasoner_probe.py --skip-train --no-curiosity
    --semantic-grade --triplet-candidate --output reports/phase0_ab_b.json
```

**Zero-regression check**: Compare `phase0_protected_manifest.json` across A and B:
- B's `closure_fired` count must be **exactly 195** (same as A)
- B's `syllogism_fired` count must be **exactly 0** (same as A)
- No LogiQA case may score lower in B than in A

**Success criterion**: B's LogiQA score `> 0.374` with zero regression on all other benchmarks.
A negative delta means the additive candidate is still too noisy — retreat to phase 2 with
the flag OFF.

### 3.6 Estimated LOC

| Change | File | LOC |
|--------|------|-----|
| Warmup script | `scripts/tmp_warmup_profiles.py` | ~30 |
| Snapshot integration | `evaluate_ravana.py` | ~5 |
| Additive candidate method | `engine.py` | ~25 |
| Gate call in `_try_fact_reasoning` | `engine.py:2014-2024` | ~5 |
| CLI flag | `evaluate_ravana.py` | ~5 |
| Probe manifest check | `scripts/phase0_reasoner_probe.py` | ~3 |
| **Total** | | **~73** |

All changes are additive (flag-gated), no existing code modified.
No changes to `fact_reasoning.py`, `in_prompt_reasoner.py`, or `triplet_inference/`.

---

## 4. Brain-Grounded Analysis: Why the Swap Model is Wrong

### 4.1 What the Brain Actually Does

The original plan (phase out T1/T2 in favor of learned operator) assumes a single
unified relational engine. The cognitive neuroscience literature suggests otherwise:

| System | Brain Region | Function | Input |
|--------|-------------|----------|-------|
| Pattern completion | Hippocampus CA3 | Reinstates memory from partial cue | Raw featural overlap |
| Relational binding | Hippocampus + PFC | Encodes S-P-O associations | Clean item representations |
| Rule extraction | Medial PFC / caudate | Learns regularities across episodes | Abstracted relational patterns |
| Working memory | dlPFC | Maintains premise set for current reasoning | Current context only |

The `_closure` mechanism is CA3-like pattern completion — it doesn't need clean triples
because it operates on FEATURE overlap (content words), not on symbolic structure.
The triplet operator is downstream of the medial PFC / caudate — it learns regularities
OVER CLEAN relational data.

You cannot shortcut the hippocampus→cortex consolidation trajectory. The triplet operator
needs CLEAN triples because its evidence counters (transitivity_pos/neg) measure
whether (A-r-B, B-r-C) implies (A-r-C). If the r profile is noise-collapsed from
"(entire paragraph, is, noise)", no chain forms, no evidence accumulates, the gate stays closed.

### 4.2 What We Are Missing

The missing piece is a **prose-to-clean-triple bridge** — a lightweight clause segmentation
and role-labeling layer that:
1. Segments complex LogiQA sentences into atomic clauses
2. Extracts the predicate-argument structure from each clause
3. Normalizes to canonical SPO form

**Constrained implementation** (fits codebase, no new deps):
The existing `in_prompt_reasoner.py` already has many of the pieces:
- `parse_causal_edges()` extracts (cause, effect) from if/when conditionals
- `parse_universal_edges()` extracts (subclass, superclass) from "All X are Y"
- `_INST` regex captures "X is a Y" patterns

These could be refactored into a **ClauseSegmenter** that:
1. Splits on coordination conjunctions ("and", "but", "however")
2. Applies the existing in_prompt_reasoner patterns per clause
3. Feeds the extracted clean triples to the triplet operator

**However**: this is an independent improvement, not a prerequisite for section 6.4.
The additive candidate phase 2 already measures whether the warmup-fed operator
can contribute to LogiQA scores. If it can't, the ClauseSegmenter is the next
investment — not a swap delay.

---

## 5. What to Do Right Now

### 5.1 Immediate Actions

1. **Run the warmup script** (create `scripts/tmp_warmup_profiles.py`):
   - Extract ~200 is-a chains from `self._cn_ontology.isa`
   - Feed as clean triples via `triplet_op.ingest_triple()`
   - Save updated snapshot

2. **Verify warmup**: Run `scripts/tmp_probe_profile_state.py` — confirm
   `"is"` profile has `transitivity_lower > 0.5`

3. **Wire additive candidate** at `engine.py:2014-2024` behind flag `--triplet-candidate`

4. **Run A/B measurement** with the protocol in 3.5

5. **Check regression**: Compare closure_fired count against 195-call manifest

### 5.2 If A/B Shows Improvement (B > A)

- Set `_use_triplet_candidate = True` as default
- File as a single commit: "feat: triplet-inference additive MC candidate"
- Run full suite one more time to confirm no regression

### 5.3 If A/B Shows No Improvement (B ≤ A)

- Keep flag OFF
- Investigate: `scripts/tmp_probe_logiqa_coverage.py` — run triplet_op.infer() on
  each LogiQA case offline, report how many cases it could answer vs. how many
  `_closure` already answers
- The bottleneck is likely that warmup profiles (`"is"`) don't match LogiQA's logical
  predicates ("supports", "weakens", "contradicts", "assumes")
- Next: grow profiles via web ingestion of logical-relation texts, not via warmup

---

## 6. Files Referenced

### Production code (read-only, not modified)

| File | Purpose |
|------|---------|
| `ravana/src/ravana/core/triplet_inference/*.py` | 11 files — triplet operator |
| `ravana/src/ravana/core/fact_reasoning.py` | `_closure` definition + `select_option`, `conditional_answer` |
| `ravana/src/ravana/core/in_prompt_reasoner.py` | `answer_universal_syllogism` definition |
| `ravana/src/ravana/chat/engine.py` | `_try_fact_reasoning`:1929, save:4760, load:5224 |
| `ravana/src/ravana/chat/engine_memory.py` | `_try_combined_fact_query`:469, syllogism call:588 |
| `ravana/src/ravana/chat/web_learning.py` | OpenIE ingestion at line 992-993 |
| `ravana/src/ravana/web/openie.py` | C-lite OpenIE extractor |
| `ravana/src/ravana/core/proposition_parser.py` | PropositionParser — full file read |
| `ravana/src/ravana/core/hrr_reasoner.py` | HRRReasoner — query_chain for cross-signal |
| `scripts/evaluate_ravana.py` | Snapshot build (1041), restore (1053), benchmark runner |

### Probe scripts created

| File | Purpose |
|------|---------|
| `scripts/tmp_probe_triple_yield.py` | PropositionParser yield on 20 LogiQA cases |
| `scripts/tmp_probe_profile_state.py` | Triplet profile state before/after LogiQA ingestion |

### Reports

| File | Source |
|------|--------|
| `reports/phase0_baseline.json` | Full 9-benchmark run, overall 0.672 |
| `reports/phase0_protected_manifest.json` | syllogism_fired=0, closure_fired=195 |
