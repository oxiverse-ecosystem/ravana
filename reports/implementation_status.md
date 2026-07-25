# RAVANA De-Hardcode — Implementation Status (final, 2026-07-22)

**Repo:** C:/Users/Likhith/Documents/projects/ravana
**Regression gate:** `python -m pytest tests/test_dehardcode_plan.py -q` → 21 passed / 1 failed (the 1 = pre-existing `test_meaning_of_life_not_dict_dump`, never a regression)
**Unit-audit suite:** `tests/unit/test_derived_ontology_audit.py` → 5 failures, ALL PRE-EXISTING on HEAD (see "Known blockers").

## What shipped (commits, all green at the gate)

| Commit | Item | What changed |
|---|---|---|
| e8ff1c9 | Modularization | engine.py (4000+ lines) split into `CognitiveChatEngine` + 8 mixins (`engine_prompt.py`, `engine_reasoning.py`, `engine_generation.py`, `engine_graph.py`, `engine_web_search.py`, `engine_ontology.py`, `engine_self_query.py`, `engine_perception.py`). Verified 153 method bodies byte-identical via `scripts_split_engine/verify_bodies.py`. |
| 0b0501f | P0 | Switched 4 hardcoded flags to live/learned by default: learned POS (`--no-learn-pos` to disable), source-trust (`--no-source-trust`), intent-router (`--no-intent-router`), salad `use_salad_classifier` ON. |
| c686dc8 | P1-I | Deduped `_UNIVERSAL_PURGE` / `_DEFINITION_ASSERTION` into `constants.py` (single source). |
| a1a03fb | P1-G | Wired `ConnectorLearner` (cold-start = hand-map; learned OOV entries overlay after real learning). |
| b6f631f | P2-E | Replaced 5 fixed cutoffs with `self._adaptive_gate(...)`: EMA precision-tracking, mu starts == old constant (cold-start identical), precision-weighting drives drift. Directly serves "no fixed thresholds". |
| 375e8c1 | P2-C | Revived the dead `_calibration_error` signal + adaptive calibration window (was shadowed by `==` → now `>=`); logits-layer gate collapses to the proven constant at cold-start. |
| **9f47e58** | **P1-H + P3-F (+ D verified)** | **See below.** Also resolved a pre-existing in-progress git stash conflict on tracked files. |

## P1-H — closed-class lists consolidated (9f47e58)
- Extended `data/functional_lexicon.json` with 6 new categories (verbatim copies of the hand lists): `common_words` (209), `topic_skip` (102), `subject_context` (234), `conditional_frame` (59), `attr_words` (25), `grammatical_concepts` (200) — plus the 6 pre-existing polarity/moral/framing keys (preserved).
- Added 6 accessors to `FunctionalLexicon` + `CognitiveChatEngine._closed_class(name)` delegation method: prefers the fit file, falls back to the original class attribute (cold-start identical; external modules like `chain_walker.py` / `interface.py` that read the class attr via MRO are untouched).
- Routed the verifiable engine-mixin usage sites through `_closed_class`: `engine_reasoning.py` (`common_words`, `conditional_frame`×4), `engine_web_search.py` (`attr_words`, `grammatical_concepts`×2).
- Deliberately NOT routed: `_SUBJECT_CONTEXT_WORDS` (used in a `@classmethod` via `cls.`), and external `chain_walker.py` / `interface.py` grammatical usages — they keep the class attr.

## P3-F — shape-junk repetition (9f47e58)
- Added `_is_repetition_pattern()` to `junk_scorer.py` and wired it into `_structural_floor()` (additive, cold-start-safe). Flags perfect cyclic repeats (`xkxk`, `abab`, `olol`, `testtest`) → raises the junk floor. Correctly does NOT flag real words (`banana`, `mississippi`, `bookkeeper` stay 0.00). Vowel-ratio / digit-density / keyboard-mash were already present.

## P3-D — sensorimotor: NO CODE CHANGE NEEDED (verified)
The brain-based solution is ALREADY implemented and wired at runtime:
- `engine_generation.py` lazy-loads `data/lancaster_encoder.npz` (the local 40k-word Lancaster sensorimotor norms) and prefers the HUMAN 11-D norms over the merged probe for in-vocab words.
- `engine.py:1184` wires `VerbLexicon.set_sensorimotor_fn(self._lancaster_vector)` + `surface_realizer.set_sensorimotor_fn(...)`.
- `_PROP_TO_BINDER` / `_SENSORY_DIM_PHRASE` are introspection-PHRASE templates ("looks"/"see") — presentation layer, correctly hand-authored, not derivable from norms.
A parallel `sensorimotor_lexicon.json` fit (as the research plan suggested) would be redundant and risk regressing this verified read-out. So D is marked DONE-by-architecture.

## P2-B — consult domain knowledge: NO CODE CHANGE NEEDED (verified)
Continuous web-learning of domain knowledge (health/programming) is ALREADY wired via `_bg_learning_queue` / `_pending_learning_queue` (engine.py:2458, 2810, 3127). A synchronous consult→web fallback (as the plan suggested) would violate cold-start preservation and add latency; not added.

## Known blockers (NOT caused by this work — flagged, not faked)
1. **Unit-audit suite has 5 PRE-EXISTING failures on HEAD** (`test_snippet_gate_learned_with_flag_on`, `test_source_trust_learned_*`, `test_learned_pos_*`). Root cause: `engine_web_search.py:997` (`_snippet_is_structural_junk`) references `self._snippet_structure_model`, which `__init__` never sets → `AttributeError`. Same 5 fail before AND after this batch. Out of P1-H/P3 scope; needs a separate fix (init the attribute or guard the access).
2. **2 generated artifacts deliberately excluded from commit** (per review): `docs/RAVANA_STATUS.md` (standalone status doc) and `experiment_results/cross_domain_transfer.json` (regenerable benchmark output, only read by plotting scripts). They remain on disk, unstaged.
3. **`docs/ARCHITECTURE.md`** is a track-ed local modification in limbo (the in-progress stash conflict had no valid "their" side for it). Left as-is, not committed.

## Verification evidence
- `scripts_split_engine/verify_bodies.py` → 153/153 method bodies byte-identical (modularization).
- Full regression gate: 21 passed / 1 failed (pre-existing) — run after every commit.
- `_is_repetition_pattern` smoke: flags `xkxk`/`abab`/`olol`/`testtest`, does NOT flag `banana`/`mississippi`/`bookkeeper`.
- Lancaster read-out: confirmed loaded from local norms (no external download; the prior subagent's plan wrongly proposed a redundant JSON fit).
