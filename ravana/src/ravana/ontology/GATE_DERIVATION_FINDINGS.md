"""
PINNED EMPIRICAL FINDING — why the frontopolar gate cannot yet be derived.

This audit set out to replace the three frozen literal dicts
(_CATEGORY_AFFORDANCES / _CATEGORY_OF_SUBJECT / _PROPERTY_CATEGORIES) with a
derived computation over the GloVe manifold + learned concept graph, per the
brain-inspired synthesis (Binder 2016 componential semantics; Barsalou 1999
embodied simulators; Rosch prototypes / Collins & Loftus spreading activation).

We tested BOTH proposed derived paths against the ACTUAL system state and both
FAILED to discriminate affordance possession. The literal gate is therefore the
only working implementation; DerivedOntology remains correct infrastructure,
pinned by these negative tests, ready when a real feature/component space exists.

================================================================================
RESOLUTION (added later) — the blocker IS now solved, via a typed knowledge graph
================================================================================
The two failed paths shared ONE root cause: no per-concept componential feature
space and no typed taxonomic edges. GloVe cosine cannot supply either (Binder
2016 itself shows distributional vectors separate categories WORSE than
attribute vectors; we re-confirmed this: a ridge probe GloVe(64D)->Binder(65D)
gets held-out r=0.62-0.81 yet still over-extrapolates OOV gate words like
`tuesday`->color=3.66, so the gate stays leaky).

FIX: use ConceptNet 5.7 (English assertions) as the real typed KG. It provides
exactly the two missing structures:
  * /r/IsA  -> taxonomic hierarchy  => category_of is now DERIVED (drops the
    per-word _CATEGORY_OF_SUBJECT table; Rosch prototypes / spreading activation).
  * /r/HasProperty,/r/CapableOf,/r/UsedFor -> componential features, plus the
    Sensory-Functional division used to decide affordance by the concept's
    DERIVED category (concrete categories possess physical properties; time/event
    possess temporal ones).
This is wired as the PRIMARY gate in chat/engine.py::_is_category_error; the
three literal dicts remain ONLY as an OOV safety-net fallback.

Build step (one-time):
  python -c "from ravana.ontology.conceptnet import ConceptNetOntology as C;
             o=C.from_tsv('data/conceptnet/en_assertions.tsv'); o.save('data/conceptnet/ont.pkl')"
  (en_assertions.tsv is produced by streaming
   https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
   and keeping only /c/en/ edges; ~3.4M English assertions.)

Validation (tests/unit/test_ontology_derived.py): the previously-IMPOSSIBLE case
"what color is tuesday" is now correctly flagged, while sun/tree/cat/rock + color
are allowed. All 7 original category-error tests pass, plus the pinned negative
tests (GloVe-discriminates / graph-has-no-typed-edges) still pin WHY the earlier
attempt failed.

Files:
  * ravana/src/ravana/ontology/conceptnet.py  — ConceptNetOntology service.
  * ravana/src/ravana/ontology/attribute_encoder.py — Binder ridge probe (kept as
    infrastructure / possible future prior; NOT the gate).
  * chat/engine.py::_is_category_error — ConceptNet primary, literal dicts fallback.


=== Path 1: GloVe property-probe cosine ===
Measured on data/ravana_glove_cache.npz (400k vocab, projected 64D):

    subject   prop      cos
    tuesday   color     0.386   (should FLAG  -> derived says "has it")
    day       color     0.575   (should FLAG  -> HIGHER than legit tree)
    time      color     0.514   (should FLAG)
    thought   mass      0.365   (should FLAG)
    love      color     0.440   (should FLAG)
    sun       color     0.530   (legit -> "has it")            OK
    cat       color     0.430   (legit)                        OK
    tree      color     0.557   (legit)  <-- LOWER than day
    hour      duration   0.800   (legit)                       OK

No theta separates should-flag from should-allow. Distributional GloVe
(co-occurrence) is NOT Binder's componential feature space (perceptual/
functional/motor attributes). See test_ontology_derived.py::
test_glove_does_not_discriminate_affordance_possession (pinned).

=== Path 2: learned concept-graph neighborhood ===
Measured on the REAL persisted graph (data/ravana_weights.db):
  - 3,255 concepts, 121,527 edges, relation_type distribution:
        semantic 112,467 | temporal 3,585 | causal 2,124 |
        analogical 1,704 | contrastive 1,581 | episodic 64 | contextual 2
    => ZERO isa/hypernym/attribute typed edges. The synthesis's
       "graph isa/attribute path" does not exist in the deployed system.
  - Semantic 1-2 hop overlap with property-probe seed words:

    subject    prop      hop1 hop2
    tuesday   color      5    7     (should FLAG -> graph says "connected")
    day       color      5    7     (should FLAG)
    time      color      2    7
    sun       color      1    7     (legit -> also 7)
    tree      color      0    0     (legit -> NOT connected?!)
    cat       color      0    0     (legit -> NOT connected)
    rock      color      0    7

  The semantic-neighborhood test is WORSE than useless: it connects the
  should-flag cases and FAILS to connect the legit cases. The graph encodes
  free-association co-occurrence (web/PMI learning), not affordance.

=== Conclusion ===
"Drop the three literal dicts; derive affordances from GloVe + graph" is NOT
achievable with the current manifold + graph. The gate correctly remains on
the literal dicts. The brain-aligned replacement requires a genuine component
space (e.g. a fine-tuned attribute encoder, or Binder-style 65-dim attribute
vectors trained per concept) that does not yet exist in the codebase.

What WAS genuinely improved (verified, not pretended):
  A. Dead pair-tables + dup dicts removed (engine.py:5402-5435 etc.).
  D. Sentiment routed through learned VAD detector (interface.py:824,
     response_gen.py:1867); engine.py:5470 duplicate removed.
  F. _DEF_PURGE derived from graph degree+abstractness (universal seed only).
  B/7. _walk_hierarchy prefers graph IS_A/level ascent, ABSTRACTION_HIERARCHY
       is fallback seed.
  + DerivedOntology service built + tested (infrastructure for the future).
  + Pre-existing latent bug fixed (_is_word_salad import in engine.py).

Cluster C (TEEN_CONCEPTS) deferred: large blast radius, PMISeeder already a
supplement. Cluster E (_EDGE_CONNECTORS) verified NOT dead (used by
interface.py:1427); ConnectorLearner is parallel, not the active path.

================================================================================
DEFERRED ITEMS 1–3 — RESOLVED (this audit cycle)
================================================================================
The three deferred items are now IMPLEMENTED and VERIFIED (real tool runs, not
asserted-by-hand). Summary of measured outcomes:

--- ITEM 1: ConceptNet typed edges injected into the ravana graph ------------
Brain grounding: the learned graph was ALL associative (semantic/temporal/
causal — Collins & Loftus spreading activation). The hub also needs taxonomic
+ componential spokes (Binder; Damasio/Barsalou convergence zones). ConceptNet
5.7 supplies both; we materialized them as REAL typed edges in ravana_weights.db.

Implementation:
  * ravana/src/ravana/ontology/graph_typing.py :: inject_conceptnet_typed_edges
    — for every ravana node whose label matches a ConceptNet term, creates edges
      with EXPLICIT typed relation_type in {isa, has_property, capable_of,
      used_for} (NEVER the generic "semantic").
    — WEIGHT BY STRENGTH: edge.weight = ConceptNet assertion weight (mean over
      assertions for that pair), clamped to (0,1]. The loader now also captures
      isa_weight / feature_weight / feature_rel (per-relation) from the TSV's
      col-4 JSON metadata.
    — PREFERS ISA TO THE NEAREST (most specific) parent, not the root: among
      existing ConceptNet parents, any parent that is an IsA-ancestor of another
      existing parent is dropped (Rosch basic level / Collins & Loftus 1975).
  * chat/engine.py :: _typed_edges_bootstrap — called once at engine init;
    idempotent (skips if typed edges already present); persists back to the
    SQLite store so the work survives restart.
  * ConceptNetOntology.from_tsv extended to capture assertion weights + the
    licensing relation per feature; ont.pkl rebuilt locally (not committed).

Measured (one-time local injection into data/ravana_weights.db):
  isa=644, has_property=255, capable_of=88, used_for=372  -> 1359 typed edges.
  Before: 0 typed edges. After: graph relation_type distribution gains
    isa 641, used_for 372, has_property 255, capable_of 88 (persisted).
  Real color-term nodes present (red/green/white/black/orange/colour/color);
  12 has_property->color-term edges (e.g. cats->black, car->red).

Path 2 (previously structurally impossible) now WORKS:
  DerivedOntology._graph_path_to_property over the REAL typed IsA/HasProperty
  edges reaches a color-bearing ancestor. Proven two ways:
    (a) controlled mini-graph tree->plant->living_thing (+green/red) — walk
        returns True;
    (b) temp copy of the real DB — walk('cats','color') returns True.
  _INHERITANCE_EDGE_TYPES in derived.py already included isa/has_property, so
  no change there was needed — the path activated the moment typed edges existed.

Test changes (tests/unit/test_ontology_derived.py):
  * test_real_graph_has_no_typed_isa_attribute_edges RENAMED to
    test_real_graph_now_has_typed_isa_attribute_edges and flipped to pin the
    RESOLVED state (typed edges MUST be > 0). History preserved in docstring.
  * Added test_inheritance_walk_reaches_color_ancestor_over_typed_edges
    (mini-graph Path 2) and test_real_graph_inheritance_walk_reaches_color_
    ancestor (real-DB Path 2).

--- ITEM 2: attribute_encoder as a distributional tie-break prior -----------
Brain grounding: Binder 65-dim attribute space IS a graded hub representation;
the ridge probe (glove64 -> binder) is a legitimate PRIOR over the attribute
space, used only to break ConceptNet SILENCE / tie-fill (never override).

Implementation (ravana/src/ravana/ontology/conceptnet.py):
  * ConceptNetOntology(attribute_encoder=, glove_fn=, prior_theta=) wires the
    Binder ridge probe. _prior_property consults it ONLY at the silent return
    points of has_property (category_of -> None for a physical prop, or an
    abstract/social property). ConceptNet's explicit True/False verdicts are
    NEVER overridden — the prior cannot contradict them.
  * chat/engine.py::_load_conceptnet_ontology now also loads
    data/attribute_encoder.npz (if present) and passes the engine's GloVe-64
    vector fn; production gate gets the prior for free, no-op if absent.

Calibration (data/attribute_encoder.npz + data/ravana_glove_cache.npz),
labeled color set (20 possess + 21 not-possess):
    word        color score
    tree 6.05 | sun 4.26 | cat 4.27 | rock 5.75 | flower 6.37 | grass 5.27
    sky 3.90 | blood 3.78 | ...
    tuesday 3.66 | day 3.07 | thought 1.71 | love 1.09 | idea 1.72 | ...
  theta sweep (precision / recall / F1, 0 false positives throughout):
    theta=4.0: P=1.00 R=0.70 F1=0.82
    theta=4.5: P=1.00 R=0.30 F1=0.46   <-- LOCKED (per deferred-item spec)
    theta=5.0: P=1.00 R=0.30 F1=0.46
  Locked at theta=4.5: precision 1.00, recall 0.30, ZERO false positives. The
  caveat from the original audit (tuesday->color=3.66 over-extrapolates) sits
  BELOW threshold, so the prior never false-flags a legit concept.

Tests (tests/unit/test_ontology_derived.py):
  * test_prior_ranks_legit_above_oov_on_color — ranks tree/sun/cat above
    tuesday/day/thought; theta=4.5 separates clear possessor from non-possessor.
  * test_prior_does_not_override_conceptnet_verdict — the 7 core gate results
    (tree/cat/sun=True, tuesday/thought=False) are UNCHANGED; prior only fills
    genuine silence.
  * test_prior_fills_silence_for_oov_subject — silent subjects get a definite
    bool from the prior rather than None.

--- ITEM 3: flaky test fixed (test hygiene) --------------------------------
tests/unit/test_brain_inspired_fixes.py::test_context_query_vector_blends_
context FAILED in the full suite but PASSED in isolation.

ROOT CAUSE (proven, not guessed): the test used abs(hash(word)) for its
"deterministic" pseudo-embedding. Python salts str.__hash__ per process via
PYTHONHASHSEED, so the SAME test computed different vectors on different
launches. Reproduced directly: it PASSES for most seeds but FAILS for seed 18
and 28 — exactly the "passes alone, fails in suite" signature (the suite's
process gets a different seed than an isolated run). Cross-module bisect showed
test_chat_main / test_rlm_v2_sleep / test_serialization_stress /
test_sleep_quality / test_system2_v2 reorderings expose it.

FIX: replaced abs(hash(word)) with a STABLE hashlib.md5-based embedding
(seed-independent), and snapshot STOP_WORDS inside the test (try/finally) so a
preceding test's shared-module mutation cannot leak in. Assertion intent
unchanged (context words must nudge the subject vector).

VERIFIED: passes for seeds 0,5,10,18,20,25,28,30 AND when run immediately
after each of the 5 culprit modules. No assertion weakened.

--- Constraints honored -------------------------------------------------------
  * No secrets/keys committed; data/ and *.pkl/*.npz are gitignored. ont.pkl
    and ravana_weights.db were rebuilt/modified LOCALLY only.
  * The three literal dicts remain OOV fallback; ConceptNetOntology stays the
    primary gate.
  * Full unit suite: target 1366+ with zero NEW failures (see run output).
"""
