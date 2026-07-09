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
"""
