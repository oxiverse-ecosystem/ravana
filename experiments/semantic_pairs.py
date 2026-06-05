"""
Semantic Pair Curation for Domain-Aware Encoder Training (Option 3).

Each pair connects a Science concept (Domain A) with a Social concept (Domain B)
that shares deep structural/analogical similarity. The contrastive regularizer
pulls these pairs together in the encoder's latent space while pushing apart
morphologically similar but semantically unrelated words.

Every word in these pairs is verified to exist in the training vocabulary
(experiment_cross_domain_v2.py triples).

Reference: Option 3 design from the morphological bias analysis.
"""

# ── Science → Social Domain Analogies ────────────────────────────────────
# These are the positive pairs used in L_contrastive:
#   L = -log(sim(concept_A, concept_B_analog)) + log(sim(concept_A, concept_B_wrong))
SEMANTIC_PAIRS = [
    ("friction", "conflict"),       # both create resistance
    ("heat", "anger"),              # both are intense, energetic
    ("expansion", "growth"),        # both are increase processes
    ("cold", "loneliness"),         # both are absence of warmth/connection
    ("rust", "betrayal"),           # both are degradation/corrosion
    ("combustion", "resentment"),   # both are reactive, festering
    ("pressure", "stress"),         # both are forces causing change
    ("erosion", "neglect"),         # both are gradual wearing away
    ("gravity", "loyalty"),         # both are binding/attractive forces
    ("light", "hope"),             # both enable, illuminate, guide
]

# ── Within-Domain Positive Pairs (same concept, synonymous paths) ─────────
# Used to reinforce that the encoder should map synonyms close together,
# separate from the cross-domain analogies.
WITHIN_DOMAIN_PAIRS = [
    ("heat", "warmth"),             # same domain: thermal energy
    ("anger", "resentment"),        # same domain: negative emotion
    ("trust", "loyalty"),           # same domain: positive relationship
    ("sadness", "loneliness"),      # same domain: negative affect
    ("conflict", "stress"),         # same domain: social friction
]

# ── Combined set for training ────────────────────────────────────────────
ALL_PAIRS = SEMANTIC_PAIRS + WITHIN_DOMAIN_PAIRS
