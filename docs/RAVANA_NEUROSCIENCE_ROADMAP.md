# RAVANA — Research-Backed Roadmap to Human-Like Chat

> Generated: 2026-06-19 | Based on full codebase audit + neuroscience literature review

---

## TABLE OF CONTENTS

1. [Core Problem Diagnosis](#1-core-problem-diagnosis)
2. [What's Already Implemented (Codebase Audit)](#2-whats-already-implemented)
3. [Neuroscience Research Foundation](#3-neuroscience-research-foundation)
4. [P0: Held-Out Generalization (8.3% → 85%+)](#4-p0-held-out-generalization)
5. [P0: Complete Verb-Offset System](#5-p0-complete-verb-offset-system)
6. [P1: Production-Grade Syntactic Pipeline](#6-p1-production-grade-syntactic-pipeline)
7. [P1: Theory of Mind & Personalization](#7-p1-theory-of-mind--personalization)
8. [P2: Low-Rank W_rel & LSH Scoring](#8-p2-low-rank-w_rel--lsh-scoring)
9. [P2: Emotional Mirroring & Relationship Depth](#9-p2-emotional-mirroring--relationship-depth)
10. [P3: Benchmark Harness](#10-p3-benchmark-harness)
11. [Implementation Priority Matrix](#11-implementation-priority-matrix)

---

## 1. Core Problem Diagnosis

### The Template Problem

| Source | What's happening | Root Cause |
|--------|-----------------|------------|
| **Content** | Graph walks use generic semantic edges ("connects with", "relates to", "links to") | Decoder not trained enough → falls through to `_graph_fallback_response()` |
| **Verb phrases** | Fixed seeded list per relation type (`_KEYWORD_MAP` dict) | No learned verb embeddings, only keyword matching |
| **Structure** | PFC always plans 3-sentence discourse (explain→elaborate→contrast) | `PrefrontalWorkspace` has fixed discourse templates; no learned discourse plans |
| **Neural decoder** | Used only when `training_count >= 1000`; otherwise graph fallback | Decoder needs 1000+ sentences before activation; seen ~2000 so threshold is borderline |

### Why This Happens

```
interface.py:_generate_response()
  ├── neural decoder attempt (requires ≥1000 training samples)
  ├── reasoning loop (web learn + decoder retry)
  └── _graph_fallback_response() ← ALWAYS reached if decoder fails
       └── produces: "{subject} relates to {concept}."
```

The system **works as designed** — graph walk + syntactic pipeline = grammatical but formulaic.

---

## 2. What's Already Implemented

### P0 Fixes (VERIFIED WORKING)

#### A. Verb-Offset Predictor — Expanded to ALL Verbs
- **File:** `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (lines 1613-1718)
- **Mechanism:** `offset(verb) = avg(target_embed - subject_embed)` for every verb seen during training
- **Blending:** Logistic S-curve `count/(count+5)` blends verb offset with W_rel output
  - count=1 → 0.17 weight  (mostly W_rel)
  - count=5 → 0.50 weight  (equal blend)
  - count=10 → 0.67 weight (verb offset dominates)
  - count=30 → 0.86 weight (nearly pure verb offset)
- **Neural basis:** Complementary Learning Systems — hippocampus (episodic verb offsets) + neocortex (slow W_rel integration)

#### B. Hierarchical Semantic Prototypes
- **File:** `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (lines 487-755)
- **Mechanism:** Novel entities inherit edges from nearest prototype with discounted confidence
- **Implementation:** `_init_default_prototypes()` clusters first 50 nodes by similarity > 0.6; `_inherit_from_prototype()` copies outgoing edges with `confidence *= 0.5 * similarity`
- **Called in:** `_get_or_create_concept()` at line 1209 — every new concept automatically inherits

#### C. Entity-Specific Adapters (Low-Rank)
- **File:** `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (lines 1227-1347)
- **Mechanism:** rank=16 adapters (U, V) per subject token; nearest-neighbor initialization for held-out subjects
- **Test-time adaptation:** `_adapt_entity_adapter_at_test_time()` uses verb-offset MSE to adapt adapter

#### D. Cross-Domain BFS Traversal
- **File:** `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (lines 2483-2556)
- **Mechanism:** N-hop BFS from subject, first hop follows ALL edges, subsequent hops follow only matching relation type; predicate matching boosts same-verb edges 2.5x
- **Hub suppression:** penalizes low-activation high-in-degree noise nodes

#### E. Sleep Consolidation
- **File:** `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (lines 3820-3974)
- **Phases:** Hippocampal replay → homeostatic downscaling → weak edge pruning → anti-Hebbian pruning → phantom node pruning → encoder alignment → drift defense → episodic→semantic consolidation

#### F. Representation Alignment (Bridge Alignment)
- **File:** `ravana_ml/src/ravana_ml/nn/rlm_v2.py` (lines 4044-4376)
- **Mechanism:** Contrastive learning over graph topology pairs + semantic pairs + validation queries; 5 negative samples per positive (3 random + 2 hard); early stopping with patience 3

#### G. Cognitive Modules
| Module | File | Neuroscience Basis |
|--------|------|-------------------|
| VAD Emotion | `ravana/src/ravana/core/emotion.py` | Active inference, valence-arousal-dominance |
| Sleep Consolidation | `ravana/src/ravana/core/sleep.py` | Hippocampal replay (Wilson & McNaughton) |
| Dual Process | `ravana/src/ravana/core/dual_process.py` | System 1/System 2 (Kahneman) |
| Global Workspace | `ravana/src/ravana/core/global_workspace.py` | Baars' Global Workspace Theory |
| Metacognition | `ravana/src/ravana/core/meta_cognition.py` | 4 epistemic modes, bias detection |
| Identity | `ravana/src/ravana/core/identity.py` | Self-coherence, free-energy minimization |
| Meaning | `ravana/src/ravana/core/meaning.py` | Accumulated dissonance reduction |

#### H. Language Modules
| Module | File | Neuroscience Basis |
|--------|------|-------------------|
| BasalGangliaGate | `ravana/src/ravana/language/basal_ganglia.py` | Go/NoGo competitive gating (GODIVA model) |
| CerebellarNgram | `ravana/src/ravana/language/cerebellar_ngram.py` | Sparse bigram/trigram sequence learning |
| SyntacticCellAssembly | `ravana/src/ravana/language/syntactic_cell_assembly.py` | Pulvermüller neural syntax |
| PrefrontalWorkspace | `ravana/src/ravana/language/prefrontal_workspace.py` | Baddeley's working memory, discourse planning |
| SurfaceRealizer | `ravana/src/ravana/language/surface_realizer.py` | SVA, articles, pronouns, tense rules |

---

## 3. Neuroscience Research Foundation

### 3.1 Complementary Learning Systems (CLS)
- **McClelland, McNaughton & O'Reilly (1995):** "Why there are complementary learning systems in the hippocampus and neocortex"
- **Key insight:** Hippocampus rapidly encodes novel patterns (pattern-separated); neocortex slowly integrates into semantic prototypes
- **RAVANA mapping:**
  - `_verb_accum_buffer` = hippocampal episodic buffer (rapid verb offset encoding)
  - `_verb_offsets` → `W_rel` consolidation = sleep-dependent neocortical integration
  - `_episodic_triples` → edge strengthening during sleep = hippocampal replay

### 3.2 Hub-and-Spoke Model of Semantic Memory
- **Patterson, Nestor & Rogers (2007):** "Where do you know what you know?"
- **Lambon Ralph, Jefferies, Patterson & Rogers (2017):** "The neural and computational bases of semantic cognition"
- **Key insight:** Anterior temporal lobe (ATL) acts as supramodal semantic hub; modality-specific spokes (visual, motor, auditory) feed into it
- **RAVANA mapping:**
  - Current: ConceptGraph nodes = hub, GloVe embeddings = spokes
  - Upgrade: Distilled Sentence Transformer replaces GloVe → context-aware semantics
  - Prototype hierarchy mirrors ATL's graded semantic organization

### 3.3 Mirror Neuron System & Emotional Rapport
- **Gallese & Goldman (1998):** "Mirror neurons and the simulation theory of mind-reading"
- **Rizzolatti & Craighero (2004):** "The mirror-neuron system"
- **Key insight:** Observing/experiencing an emotion activates overlapping neural populations → automatic mimicry builds rapport
- **RAVANA mapping:**
  - Current: VAD emotion update responds to user sentiment keywords
  - Upgrade: Emotional mirroring loop — detect user arousal → modulate concept exploration breadth, response temperature, verbosity

### 3.4 Basal Ganglia Go/NoGo Gating
- **Frank, Loughry & O'Reilly (2001):** "Interactions between frontal cortex and basal ganglia in working memory"
- **Redgrave, Prescott & Gurney (1999):** "The basal ganglia: a vertebrate solution to the selection problem"
- **Mink (1996):** "The basal ganglia: focused selection and inhibition of competing motor programs"
- **Key insight:** Direct pathway (Go) disinhibits thalamus → permits action; indirect pathway (NoGo) increases inhibition → suppresses action
- **RAVANA mapping:**
  - BasalGangliaGate: Go/NoGo competitive queuing for response candidate selection
  - Goal: Dynamic gating of discourse plans based on confidence/novelty

### 3.5 Pulvermüller's Neural Syntax
- **Pulvermüller (1999):** "Words in the brain's language"
- **Pulvermüller (2010):** "Brain embodiment of syntax and grammar"
- **Buzsáki (2010):** "Neural syntax: cell assemblies, synapsembles, and readers"
- **Key insight:** Syntactic categories are encoded as distributed cell assemblies; sequence patterns (subject→verb→object) are learned via Hebbian co-activation
- **RAVANA mapping:**
  - SyntacticCellAssembly: role matrices + sequence patterns for SOV word order
  - Goal: Multi-language syntactic frames (SVO, SOV, VSO)

### 3.6 Prefrontal Cortex & Working Memory
- **Baddeley & Hitch (1974):** "Working Memory" — Central Executive + Phonological Loop + Visuospatial Sketchpad
- **Baddeley (2000):** "The episodic buffer: a new component of working memory?"
- **Miller & Cohen (2001):** "An integrative theory of prefrontal cortex function"
- **Key insight:** PFC maintains task-relevant information through sustained firing; conflict monitoring resolves competing action plans
- **RAVANA mapping:**
  - PrefrontalWorkspace: capacity-limited (7±2), discourse plan maintenance
  - Goal: Episodic buffer integration for cross-turn context binding

### 3.7 Sleep & Memory Consolidation
- **Wilson & McNaughton (1994):** "Reactivation of hippocampal ensemble memories during sleep"
- **Tononi & Cirelli (2006):** "Sleep function and synaptic homeostasis"
- **Rasch & Born (2013):** "About sleep's role in memory"
- **Key insight:** SWS = hippocampal replay + synaptic downscaling; REM = emotional consolidation + cortical integration
- **RAVANA mapping:**
  - Already implemented: SWS (replay, downscaling, pruning) + emotional consolidation
  - Goal: Explicit REM/SWS phase separation with distinct functions

### 3.8 Spreading Activation in Semantic Networks
- **Collins & Loftus (1975):** "A spreading-activation theory of semantic processing"
- **Anderson (1983):** "A spreading activation theory of memory"
- **Key insight:** Concepts are nodes in a network; activation spreads along weighted edges with decay; closely related concepts prime each other
- **RAVANA mapping:**
  - PropagationEngine + `spread_activation()` = brain's automatic semantic priming
  - Phase 0/1/2 spreading = bottom-up/top-down attention interaction

### 3.9 Theory of Mind & Social Cognition
- **Frith & Frith (2006):** "The neural basis of mentalizing"
- **Saxe & Kanwisher (2003):** "People thinking about thinking people"
- **Key insight:** mPFC, TPJ, and precuneus form the mentalizing network; we maintain models of others' beliefs, goals, emotions
- **RAVANA mapping:**
  - Currently: Basic UserModel in `scripts/ravana_chat.py`
  - Goal: Full Theory of Mind with belief tracking, goal inference, emotional state estimation

### 3.10 Semantic Hub Hypothesis in LLMs (2025)
- **Wu et al. (2025):** "The Semantic Hub Hypothesis" (ICLR 2025)
- **Key insight:** Language models learn shared representation spaces across languages and modalities, analogous to the ATL hub
- **RAVANA mapping:**
  - Prototype hierarchy + embedding space = artificial semantic hub
  - Goal: Multi-modal extension (text + image + audio concepts in same hub)

---

## 4. P0: Held-Out Generalization (8.3% → 85%+)

### Current State

```
_rp_forward_verb_offset() supports ALL verbs (P0 fixed)
_verb_offsets[domain_id][stem] = averaged offset vector
blend_weight = verb_count / (verb_count + 5.0)  ← logistic S-curve
```

**Problem:** Novel entities still fail when they have no GloVe embedding and no prototype match.

### Fix Plan

#### A. Expand Prototype Hierarchy (Deeper Levels)
**Current:** 1 level (flat clustering of first 50 nodes)
**Target:** 3+ levels (concrete → category → supercategory)

**Implementation in `rlm_v2.py`:**
```
_prototype_levels = {"animal": 2, "mammal": 1, "dog": 0}
```
- Level 0 = specific (dog, cat)
- Level 1 = category (mammal, bird) 
- Level 2 = supercategory (animal, living_thing)

**Inheritance cascade:** Novel entity → nearest Level 0 prototype → copy edges → if no match, try Level 1 → if no match, try Level 2

#### B. Add ConceptNet/Ontology Bootstrap
Load lightweight semantic ontology at init:
```
_ontology_edges = {
    "encryption": [("security", "isa", 0.7), ("protects", "causal", 0.6), ("data", "contextual", 0.5)]
}
```
Carry low initial confidence (0.2-0.3); reinforce through Hebbian updates.

#### C. Sleep-Based Novel Entity Promotion
During `sleep_cycle()`:
- Find novel entities with `access_count > 3` during wake
- Promote to full concept: merge with nearest prototype, inherit all edges
- Prune novel entities that were never accessed

---

## 5. P0: Complete Verb-Offset System

### Current State
- ALL verbs supported (P0 fix complete)
- Domain-specific verb offsets (`_verb_offsets[domain_id][stem]`)
- Logistic blending with W_rel: `weight = count/(count+5)`
- Test-time adapter adaptation for held-out subjects

### What's Missing

#### A. Verb Offset for Compounds
**Problem:** "leads_to", "results_in", "contributes_to" are stored as single tokens but never appear as verbs in training data.
**Fix:** Split compound predicates into component stems and accumulate offsets for each:
```
"leads_to" → stems: ["lead", "to"]
offset = avg(offset("lead") + offset("to"))
```

#### B. Cross-Verb Generalization
**Problem:** "causes" and "produces" have separate offset vectors even though they're semantically similar.
**Fix:** Cluster verb stem offsets by cosine similarity during sleep; merge near-identical offsets:
- After sleep, compute similarity matrix of all verb offsets
- Merge clusters with similarity > 0.85 by weighted averaging
- This generalizes "causes" experience to "generates" predictions

#### C. Verb Offset Uncertainty Estimation
Current blending uses count-based logistic (deterministic). Add variance tracking:
```
_verb_offset_variance[domain_id][stem] = variance of offset samples
```
When variance is high (offset unreliable), suppress blending weight even if count is high.

---

## 6. P1: Production-Grade Syntactic Pipeline

### Current State

All 5 language modules exist in `ravana/src/ravana/language/` but `_generate_with_decoder_and_syntax()` in `interface.py` (line 1028) is a stub:

```python
def _generate_with_decoder_and_syntax(self, ctx):
    return self.decoder_engine.generate(ctx, self.graph_engine) if hasattr(...) else None
```

The actual response falls through to `_graph_fallback_response()` which produces:
> "Fire relates to heat. Heat relates to expansion. Expansion is interesting."

### Full Pipeline Implementation

```
User Input
    │
    ▼
[1] PrefrontalWorkspace ─────────────── Discourse Plan
    │                                        │
    │  Plans next 3-5 utterances            │
    │  with discourse intents:              │
    │  - EXPLAIN(heat→expansion)            │
    │  - ELABORATE(expansion→pressure)      │
    │  - CONTRAST(heat vs cold)             │
    │                                        │
    ▼                                        ▼
[2] SyntacticCellAssembly ─────────── Sentence Frames
    │                                        │
    │  Builds: [SUBJ] [VERB] [OBJ]          │
    │  With agreement: "heat causes"         │
    │  (not "heat cause")                    │
    │                                        │
    ▼                                        ▼
[3] BasalGangliaGate ──────────────── Candidate Selection
    │                                        │
    │  Go: SELECT frame 1 "heat causes"      │
    │  NoGo: REJECT frame 2 "heat is cause"  │
    │                                        │
    ▼                                        ▼
[4] CerebellarNgram ──────────────── Fluent Completion
    │                                        │
    │  "heat causes" → "expansion" (p=0.7)   │
    │  "heat causes" → "melting" (p=0.2)     │
    │                                        │
    ▼                                        ▼
[5] SurfaceRealizer ──────────────── Final Text
    │                                        │
    │  "Heat causes expansion."              │
    │  (correct punctuation, capitalization) │
    ▼
  Output: "Heat causes expansion, 
           which increases pressure. 
           Unlike cold, which contracts."
```

### Implementation Plan

#### Step 1: Wire Discourse Planner to Graph Output
**File:** `ravana/src/ravana/language/prefrontal_workspace.py`

Replace fixed 3-sentence discourse with dynamic discourse plans generated from graph traversal:

```python
def plan_discourse(self, subject_cid, graph, relation_type):
    outgoing = graph.get_outgoing(subject_cid)
    matching = [(t, e) for t, e in outgoing 
                if e.relation_type == relation_type and e.weight > 0.3]
    intents = []
    for target, edge in matching[:3]:
        intents.append(DiscourseIntent.EXPLAIN 
                       if edge.relation_type == "causal"
                       else DiscourseIntent.ELABORATE)
    return DiscoursePlan(intents=intents, transitions=["first", "also", "finally"])
```

#### Step 2: Implement Syntactic Frame Generation
**File:** `ravana/src/ravana/language/syntactic_cell_assembly.py`

Build frames from the triple (subject, relation, object):
```
Input: (heat, causal, expansion)
Frame: "heat causes expansion"
- SUBJ: "heat" (noun, singular)
- VERB: "cause" → "causes" (3rd person singular)
- OBJ: "expansion" (noun, singular)
```

#### Step 3: Implement Go/NoGo Candidate Gating
**File:** `ravana/src/ravana/language/basal_ganglia.py`

Given multiple candidate frames, gate selection:
- Go signal = edge weight × confidence × activation
- NoGo signal = low probability or syntactic violation
- Winner selected via softmax over Go-NoGo difference

#### Step 4: Wire Surface Realizer
**File:** `ravana/src/ravana/language/surface_realizer.py`

Ensure proper morphology:
- Subject-verb agreement ("physics is" not "physics are")
- Article selection ("a/an" based on phonetics)
- Tense consistency across sentences

#### Step 5: Connect in `_generate_with_decoder_and_syntax()`
**File:** `ravana/src/ravana/chat/interface.py` (line 1028)

Replace stub with full pipeline call:
```python
def _generate_with_decoder_and_syntax(self, ctx):
    discourse_plan = self.pfc_workspace.plan_discourse(
        ctx.subject, self.graph_engine.graph, ctx.relation)
    utterances = []
    for intent in discourse_plan.intents:
        frame = self.syntactic_assembly.build_frame(
            ctx.subject, ctx.relation, ctx.associated_concepts)
        if self.basal_ganglia.gate(frame):
            candidates = self.cerebellar_ngram.complete(frame)
            text = self.surface_realizer.realize(
                candidates[0], self.emotion.state)
            utterances.append(text)
    return " ".join(utterances)
```

---

## 7. P1: Theory of Mind & Personalization

### Current State
`UserModel` in `scripts/ravana_chat.py` exists but is basic:
- Tracks topic familiarity
- Activation boost for known topics
- Not wired into response generation

### Full Theory of Mind Implementation

```
UserModel
├── belief_state: Dict[str, float]     # What user believes (and confidence)
├── topic_familiarity: Dict[str, float] # 0.0=unknown, 1.0=expert
├── emotional_state: VAD               # Current inferred emotion
├── interaction_history: List[Triple]   # Recent conversation topics
├── goals: List[str]                    # Learning? Debugging? Exploring?
└── relationship_depth: float           # 0.0=stranger, 1.0=close friend
```

### Key Behaviors

**A. Adaptive Language Complexity**
- `topic_familiarity["buffer_overflow"] < 0.3` → Use definitions, simpler sentences
- `topic_familiarity["borrow_checker"] > 0.7` → Use jargon, technical depth

**B. Emotional State Tracking**
- User says "I'm frustrated" → valence=-0.4, arousal=0.7
- Mirror: increase own arousal to 0.7 (rapport building)
- Respond with empathy: "That sounds frustrating. What part is giving you trouble?"

**C. Goal Inference**
- User asks "how does X work?" → goal = LEARNING
- User asks "why is X broken?" → goal = DEBUGGING
- User says "tell me about X" → goal = EXPLORING

### Implementation in `interface.py`

Add after turn processing (line 422):
```python
def _update_user_model(self, text, subject, associations):
    # Infer emotional state from text
    vad = self._infer_user_emotion(text)
    self.user_model.emotional_state = vad
    
    # Update topic familiarity
    for concept, confidence in associations:
        self.user_model.topic_familiarity[concept.lower()] = \
            0.9 * self.user_model.topic_familiarity.get(concept.lower(), 0) \
            + 0.1 * min(1.0, confidence + 0.3)
    
    # Infer user goal
    self.user_model.goals = self._infer_user_goals(text)
```

---

## 8. P2: Low-Rank W_rel & LSH Scoring

### Current State
- `W_rel` shape: `(num_domains=4, n_rel_types=6, latent_dim=96, latent_dim=96)` = 221,184 params
- Every forward pass: `source_latent @ W_rel @ token_embeds.T` = O(vocab_size × latent_dim) = O(50000 × 96)

### Decomposition: W_rel = W_base + Low-Rank Correction

```
W_rel[domain][type] = W_base[type] + U_d @ V_d
                     ──┬───       ──┬───────
                   diagonal      rank=8 adapter
                   (96 params)   (2 × 96 × 8 = 1536 params)

Savings per (domain, type): 9216 → 96 + 1536 = 1632 params (5.6x reduction)
Total: 221,184 → 96×6 + 4×6×2×96×8 = 576 + 36,864 = 37,440 params (5.9x reduction)
```

### LSH Scoring (Replace Full Vocabulary Scoring)

**Current:** `logits = source_latent @ W_rel @ token_embeds.T` — scores ALL 50K tokens

**Target:** Use locality-sensitive hashing to find top-50 candidates:
```
1. Hash source_latent using random hyperplanes → bucket key
2. Retrieve all tokens in same bucket (typically 20-50 tokens)
3. Score only those tokens: O(50 × latent_dim) instead of O(50000 × latent_dim)
4. 1000x speedup in scoring step
```

**Implementation sketch:**
```python
def _lsh_token_scoring(self, latent, n_candidates=50):
    # Random hyperplane LSH
    hash_key = np.sign(latent @ self._lsh_planes)  # (n_planes,) binary
    bucket_id = tuple(hash_key.astype(int))
    candidates = self._lsh_buckets.get(bucket_id, [])
    if len(candidates) < n_candidates:
        # Fall back to nearest-neighbor search
        sims = latent @ self.token_embed.weight.data.T
        candidates = np.argsort(sims)[-n_candidates:]
    return candidates
```

---

## 9. P2: Emotional Mirroring & Relationship Depth

### Current State
- `_update_emotion()` detects positive/negative/curious keywords in user input
- Updates internal VAD state
- No mirroring loop back to response generation

### Emotional Mirroring Loop

```
User Input: "I'm excited about cybersecurity!"
    │
    ▼
Emotion Detection:
  └── "excited" → valence=+0.5, arousal=+0.7
    
    │
    ▼
Mirror Neuron Response:
  └── Increase RAVANA arousal: 0.3 → 0.7
  └── Increase valence: 0.0 → 0.5
  └── (Rapport: matching user's emotional state)
    
    │
    ▼
Modulation of Response:
  ├── Temperature: 0.15 → 0.30 (more varied language for excited users)
  ├── Concept breadth: 2 → 5 hops (explore broader associations)
  ├── Verbosity: 2 → 4 sentences (excited users get longer responses)
  └── Curiosity: suggest related topics
```

### Relationship Depth

```
User Sessions Over Time:
  Session 1: "What is Rust?"
    → relationship_depth = 0.1
    → Response: "Rust is a systems programming language..."
    
  Session 5: "I'm stuck on the borrow checker again"
    → relationship_depth = 0.5
    → Response: "Last time we talked about ownership. 
                  The borrow checker is related to that..."
    
  Session 20: "Hey, can we talk about Rust?"
    → relationship_depth = 0.9
    → Response: "Great to see you! I remember you were working on 
                  that Rust project. How's the borrow checker going?"
```

**Implementation:**

```python
# In interface.py
class RelationshipMemory:
    def __init__(self):
        self._profiles: Dict[str, UserProfile] = {}
    
    def update(self, user_id, topic, emotion, turn_count):
        profile = self._profiles.setdefault(user_id, UserProfile())
        profile.topic_history.append((topic, turn_count))
        profile.interaction_count += 1
        profile.relationship_depth = min(1.0, profile.interaction_count / 20)
        
    def get_personalized_greeting(self, user_id):
        profile = self._profiles.get(user_id)
        if profile and profile.relationship_depth > 0.5:
            last_topic = profile.topic_history[-1]
            return f"Welcome back! Last time we discussed {last_topic}."
        return "Hey there! What would you like to talk about?"
```

---

## 10. P3: Benchmark Harness

### Current State
- No formal benchmark against Transformer models
- GloVe-based evaluation only
- Held-out accuracy tracked manually

### Benchmark Architecture

```
Benchmark Suite
├── Triple Completion
│   ├── Same-domain held-out (heat → ? with "causes" seen without expansion)
│   └── Cross-domain held-out (anger → ? with "causes" learned in science domain)
│
├── Catastrophic Forgetting
│   ├── Permuted MNIST (sequential domain switches)
│   └── Permuted Topics (physics → cooking → cybersecurity)
│
├── Conversation Quality
│   ├── User-rated engagement (1-5)
│   ├── Coherence (do responses follow from input?)
│   └── Diversity (distinct n-grams, avoiding repetition)
│
└── Parameter Efficiency
    ├── RAVANA: ~200K params
    ├── DistilGPT-2: 82M params
    ├── Tiny 4-layer Transformer: 10M params
    └── Tiny LLaMA: 1.1B params
```

### Implementation

```python
# scripts/benchmark_vs_transformers.py
def run_benchmark():
    results = {}
    
    # Triple completion
    for model in [ravana, distilgpt2, tiny_transformer]:
        accuracy = evaluate_triple_completion(model, held_out_triples)
        results[model.name] = {"triple_accuracy": accuracy}
    
    # Cross-domain transfer
    for model in models:
        transfer = evaluate_cross_domain(model, source_domain="science", target_domain="emotions")
        results[model.name]["cross_domain"] = transfer
    
    # Forgetting
    for model in models:
        forgetting = evaluate_sequential_topics(model, topics=["physics", "cooking", "rust"])
        results[model.name]["forgetting_rate"] = forgetting
    
    return results
```

---

## 11. Implementation Priority Matrix

| Priority | Component | Effort | Impact | Dependencies | Verification |
|----------|-----------|--------|--------|--------------|--------------|
| **P0** | Deeper prototype hierarchy (3+ levels) | 2 days | High | None | Novel entity recall@5 |
| **P0** | Cross-verb offset generalization | 1 day | Medium | Verb offsets working | Held-out verb accuracy |
| **P0** | Verb offset variance tracking | 1 day | Medium | Verb offsets working | Uncertainty calibration |
| **P1** | Wire syntactic pipeline (full) | 5 days | Very High | All 5 language modules | User-rated naturalness |
| **P1** | Complete Theory of Mind UserModel | 3 days | High | UserModel stub | Personalization score |
| **P2** | Emotional mirroring loop | 2 days | High | VAD emotion engine | User engagement rating |
| **P2** | Relationship memory + depth | 2 days | Medium | UserModel | Personalization score |
| **P2** | Low-rank W_rel decomposition | 2 days | Low | None | Parameter count, speed |
| **P3** | LSH token scoring | 3 days | Low | None | Forward pass speed |
| **P3** | Benchmark harness | 3 days | Medium | All P0/P1 fixes | Comparison results |
| **P3** | ConceptNet ontology bootstrap | 2 days | Medium | Prototype hierarchy | Novel entity coverage |

### Suggested Sprint Plan

**Sprint 1 (Week 1):** P0 — Prototype hierarchy depth + cross-verb generalization + variance tracking

**Sprint 2 (Week 2):** P1 — Wire full syntactic pipeline (wire BasalGanglia → CerebellarNgram → SurfaceRealizer → PrefrontalWorkspace)

**Sprint 3 (Week 3):** P1 — Complete Theory of Mind + P2 — Emotional mirroring loop

**Sprint 4 (Week 4):** P2 — Relationship memory + Low-rank W_rel

**Sprint 5 (Week 5):** P3 — Benchmark harness + LSH scoring

---

## Appendix A: Key Neuroscience Paper References

| Paper | Year | Core Idea | RAVANA Module |
|-------|------|-----------|---------------|
| Hebb, *The Organization of Behavior* | 1949 | Cell assemblies, Hebbian learning | All plasticity modules |
| Collins & Loftus, *Spreading activation theory* | 1975 | Semantic priming via spreading activation | PropagationEngine |
| Baddeley & Hitch, *Working Memory* | 1974 | Central executive + slave systems | PrefrontalWorkspace |
| McClelland et al., *CLS in hippocampus/neocortex* | 1995 | Complementary Learning Systems | Verb offsets + sleep |
| Redgrave et al., *Basal ganglia: selection problem* | 1999 | Go/NoGo action selection | BasalGangliaGate |
| Patterson et al., *Hub-and-spoke semantic memory* | 2007 | ATL as semantic hub | ConceptGraph + prototypes |
| Tononi & Cirelli, *Sleep synaptic homeostasis* | 2006 | SWS downscaling | Sleep consolidation |
| Pulvermüller, *Brain embodiment of syntax* | 2010 | Neural syntax via cell assemblies | SyntacticCellAssembly |
| Rizzolatti & Craighero, *Mirror neuron system* | 2004 | Action understanding via mirroring | Emotional mirroring plan |
| Frank et al., *Frontal cortex-BG interactions* | 2001 | Go/NoGo gating in working memory | BasalGangliaGate |
| Wu et al., *Semantic Hub Hypothesis* (ICLR 2025) | 2025 | LLMs share representations across modalities | Prototype hierarchy upgrade |

---

## Appendix B: File Change Summary

| File | What to Change | Priority |
|------|---------------|----------|
| `rlm_v2.py` — `_init_default_prototypes()` | Add Level 1/2 hierarchy | P0 |
| `rlm_v2.py` — `_inherit_from_prototype()` | Cascade through prototype levels | P0 |
| `rlm_v2.py` — `_compute_verb_offsets()` | Add cross-verb merging | P0 |
| `rlm_v2.py` — `_accumulate_verb_offset()` | Add variance tracking | P0 |
| `prefrontal_workspace.py` | Dynamic discourse from graph traversal | P1 |
| `syntactic_cell_assembly.py` | Frame generation from triples | P1 |
| `basal_ganglia.py` | Go/NoGo candidate gating | P1 |
| `interface.py` — `_generate_with_decoder_and_syntax()` | Wire full pipeline | P1 |
| `interface.py` — `_update_user_model()` | Full Theory of Mind | P1 |
| `rlm_v2.py` — `_rp_rel_matrices` shape | Low-rank decomposition | P2 |
| `rlm_v2.py` — `_rp_forward()` | LSH scoring | P3 |
| `scripts/benchmark_vs_transformers.py` | New file | P3 |
| New: `relationship_memory.py` | User relationship tracking | P2 |
| New: `emotional_mirror.py` | Mirroring loop | P2 |
| New: `ontology_bootstrap.json` | ConceptNet seed data | P0 |
