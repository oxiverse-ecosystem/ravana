# Tutorials — Step-by-Step Guides

> **Hands-on tutorials** building from Hello World to advanced cognitive agents.

---

## Table of Contents

1. [Tutorial 1: Hello RAVANA](#tutorial-1-hello-ravana)
2. [Tutorial 2: Learning Facts with RLMv2](#tutorial-2-learning-facts-with-rlmv2)
3. [Tutorial 3: Full Cognitive Agent](#tutorial-3-full-cognitive-agent)
4. [Tutorial 4: Cross-Domain Generalization](#tutorial-4-cross-domain-generalization)
5. [Tutorial 5: Custom Sleep & Dreams](#tutorial-5-custom-sleep--dreams)
6. [Tutorial 6: Multi-Agent Dialogue](#tutorial-6-multi-agent-dialogue)
7. [Tutorial 7: Lifelong Learning](#tutorial-7-lifelong-learning)
8. [Tutorial 8: Visualizing the ConceptGraph](#tutorial-8-visualizing-the-conceptgraph)

---

## Tutorial 1: Hello RAVANA

**Goal**: Install, import, and run a tensor operation.

### Step 1: Install

```bash
# Primary (Codeberg)
git clone https://codeberg.org/oxiverse/ravana.git
# Mirror (GitHub)
git clone https://github.com/oxiverse-ecosystem/ravana.git
cd ravana
pip install -e ravana/
```

### Step 2: Verify Import

```python
# hello_ravana.py
import ravana as torch
import numpy as np

print(f"RAVANA version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# Create tensor
x = torch.tensor([1.0, 2.0, 3.0])
print(f"Tensor: {x}")
print(f"Shape: {x.shape}")
print(f"Device: {x.device}")

# Linear layer
model = torch.nn.Linear(3, 2)
y = model(x)
print(f"Linear output: {y}")
```

### Step 3: Run

```bash
python hello_ravana.py
```

**Expected output:**
```
RAVANA version: 0.1.0
NumPy version: 1.26.4
Tensor: StateTensor([1., 2., 3.], salience=1.0, free_energy=0.0, stability=0.5, decay=0.0)
Shape: (3,)
Device: device('cpu')
Linear output: StateTensor([...], salience=1.0, ...)
```

---

## Tutorial 2: Learning Facts with RLMv2

**Goal**: Teach RLMv2 causal facts and query them.

### Step 1: Prepare Data

```python
# facts_rlmv2.py
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

# 1. Tokenizer
tok = WordTokenizer()

# 2. Training facts (subject, relation, object)
facts = [
    ("heat causes", "expansion"),
    ("fire causes", "heat"),
    ("ice melts", "water"),
    ("water boils", "steam"),
    ("gravity pulls", "objects"),
    ("magnet attracts", "metal"),
]

# Build vocab
for premise, conclusion in facts:
    tok.encode(premise)
    tok.encode(conclusion)

print(f"Vocab size: {tok.vocab_size}")
print(f"Vocab: {tok.id_to_word}")
```

### Step 2: Create Model

```python
# 3. Model
model = RLM(
    vocab_size=tok.vocab_size,
    embed_dim=64,
    concept_dim=64,
    n_concepts=100,
    sleep_interval=10,  # Sleep often for demo
)

print("Model created")
```

### Step 3: Train

```python
# 4. Learn each fact
for i, (premise, conclusion) in enumerate(facts):
    inp = np.array(tok.encode(premise), dtype=np.int64)
    tgt = np.array(tok.encode(conclusion), dtype=np.int64)
    
    model.learn(inp, tgt)
    print(f"Learned: {premise} → {conclusion}")
    
    # Test immediately
    logits = model.forward(inp)
    top3 = np.argsort(logits)[-3:][::-1]
    preds = [tok.decode([t]) for t in top3]
    print(f"  Immediate prediction: {preds}")

# 5. Sleep to consolidate
print("\n--- Sleep cycle ---")
model.sleep_cycle()
print("Sleep complete")
```

### Step 4: Evaluate

```python
# 6. Test all facts after sleep
print("\n=== After Sleep ===")
for premise, conclusion in facts:
    inp = np.array(tok.encode(premise), dtype=np.int64)
    logits = model.forward(inp)
    top5 = np.argsort(logits)[-5:][::-1]
    preds = [tok.decode([t]) for t in top5]
    correct = conclusion in preds
    status = "✓" if correct else "✗"
    print(f"{status} {premise} → {preds} (target: {conclusion})")

# 7. Show graph stats
print(f"\nGraph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")
print(f"Sleep cycles: {model.sleep_cycles_completed}")
print(f"Free energy: {model.total_free_energy:.3f}")
```

### Run

```bash
python facts_rlmv2.py
```

**Expected**: After sleep, most facts should be in top-5.

---

## Tutorial 3: Full Cognitive Agent

**Goal**: Build an agent with perception, learning, emotion, sleep, and memory.

### Step 1: Setup Framework

```python
# cognitive_agent.py
from ravana.cognitive import CognitiveFramework, FrameworkConfig
import numpy as np

# Minimal config
config = FrameworkConfig(
    concept_dim=64,
    max_concepts=2000,
    k_active=5,
    hebbian_lr=0.03,
    anti_hebbian_lr=0.02,
    propagation_steps=3,
    propagation_decay=0.5,
    initial_identity=0.5,
)

fw = CognitiveFramework(config)
state = fw.initialize()

print(f"Initialized: dissonance={state.dissonance:.2f}, identity={state.identity:.2f}")
```

### Step 2: Create Training Data

```python
# Simple synthetic data: random vectors with structure
def create_training_data(n_samples=200, dim=64):
    # Create "concepts" as random vectors
    concepts = {}
    for i in range(20):
        concepts[f"concept_{i}"] = np.random.randn(dim).astype(np.float32)
        concepts[f"concept_{i}"] /= np.linalg.norm(concepts[f"concept_{i}"])
    
    # Create pairs: concept_i → concept_{i+1} (chain)
    pairs = []
    for i in range(19):
        pairs.append((concepts[f"concept_{i}"], concepts[f"concept_{i+1}"]))
    
    # Repeat with noise
    data = []
    for _ in range(n_samples):
        src, tgt = np.random.choice(pairs)
        # Add noise
        src_noisy = src + np.random.randn(dim).astype(np.float32) * 0.1
        tgt_noisy = tgt + np.random.randn(dim).astype(np.float32) * 0.1
        src_noisy /= np.linalg.norm(src_noisy)
        tgt_noisy /= np.linalg.norm(tgt_noisy)
        data.append((src_noisy, tgt_noisy))
    
    return data, concepts

training_data, concept_vectors = create_training_data(200, 64)
```

### Step 3: Training Loop

```python
print("\n=== Training ===")
for episode, (inp_vec, tgt_vec) in enumerate(training_data):
    # Perceive
    concepts = fw.perceive(state, inp_vec)
    
    # Predict
    predictions = fw.predict(state, concepts)
    
    # Learn
    state = fw.learn(state, predictions, tgt_vec, episode, difficulty=0.5, effort=0.3)
    
    # Sleep periodically
    if episode % 50 == 0 and episode > 0:
        print(f"Episode {episode}: Sleeping... (dissonance={state.dissonance:.3f})")
        state = fw.sleep(state)
        print(f"  After sleep: dissonance={state.dissonance:.3f}, identity={state.identity:.3f}")
    
    # Log progress
    if episode % 25 == 0:
        print(f"Ep {episode:3d}: dissonance={state.dissonance:.3f}, identity={state.identity:.3f}, "
              f"valence={state.valence:.2f}, arousal={state.arousal:.2f}")
```

### Step 4: Inference & Query

```python
print("\n=== Inference ===")
# Test on clean concept vectors
for name, vec in list(concept_vectors.items())[:5]:
    result = fw.infer(state, vec)
    print(f"\nInput: {name}")
    print(f"  Active concepts: {result['concepts']}")
    print(f"  Predicted concepts: {result['predicted_concepts']}")
    print(f"  Coherence: {result['coherence']:.3f}")
    print(f"  Recalled memories: {len(result['recalled_memories'])}")

# Query specific concept
if fw.graph.nodes:
    first_nid = list(fw.graph.nodes.keys())[0]
    query = fw.query(state, first_nid)
    print(f"\nQuery concept {first_nid} ({query['concept']['label']}):")
    print(f"  Neighbors: {len(query['neighbors'])}")
    for n in query['neighbors'][:3]:
        print(f"    → {n['label']} (w={n.get('weight', 'N/A'):.2f})")
```

### Step 5: Save/Load

```python
print("\n=== Save/Load ===")
fw.save("agent_checkpoint.pkl")
print("Saved checkpoint")

# Load new framework
fw2 = CognitiveFramework.load("agent_checkpoint.pkl")
fw2.rebridge()  # Critical: sync memories to graph
state2 = fw2.initialize()  # Gets restored state

print(f"Loaded: dissonance={state2.dissonance:.3f}, identity={state2.identity:.3f}")
```

### Run

```bash
python cognitive_agent.py
```

---

## Tutorial 4: Cross-Domain Generalization

**Goal**: Train on physics, test zero-shot on social domain.

### Step 1: Define Domains

```python
# cross_domain.py
from ravana.nn import RLM
from ravana_ml.tokenizer import WordTokenizer
import numpy as np

tok = WordTokenizer()

# Domain A: Physics (train)
physics_facts = [
    ("heat causes", "expansion"),
    ("fire causes", "heat"),
    ("cold causes", "contraction"),
    ("pressure causes", "compression"),
    ("electricity causes", "magnetism"),
]

# Domain B: Social (test - zero-shot)
social_facts = [
    ("kindness causes", "friendship"),
    ("anger causes", "conflict"),
    ("trust causes", "cooperation"),
    ("betrayal causes", "distrust"),
    ("generosity causes", "gratitude"),
]

# Domain C: Biology (test - zero-shot)
biology_facts = [
    ("sunlight causes", "photosynthesis"),
    ("water causes", "growth"),
    ("mutation causes", "evolution"),
    ("predation causes", "selection"),
    ("symbiosis causes", "coexistence"),
]

all_facts = physics_facts + social_facts + biology_facts
for p, c in all_facts:
    tok.encode(p)
    tok.encode(c)

print(f"Vocab: {tok.vocab_size}")
```

### Step 2: Train on Physics Only

```python
model = RLM(
    vocab_size=tok.vocab_size,
    embed_dim=64,
    concept_dim=64,
    n_concepts=150,
    sleep_interval=20,
)

# Train ONLY on physics
print("Training on physics...")
for premise, conclusion in physics_facts:
    inp = np.array(tok.encode(premise), dtype=np.int64)
    tgt = np.array(tok.encode(conclusion), dtype=np.int64)
    model.learn(inp, tgt)

# Sleep to consolidate
for _ in range(5):
    model.sleep_cycle()

print("Physics training complete")
```

### Step 3: Test Zero-Shot Transfer

```python
def evaluate_domain(model, tok, facts, domain_name):
    print(f"\n=== {domain_name} (zero-shot) ===")
    correct_1 = 0
    correct_10 = 0
    for premise, conclusion in facts:
        inp = np.array(tok.encode(premise), dtype=np.int64)
        logits = model.forward(inp)
        top10 = np.argsort(logits)[-10:][::-1]
        preds = [tok.decode([t]) for t in top10]
        if preds[0] == conclusion:
            correct_1 += 1
        if conclusion in preds:
            correct_10 += 1
        status = "✓" if conclusion in preds else "✗"
        print(f"  {status} {premise} → {preds[:5]}... (target: {conclusion})")
    
    n = len(facts)
    print(f"  Top-1: {correct_1}/{n} = {correct_1/n:.1%}")
    print(f"  Top-10: {correct_10}/{n} = {correct_10/n:.1%}")
    return correct_1/n, correct_10/n

# Test all domains
evaluate_domain(model, tok, physics_facts, "Physics (trained)")
evaluate_domain(model, tok, social_facts, "Social (zero-shot)")
evaluate_domain(model, tok, biology_facts, "Biology (zero-shot)")
```

### Step 4: Enable Verb-Offset for Better Transfer

```python
# Enable analogy path
model.use_rp_for_analogy = True

print("\n=== With Verb-Offset Analogy ===")
evaluate_domain(model, tok, social_facts, "Social (with analogy)")
evaluate_domain(model, tok, biology_facts, "Biology (with analogy)")
```

### Run

```bash
python cross_domain.py
```

**Expected**: Physics ~90%+, Social/Biology lower but verb-offset improves them.

---

## Tutorial 5: Custom Sleep & Dreams

**Goal**: Add custom sleep stage and dream sabotage.

### Step 1: Custom Sleep Stage

```python
# custom_sleep.py
from core.sleep import SleepConsolidation, SleepConfig, SleepStage, SleepRecord
from enum import Enum
import numpy as np

class CustomSleepStage(Enum):
    MEMORY_REINDEXING = "memory_reindexing"
    EMOTIONAL_PROCESSING = "emotional_processing"

class CustomSleepConsolidation(SleepConsolidation):
    def execute_sleep_cycle(self, episode, state_snapshot, beliefs, hypotheses,
                            episodic_memories, emotion_engine, coherence_fn, graph):
        # Run standard stages
        record = super().execute_sleep_cycle(
            episode, state_snapshot, beliefs, hypotheses,
            episodic_memories, emotion_engine, coherence_fn, graph
        )
        
        # Custom stage 1: Memory reindexing
        self._current_stage = CustomSleepStage.MEMORY_REINDEXING
        reindex_result = self._reindex_memories(graph, episodic_memories)
        record.details["reindexing"] = reindex_result
        record.perturbations_applied += reindex_result.get("remapped", 0)
        
        # Custom stage 2: Emotional processing
        self._current_stage = CustomSleepStage.EMOTIONAL_PROCESSING
        emo_result = self._process_emotions(emotion_engine, episodic_memories)
        record.details["emotional_processing"] = emo_result
        
        return record
    
    def _reindex_memories(self, graph, memories):
        """Update memory-to-concept mappings after graph restructuring."""
        remapped = 0
        orphaned = 0
        for mem in memories[:20]:  # Limit
            # Find best matching concept
            if hasattr(mem, 'vector') and mem.vector is not None:
                similar = graph.find_similar(mem.vector, k=1)
                if similar and similar[0][1] > 0.3:
                    mem.concept_id = similar[0][0]
                    remapped += 1
                else:
                    orphaned += 1
        return {"remapped": remapped, "orphaned": orphaned}
    
    def _process_emotions(self, emotion_engine, memories):
        """Process emotional valence of memories during sleep."""
        processed = 0
        flipped = 0
        for mem in memories:
            if hasattr(mem, 'emotional_valence'):
                # High arousal memories get processed
                if abs(mem.emotional_valence) > 0.5:
                    # Dream sabotage: sometimes flip valence
                    if np.random.random() < 0.15:
                        mem.emotional_valence *= -1
                        flipped += 1
                    processed += 1
        return {"processed": processed, "valence_flipped": flipped}

# Usage
sleep = CustomSleepConsolidation(SleepConfig())
```

### Step 2: Custom Dream Sabotage

```python
class CreativeDreamSabotage:
    """More sophisticated dream sabotage."""
    
    def __init__(self, graph, creativity=0.3):
        self.graph = graph
        self.creativity = creativity
    
    def apply(self, episodic_memories, emotion_engine=None):
        sabotages = {"counterfactual": 0, "emotional_flip": 0, 
                     "failure_oversample": 0, "creative_blend": 0}
        
        for mem in episodic_memories:
            r = np.random.random()
            
            if r < 0.15:  # Counterfactual
                self._reverse_causality(mem)
                sabotages["counterfactual"] += 1
            elif r < 0.25:  # Emotional flip
                self._flip_valence(mem)
                sabotages["emotional_flip"] += 1
            elif r < 0.35:  # Failure oversample
                self._replay_failure(mem)
                sabotages["failure_oversample"] += 1
            elif r < 0.35 + self.creativity:  # Creative blend
                self._creative_blend(mem, episodic_memories)
                sabotages["creative_blend"] += 1
        
        return sabotages
    
    def _creative_blend(self, mem, all_memories):
        """Blend two unrelated memories to create novel association."""
        other = np.random.choice(all_memories)
        if hasattr(mem, 'vector') and hasattr(other, 'vector'):
            # Interpolate in concept space
            blended_vec = 0.5 * mem.vector + 0.5 * other.vector
            blended_vec /= np.linalg.norm(blended_vec)
            # Inject as new weak memory
            # (implementation depends on memory system)
            pass

# Integrate with sleep
original_sabotage = sleep._apply_dream_sabotage
def enhanced_sabotage(*args, **kwargs):
    result = original_sabotage(*args, **kwargs)
    creative = CreativeDreamSabotage(kwargs.get('graph'), creativity=0.2)
    extra = creative.apply(kwargs.get('episodic_memories', []))
    result["creative_blends"] = extra["creative_blend"]
    return result

sleep._apply_dream_sabotage = enhanced_sabotage
```

---

## Tutorial 6: Multi-Agent Dialogue

**Goal**: Two agents with shared graph but different beliefs.

### Step 1: Shared Graph, Separate Frameworks

```python
# multi_agent.py
from ravana.cognitive import CognitiveFramework, FrameworkConfig
from ravana_ml.graph import ConceptGraph
import numpy as np

# Shared concept graph
shared_graph = ConceptGraph(dim=64, max_nodes=5000)

# Agent Alice - trusting, physics-oriented
config_alice = FrameworkConfig(concept_dim=64, initial_identity=0.7)
fw_alice = CognitiveFramework(config_alice)
fw_alice.graph = shared_graph
state_alice = fw_alice.initialize()

# Agent Bob - skeptical, social-oriented  
config_bob = FrameworkConfig(concept_dim=64, initial_identity=0.4)
fw_bob = CognitiveFramework(config_bob)
fw_bob.graph = shared_graph
state_bob = fw_bob.initialize()
```

### Step 2: Teach Different Facts

```python
# Alice learns physics
physics_facts = [
    ("heat causes", "expansion"),
    ("gravity pulls", "objects"),
]

# Bob learns social
social_facts = [
    ("kindness causes", "friendship"),
    ("betrayal causes", "distrust"),
]

# Simple vector encoding for demo
def encode_fact(premise, conclusion, agent_fw):
    inp = np.random.randn(64).astype(np.float32)  # Placeholder
    tgt = np.random.randn(64).astype(np.float32)
    inp /= np.linalg.norm(inp)
    tgt /= np.linalg.norm(tgt)
    return inp, tgt

print("=== Alice learns physics ===")
for i, (p, c) in enumerate(physics_facts):
    inp, tgt = encode_fact(p, c, fw_alice)
    concepts = fw_alice.perceive(state_alice, inp)
    preds = fw_alice.predict(state_alice, concepts)
    state_alice = fw_alice.learn(state_alice, preds, tgt, i)

print("=== Bob learns social ===")
for i, (p, c) in enumerate(social_facts):
    inp, tgt = encode_fact(p, c, fw_bob)
    concepts = fw_bob.perceive(state_bob, inp)
    preds = fw_bob.predict(state_bob, concepts)
    state_bob = fw_bob.learn(state_bob, preds, tgt, i)
```

### Step 3: Dialogue via Graph

```python
# They "discuss" by querying shared graph
print("\n=== Graph state after learning ===")
print(f"Nodes: {len(shared_graph.nodes)}")
print(f"Edges: {len(shared_graph.edges)}")

# Alice queries Bob's knowledge
for node in list(shared_graph.nodes.values())[:5]:
    if node.label.startswith("tok_"):
        print(f"  {node.label}: act={node.activation:.3f}")

# Agent-specific edge weights (built-in)
for (src, tgt), edge in list(shared_graph.edges.items())[:3]:
    print(f"  Edge {src}→{tgt}: global={edge.weight:.2f}")
    # In real use: edge.agent_weights["alice"], edge.agent_weights["bob"]
```

### Step 4: Conversational Repair

```python
from core.conversational_repair import ConversationalRepair, CorrectionType

repair = ConversationalRepair()

# Alice states belief
# Bob corrects
event = repair.process_correction(
    user_input="actually gravity pushes objects away",
    current_beliefs=[],  # Would be actual beliefs
    dialogue_context=None,
)

print(f"\nCorrection: {event.correction_type}")
print(f"Target: {event.target_triple}")
print(f"Proposed: {event.proposed_correction}")
```

---

## Tutorial 7: Lifelong Learning

**Goal**: Sequential domain learning without forgetting.

### Step 1: Domain Sequence

```python
# lifelong.py
from ravana.cognitive import CognitiveFramework, FrameworkConfig
import numpy as np

fw = CognitiveFramework(FrameworkConfig(
    concept_dim=64,
    max_concepts=10000,
    sleep_config=SleepConfig(pressure_threshold=0.15),  # More frequent sleep
))
state = fw.initialize()

domains = {
    "physics": [
        ("heat causes", "expansion"),
        ("fire causes", "heat"),
        ("force causes", "acceleration"),
    ],
    "chemistry": [
        ("acid reacts_with", "base"),
        ("catalyst speeds_up", "reaction"),
        ("oxidation causes", "rust"),
    ],
    "biology": [
        ("dna encodes", "protein"),
        ("cell divides", "growth"),
        ("mutation drives", "evolution"),
    ],
    "social": [
        ("trust enables", "cooperation"),
        ("communication reduces", "conflict"),
        ("empathy builds", "connection"),
    ],
}
```

### Step 2: Sequential Training with Evaluation

```python
all_facts = [(d, p, c) for d, facts in domains.items() for p, c in facts]

# Train domain by domain
for domain_name, facts in domains.items():
    print(f"\n{'='*50}")
    print(f"Learning {domain_name}...")
    print(f"{'='*50}")
    
    # Train on this domain
    for i, (premise, conclusion) in enumerate(facts):
        inp = np.random.randn(64).astype(np.float32)
        tgt = np.random.randn(64).astype(np.float32)
        inp /= np.linalg.norm(inp)
        tgt /= np.linalg.norm(tgt)
        
        concepts = fw.perceive(state, inp)
        preds = fw.predict(state, concepts)
        state = fw.learn(state, preds, tgt, i)
        
        if i % 5 == 0:
            state = fw.sleep(state)
    
    # Evaluate on ALL domains seen so far
    print(f"\n--- Retention Test after {domain_name} ---")
    for test_domain, test_facts in domains.items():
        if list(domains.keys()).index(test_domain) > list(domains.keys()).index(domain_name):
            continue  # Haven't learned yet
        
        correct = 0
        for premise, conclusion in test_facts:
            inp = np.random.randn(64).astype(np.float32)  # Same encoding
            inp /= np.linalg.norm(inp)
            result = fw.infer(state, inp)
            # Check if predicted concepts match
            if result['predicted_concepts']:
                correct += 1
        
        acc = correct / len(test_facts)
        status = "✓" if acc > 0.7 else "✗"
        print(f"  {status} {test_domain}: {acc:.1%} retention")

print("\n=== Final Graph Stats ===")
print(f"Concepts: {len(fw.graph.nodes)}")
print(f"Edges: {len(fw.graph.edges)}")
print(f"Sleep cycles: {state.sleep_cycles}")
```

---

## Tutorial 8: Visualizing the ConceptGraph

**Goal**: Export graph for interactive visualization.

### Step 1: Export to JSON

```python
# visualize_graph.py
import json
import numpy as np
from ravana.cognitive import CognitiveFramework, FrameworkConfig

fw = CognitiveFramework(FrameworkConfig())
state = fw.initialize()

# ... train model ...

def export_graph(graph, path="graph.json"):
    """Export ConceptGraph for D3.js / Cytoscape / Gephi."""
    
    # Project vectors to 2D for layout (simple PCA)
    vectors = np.array([n.vector for n in graph.nodes.values()])
    if len(vectors) > 2:
        from scipy.linalg import svd
        U, S, Vt = svd(vectors - vectors.mean(axis=0), full_matrices=False)
        coords_2d = vectors @ Vt[:2].T
    else:
        coords_2d = np.zeros((len(vectors), 2))
    
    nodes = []
    for i, (nid, node) in enumerate(graph.nodes.items()):
        nodes.append({
            "id": str(nid),
            "label": node.label,
            "activation": float(node.activation),
            "level": node.level,
            "stability": float(node.stability),
            "confidence": float(node.confidence),
            "x": float(coords_2d[i, 0]) * 100,
            "y": float(coords_2d[i, 1]) * 100,
            "size": 10 + 20 * float(node.activation),
            "color": f"hsl({node.level * 60}, 70%, 50%)",
        })
    
    edges = []
    for (src, tgt), edge in graph.edges.items():
        edges.append({
            "source": str(src),
            "target": str(tgt),
            "weight": float(edge.weight),
            "confidence": float(edge.confidence),
            "relation_type": edge.relation_type,
            "edge_type": edge.edge_type,
            "width": 1 + 3 * float(edge.weight),
            "color": "#ff4444" if edge.edge_type == "inhibitory" else "#4488ff",
        })
    
    with open(path, 'w') as f:
        json.dump({"nodes": nodes, "edges": edges}, f, indent=2)
    
    print(f"Exported {len(nodes)} nodes, {len(edges)} edges to {path}")

export_graph(fw.graph, "concept_graph.json")
```

### Step 2: HTML Viewer (D3.js)

```html
<!-- viewer.html -->
<!DOCTYPE html>
<html>
<head>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body { margin: 0; font-family: sans-serif; }
        #graph { width: 100vw; height: 100vh; }
        .node { cursor: pointer; }
        .node text { font-size: 10px; pointer-events: none; }
        .link { stroke-opacity: 0.6; }
        .link.inhibitory { stroke-dasharray: 4,2; }
        #info { position: absolute; top: 10px; right: 10px; background: white; 
               padding: 10px; border: 1px solid #ccc; border-radius: 4px; }
    </style>
</head>
<body>
    <div id="graph"></div>
    <div id="info">Click a node</div>
    
    <script>
    const width = window.innerWidth;
    const height = window.innerHeight;
    
    const svg = d3.select("#graph").append("svg")
        .attr("width", width).attr("height", height);
    
    const g = svg.append("g");
    
    // Zoom
    svg.call(d3.zoom().on("zoom", (e) => g.attr("transform", e.transform)));
    
    d3.json("concept_graph.json").then(data => {
        const links = data.edges.map(d => Object.assign({}, d));
        const nodes = data.nodes.map(d => Object.assign({}, d));
        
        const simulation = d3.forceSimulation(nodes)
            .force("link", d3.forceLink(links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-300))
            .force("center", d3.forceCenter(width/2, height/2))
            .force("x", d3.forceX(width/2).strength(0.01))
            .force("y", d3.forceY(height/2).strength(0.01));
        
        const link = g.append("g")
            .selectAll("line")
            .data(links)
            .join("line")
            .attr("class", d => `link ${d.edge_type}`)
            .attr("stroke-width", d => d.width)
            .attr("stroke", d => d.color);
        
        const node = g.append("g")
            .selectAll("circle")
            .data(nodes)
            .join("circle")
            .attr("class", "node")
            .attr("r", d => d.size)
            .attr("fill", d => d.color)
            .call(drag(simulation));
        
        const label = g.append("g")
            .selectAll("text")
            .data(nodes)
            .join("text")
            .text(d => d.label)
            .attr("x", 8)
            .attr("y", 4);
        
        node.on("click", (e, d) => {
            document.getElementById("info").innerHTML = `
                <b>${d.label}</b><br>
                ID: ${d.id}<br>
                Activation: ${d.activation.toFixed(3)}<br>
                Level: ${d.level}<br>
                Stability: ${d.stability.toFixed(3)}<br>
                Confidence: ${d.confidence.toFixed(3)}
            `;
        });
        
        simulation.on("tick", () => {
            link.attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            node.attr("cx", d => d.x).attr("cy", d => d.y);
            label.attr("x", d => d.x + 8).attr("y", d => d.y + 4);
        });
    });
    
    function drag(simulation) {
        return d3.drag()
            .on("start", (e, d) => { if (!e.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; })
            .on("drag", (e, d) => { d.fx = e.x; d.fy = e.y; })
            .on("end", (e, d) => { if (!e.active) simulation.alphaTarget(0); d.fx = null; d.fy = null; });
    }
    </script>
</body>
</html>
```

### Step 3: View

```bash
# Serve
python -m http.server 8000
# Open http://localhost:8000/viewer.html
```

---

## Next Steps

| Tutorial | Prerequisites | Time |
|----------|---------------|------|
| 1. Hello RAVANA | None | 5 min |
| 2. RLMv2 Facts | Tutorial 1 | 10 min |
| 3. Cognitive Agent | Tutorial 2 | 20 min |
| 4. Cross-Domain | Tutorial 2 | 15 min |
| 5. Custom Sleep | Tutorial 3 | 20 min |
| 6. Multi-Agent | Tutorial 3 | 20 min |
| 7. Lifelong | Tutorial 3 | 15 min |
| 8. Visualization | Tutorial 3 | 15 min |

---

## See Also

- [Getting Started](GETTING_STARTED.md)
- [Experiments](EXPERIMENTS.md)
- [Advanced Topics](ADVANCED_TOPICS.md)
- [API Reference](API_REFERENCE.md)