# Advanced Topics

> **Customization, extension, and advanced usage patterns** for RAVANA.

---

## Table of Contents

1. [Custom Tokenizers](#custom-tokenizers)
2. [Custom Plasticity Rules](#custom-plasticity-rules)
3. [Custom Sleep Stages](#custom-sleep-stages)
4. [Custom Governor Constraints](#custom-governor-constraints)
5. [Multi-Agent Scenarios](#multi-agent-scenarios)
6. [Lifelong / Continual Learning](#lifelong--continual-learning)
7. [Graph Surgery & Inspection](#graph-surgery--inspection)
8. [Performance Optimization](#performance-optimization)
9. [Integration with External Systems](#integration-with-external-systems)
10. [Debugging & Diagnostics](#debugging--diagnostics)

---

## Custom Tokenizers

### Implementing `TokenizerInterface`

```python
from ravana_ml.tokenizer import TokenizerInterface
from typing import List

class MyTokenizer(TokenizerInterface):
    def __init__(self, vocab_file: str = None):
        self.word_to_id = {}
        self.id_to_word = {}
        self._next_id = 0
        if vocab_file:
            self.load_vocab(vocab_file)
    
    def encode(self, text: str) -> List[int]:
        # Your tokenization logic
        tokens = self._tokenize(text)
        ids = []
        for t in tokens:
            if t not in self.word_to_id:
                self.word_to_id[t] = self._next_id
                self.id_to_word[self._next_id] = t
                self._next_id += 1
            ids.append(self.word_to_id[t])
        return ids
    
    def decode(self, token_ids: List[int]) -> str:
        return " ".join(self.id_to_word.get(tid, "[UNK]") for tid in token_ids)
    
    @property
    def vocab_size(self) -> int:
        return max(1, self._next_id)
    
    def save_vocab(self, path: str):
        import json
        with open(path, 'w') as f:
            json.dump(self.word_to_id, f)
    
    def load_vocab(self, path: str):
        import json
        with open(path) as f:
            self.word_to_id = json.load(f)
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
        self._next_id = max(self.word_to_id.values()) + 1

# Usage
tok = MyTokenizer()
tok.encode("custom tokenization")
model = RLM(vocab_size=tok.vocab_size, ...)
```

### Subword / BPE Custom

```python
# For custom BPE, use tokenizers library (Rust, fast)
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=10000, special_tokens=["[UNK]", "[PAD]"])
tokenizer.train_from_iterator(corpus, trainer)

# Wrap for RAVANA
class FastBPETokenizer(TokenizerInterface):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    def encode(self, text): return self.tokenizer.encode(text).ids
    def decode(self, ids): return self.tokenizer.decode(ids)
    @property
    def vocab_size(self): return self.tokenizer.get_vocab_size()
```

---

## Custom Plasticity Rules

### Extending Base Plasticity

```python
from ravana_ml.plasticity import HebbianPlasticity
from ravana_ml.graph import ConceptGraph

class NeuromodulatedHebbian(HebbianPlasticity):
    """Hebbian plasticity gated by neuromodulator (dopamine-like)."""
    
    def __init__(self, graph: ConceptGraph, lr: float = 0.03, neuromodulator=None):
        super().__init__(graph, lr)
        self.neuromodulator = neuromodulator or (lambda: 1.0)
    
    def update(self, source: int, target: int, coactivation: float):
        # Gate by neuromodulator
        mod = self.neuromodulator()
        effective_lr = self.lr * mod
        super().update(source, target, coactivation * mod)

# Usage
def dopamine_signal():
    # High when reward > expected
    return 1.5 if reward > expected else 0.5

hebbian = NeuromodulatedHebbian(graph, lr=0.03, neuromodulator=dopamine_signal)
```

### Structural Plasticity with Custom Criteria

```python
from ravana_ml.plasticity import StructuralPlasticity

class SemanticStructuralPlasticity(StructuralPlasticity):
    """Only form edges between semantically compatible concepts."""
    
    def __init__(self, graph, compatibility_fn, **kwargs):
        super().__init__(graph, **kwargs)
        self.compatibility_fn = compatibility_fn
    
    def should_form_edge(self, source_id, target_id):
        source = self.graph.get_node(source_id)
        target = self.graph.get_node(target_id)
        return self.compatibility_fn(source, target)
    
    def step(self):
        # Custom formation logic
        formed = 0
        for source_id in self.graph.nodes:
            for target_id in self.graph.nodes:
                if source_id != target_id and not self.graph.has_edge(source_id, target_id):
                    if self.should_form_edge(source_id, target_id):
                        self.graph.add_edge(source_id, target_id, weight=0.1)
                        formed += 1
        return super().step()[0], formed

# Usage
def compatible(source, target):
    # Only connect if same hierarchical level or parent-child
    return abs(source.level - target.level) <= 1

structural = SemanticStructuralPlasticity(graph, compatible)
```

---

## Custom Sleep Stages

### Adding a Sleep Stage

```python
from core.sleep import SleepConsolidation, SleepConfig, SleepStage, SleepRecord
from enum import Enum

class CustomSleepStage(Enum):
    MEMORY_REINDEXING = "memory_reindexing"  # New stage

class ExtendedSleepConsolidation(SleepConsolidation):
    def execute_sleep_cycle(self, *args, **kwargs):
        # Run parent stages
        record = super().execute_sleep_cycle(*args, **kwargs)
        
        # Add custom stage
        self._current_stage = CustomSleepStage.MEMORY_REINDEXING
        reindex_result = self._reindex_memories(kwargs.get('graph'))
        record.details["reindexing"] = reindex_result
        
        return record
    
    def _reindex_memories(self, graph):
        """Rebuild memory-to-concept mappings after graph changes."""
        # Custom logic
        return {"remapped": 42, "orphaned": 3}

# Usage
sleep = ExtendedSleepConsolidation(SleepConfig())
```

### Custom Dream Sabotage

```python
class CreativeDreamSabotage:
    def __init__(self, graph, creativity_level=0.3):
        self.graph = graph
        self.creativity = creativity_level
    
    def apply(self, memories, concept_graph):
        sabotages = 0
        for mem in memories:
            if np.random.random() < self.creativity:
                # Creative recombination: blend two unrelated memories
                other = np.random.choice(memories)
                blended = self._blend_memories(mem, other)
                self._inject_blended_memory(blended, concept_graph)
                sabotages += 1
        return {"sabotages_applied": sabotages}
    
    def _blend_memories(self, mem1, mem2):
        # Vector interpolation in concept space
        return (mem1.vector + mem2.vector) / 2
```

---

## Custom Governor Constraints

### Adding Domain-Specific Constraints

```python
from core.governor import Governor, GovernorConfig, CognitiveSignals, RegulatedOutput

class DomainAwareGovernor(Governor):
    def __init__(self, config, domain_constraints=None):
        super().__init__(config)
        self.domain_constraints = domain_constraints or {}
    
    def regulate(self, current_dissonance, current_identity, signals, episode=0):
        # Apply domain-specific constraints BEFORE standard regulation
        if signals.source_domain in self.domain_constraints:
            constraint = self.domain_constraints[signals.source_domain]
            signals = self._apply_domain_constraint(signals, constraint)
        
        return super().regulate(current_dissonance, current_identity, signals, episode)
    
    def _apply_domain_constraint(self, signals, constraint):
        # e.g., physics domain: dissonance must stay low
        if constraint.get("max_dissonance"):
            signals.dissonance_delta = min(
                signals.dissonance_delta,
                constraint["max_dissonance"] - signals.current_dissonance
            )
        return signals

# Usage
governor = DomainAwareGovernor(
    GovernorConfig(),
    domain_constraints={
        "physics": {"max_dissonance": 0.3},      # Strict
        "creative": {"max_dissonance": 0.7},      # Permissive
        "social": {"min_identity": 0.5},          # Identity matters
    }
)
```

### Soft Constraints via Penalties

```python
class SoftConstraintGovernor(Governor):
    def __init__(self, config, soft_constraints=None):
        super().__init__(config)
        self.soft_constraints = soft_constraints or []
    
    def _apply_soft_constraints(self, dd, id_val, current_d, current_i):
        penalty_d, penalty_i = 0.0, 0.0
        for constraint in self.soft_constraints:
            # constraint: (condition_fn, penalty_fn)
            if constraint[0](current_d, current_i):
                p_d, p_i = constraint[1](current_d, current_i)
                penalty_d += p_d
                penalty_i += p_i
        return dd - penalty_d, id_val - penalty_i

# Usage
soft_constraints = [
    # If dissonance high AND identity low → penalize further identity loss
    (lambda d, i: d > 0.6 and i < 0.4, lambda d, i: (0.0, 0.05)),
    # If identity very high → penalize dissonance increase (complacency)
    (lambda d, i: i > 0.9, lambda d, i: (0.02, 0.0)),
]
governor = SoftConstraintGovernor(GovernorConfig(), soft_constraints)
```

---

## Multi-Agent Scenarios

### ConceptGraph with Agent-Specific Weights

```python
from ravana_ml.graph import ConceptGraph, ConceptEdge

graph = ConceptGraph(dim=64, max_nodes=10000)

# Edge with agent-specific weights
edge = graph.add_edge(1, 2, weight=0.8, relation_type="causal")
edge.agent_weights = {
    "user_alice": 0.9,    # Alice strongly believes this
    "user_bob": 0.3,      # Bob is skeptical
    "global": 0.8,        # Consensus
}

# Query for specific agent
alice_weight = edge.get_weight_for_agent("user_alice")  # 0.9
bob_weight = edge.get_weight_for_agent("user_bob")      # 0.3

# Update for specific agent (e.g., after correction)
edge.update_weight_for_agent("user_bob", 0.2)  # Bob's weight → 0.5
```

### Multi-Agent CognitiveFrameworks

```python
from ravana.cognitive import CognitiveFramework, FrameworkConfig

# Shared graph, separate cognitive states
shared_graph = ConceptGraph(dim=64, max_nodes=10000)

config_alice = FrameworkConfig(concept_dim=64, max_concepts=10000)
config_bob = FrameworkConfig(concept_dim=64, max_concepts=10000)

# Manually wire shared graph
fw_alice = CognitiveFramework(config_alice)
fw_bob = CognitiveFramework(config_bob)

fw_alice.graph = shared_graph
fw_bob.graph = shared_graph

# Each has own identity, emotion, memory
state_alice = fw_alice.initialize()
state_bob = fw_bob.initialize()

# They learn from same graph but develop different cognitive states
```

### Social Epistemology (Built-in)

```python
from core.social_epistemology import SocialEpistemology, SocialEpistemologyConfig

social = SocialEpistemology(SocialEpistemologyConfig())

# Track beliefs across agents
social.register_agent("alice", initial_trust=0.8)
social.register_agent("bob", initial_trust=0.6)

# Submit belief
social.submit_belief(
    agent="alice",
    content="heat causes expansion",
    confidence=0.9,
    epistemic_status="fact",
)

# Detect deception
deception = social.detect_deception(
    speaker="bob",
    claim="heat causes contraction",  # contradicts alice
    evidence={"alice": 0.9, "physics_textbook": 0.95},
)

# Trust update
social.update_trust("bob", deception_score=deception)
```

---

## Lifelong / Continual Learning

### Domain-Incremental Learning

```python
from ravana.cognitive import CognitiveFramework, FrameworkConfig
from ravana_ml.tokenizer import WordTokenizer

fw = CognitiveFramework(FrameworkConfig())
state = fw.initialize()

domains = ["physics", "chemistry", "biology", "social", "psychology"]

for domain in domains:
    print(f"\n=== Learning {domain} ===")
    domain_data = load_domain_data(domain)  # Your data loader
    
    for episode, (inp_vec, tgt_vec) in enumerate(domain_data):
        concepts = fw.perceive(state, inp_vec)
        preds = fw.predict(state, concepts)
        state = fw.learn(state, preds, tgt_vec, episode)
        
        if episode % 50 == 0:
            state = fw.sleep(state)
    
    # Evaluate on ALL previous domains (test retention)
    for prev_domain in domains[:domains.index(domain)+1]:
        acc = evaluate_on_domain(fw, state, prev_domain)
        print(f"  {prev_domain} retention: {acc:.2%}")

# Key: interleaved replay in sleep() prevents forgetting
```

### Elastic Weight Consolidation (EWC) on Graph

```python
# Built into ConceptEdge:
edge.fisher_importance  # Computed during domain transition
edge.old_weight         # Snapshot at domain boundary

# During sleep, EWC penalty applied:
def ewc_penalty(edge, current_weight):
    return edge.fisher_importance * (current_weight - edge.old_weight) ** 2

# Used in homeostatic_downscale to protect important edges
```

### Bayesian Edge Posteriors

```python
# Each edge maintains Beta(alpha, beta) posterior
# Updated during learning:
edge.posterior_alpha += success_count
edge.posterior_beta += failure_count

# Posterior mean = alpha / (alpha + beta)
# Uncertainty = variance of Beta

# During sleep, prune high-uncertainty, low-mean edges
if edge.posterior_mean < 0.2 and edge.posterior_uncertainty > 0.1:
    graph.remove_edge(edge.source, edge.target)
```

---

## Graph Surgery & Inspection

### Inspecting Graph State

```python
# Node inspection
node = graph.get_node(nid)
print(f"Node {nid}: {node.label}")
print(f"  Activation: {node.activation:.3f}")
print(f"  Stability: {node.stability:.3f}")
print(f"  Drift: {node.drift_magnitude:.3f}")
print(f"  Level: {node.level}, Parent: {node.parent}, Children: {node.children}")

# Edge inspection
edge = graph.get_edge(src, tgt)
print(f"Edge {src}→{tgt}: {edge.relation_type}")
print(f"  Weight: {edge.weight:.3f}, Conf: {edge.confidence:.3f}")
print(f"  Rel vec norm: {np.linalg.norm(edge.relation_vector):.3f}")
print(f"  Predicate: {edge.predicate_token_id}")

# Neighborhood
neighbors = graph.get_neighbors(nid, max_hops=2)
for n in neighbors:
    print(f"  {n.id}: {n.label} (act={n.activation:.3f})")

# Subgraph extraction
subgraph = graph.extract_subgraph(center_nid, radius=2)
```

### Graph Surgery

```python
# Merge two concepts (e.g., after detecting synonymy)
def merge_concepts(graph, nid_a, nid_b, new_label):
    node_a = graph.get_node(nid_a)
    node_b = graph.get_node(nid_b)
    
    # Blend vectors
    new_vec = (node_a.vector + node_b.vector) / 2
    new_vec /= np.linalg.norm(new_vec)
    
    # Create merged node
    new_nid = graph.add_node(new_vec, label=new_label)
    
    # Redirect edges
    for (src, tgt), edge in list(graph.edges.items()):
        if tgt == nid_a or tgt == nid_b:
            graph.add_edge(src, new_nid, edge.weight, edge.relation_type)
            graph.remove_edge(src, tgt)
        if src == nid_a or src == nid_b:
            graph.add_edge(new_nid, tgt, edge.weight, edge.relation_type)
            graph.remove_edge(src, tgt)
    
    # Remove old
    graph.remove_node(nid_a)
    graph.remove_node(nid_b)
    
    return new_nid

# Split concept (e.g., polysemy resolution)
def split_concept(graph, nid, label_a, label_b, criterion_fn):
    node = graph.get_node(nid)
    
    # Partition edges by criterion
    edges_a, edges_b = [], []
    for (src, tgt), edge in graph.edges.items():
        if src == nid or tgt == nid:
            other = tgt if src == nid else src
            other_node = graph.get_node(other)
            if criterion_fn(other_node):
                edges_a.append((src, tgt, edge))
            else:
                edges_b.append((src, tgt, edge))
    
    # Create two new nodes
    vec = node.vector
    nid_a = graph.add_node(vec + np.random.randn(64)*0.01, label=label_a)
    nid_b = graph.add_node(vec + np.random.randn(64)*0.01, label=label_b)
    
    # Reassign edges
    for src, tgt, edge in edges_a:
        new_src = nid_a if src == nid else src
        new_tgt = nid_a if tgt == nid else tgt
        graph.add_edge(new_src, new_tgt, edge.weight, edge.relation_type)
    for src, tgt, edge in edges_b:
        new_src = nid_b if src == nid else src
        new_tgt = nid_b if tgt == nid else tgt
        graph.add_edge(new_src, new_tgt, edge.weight, edge.relation_type)
    
    graph.remove_node(nid)
    return nid_a, nid_b
```

---

## Performance Optimization

### NumPy Optimization

```python
# Use vectorized operations
# BAD
for i in range(n):
    for j in range(m):
        result[i, j] = a[i] * b[j]

# GOOD
result = np.outer(a, b)  # (n, m)

# Pre-allocate arrays
# BAD
results = []
for x in data:
    results.append(compute(x))

# GOOD
results = np.empty(len(data))
for i, x in enumerate(data):
    results[i] = compute(x)
```

### Graph Operation Caching

```python
# RLMv2 caches norms per forward pass
model._token_embed_norms = None  # Auto-computed once per forward

# Custom caching for repeated similarity searches
class CachedConceptGraph(ConceptGraph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._similarity_cache = {}
    
    def find_similar(self, vector, k=10):
        key = (vector.tobytes(), k)
        if key in self._similarity_cache:
            return self._similarity_cache[key]
        result = super().find_similar(vector, k)
        self._similarity_cache[key] = result
        if len(self._similarity_cache) > 1000:
            self._similarity_cache.clear()  # Prevent memory leak
        return result
```

### Batch Processing

```python
# Process multiple inputs in batch
def batch_learn(model, input_batch, target_batch):
    for inp, tgt in zip(input_batch, target_batch):
        model.learn(inp, tgt)
    model.sleep_cycle()  # Single sleep for batch

# Vectorized graph operations (where possible)
active_matrix = np.stack([graph.nodes[nid].vector for nid in active_nids])
# similarity = active_matrix @ graph.node_matrix.T
```

### Profiling

```python
import cProfile, pstats

profiler = cProfile.Profile()
profiler.enable()

# Run experiment
run_experiment()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(30)  # Top 30 functions

# Key functions to watch:
# - graph.spread_activation
# - graph.find_similar (cosine similarity)
# - model.sleep_cycle
# - np.linalg.norm (called heavily)
```

---

## Integration with External Systems

### Exporting Graph for Visualization

```python
import json

def export_graph_for_vis(graph, path):
    nodes = []
    for nid, node in graph.nodes.items():
        nodes.append({
            "id": nid,
            "label": node.label,
            "activation": float(node.activation),
            "level": node.level,
            "x": float(node.vector[0]) * 100,  # 2D projection
            "y": float(node.vector[1]) * 100,
        })
    
    edges = []
    for (src, tgt), edge in graph.edges.items():
        edges.append({
            "source": src,
            "target": tgt,
            "weight": float(edge.weight),
            "type": edge.relation_type,
            "inhibitory": edge.edge_type == "inhibitory",
        })
    
    with open(path, 'w') as f:
        json.dump({"nodes": nodes, "edges": edges}, f)

# Load in D3.js, Cytoscape, Gephi, etc.
```

### Embedding in Web Service

```python
# FastAPI example
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

app = FastAPI()
fw = CognitiveFramework()
state = fw.initialize()

class Query(BaseModel):
    text: str

@app.post("/infer")
async def infer(query: Query):
    # Tokenize & embed (simplified)
    inp_vec = embed_text(query.text)  # Your embedding fn
    result = fw.infer(state, inp_vec)
    return {
        "concepts": result["concepts"],
        "predictions": result["predicted_concepts"],
        "coherence": float(result["coherence"]),
    }

@app.post("/learn")
async def learn(query: Query):
    global state
    inp_vec = embed_text(query.text)
    # ... need target ...
    concepts = fw.perceive(state, inp_vec)
    preds = fw.predict(state, concepts)
    # state = fw.learn(state, preds, target_vec, episode)
    return {"status": "learned"}
```

### Logging to Weights & Biases

```python
import wandb

wandb.init(project="ravana", config=config_dict)

for episode in range(n_episodes):
    # ... training ...
    
    wandb.log({
        "episode": episode,
        "dissonance": state.dissonance,
        "identity": state.identity,
        "top1_acc": top1,
        "top10_acc": top10,
        "sleep_cycles": state.sleep_cycles,
    })
    
    if episode % 100 == 0:
        # Log graph visualization
        export_graph_for_vis(model.graph, "graph.json")
        wandb.save("graph.json")
```

---

## Debugging & Diagnostics

### Verbose Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("ravana")

# Module-specific
logging.getLogger("ravana_ml.graph").setLevel(logging.DEBUG)
logging.getLogger("core.governor").setLevel(logging.DEBUG)
logging.getLogger("core.sleep").setLevel(logging.DEBUG)
```

### Debugging Graph State

```python
def debug_graph(graph, prefix=""):
    print(f"{prefix}Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    
    # Active nodes
    active = [(nid, n.activation) for nid, n in graph.nodes.items() if n.activation > 0.1]
    print(f"  Active: {sorted(active, key=lambda x: -x[1])[:5]}")
    
    # High free energy nodes
    high_fe = [(nid, n.prediction_free_energy) for nid, n in graph.nodes.items() 
               if n.prediction_free_energy > 0.5]
    print(f"  High FE: {high_fe[:5]}")
    
    # Edge weight distribution
    weights = [e.weight for e in graph.edges.values()]
    print(f"  Edge weights: mean={np.mean(weights):.3f}, max={np.max(weights):.3f}")

# Call during training
if episode % 10 == 0:
    debug_graph(model.graph, f"Ep {episode}")
```

### Governor Clamp Diagnostics

```python
# Already built-in!
print(gov.get_clamp_report())

# Or programmatic
metrics = gov.get_clamp_metrics()
if metrics["alignment_score"] < 0.8:
    print("WARNING: Governor frequently overriding upstream")
    print(f"  Clamp rate: {metrics['clamp_rate']:.2%}")
    print(f"  Mean correction: {metrics['mean_correction']:.4f}")
```

### Sleep Record Analysis

```python
for record in sleep.sleep_history:
    print(f"Ep {record.episode}: "
          f"coherence {record.pre_coherence:.3f}→{record.post_coherence:.3f}, "
          f"perturbations={record.perturbations_applied}, "
          f"sabotages={record.sabotages_applied}, "
          f"rollback={record.rollback_occurred}")
    
    # Check for problems
    if record.rollback_occurred:
        print(f"  ⚠️ ROLLBACK at episode {record.episode}")
    if record.post_coherence < record.pre_coherence - 0.1:
        print(f"  ⚠️ COHERENCE DROP >0.1")
```

---

## See Also

- [Architecture](ARCHITECTURE.md)
- [ML Framework](ML_FRAMEWORK.md)
- [Cognitive Core](COGNITIVE_CORE.md)
- [API Reference](API_REFERENCE.md)
- [Developer Guide](DEVELOPER_GUIDE.md)