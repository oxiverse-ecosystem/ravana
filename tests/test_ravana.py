#!/usr/bin/env python3
"""Integration test: verify ravana works as a PyTorch drop-in + RLM training."""

import sys
import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
np.random.seed(42)

print("=" * 60)
print("RAVANA Framework Integration Test")
print("=" * 60)

# ─── Test 1: Import as torch ─────────────────────────────────────────
print("\n[1/7] Import as torch...")
import ravana as torch
from ravana import nn, tensor, Tensor
from ravana import StateTensor
print("  ✓ ravana.tensor:", hasattr(torch, 'tensor'))
print("  ✓ ravana.nn:", hasattr(torch, 'nn'))

# ─── Test 2: Tensor operations ───────────────────────────────────────
print("\n[2/7] Tensor operations...")
x = torch.tensor([1, 2, 3], requires_grad=True)
y = torch.tensor([4, 5, 6])
print(f"  x = {x.data}")
print(f"  y = {y.data}")
assert isinstance(x, Tensor), f"Expected StateTensor, got {type(x)}"
assert isinstance(y, torch.RawTensor), f"Expected RawTensor, got {type(y)}"
print("  ✓ requires_grad=True → StateTensor, requires_grad=False → RawTensor")

z = x + y
print(f"  x + y = {z.data}")
assert isinstance(z, Tensor), "Arithmetic should preserve StateTensor"
print("  ✓ Arithmetic preserves StateTensor")

m = torch.randn(3, 4)
print(f"  randn(3,4) shape: {m.shape}")
assert m.shape == (3, 4)
print("  ✓ randn works")

eye = torch.eye(3)
print(f"  eye(3) shape: {eye.shape}, diag: {eye.data.diagonal()}")
assert eye.shape == (3, 3)
print("  ✓ eye works")

# ─── Test 3: Module API ─────────────────────────────────────────────
print("\n[3/7] Module API...")
linear = nn.Linear(4, 8)
print(f"  Linear: {linear}")
out = linear(m)
print(f"  forward shape: {out.shape}")
assert out.shape == (3, 8), f"Expected (3,8), got {out.shape}"
print("  ✓ Linear forward pass")

# Parameter check
params = list(linear.parameters())
print(f"  Parameters: {len(params)} tensors")
assert len(params) == 2, f"Expected 2 params (weight+bias), got {len(params)}"
print("  ✓ Module.parameters() works")

# Sequential
seq = nn.Sequential(
    nn.Linear(4, 8),
    nn.Linear(8, 2),
)
out2 = seq(m)
print(f"  Sequential forward shape: {out2.shape}")
assert out2.shape == (3, 2)
print("  ✓ Sequential works")

# ─── Test 4: Pressure learning ──────────────────────────────────────
print("\n[4/7] Pressure learning...")
lin = nn.Linear(4, 2)
x_in = torch.randn(3, 4)
target = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

out_before = lin(x_in).data.copy()
error = out_before - target.data
lin.accumulate_free_energy(error)
print(f"  Pressure after accumulate: {lin._free_energy:.4f}")
assert lin._free_energy > 0, "Pressure should accumulate"
print("  ✓ accumulate_free_energy works")

lin.sleep_cycle()
out_after = lin(x_in).data
print(f"  Weight change: {np.abs(out_after - out_before).mean():.6f}")
assert not np.allclose(out_before, out_after), "Weights should change after sleep"
print("  ✓ sleep_cycle changes weights")

# ─── Test 5: ConceptGraph ──────────────────────────────────────────
print("\n[5/7] ConceptGraph...")
from ravana.graph import ConceptGraph
from ravana.propagation import PropagationEngine
from ravana.free_energy import FreeEnergyAccumulator
from ravana.plasticity import HebbianPlasticity, StructuralPlasticity

graph = ConceptGraph(dim=16, max_nodes=100)
for i in range(20):
    graph.add_node(np.random.randn(16).astype(np.float32) * 0.1, label=f"concept_{i}")
print(f"  Graph: {graph}")
assert len(graph.nodes) == 20, f"Expected 20 nodes, got {len(graph.nodes)}"

# Add edges
for i in range(5):
    graph.add_edge(i, i + 1, weight=0.8 + 0.1 * i / 5)
print(f"  Edges: {len(graph.edges)}")
assert len(graph.edges) == 5

# Activate and spread
graph.activate(0, 1.0)
graph.spread_activation(steps=3, k_active=5)
active = [n.id for n in graph.nodes.values() if n.activation > 0.01]
print(f"  Active nodes after spread: {active}")
assert len(active) > 0, "Should have active nodes"

# Propagation engine
prop = PropagationEngine(graph)
result = prop.propagate(np.random.randn(16).astype(np.float32) * 0.1)
print(f"  Propagation result: {result}")
assert len(result) > 0, "Propagation should return active concepts"

# Pressure accumulator
pa = FreeEnergyAccumulator(graph)
pa.accumulate_semantic(0.8)
pa.accumulate_linguistic(0.3)
print(f"  Pressure total: {pa.total:.2f}")
assert pa.total > 0, "Pressure should accumulate"
print("  ✓ All graph systems work")

# ─── Test 6: Hebbian learning ───────────────────────────────────────
print("\n[6/7] Hebbian plasticity...")
hb = HebbianPlasticity(graph, lr=0.01)
graph.activate(0, 0.9)
graph.activate(1, 0.8)
delta = hb.update(0, 1)
print(f"  Hebbian delta: {delta:.6f}")
assert delta != 0, "Hebbian update should produce a delta"

sp = StructuralPlasticity(graph)
pruned, formed = sp.step()
print(f"  Structural: pruned={pruned}, formed={formed}")
print("  ✓ Plasticity works")

# ─── Test 7: RLM (RAVANA Language Model) ────────────────────────────
print("\n[7/7] RAVANA Language Model...")

def test_rlm_convergence():
    """Test RLM learns distant causal transitions (not geometric coincidence)."""
    train_seq = [1, 50, 25, 10, 40]  # transitions are FAR apart on the unit circle
    n_repeats = 20

    rlm = nn.RLM(
        vocab_size=64, embed_dim=32, concept_dim=32, n_concepts=128,
        n_hidden=32, n_layers=1, max_seq_len=16,
        free_energy_threshold=5.0, sleep_interval=15,
    )

    # Train
    for _ in range(n_repeats):
        for i in range(len(train_seq) - 1):
            inp = np.array([train_seq[i]], dtype=np.int64)
            nxt = np.array([train_seq[i + 1]], dtype=np.int64)
            rlm.learn(inp, nxt)

    # Verify edges match causal transitions
    edges_found = 0
    for i in range(len(train_seq) - 1):
        src, tgt = train_seq[i], train_seq[i + 1]
        src_e = rlm.token_embed(StateTensor(np.array([src]))).data[0]
        tgt_e = rlm.token_embed(StateTensor(np.array([tgt]))).data[0]
        src_cid = rlm._nearest_concept(src_e)[0]  # (cid, sim) tuple
        tgt_cid = rlm._nearest_concept(tgt_e)[0]  # (cid, sim) tuple
        if rlm.graph.get_edge(src_cid, tgt_cid):
            edges_found += 1

    total = len(train_seq) - 1
    print(f"  Causal edges: {edges_found}/{total}")
    return edges_found, total, rlm

edges, total, rlm = test_rlm_convergence()
print(f"  RLM: {rlm}")
assert edges >= total * 0.75, f"Failed: only {edges}/{total} causal edges learned"
print("  ✓ RLM converges (learns distant causal transitions)")

# ─── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL TESTS PASSED ✓")
print("=" * 60)
print(f"\nFramework: {torch.__version__}")
print(f"Tensors: RawTensor + StateTensor (cognitive fields)")
print(f"Modules: Linear, Embedding, LayerNorm, Dropout, Sequential")
print(f"Graph: ConceptGraph ({len(graph.nodes)} nodes, {len(graph.edges)} edges)")
print(f"Learning: Hebbian + pressure-driven (no backprop)")
print(f"RLM: RAVANA Language Model with conceptual prediction")
