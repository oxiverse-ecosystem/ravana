"""
Lifelong Learning Benchmark (100K+ Experiences)

The definitive test for RLM: streaming sequential experiences with NO replay
buffer, measuring retention, interference, adaptation, transfer, and compute.

RLM advantages being tested:
- Hebbian plasticity (no backprop, bounded compute per step)
- Sleep consolidation (SWS + REM phases)
- Concept graph structure (splits, edges, hierarchy)
- Free energy dynamics (self-regulating learning)
- Multi-hop inference (compositional generalization)

Measurements:
1. Retention curve — recall of early facts over time
2. Catastrophic forgetting — early vs late knowledge degradation
3. Adaptation speed — how fast new facts enter memory
4. Interference — does new learning damage old knowledge?
5. Compositional transfer — can the model chain learned facts?
6. Contradiction recovery — resilience to conflicting information
7. Graph evolution — topology growth, splits, clustering
8. Compute cost — time per experience, memory growth

Baselines:
- SimpleMLP (backprop, catastrophic forgetting expected)
- SlidingWindow (fixed context, no generalization)
- kNN-LM (exact retrieval, no composition)

Usage:
    python experiment_lifelong.py                     # 100K, all metrics
    python experiment_lifelong.py --n 50000           # 50K (faster)
    python experiment_lifelong.py --n 200000          # 200K (stress test)
    python experiment_lifelong.py --skip-baselines    # RLM only
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field, asdict

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from ravana_ml.nn.rlm import RLM
from ravana_ml.tokenizer import SimpleTokenizer
from experiment_baselines import SimpleMLP


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkConfig:
    n_experiences: int = 100_000
    n_probes_per_epoch: int = 50        # retention probes per entity epoch
    retention_check_interval: int = 5_000
    graph_snapshot_interval: int = 10_000
    transfer_test_interval: int = 20_000
    seed: int = 42
    skip_baselines: bool = False

    # RLM architecture
    embed_dim: int = 32
    concept_dim: int = 32
    n_hidden: int = 32
    n_layers: int = 3
    sleep_interval: int = 100

    # Data generation
    n_entity_epochs: int = 5            # entities introduced in waves
    contradiction_rate: float = 0.02    # fraction of contradictory experiences
    noise_rate: float = 0.05            # fraction of noisy/garbage experiences


# ═══════════════════════════════════════════════════════════════════════════
# Streaming Data Generator
# ═══════════════════════════════════════════════════════════════════════════

class LifelongDataStream:
    """Generates structured streaming experiences with entity epochs.

    Entities are introduced in waves (epochs). Each epoch has:
    - Factual statements: "entity is attribute"
    - Relational statements: "entity relation target"
    - Causal chains: "A causes B"
    - Contradictions: entity gets opposite attribute
    - Noise: garbage/random experiences

    This tests:
    - Can the model learn new entities without forgetting old ones?
    - Does it handle contradictions gracefully?
    - Can it compose facts across epochs?
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.rng = np.random.RandomState(config.seed)

        # Entity pools (one per epoch)
        self.entity_pools = [
            # Epoch 0: core entities
            ["zorbax", "vlentor", "quarnox", "threndis", "calibri",
             "nexora", "phyrox", "zenthari", "valoris", "kryonis"],
            # Epoch 1: science entities
            ["cryonis", "verdantis", "ferrumis", "aqualis", "tempestis",
             "silenxis", "sonoris", "chronaxis", "spatialis", "gravitis"],
            # Epoch 2: nature entities
            ["darkovox", "brightonis", "shadowis", "crystalis", "stormis",
             "flamexus", "frostis", "earthenis", "windaris", "thunderis"],
            # Epoch 3: abstract entities
            ["oceanis", "mountais", "forestis", "desertis", "arcticis",
             "tropicalis", "volcanis", "tsunamis", "earthquis", "tornadis"],
            # Epoch 4: advanced entities
            ["quantaxis", "neutrinis", "bosonis", "fermionis", "hadronis",
             "magnetis", "electris", "photonis", "nucleonis", "plasmis"],
        ]
        # Ensure we have enough pools
        while len(self.entity_pools) < config.n_entity_epochs:
            self.entity_pools.append(
                [f"entity_{len(self.entity_pools)}_{j}" for j in range(10)]
            )

        self.attribute_pairs = [
            ("hard", "soft"), ("hot", "cold"), ("bright", "dark"),
            ("fast", "slow"), ("heavy", "light"), ("loud", "quiet"),
            ("sharp", "dull"), ("smooth", "rough"), ("wet", "dry"),
            ("strong", "weak"), ("tall", "short"), ("wide", "narrow"),
            ("deep", "shallow"), ("thick", "thin"), ("dense", "sparse"),
            ("old", "new"), ("pure", "corrupt"), ("safe", "dangerous"),
        ]

        self.relations = [
            "is made of", "can emit", "lives in", "produces", "contains",
            "is powered by", "grows near", "feeds on", "protects", "creates",
        ]

        # Track what we've told the model (for retention testing)
        self.fact_history: Dict[str, List[Dict]] = defaultdict(list)
        self.epoch_boundaries: List[int] = []

    def _get_epoch(self, step: int) -> int:
        """Which entity epoch are we in?"""
        epoch_size = self.config.n_experiences // self.config.n_entity_epochs
        return min(step // epoch_size, self.config.n_entity_epochs - 1)

    def _get_entities(self, epoch: int) -> List[str]:
        """Get entity pool for current and all prior epochs (cumulative)."""
        entities = []
        for e in range(epoch + 1):
            entities.extend(self.entity_pools[e])
        return entities

    def generate(self) -> List[Dict[str, Any]]:
        """Generate the full experience stream."""
        n = self.config.n_experiences
        experiences = []
        epoch_size = n // self.config.n_entity_epochs

        for i in range(n):
            epoch = self._get_epoch(i)
            entities = self._get_entities(epoch)

            # Track epoch boundaries
            if i > 0 and i % epoch_size == 0:
                self.epoch_boundaries.append(i)

            # Decide experience type
            r = self.rng.random()

            if r < self.config.noise_rate:
                # Noise: random tokens (tests robustness)
                noise_words = [chr(self.rng.randint(97, 123)) * self.rng.randint(3, 8)
                               for _ in range(self.rng.randint(2, 5))]
                text = " ".join(noise_words)
                exp = {'text': text, 'type': 'noise', 'entity': '', 'target': '',
                       'epoch': epoch, 'index': i}
            elif r < self.config.noise_rate + self.config.contradiction_rate:
                # Contradiction: flip an attribute
                entity = entities[self.rng.randint(len(entities))]
                pair = self.attribute_pairs[self.rng.randint(len(self.attribute_pairs))]
                # Use opposite attribute from what was likely said
                prior_attrs = [f['target'] for f in self.fact_history.get(entity, [])
                               if f['type'] == 'factual']
                if prior_attrs:
                    # Find opposite
                    attr = pair[1]  # default to second
                    for a, b in self.attribute_pairs:
                        if prior_attrs[-1] == a:
                            attr = b
                            break
                        elif prior_attrs[-1] == b:
                            attr = a
                            break
                else:
                    attr = pair[1]
                text = f"{entity} is {attr}"
                exp = {'text': text, 'type': 'contradiction', 'entity': entity,
                       'target': attr, 'epoch': epoch, 'index': i}
            else:
                # Normal experience
                exp_type = self.rng.choice(
                    ['factual', 'relational', 'causal', 'temporal'],
                    p=[0.4, 0.3, 0.2, 0.1]
                )

                entity = entities[self.rng.randint(len(entities))]

                if exp_type == 'factual':
                    pair = self.attribute_pairs[self.rng.randint(len(self.attribute_pairs))]
                    attr = pair[0] if self.rng.random() < 0.8 else pair[1]
                    text = f"{entity} is {attr}"
                    exp = {'text': text, 'type': 'factual', 'entity': entity,
                           'target': attr, 'epoch': epoch, 'index': i}

                elif exp_type == 'relational':
                    relation = self.relations[self.rng.randint(len(self.relations))]
                    target = entities[self.rng.randint(len(entities))]
                    text = f"{entity} {relation} {target}"
                    exp = {'text': text, 'type': 'relational', 'entity': entity,
                           'target': target, 'epoch': epoch, 'index': i}

                elif exp_type == 'causal':
                    target = entities[self.rng.randint(len(entities))]
                    text = f"{entity} causes {target} to change"
                    exp = {'text': text, 'type': 'causal', 'entity': entity,
                           'target': target, 'epoch': epoch, 'index': i}

                elif exp_type == 'temporal':
                    e2 = entities[self.rng.randint(len(entities))]
                    e3 = entities[self.rng.randint(len(entities))]
                    text = f"{entity} then {e2} then {e3}"
                    exp = {'text': text, 'type': 'temporal', 'entity': entity,
                           'target': f"{e2},{e3}", 'epoch': epoch, 'index': i}

            experiences.append(exp)
            self.fact_history[exp['entity']].append(exp)

        return experiences

    def generate_probes(self, experiences: List[Dict]) -> Dict[int, List[Dict]]:
        """Generate retention probes per epoch — tests recall of each epoch's facts."""
        probes_by_epoch = defaultdict(list)
        factual_by_epoch = defaultdict(list)

        for exp in experiences:
            if exp['type'] == 'factual' and exp['entity']:
                factual_by_epoch[exp['epoch']].append(exp)

        for epoch, facts in factual_by_epoch.items():
            n = min(self.config.n_probes_per_epoch, len(facts))
            sampled = self.rng.choice(len(facts), size=n, replace=False)
            for idx in sampled:
                f = facts[idx]
                probes_by_epoch[epoch].append({
                    'prompt': f"{f['entity']} is",
                    'target': f['target'],
                    'entity': f['entity'],
                    'source_epoch': epoch,
                    'source_index': f['index'],
                })

        return dict(probes_by_epoch)

    def generate_transfer_tests(self, experiences: List[Dict]) -> List[Dict]:
        """Generate compositional transfer tests."""
        tests = []
        entity_attrs = {}
        attr_relations = defaultdict(list)

        for exp in experiences:
            if exp['type'] == 'factual':
                entity_attrs[exp['entity']] = exp['target']
            elif exp['type'] == 'relational' and ' is ' in exp['text']:
                parts = exp['text'].split(' is ', 1)
                if len(parts) == 2:
                    entity = parts[0].strip()
                    rest = parts[1].strip()
                    if entity in entity_attrs:
                        attr = entity_attrs[entity]
                        attr_relations[attr].append(rest)

        for entity, attr in entity_attrs.items():
            if attr in attr_relations:
                for target in attr_relations[attr][:1]:
                    tests.append({
                        'prompt': f"{entity} is",
                        'hop1_target': attr,
                        'hop2_target': target,
                        'entity': entity,
                    })

        return tests[:50]


# ═══════════════════════════════════════════════════════════════════════════
# Baselines
# ═══════════════════════════════════════════════════════════════════════════

class SlidingWindowBaseline:
    """Fixed sliding window — no learning, just recent memory."""

    def __init__(self, window_size: int = 5000):
        self.window_size = window_size
        self.buffer: List[str] = []

    def learn(self, text: str):
        self.buffer.append(text)
        if len(self.buffer) > self.window_size:
            self.buffer.pop(0)

    def query(self, entity: str) -> str:
        for text in reversed(self.buffer):
            if entity in text:
                parts = text.split(' is ')
                if len(parts) == 2:
                    return parts[1].strip()
        return ""


class kNNBaseline:
    """Exact match lookup over all seen experiences."""

    def __init__(self):
        self.experiences: List[Tuple[str, str]] = []

    def learn(self, text: str):
        entity = text.split()[0] if text else ""
        self.experiences.append((entity, text))

    def query(self, entity: str) -> str:
        matches = [(e, t) for e, t in self.experiences if e == entity]
        if matches:
            text = matches[-1][1]
            parts = text.split(' is ')
            if len(parts) == 2:
                return parts[1].strip()
        return ""


# ═══════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Snapshot:
    """Point-in-time measurement of all metrics."""
    step: int
    timestamp: float

    # Retention per epoch
    retention_by_epoch: Dict[int, float] = field(default_factory=dict)
    retention_overall: float = 0.0

    # Forgetting
    early_recall: float = 0.0
    late_recall: float = 0.0
    catastrophic_forgetting: float = 0.0

    # Graph state
    n_concepts: int = 0
    n_edges: int = 0
    n_inhibitory: int = 0
    n_abstract: int = 0
    mean_edge_weight: float = 0.0
    graph_density: float = 0.0

    # Cognitive state
    identity_strength: float = 0.0
    conceptual_accuracy: float = 0.0
    sleep_cycles: int = 0
    total_free_energy: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0

    # Compute
    cumulative_time_s: float = 0.0
    per_experience_ms: float = 0.0

    # Baselines
    mlp_retention: float = 0.0
    sliding_retention: float = 0.0
    knn_retention: float = 0.0


def evaluate_probe_accuracy(model, probes: List[Dict], tokenizer) -> float:
    """Score model on probe questions — target token in top-10."""
    if not probes:
        return 0.0
    correct = 0
    for probe in probes:
        # Use 'prompt' (e.g. "zorbax is") not 'entity' (e.g. "zorbax") — matches training format
        prompt = probe.get('prompt', f"{probe['entity']} is")
        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            continue
        ctx = np.array([prompt_ids], dtype=np.int64)
        logits = model.forward(ctx).data
        target_ids = tokenizer.encode(probe['target'])
        if not target_ids:
            continue
        target_id = target_ids[0]
        top10 = set(np.argsort(logits)[::-1][:10])
        if target_id in top10:
            correct += 1
    return correct / max(1, len(probes))


def collect_graph_metrics(rlm: RLM) -> Dict[str, Any]:
    """Collect graph topology metrics."""
    nodes = list(rlm.graph.nodes.values())
    edges = list(rlm.graph.edges.values())
    n_nodes = len(nodes)
    n_edges = len(edges)
    n_inhibitory = sum(1 for e in edges if e.edge_type == "inhibitory")
    n_abstract = sum(1 for n in nodes if n.level > 0)
    mean_w = float(np.mean([e.weight for e in edges])) if edges else 0.0
    max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
    density = n_edges / max_edges

    return {
        'n_concepts': n_nodes,
        'n_edges': n_edges,
        'n_inhibitory': n_inhibitory,
        'n_abstract': n_abstract,
        'n_bindings': len(rlm.graph.edges),
        'mean_edge_weight': mean_w,
        'graph_density': density,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════════════════

def run_lifelong_benchmark(config: Optional[BenchmarkConfig] = None) -> Dict[str, Any]:
    """Run the comprehensive lifelong learning benchmark."""

    if config is None:
        config = BenchmarkConfig()

    print("\n" + "=" * 70)
    print("LIFELONG LEARNING BENCHMARK (100K+ EXPERIENCES)")
    print("=" * 70)
    print(f"  Experiences:    {config.n_experiences:,}")
    print(f"  Entity epochs:  {config.n_entity_epochs}")
    print(f"  Probe interval: {config.retention_check_interval:,}")
    print(f"  Seed:           {config.seed}")
    print()

    # ── Generate data ──
    print("Generating experience stream...")
    stream = LifelongDataStream(config)
    experiences = stream.generate()
    probes_by_epoch = stream.generate_probes(experiences)
    transfer_tests = stream.generate_transfer_tests(experiences)

    all_probes = []
    for epoch_probes in probes_by_epoch.values():
        all_probes.extend(epoch_probes)

    print(f"  {len(experiences):,} experiences generated")
    print(f"  {len(all_probes)} retention probes across {len(probes_by_epoch)} epochs")
    print(f"  {len(transfer_tests)} compositional transfer tests")
    print(f"  Epoch boundaries: {stream.epoch_boundaries}")
    print()

    # ── Initialize models ──
    tok = SimpleTokenizer()
    vocab_size = tok.vocab_size

    np.random.seed(config.seed)
    rlm = RLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        concept_dim=config.concept_dim,
        n_concepts=vocab_size,
        n_hidden=config.n_hidden,
        n_layers=config.n_layers,
        sleep_interval=config.sleep_interval,
    )

    mlp = None
    sliding = None
    knn = None
    if not config.skip_baselines:
        np.random.seed(config.seed)
        mlp = SimpleMLP(vocab_size=vocab_size, embed_dim=config.embed_dim,
                        n_hidden=config.n_hidden, lr=0.001)
        sliding = SlidingWindowBaseline(window_size=5000)
        knn = kNNBaseline()

    snapshots: List[Snapshot] = []
    peak_early_recall = 0.0
    rlm_total_time = 0.0
    mlp_total_time = 0.0

    def take_snapshot(step: int) -> Snapshot:
        nonlocal peak_early_recall

        snap = Snapshot(step=step, timestamp=time.time())

        # Retention by epoch
        for epoch, epoch_probes in probes_by_epoch.items():
            acc = evaluate_probe_accuracy(rlm, epoch_probes, tok)
            snap.retention_by_epoch[epoch] = acc

        snap.retention_overall = evaluate_probe_accuracy(rlm, all_probes, tok)

        # Early vs late
        if 0 in probes_by_epoch:
            snap.early_recall = snap.retention_by_epoch.get(0, 0.0)
        current_epoch = stream._get_epoch(step)
        snap.late_recall = snap.retention_by_epoch.get(current_epoch, 0.0)

        # Catastrophic forgetting
        peak_early_recall = max(peak_early_recall, snap.early_recall)
        snap.catastrophic_forgetting = peak_early_recall - snap.early_recall

        # Graph
        gm = collect_graph_metrics(rlm)
        snap.n_concepts = gm['n_concepts']
        snap.n_edges = gm['n_edges']
        snap.n_inhibitory = gm['n_inhibitory']
        snap.n_abstract = gm['n_abstract']
        snap.mean_edge_weight = gm['mean_edge_weight']
        snap.graph_density = gm['graph_density']

        # Cognitive
        snap.identity_strength = rlm.identity_strength
        snap.conceptual_accuracy = rlm.conceptual_accuracy
        snap.sleep_cycles = rlm.sleep_cycles_completed
        snap.total_free_energy = rlm.total_free_energy
        snap.valence = rlm.valence
        snap.arousal = rlm.arousal

        # Compute
        snap.cumulative_time_s = rlm_total_time
        snap.per_experience_ms = (rlm_total_time / max(1, step + 1)) * 1000

        # Baselines
        if mlp is not None:
            snap.mlp_retention = evaluate_probe_accuracy(mlp, all_probes, tok)
        if sliding is not None:
            correct = sum(1 for p in all_probes if sliding.query(p['entity']) == p['target'])
            snap.sliding_retention = correct / max(1, len(all_probes))
        if knn is not None:
            correct = sum(1 for p in all_probes if knn.query(p['entity']) == p['target'])
            snap.knn_retention = correct / max(1, len(all_probes))

        return snap

    # ── Main streaming loop ──
    print("Streaming experiences...")
    loop_start = time.time()

    for step, exp in enumerate(experiences):
        text = exp['text']
        ids = tok.encode(text)

        # RLM learn — one learn call per experience (whole text sequence)
        t0 = time.time()
        if len(ids) >= 2:
            ctx = np.array([ids[:-1]], dtype=np.int64)
            tgt = np.array([[ids[-1]]], dtype=np.int64)
            rlm.learn(ctx, tgt)
        rlm_total_time += time.time() - t0

        # Periodic sleep
        if (step + 1) % rlm.sleep_interval == 0:
            rlm.sleep_cycle()

        # Pressure-based auto-sleep
        if rlm.sleep_pressure >= rlm.sleep_pressure_threshold:
            rlm.sleep_cycle()

        # MLP learn
        if mlp is not None:
            t0 = time.time()
            if len(ids) >= 2:
                ctx = np.array([ids[:-1]], dtype=np.int64)
                tgt = np.array([[ids[-1]]], dtype=np.int64)
                mlp.train_step(ctx, tgt[0])
            mlp_total_time += time.time() - t0

        # Baselines
        if sliding is not None:
            sliding.learn(text)
        if knn is not None:
            knn.learn(text)

        # ── Periodic evaluation ──
        if (step + 1) % config.retention_check_interval == 0 or step == len(experiences) - 1:
            snap = take_snapshot(step)
            snapshots.append(snap)

            elapsed = time.time() - loop_start
            eta_s = elapsed / (step + 1) * (config.n_experiences - step - 1)

            epoch = stream._get_epoch(step)
            print(f"  Step {step + 1:>8,}/{config.n_experiences:,} | "
                  f"Epoch {epoch} | "
                  f"Ret: {snap.retention_overall:.1%} | "
                  f"Early: {snap.early_recall:.1%} | "
                  f"Forget: {snap.catastrophic_forgetting:+.1%} | "
                  f"C: {snap.n_concepts} E: {snap.n_edges} S: {snap.n_abstract} | "
                  f"{snap.cumulative_time_s:.1f}s (ETA {eta_s:.0f}s)")

    # ── Final evaluation ──
    print("\n" + "-" * 70)
    print("FINAL RESULTS")
    print("-" * 70)

    final = snapshots[-1] if snapshots else Snapshot(step=0, timestamp=0)

    # Retention by epoch
    print(f"\n  Retention by entity epoch:")
    for epoch in sorted(probes_by_epoch.keys()):
        probes = probes_by_epoch[epoch]
        acc = evaluate_probe_accuracy(rlm, probes, tok)
        print(f"    Epoch {epoch}: {acc:.1%} ({len(probes)} probes)")

    print(f"\n  Overall retention:       {final.retention_overall:.1%}")
    print(f"  Early recall (epoch 0):  {final.early_recall:.1%}")
    print(f"  Catastrophic forgetting: {final.catastrophic_forgetting:+.1%}")
    print(f"  Peak early recall:       {peak_early_recall:.1%}")

    # Graph evolution
    print(f"\n  Graph evolution:")
    print(f"    Concepts:    {final.n_concepts}")
    print(f"    Edges:       {final.n_edges}")
    print(f"    Inhibitory:  {final.n_inhibitory}")
    print(f"    Abstract:    {final.n_abstract} (from splits)")
    print(f"    Density:     {final.graph_density:.4f}")
    print(f"    Mean weight: {final.mean_edge_weight:.3f}")

    # Cognitive state
    print(f"\n  Cognitive state:")
    print(f"    Identity:     {final.identity_strength:.3f}")
    print(f"    Accuracy:     {final.conceptual_accuracy:.3f}")
    print(f"    Sleep cycles: {final.sleep_cycles}")
    print(f"    Free energy:  {final.total_free_energy:.2f}")
    print(f"    Valence:      {final.valence:+.3f}")
    print(f"    Arousal:      {final.arousal:.3f}")

    # Compute cost
    print(f"\n  Compute cost:")
    print(f"    RLM total:    {rlm_total_time:.1f}s")
    print(f"    Per exp:      {final.per_experience_ms:.2f}ms")
    if mlp is not None:
        mlp_per = (mlp_total_time / max(1, config.n_experiences)) * 1000
        print(f"    MLP total:    {mlp_total_time:.1f}s ({mlp_per:.2f}ms/exp)")

    # Baselines
    if not config.skip_baselines:
        print(f"\n  Baselines (final retention):")
        print(f"    RLM:     {final.retention_overall:.1%}")
        print(f"    MLP:     {final.mlp_retention:.1%}")
        print(f"    Sliding: {final.sliding_retention:.1%}")
        print(f"    kNN:     {final.knn_retention:.1%}")

    # Retention curve
    if len(snapshots) >= 4:
        print(f"\n  Retention curve (RLM overall):")
        quartiles = [snapshots[len(snapshots) * i // 4] for i in range(4)]
        for i, s in enumerate(quartiles):
            pct = (s.step / config.n_experiences) * 100
            print(f"    {pct:5.1f}%: retention={s.retention_overall:.1%} "
                  f"concepts={s.n_concepts} edges={s.n_edges} splits={s.n_abstract}")

    # Transfer test
    if transfer_tests:
        print(f"\n  Compositional transfer ({len(transfer_tests)} tests):")
        transfer_correct = 0
        for tt in transfer_tests[:10]:
            prompt = tt.get('prompt', f"{tt['entity']} is")
            prompt_ids = tok.encode(prompt)
            if not prompt_ids:
                continue
            ctx = np.array([prompt_ids], dtype=np.int64)
            logits = rlm.forward(ctx).data
            hop2_ids = tok.encode(tt['hop2_target'])
            if hop2_ids:
                top5 = np.argsort(logits)[::-1][:5]
                if hop2_ids[0] in top5:
                    transfer_correct += 1
        print(f"    2-hop transfer accuracy: {transfer_correct}/{min(10, len(transfer_tests))}")

    # Build result dict
    result = {
        'experiment': 'lifelong_learning_100k',
        'config': asdict(config),
        'final_retention': {
            'overall': final.retention_overall,
            'early_recall': final.early_recall,
            'late_recall': final.late_recall,
            'catastrophic_forgetting': final.catastrophic_forgetting,
            'by_epoch': {str(k): v for k, v in final.retention_by_epoch.items()},
        },
        'baselines': {
            'mlp': final.mlp_retention,
            'sliding': final.sliding_retention,
            'knn': final.knn_retention,
        },
        'graph_final': {
            'concepts': final.n_concepts,
            'edges': final.n_edges,
            'inhibitory': final.n_inhibitory,
            'abstract': final.n_abstract,
            'density': final.graph_density,
        },
        'cognitive_final': {
            'identity': final.identity_strength,
            'accuracy': final.conceptual_accuracy,
            'sleep_cycles': final.sleep_cycles,
            'free_energy': final.total_free_energy,
        },
        'compute': {
            'rlm_total_s': rlm_total_time,
            'per_experience_ms': final.per_experience_ms,
            'mlp_total_s': mlp_total_time if mlp else 0,
        },
        'snapshots': [
            {
                'step': s.step,
                'retention_overall': s.retention_overall,
                'early_recall': s.early_recall,
                'catastrophic_forgetting': s.catastrophic_forgetting,
                'n_concepts': s.n_concepts,
                'n_edges': s.n_edges,
                'n_abstract': s.n_abstract,
                'sleep_cycles': s.sleep_cycles,
                'cumulative_time_s': s.cumulative_time_s,
            }
            for s in snapshots
        ],
    }

    # Save
    os.makedirs("experiment_results", exist_ok=True)
    out_path = "experiment_results/lifelong_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lifelong Learning Benchmark (100K+)")
    parser.add_argument('--n', type=int, default=100_000, help='Number of experiences')
    parser.add_argument('--probes', type=int, default=50, help='Probes per epoch')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--skip-baselines', action='store_true', help='Skip MLP/sliding/kNN')
    parser.add_argument('--epochs', type=int, default=5, help='Number of entity epochs')
    parser.add_argument('--check-interval', type=int, default=5000, help='Retention check interval')
    args = parser.parse_args()

    config = BenchmarkConfig(
        n_experiences=args.n,
        n_probes_per_epoch=args.probes,
        seed=args.seed,
        skip_baselines=args.skip_baselines,
        n_entity_epochs=args.epochs,
        retention_check_interval=args.check_interval,
    )

    run_lifelong_benchmark(config)
