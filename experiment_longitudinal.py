"""
Longitudinal Concept Evolution Experiment

Runs RLM for 100K+ autonomous learning cycles, tracking semantic geometry,
cognitive state, and emergent dynamics over time.

Usage:
    python experiment_longitudinal.py                      # 100K cycles, random input
    python experiment_longitudinal.py --cycles 1000000     # 1M cycles
    python experiment_longitudinal.py --resume checkpoints/longitudinal/checkpoint_50000.zip
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, Any, List, Optional

import numpy as np

from ravana_ml.nn import RLM


# ── Configuration ──────────────────────────────────────────────

@dataclass
class LongitudinalConfig:
    # RLM architecture
    vocab_size: int = 256
    embed_dim: int = 64
    concept_dim: int = 32
    n_concepts: int = 200
    n_hidden: int = 128
    n_layers: int = 2
    sleep_interval: int = 100

    # Experiment
    n_cycles: int = 100_000
    checkpoint_every: int = 10_000
    metrics_every: int = 100
    diagnostics_every: int = 1_000

    # Input generation
    input_mode: str = "random"  # "random", "structured", "adversarial"

    # Checkpointing
    checkpoint_dir: str = "checkpoints/longitudinal"
    resume_from: Optional[str] = None


# ── Input Generation ───────────────────────────────────────────

class InputGenerator:
    """Generates token sequences for autonomous learning."""

    def __init__(self, vocab_size: int, mode: str = "random", seed: int = 42):
        self.vocab_size = vocab_size
        self.mode = mode
        self.rng = np.random.RandomState(seed)
        self._cycle = 0
        # For structured mode
        self._pattern_len = max(3, vocab_size // 20)
        self._pattern = self.rng.randint(0, vocab_size, size=self._pattern_len)
        # For adversarial mode
        self._shift_interval = 5000
        self._current_offset = 0

    def next(self) -> tuple:
        """Return (token_ids, next_token_ids) as numpy arrays."""
        if self.mode == "random":
            src = self.rng.randint(0, self.vocab_size)
            tgt = self.rng.randint(0, self.vocab_size)
        elif self.mode == "structured":
            # Periodic patterns with noise
            idx = self._cycle % self._pattern_len
            src = int(self._pattern[idx])
            tgt = int(self._pattern[(idx + 1) % self._pattern_len])
            # 20% noise
            if self.rng.random() < 0.2:
                src = self.rng.randint(0, self.vocab_size)
            if self.rng.random() < 0.2:
                tgt = self.rng.randint(0, self.vocab_size)
        elif self.mode == "adversarial":
            # Shifting distribution — every shift_interval cycles, rotate vocabulary
            if self._cycle > 0 and self._cycle % self._shift_interval == 0:
                self._current_offset = (self._current_offset + self.vocab_size // 3) % self.vocab_size
            src = (self.rng.randint(0, self.vocab_size) + self._current_offset) % self.vocab_size
            tgt = (self.rng.randint(0, self.vocab_size) + self._current_offset) % self.vocab_size
        else:
            raise ValueError(f"Unknown input mode: {self.mode}")

        self._cycle += 1
        return np.array([src], dtype=np.int64), np.array([tgt], dtype=np.int64)


# ── Metrics ────────────────────────────────────────────────────

def collect_cognitive_metrics(rlm: RLM) -> Dict[str, Any]:
    """Collect lightweight cognitive state metrics."""
    return {
        "cycle": rlm._step_counter,
        "identity_strength": rlm.identity_strength,
        "identity_momentum": rlm.identity_momentum,
        "valence": rlm.valence,
        "arousal": rlm.arousal,
        "dominance": rlm.dominance,
        "accumulated_meaning": rlm.accumulated_meaning,
        "sleep_pressure": rlm.sleep_pressure,
        "dissonance_ema": rlm.dissonance_ema,
        "regulation_mode": rlm.regulation_mode,
        "conceptual_accuracy": rlm.conceptual_accuracy,
        "total_free_energy": rlm.total_free_energy,
        "semantic_fe": rlm.free_energy_engine.semantic_free_energy,
        "episodic_fe": rlm.free_energy_engine.episodic_free_energy,
        "contradiction_fe": rlm.free_energy_engine.contradiction_free_energy,
        "n_edges": len(rlm.graph.edges),
        "n_nodes": len(rlm.graph.nodes),
        "sleep_cycles": rlm.sleep_cycles_completed,
        "episodic_buffer_size": len(rlm._episodic_buffer),
        "semantic_memory_count": len(rlm._semantic_memories),
    }


def collect_diagnostics(rlm: RLM) -> Dict[str, Any]:
    """Collect full graph diagnostics (expensive, call less often)."""
    diag = rlm.graph.graph_diagnostics()
    phase = rlm.graph.classify_phase(diag)
    return {
        "graph_entropy": diag.get("graph_entropy", 0.0),
        "clustering_coefficient": diag.get("clustering_coefficient", 0.0),
        "contradiction_density": diag.get("contradiction_density", 0.0),
        "relation_separation": diag.get("relation_separation", 0.0),
        "attractor_stability": diag.get("attractor_stability", 0.0),
        "core_active_alignment": diag.get("core_active_alignment", 0.0),
        "branching_factor": diag.get("branching_factor", 0.0),
        "branching_max": diag.get("branching_max", 0),
        "shortcut_ratio": diag.get("shortcut_ratio", 0.0),
        "edge_weight_mean": diag.get("edge_weight_mean", 0.0),
        "edge_weight_std": diag.get("edge_weight_std", 0.0),
        "basin_depth_mean": diag.get("basin_depth_mean", 0.0),
        "shallow_fraction": diag.get("shallow_fraction", 0.0),
        "curvature_trend": diag.get("curvature_trend", 0.0),
        "curvature_volatility": diag.get("curvature_volatility", 0.0),
        "inference_specificity_mean": diag.get("inference_specificity_mean", 0.0),
        "activation_mean": diag.get("activation_mean", 0.0),
        "phase": phase.get("phase", "unknown"),
        "phase_confidence": phase.get("confidence", 0.0),
        # Governor suppression monitoring
        "exploration_diversity": diag.get("graph_entropy", 0.0),
        "relation_type_distribution": diag.get("relation_type_distribution", {}),
    }


# ── Experiment Runner ──────────────────────────────────────────

class LongitudinalExperiment:
    def __init__(self, config: LongitudinalConfig):
        self.config = config
        self.metrics_history: List[Dict] = []
        self.start_cycle = 0
        self._metrics_path = os.path.join(config.checkpoint_dir, "metrics.jsonl")
        self._summary_path = os.path.join(config.checkpoint_dir, "summary.json")

        # Create or load RLM
        if config.resume_from:
            print(f"Resuming from {config.resume_from}")
            self.rlm = RLM.load_zip(config.resume_from)
            self._load_metrics_history()
            self.start_cycle = self.rlm._step_counter
            print(f"  Resumed at cycle {self.start_cycle}")
        else:
            self.rlm = RLM(
                vocab_size=config.vocab_size,
                embed_dim=config.embed_dim,
                concept_dim=config.concept_dim,
                n_concepts=config.n_concepts,
                n_hidden=config.n_hidden,
                n_layers=config.n_layers,
                sleep_interval=config.sleep_interval,
            )

        # Input generator (reseed from start_cycle for reproducibility)
        self.generator = InputGenerator(config.vocab_size, config.input_mode,
                                        seed=42 + self.start_cycle)

        # Ensure output directory exists
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def _load_metrics_history(self):
        """Load existing metrics from JSONL file."""
        if os.path.exists(self._metrics_path):
            self.metrics_history = []
            with open(self._metrics_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.metrics_history.append(json.loads(line))
            print(f"  Loaded {len(self.metrics_history)} metric samples")

    def _append_metrics(self, metrics: Dict):
        """Append metrics to JSONL file (append-only, crash-safe)."""
        self.metrics_history.append(metrics)
        with open(self._metrics_path, 'a') as f:
            f.write(json.dumps(metrics) + '\n')

    def _checkpoint(self, cycle: int):
        """Save model checkpoint."""
        path = os.path.join(self.config.checkpoint_dir, f"checkpoint_{cycle}.zip")
        self.rlm.save_zip(path)
        print(f"  Checkpoint saved: {path}")

    def run(self):
        """Run the longitudinal experiment."""
        total = self.config.n_cycles
        t_start = time.time()
        last_print = t_start

        print(f"Starting longitudinal experiment: {total} cycles")
        print(f"  Architecture: vocab={self.config.vocab_size}, embed={self.config.embed_dim}, "
              f"concept={self.config.concept_dim}, hidden={self.config.n_hidden}")
        print(f"  Input mode: {self.config.input_mode}")
        print(f"  Metrics every {self.config.metrics_every}, "
              f"diagnostics every {self.config.diagnostics_every}, "
              f"checkpoint every {self.config.checkpoint_every}")
        print()

        for cycle in range(self.start_cycle, total):
            # Generate input and learn
            src, tgt = self.generator.next()
            self.rlm.learn(src, tgt)

            # Record cognitive metrics
            if cycle % self.config.metrics_every == 0:
                m = collect_cognitive_metrics(self.rlm)
                self._append_metrics(m)

            # Record full diagnostics (less often)
            if cycle % self.config.diagnostics_every == 0:
                diag = collect_diagnostics(self.rlm)
                # Merge diagnostics into the last metrics entry
                if self.metrics_history:
                    self.metrics_history[-1].update(diag)
                    # Update the JSONL file (rewrite last line — acceptable for infrequent updates)
                    with open(self._metrics_path, 'r') as f:
                        lines = f.readlines()
                    if lines:
                        lines[-1] = json.dumps(self.metrics_history[-1]) + '\n'
                        with open(self._metrics_path, 'w') as f:
                            f.writelines(lines)

            # Checkpoint
            if cycle > 0 and cycle % self.config.checkpoint_every == 0:
                self._checkpoint(cycle)

            # Progress reporting
            if cycle % 10000 == 0:
                now = time.time()
                elapsed = now - t_start
                cycles_done = cycle - self.start_cycle
                rate = cycles_done / elapsed if elapsed > 0 else 0
                remaining = (total - cycle) / rate if rate > 0 else 0
                m = self.metrics_history[-1] if self.metrics_history else {}
                print(f"[{cycle:>8}/{total}] "
                      f"{rate:.0f} cyc/s, ETA {remaining/60:.1f}min | "
                      f"id={m.get('identity_strength', 0):.3f} "
                      f"V={m.get('valence', 0):.3f} "
                      f"A={m.get('arousal', 0):.3f} "
                      f"acc={m.get('conceptual_accuracy', 0):.3f} "
                      f"fe={m.get('total_free_energy', 0):.1f} "
                      f"edges={m.get('n_edges', 0)} "
                      f"sleep={m.get('sleep_cycles', 0)}")

        # Final checkpoint
        self._checkpoint(total)

        # Summary
        elapsed = time.time() - t_start
        summary = {
            "total_cycles": total,
            "elapsed_seconds": elapsed,
            "cycles_per_second": total / elapsed if elapsed > 0 else 0,
            "final_metrics": self.metrics_history[-1] if self.metrics_history else {},
            "metric_samples": len(self.metrics_history),
            "config": asdict(self.config),
        }
        with open(self._summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\nExperiment complete in {elapsed/60:.1f} minutes")
        print(f"  {len(self.metrics_history)} metric samples recorded")
        print(f"  Summary: {self._summary_path}")
        print(f"  Metrics: {self._metrics_path}")

        return summary


# ── CLI ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Longitudinal Concept Evolution Experiment")
    parser.add_argument("--cycles", type=int, default=100_000, help="Total learning cycles")
    parser.add_argument("--vocab", type=int, default=256, help="Vocabulary size")
    parser.add_argument("--embed", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--concept", type=int, default=32, help="Concept dimension")
    parser.add_argument("--concepts", type=int, default=200, help="Number of concepts")
    parser.add_argument("--hidden", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--sleep-interval", type=int, default=100, help="Sleep interval")
    parser.add_argument("--input-mode", choices=["random", "structured", "adversarial"],
                        default="random", help="Input generation mode")
    parser.add_argument("--checkpoint-dir", default="checkpoints/longitudinal",
                        help="Checkpoint directory")
    parser.add_argument("--checkpoint-every", type=int, default=10_000)
    parser.add_argument("--metrics-every", type=int, default=100)
    parser.add_argument("--diagnostics-every", type=int, default=1_000)
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    config = LongitudinalConfig(
        vocab_size=args.vocab,
        embed_dim=args.embed,
        concept_dim=args.concept,
        n_concepts=args.concepts,
        n_hidden=args.hidden,
        n_layers=args.layers,
        sleep_interval=args.sleep_interval,
        n_cycles=args.cycles,
        checkpoint_every=args.checkpoint_every,
        metrics_every=args.metrics_every,
        diagnostics_every=args.diagnostics_every,
        input_mode=args.input_mode,
        checkpoint_dir=args.checkpoint_dir,
        resume_from=args.resume,
    )

    experiment = LongitudinalExperiment(config)
    experiment.run()


if __name__ == "__main__":
    main()
