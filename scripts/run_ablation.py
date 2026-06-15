#!/usr/bin/env python3
"""
RAVANA Ablation Study Runner
=============================
Systematically runs all combinations of ablation flags and compares results.

Usage:
    python scripts/run_ablation.py --queries "hello|what is trust|explain oxiverse|bye" --runs 3
    python scripts/run_ablation.py --config ablation_config.yaml
"""

import sys
import os
import json
import subprocess
import itertools
import argparse
import time
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)


@dataclass
class AblationConfig:
    """Single ablation configuration."""
    no_vad: bool = False
    no_rlm: bool = False
    no_beliefs: bool = False
    no_curiosity: bool = False
    mode: str = "stochastic"
    
    def to_args(self) -> List[str]:
        args = []
        if self.no_vad:
            args.append("--no-vad")
        if self.no_rlm:
            args.append("--no-rlm")
        if self.no_beliefs:
            args.append("--no-beliefs")
        if self.no_curiosity:
            args.append("--no-curiosity")
        if self.mode != "stochastic":
            args.extend(["--mode", self.mode])
        return args
    
    def name(self) -> str:
        parts = []
        if self.no_vad: parts.append("no_vad")
        if self.no_rlm: parts.append("no_rlm")
        if self.no_beliefs: parts.append("no_beliefs")
        if self.no_curiosity: parts.append("no_curiosity")
        if self.mode != "stochastic": parts.append(f"mode_{self.mode}")
        return "+".join(parts) if parts else "baseline"


@dataclass
class AblationResult:
    """Result of a single ablation run."""
    config: AblationConfig
    queries: List[str]
    responses: List[str]
    latencies: List[float]
    total_time: float
    error: str = ""
    
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
    
    def success_rate(self) -> float:
        if not self.responses:
            return 0.0
        successful = sum(1 for r in self.responses if r and not r.startswith("offline"))
        return successful / len(self.responses)


def run_single_ablation(config: AblationConfig, queries: List[str], 
                         data_dir: str = None, dim: int = 64,
                         timeout: int = 120) -> AblationResult:
    """Run RAVANA chat for a single ablation configuration."""
    cmd = [sys.executable, "scripts/ravana_chat.py"]
    cmd.extend(config.to_args())
    cmd.extend(["--chat", "|".join(queries), "--strategy"])
    
    if data_dir:
        cmd.extend(["--data-dir", data_dir])
    cmd.extend(["--dim", str(dim)])
    
    print(f"  Running: {' '.join(cmd)}")
    
    start = time.time()
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=timeout,
            cwd=_proj_root
        )
        elapsed = time.time() - start
        
        if result.returncode != 0:
            return AblationResult(
                config=config, queries=queries, responses=[], latencies=[],
                total_time=elapsed, error=result.stderr[:500]
            )
        
        # Parse output: look for Q: and A: lines
        responses = []
        latencies = []
        for line in result.stdout.strip().split('\n'):
            if line.startswith("A: ") or line.startswith("A ("):
                # Extract response and optional latency
                parts = line.split("|")
                resp = parts[0].replace("A: ", "").replace("A (", "").strip()
                responses.append(resp)
                # Try to extract latency from strategy tag
                lat = 0.0
                for p in parts:
                    if "lat=" in p:
                        try:
                            lat = float(p.split("lat=")[1].split()[0])
                        except:
                            pass
                latencies.append(lat)
        
        return AblationResult(
            config=config, queries=queries, responses=responses, 
            latencies=latencies, total_time=elapsed
        )
        
    except subprocess.TimeoutExpired:
        return AblationResult(
            config=config, queries=queries, responses=[], latencies=[],
            total_time=timeout, error="timeout"
        )
    except Exception as e:
        return AblationResult(
            config=config, queries=queries, responses=[], latencies=[],
            total_time=time.time() - start, error=str(e)
        )


def generate_configs(include_modes: bool = True) -> List[AblationConfig]:
    """Generate all ablation configurations."""
    configs = []
    flags = ["no_vad", "no_rlm", "no_beliefs", "no_curiosity"]
    modes = ["stochastic", "deterministic", "exploratory"] if include_modes else ["stochastic"]
    
    for r in range(len(flags) + 1):
        for combo in itertools.combinations(flags, r):
            for mode in modes:
                kwargs = {f: (f in combo) for f in flags}
                kwargs["mode"] = mode
                configs.append(AblationConfig(**kwargs))
    
    return configs


def print_comparison_table(results: List[AblationResult]):
    """Print a markdown comparison table."""
    print("\n" + "=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print(f"Total configurations: {len(results)}")
    print(f"Queries per config: {len(results[0].queries) if results else 0}\n")
    
    # Summary table
    header = f"{'Config':<35} {'Success':>8} {'Avg Latency':>12} {'Total Time':>10} {'Errors'}"
    print(header)
    print("-" * len(header))
    
    for r in sorted(results, key=lambda x: x.config.name()):
        config_name = r.config.name()[:34]
        success = f"{r.success_rate()*100:.1f}%"
        avg_lat = f"{r.avg_latency():.3f}s"
        total_t = f"{r.total_time:.1f}s"
        err = r.error[:20] if r.error else ""
        print(f"{config_name:<35} {success:>8} {avg_lat:>12} {total_t:>10} {err}")
    
    print("\n" + "=" * 100)


def save_results(results: List[AblationResult], output_path: str):
    """Save detailed results to JSON."""
    serializable = []
    for r in results:
        serializable.append({
            "config": asdict(r.config),
            "queries": r.queries,
            "responses": r.responses,
            "latencies": r.latencies,
            "total_time": r.total_time,
            "error": r.error,
            "avg_latency": r.avg_latency(),
            "success_rate": r.success_rate(),
        })
    
    with open(output_path, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "results": serializable
        }, f, indent=2)
    print(f"\nSaved detailed results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="RAVANA Ablation Study Runner")
    parser.add_argument("--queries", type=str, default="hello|what is trust|explain oxiverse|tell me about ravana|bye",
                        help="Pipe-separated test queries")
    parser.add_argument("--runs", type=int, default=1, help="Number of runs per config (for variance)")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--timeout", type=int, default=120, help="Timeout per run (seconds)")
    parser.add_argument("--output", type=str, default="ablation_results.json", help="Output JSON file")
    parser.add_argument("--modes", action="store_true", help="Include mode variations (adds 3x configs)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: only single flag ablations + baseline")
    args = parser.parse_args()
    
    queries = [q.strip() for q in args.queries.split("|")]
    
    if args.quick:
        # Baseline + single flag removals
        configs = [
            AblationConfig(),  # baseline
            AblationConfig(no_vad=True),
            AblationConfig(no_rlm=True),
            AblationConfig(no_beliefs=True),
            AblationConfig(no_curiosity=True),
        ]
        if args.modes:
            configs.extend([
                AblationConfig(mode="deterministic"),
                AblationConfig(mode="exploratory"),
            ])
    else:
        configs = generate_configs(include_modes=args.modes)
    
    print(f"Running {len(configs)} ablation configurations...")
    print(f"Queries: {queries}")
    print(f"Runs per config: {args.runs}")
    print(f"Data dir: {args.data_dir or 'default'}")
    print()
    
    all_results = []
    for i, config in enumerate(configs):
        print(f"[{i+1}/{len(configs)}] Config: {config.name()}")
        
        for run_idx in range(args.runs):
            if args.runs > 1:
                # Use unique data dir per run to avoid state contamination
                run_data_dir = f"{args.data_dir}/run_{run_idx}" if args.data_dir else f"/tmp/ravana_abl_{config.name()}_{run_idx}"
            else:
                run_data_dir = args.data_dir
            
            result = run_single_ablation(config, queries, run_data_dir, args.dim, args.timeout)
            all_results.append(result)
            
            if result.error:
                print(f"    ERROR: {result.error}")
            else:
                print(f"    Success: {result.success_rate()*100:.0f}%, Avg latency: {result.avg_latency():.3f}s")
        
        print()
    
    print_comparison_table(all_results)
    save_results(all_results, args.output)
    
    # Summary stats
    successful = [r for r in all_results if not r.error]
    if successful:
        best = max(successful, key=lambda x: x.success_rate())
        print(f"\nBest config by success rate: {best.config.name()} ({best.success_rate()*100:.1f}%)")
        fastest = min(successful, key=lambda x: x.avg_latency())
        print(f"Fastest config: {fastest.config.name()} ({fastest.avg_latency():.3f}s avg)")


if __name__ == "__main__":
    main()