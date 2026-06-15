#!/usr/bin/env python3
"""
Per-Triple Evaluation Harness for RAVANA
=========================================
Provides detailed per-triple diagnostics instead of averaged metrics.

Evaluates each triple with:
- relation_type (semantic, causal, contrastive, possessive, temporal, analogical)
- prediction_error (PE) - how well the model predicts the object
- confidence - model's confidence in the prediction
- source - where the triple was learned (seed, web, user, sleep consolidation)
- graph edge attributes (weight, edge_type, prediction_free_energy)

Usage:
    python scripts/triple_eval.py --triples-file triples.json --output report.json
    python scripts/triple_eval.py --interactive
"""

import sys
import os
import json
import argparse
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
from collections import defaultdict

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))
sys.path.insert(0, _proj_root)

from ravana_ml.graph import ConceptGraph, ConceptNode, ConceptEdge
from ravana_ml.tokenizer import WordTokenizer
from ravana_ml.nn.rlm_v2 import RLMv2, RELATION_TYPES
from scripts.ravana_chat import CognitiveChatEngine


@dataclass
class TripleEvalResult:
    """Per-triple evaluation result."""
    # Triple content
    subject: str
    relation: str
    object: str
    relation_type: str
    
    # Metrics
    prediction_error: float = 0.0
    confidence: float = 0.0
    top1_accuracy: bool = False
    top5_accuracy: bool = False
    rank: int = -1
    
    # Graph edge info
    edge_weight: Optional[float] = None
    edge_confidence: Optional[float] = None
    edge_type: Optional[str] = None
    edge_prediction_free_energy: Optional[float] = None
    edge_exist: bool = False
    
    # Source tracking
    source: str = "unknown"  # seed, web, user, sleep, inference
    learned_turn: Optional[int] = None
    
    # Model prediction details
    predicted_object: str = ""
    predicted_prob: float = 0.0
    all_candidates: List[Tuple[str, float]] = field(default_factory=list)


class TripleEvaluationHarness:
    """Evaluates triples with detailed per-triple diagnostics."""
    
    def __init__(self, engine: CognitiveChatEngine = None, dim: int = 64, data_dir: str = None):
        if engine is not None:
            self.engine = engine
            self.graph = engine.graph
            self.tokenizer = engine.tokenizer if hasattr(engine, 'tokenizer') else WordTokenizer()
        else:
            self.engine = None
            self.graph = ConceptGraph(dim=dim)
            self.tokenizer = WordTokenizer()
            
            # Build vocab for tokenizer if needed
            if self.engine is None:
                for word in ["causes", "is", "has", "like", "then", "before", "trust", "love", "fear", "knowledge", "intelligence", "creativity", "learning", "memory", "emotion", "reason"]:
                    self.tokenizer.encode(word)
    
    def evaluate_triple(self, subject: str, relation: str, object: str) -> TripleEvalResult:
        """Evaluate a single triple with full diagnostics."""
        result = TripleEvalResult(
            subject=subject,
            relation=relation,
            object=object,
            relation_type=self._classify_relation(relation)
        )
        
        # Normalize
        subj_l = subject.lower().strip()
        rel_l = relation.lower().strip()
        obj_l = object.lower().strip()
        
        # Find concept nodes
        subj_nids = self.engine._concept_keywords.get(subj_l, []) if self.engine else []
        obj_nids = self.engine._concept_keywords.get(obj_l, []) if self.engine else []
        rel_nids = self.engine._concept_keywords.get(rel_l, []) if self.engine else []
        
        # Check graph edge
        if subj_nids and obj_nids:
            edge = self.graph.get_edge(subj_nids[0], obj_nids[0])
            if edge:
                result.edge_exist = True
                result.edge_weight = edge.weight
                result.edge_confidence = edge.confidence
                result.edge_type = edge.relation_type
                result.edge_prediction_free_energy = getattr(edge, 'prediction_free_energy', None)
        
        # If we have an engine with RLM, do model-based evaluation
        if self.engine and hasattr(self.engine, 'rlm') and self.engine.rlm:
            result = self._evaluate_with_rlm(result, subj_nids, rel_nids, obj_nids)
        elif self.engine and self.engine.neural_decoder and self.engine._decoder_vocab_built:
            result = self._evaluate_with_decoder(result, subj_l, rel_l, obj_l)
        
        # Determine source
        result.source = self._determine_source(result, subj_l, rel_l, obj_l)
        
        return result
    
    def _classify_relation(self, relation: str) -> str:
        """Classify relation into type using RLM if available."""
        if self.engine and hasattr(self.engine, 'rlm') and self.engine.rlm and hasattr(self.engine, 'tokenizer'):
            r_toks = self.engine.tokenizer.encode(relation)
            if r_toks:
                idx = self.engine.rlm.classify_relation(r_toks)
                return RELATION_TYPES[idx]
        
        # Fallback heuristic
        rel_l = relation.lower()
        if rel_l in ["causes", "produces", "leads to", "results in", "creates", "generates"]:
            return "causal"
        elif rel_l in ["is", "is a", "equals", "means", "defines"]:
            return "semantic"
        elif rel_l in ["has", "contains", "owns", "possesses"]:
            return "possessive"
        elif rel_l in ["like", "similar to", "resembles", "analogous"]:
            return "analogical"
        elif rel_l in ["then", "before", "after", "when"]:
            return "temporal"
        elif rel_l in ["but", "unlike", "opposite", "contrasts"]:
            return "contrastive"
        return "semantic"
    
    def _evaluate_with_rlm(self, result: TripleEvalResult, 
                           subj_nids: List[int], rel_nids: List[int], obj_nids: List[int]) -> TripleEvalResult:
        """Evaluate using RLMv2 model."""
        rlm = self.engine.rlm
        tok = self.engine.tokenizer
        
        # Tokenize
        subj_toks = tok.encode(result.subject)
        rel_toks = tok.encode(result.relation)
        obj_toks = tok.encode(result.object)
        
        if not subj_toks or not obj_toks:
            return result
        
        # Build input sequence: subject + relation + object
        # RLM expects the full triple as input
        input_ids = subj_toks + rel_toks + obj_toks
        input_arr = np.array(input_ids, dtype=np.int64)
        
        # Forward pass
        logits = rlm.forward(input_arr[:-1])  # predict next token(s)
        
        # Get probabilities for the object tokens
        if logits is not None and hasattr(logits, 'data'):
            # The last position before object should predict object
            pred_logits = logits.data
            probs = self._softmax(pred_logits)
            
            # Check if object token is in top predictions
            obj_token = obj_toks[0] if obj_toks else -1
            if obj_token < len(probs):
                result.predicted_prob = float(probs[obj_token])
                
                # Rank
                sorted_idx = np.argsort(probs)[::-1]
                try:
                    result.rank = int(np.where(sorted_idx == obj_token)[0][0]) + 1
                except:
                    result.rank = -1
                
                result.top1_accuracy = (result.rank == 1)
                result.top5_accuracy = (result.rank <= 5 and result.rank > 0)
                
                # Predicted object (top1)
                top1_token = int(sorted_idx[0])
                result.predicted_object = tok.decode([top1_token])
                
                # Top-5 candidates
                for i, idx in enumerate(sorted_idx[:5]):
                    result.all_candidates.append((tok.decode([int(idx)]), float(probs[idx])))
                
                # Prediction error: 1 - prob of correct token (normalized)
                result.prediction_error = 1.0 - result.predicted_prob
                result.confidence = result.predicted_prob
        
        return result
    
    def _evaluate_with_decoder(self, result: TripleEvalResult, 
                                subj_l: str, rel_l: str, obj_l: str) -> TripleEvalResult:
        """Evaluate using neural decoder."""
        # Use the engine's graph walk to generate
        if hasattr(self.engine, '_walk_chain'):
            chain = self.engine._walk_chain(result.subject, set(), max_hops=1)
            if chain:
                # Check if object appears in chain
                chain_labels = [self.graph.get_node(n).label for n in chain if self.graph.get_node(n)]
                if obj_l in [l.lower() for l in chain_labels if l]:
                    result.top1_accuracy = True
                    result.confidence = 0.8  # heuristic
        
        return result
    
    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Stable softmax."""
        logits = logits - np.max(logits)
        exp_logits = np.exp(np.clip(logits, -50, 50))
        return exp_logits / (np.sum(exp_logits) + 1e-10)
    
    def _determine_source(self, result: TripleEvalResult, 
                          subj_l: str, rel_l: str, obj_l: str) -> str:
        """Determine where the triple was learned from."""
        if not result.edge_exist:
            return "unknown"
        
        # Check concept sources if available
        if self.engine and hasattr(self.engine, '_concept_sources'):
            for concept in [subj_l, rel_l, obj_l]:
                if concept in self.engine._concept_sources:
                    sources = self.engine._concept_sources[concept]
                    if "web" in str(sources):
                        return "web"
                    elif "user" in str(sources):
                        return "user"
                    elif "seed" in str(sources):
                        return "seed"
        
        # Check edge metadata
        if subj_l in self.engine._concept_keywords and obj_l in self.engine._concept_keywords:
            subj_nids = self.engine._concept_keywords[subj_l]
            obj_nids = self.engine._concept_keywords[obj_l]
            if subj_nids and obj_nids:
                edge = self.graph.get_edge(subj_nids[0], obj_nids[0])
                if edge and edge.source_metadata:
                    src_agent = edge.source_metadata.get('source_agent', 'unknown')
                    if src_agent != 'system':
                        return src_agent
        
        # Heuristic based on edge confidence
        if result.edge_confidence is not None:
            if result.edge_confidence > 0.7:
                return "seed"
            elif result.edge_confidence > 0.3:
                return "web"
            else:
                return "inference"
        
        return "inference"
    
    def evaluate_batch(self, triples: List[Tuple[str, str, str]]) -> List[TripleEvalResult]:
        """Evaluate a batch of triples."""
        results = []
        for subj, rel, obj in triples:
            result = self.evaluate_triple(subj, rel, obj)
            results.append(result)
        return results
    
    def generate_report(self, results: List[TripleEvalResult]) -> Dict[str, Any]:
        """Generate aggregated diagnostic report."""
        if not results:
            return {"error": "No results to report"}
        
        # Group by relation type
        by_type = defaultdict(list)
        by_source = defaultdict(list)
        
        for r in results:
            by_type[r.relation_type].append(r)
            by_source[r.source].append(r)
        
        # Overall metrics
        total = len(results)
        top1_acc = sum(1 for r in results if r.top1_accuracy) / total
        top5_acc = sum(1 for r in results if r.top5_accuracy) / total
        avg_pe = np.mean([r.prediction_error for r in results])
        avg_conf = np.mean([r.confidence for r in results])
        avg_rank = np.mean([r.rank for r in results if r.rank > 0]) if any(r.rank > 0 for r in results) else 0
        
        # Per-type metrics
        type_metrics = {}
        for rtype, rlist in by_type.items():
            type_metrics[rtype] = {
                "count": len(rlist),
                "top1_accuracy": sum(1 for r in rlist if r.top1_accuracy) / len(rlist),
                "top5_accuracy": sum(1 for r in rlist if r.top5_accuracy) / len(rlist),
                "avg_prediction_error": np.mean([r.prediction_error for r in rlist]),
                "avg_confidence": np.mean([r.confidence for r in rlist]),
                "avg_edge_weight": np.mean([r.edge_weight for r in rlist if r.edge_weight is not None]) if any(r.edge_weight for r in rlist) else 0,
                "avg_edge_pe": np.mean([r.edge_prediction_free_energy for r in rlist if r.edge_prediction_free_energy is not None]) if any(r.edge_prediction_free_energy for r in rlist) else 0,
            }
        
        # Per-source metrics
        source_metrics = {}
        for src, rlist in by_source.items():
            source_metrics[src] = {
                "count": len(rlist),
                "top1_accuracy": sum(1 for r in rlist if r.top1_accuracy) / len(rlist),
                "avg_prediction_error": np.mean([r.prediction_error for r in rlist]),
            }
        
        # Edge health
        edge_results = [r for r in results if r.edge_exist]
        edge_metrics = {}
        if edge_results:
            edge_metrics = {
                "count": len(edge_results),
                "avg_weight": np.mean([r.edge_weight for r in edge_results]),
                "avg_confidence": np.mean([r.edge_confidence for r in edge_results]),
                "avg_prediction_free_energy": np.mean([r.edge_prediction_free_energy for r in edge_results if r.edge_prediction_free_energy is not None]),
                "by_type": defaultdict(int),
            }
            for r in edge_results:
                if r.edge_type:
                    edge_metrics["by_type"][r.edge_type] += 1
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_triples": total,
            "overall": {
                "top1_accuracy": top1_acc,
                "top5_accuracy": top5_acc,
                "avg_prediction_error": float(avg_pe),
                "avg_confidence": float(avg_conf),
                "avg_rank": float(avg_rank),
            },
            "by_relation_type": type_metrics,
            "by_source": source_metrics,
            "edge_health": edge_metrics,
            "per_triple": [asdict(r) for r in results]
        }
    
    def print_summary(self, report: Dict[str, Any]):
        """Print human-readable summary."""
        print("\n" + "=" * 80)
        print("PER-TRIPLE EVALUATION REPORT")
        print("=" * 80)
        print(f"Timestamp: {report.get('timestamp', 'N/A')}")
        print(f"Total triples evaluated: {report.get('total_triples', 0)}")
        
        overall = report.get('overall', {})
        print(f"\nOVERALL METRICS:")
        print(f"  Top-1 Accuracy:    {overall.get('top1_accuracy', 0)*100:.1f}%")
        print(f"  Top-5 Accuracy:    {overall.get('top5_accuracy', 0)*100:.1f}%")
        print(f"  Avg Prediction PE: {overall.get('avg_prediction_error', 0):.4f}")
        print(f"  Avg Confidence:    {overall.get('avg_confidence', 0):.4f}")
        print(f"  Avg Rank:          {overall.get('avg_rank', 0):.1f}")
        
        print(f"\nBY RELATION TYPE:")
        for rtype, metrics in report.get('by_relation_type', {}).items():
            print(f"  {rtype:15s}: n={metrics['count']:3d}  Top1={metrics['top1_accuracy']*100:5.1f}%  Top5={metrics['top5_accuracy']*100:5.1f}%  PE={metrics['avg_prediction_error']:.4f}  Conf={metrics['avg_confidence']:.4f}  EdgeW={metrics['avg_edge_weight']:.3f}")
        
        print(f"\nBY SOURCE:")
        for src, metrics in report.get('by_source', {}).items():
            print(f"  {src:12s}: n={metrics['count']:3d}  Top1={metrics['top1_accuracy']*100:5.1f}%  PE={metrics['avg_prediction_error']:.4f}")
        
        edge = report.get('edge_health', {})
        if edge.get('count', 0) > 0:
            print(f"\nEDGE HEALTH ({edge['count']} edges):")
            print(f"  Avg Weight:              {edge['avg_weight']:.4f}")
            print(f"  Avg Confidence:          {edge['avg_confidence']:.4f}")
            print(f"  Avg Edge PE:             {edge['avg_prediction_free_energy']:.4f}")
            print(f"  By Type: {dict(edge['by_type'])}")


def main():
    parser = argparse.ArgumentParser(description="RAVANA Per-Triple Evaluation Harness")
    parser.add_argument("--triples-file", type=str, help="JSON file with triples [(subj, rel, obj), ...]")
    parser.add_argument("--output", type=str, default="triple_eval_report.json", help="Output report file")
    parser.add_argument("--data-dir", type=str, help="Data directory for engine")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    # Default test triples if no file provided
    default_triples = [
        ("trust", "is", "fragile"),
        ("trust", "causes", "vulnerability"),
        ("love", "is", "emotion"),
        ("knowledge", "causes", "power"),
        ("fear", "contrasts", "courage"),
        ("memory", "stores", "experience"),
        ("learning", "produces", "knowledge"),
        ("fire", "produces", "heat"),
        ("water", "is", "liquid"),
        ("dog", "is a", "animal"),
    ]
    
    if args.triples_file:
        with open(args.triples_file) as f:
            triples = json.load(f)
    else:
        triples = default_triples
    
    print(f"Evaluating {len(triples)} triples...")
    
    # Initialize engine
    engine = CognitiveChatEngine(dim=args.dim, data_dir=args.data_dir, seed=42) if args.data_dir else None
    
    harness = TripleEvaluationHarness(engine=engine, dim=args.dim, data_dir=args.data_dir)
    
    # Evaluate
    results = harness.evaluate_batch(triples)
    
    # Generate report
    report = harness.generate_report(results)
    harness.print_summary(report)
    
    # Save
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.output}")
    
    if engine:
        engine.stop_background_learning()


if __name__ == "__main__":
    main()