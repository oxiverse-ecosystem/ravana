#!/usr/bin/env python3
"""
RAVANA Baby — cognitive chat that starts like a baby and learns from the web
============================================================================
No commands. No LLM. Pure RAVANA cognitive architecture.
Starts knowing ~180 teen-level concepts with GloVe semantic embeddings
and typed graph edges (causal, contrastive, analogical, temporal, semantic),
auto-learns from the internet when it doesn't know something.

Usage:
    python scripts/ravana_chat.py
"""

import sys, os, time, random, json, re, argparse, pickle, threading, hashlib, io
import urllib.request
import socket
socket.setdefaulttimeout(4.0)
if sys.platform == "win32" and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from urllib.error import URLError
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Insert in REVERSE order of priority (last insert ends up first in sys.path)
# NOTE: modular `ravana` package (with language/core/chat) lives at ravana/src/ravana/.
# It MUST shadow the stale root ravana/ dir (whose __init__.py is broken), so we add
# ravana/src LAST (highest priority) and keep _proj_root for `scripts` package discovery.
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2", "src"))

from ravana_ml.graph import ConceptGraph, ConceptEdge
# === Phase 0: Import from engine module ===
from ravana.chat.models import FailedQuery, ChainHop, ChainTrace, CognitiveResponseContext
from ravana.chat.belief_store import BeliefStore
from ravana.chat.user_model import UserModel
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.constants import _is_word_salad, TEEN_CONCEPTS, WEB_GARBAGE, STOP_WORDS
# =============================================
from ravana_grace.core.emotion import VADEmotionEngine, VADConfig
from ravana_grace.core.identity import IdentityEngine
from ravana_grace.core.meaning import MeaningEngine, MeaningConfig
from ravana_grace.core.dual_process import DualProcessController, DualProcessConfig
from ravana_grace.core.global_workspace import GlobalWorkspace, GWConfig
from ravana_grace.core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from ravana_grace.core.sleep import SleepConsolidation, SleepConfig
from ravana.language.basal_ganglia import BasalGangliaGate
from ravana.language.cerebellar_ngram import CerebellarNgram, CerebellarState
from ravana.language.prefrontal_workspace import PrefrontalWorkspace, DiscourseIntent, DiscoursePlan, DiscourseType
from ravana.language.syntactic_cell_assembly import SyntacticCellAssembly, SyntacticFrame
from ravana.language.surface_realizer import SurfaceRealizer, DiscourseState
from ravana_ml.nn.neural_decoder import NeuralDecoder
from ravana.core import UserEmotionDetector
from ravana.core.hippocampal_buffer import HippocampalBuffer, HippocampalConfig
from ravana.core.proposition_parser import PropositionParser
from ravana.core.causal_schema import CausalSchemaLearner, CausalSchemaConfig
from ravana.core.implicature_detector import ImplicatureDetector
from ravana.core.relation_memory import RelationMemory, RelationMemoryConfig
from ravana.core.quantity_modifier import QuantityModifierSystem


# ─── Teen Vocabulary: what a teenager knows (~180 words) ───
DOMAIN_CONCEPTS = {
    "oxiverse": {
        "keywords": "oxiverse privacy-first source-available ecosystem alternative big-tech",
        "relations": [
            ("oxiverse", "ecosystem", "is_a", 0.7),
            ("oxiverse", "privacy", "causal", 0.65),
            ("oxiverse", "big tech", "contrastive", 0.55),
        ],
        "stability": 0.9,
    },
    "intentforge": {
        "keywords": "intentforge intent-driven semantic search engine discovery",
        "relations": [
            ("intentforge", "search engine", "is_a", 0.7),
            ("intentforge", "discovery", "causal", 0.6),
            ("intentforge", "oxiverse", "part_of", 0.65),
        ],
        "stability": 0.9,
    },
    "ravana": {
        "keywords": "ravana cognitive architecture backprop-free hebbian reasoning",
        "relations": [
            ("ravana", "cognitive architecture", "is_a", 0.7),
            ("ravana", "hebbian learning", "causal", 0.6),
            ("ravana", "analogical reasoning", "causal", 0.55),
            ("ravana", "oxiverse", "part_of", 0.5),
        ],
        "stability": 0.9,
    },
}

# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Pure natural language chat, no commands
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RAVANA - teenage mind on the web")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reset", action="store_true", help="Delete saved weights and start fresh")
    parser.add_argument("--chat", type=str, default=None,
        help='Send queries in batch mode. Use | to separate multiple queries. '
             'Outputs Q: and A: lines for easy parsing. E.g.: --chat "hi|what is trust|bye"')
    parser.add_argument("--strategy", action="store_true", help="Include strategy name in --chat output")
    parser.add_argument("--trace", action="store_true", help="Print edge-level chain traces")
    parser.add_argument("--trace-monitors", action="store_true",
                        help="Print the structured self-monitor log (engine.monitor_report) at exit — "
                             "shows every guard fire / swallow and why (M10 observability)")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD emotion modulation")
    parser.add_argument("--no-rlm", action="store_true", help="Disable RLMv2 triple verification")
    parser.add_argument("--no-beliefs", action="store_true", help="Disable belief store")
    parser.add_argument("--no-curiosity", action="store_true", help="Disable autonomous curiosity-driven learning")
    parser.add_argument("--snippet-pe", action="store_true",
                        help="Enable Track B Phase 2 learned snippet-quality model "
                             "(structural prediction-error gate) for web snippets. "
                             "Off by default; hardcoded filters remain the backstop.")
    parser.add_argument("--source-trust", action="store_true",
                        help="Enable Track B Phase 3 learned per-domain source-trust "
                             "(replaces the hardcoded preferred-source allowlist). "
                             "Off by default; the allowlist remains the backstop.")
    parser.add_argument("--learned-pos", action="store_true",
                        help="Enable Track B Phase 5 learned distributional POS "
                             "(replaces the hardcoded function-word set). "
                             "Off by default; the hardcoded set remains the backstop.")
    parser.add_argument("--conceptnet-primary", action="store_true",
                        help="Enable Track B Phase 6 ConceptNet as the primary "
                             "frontopolar feasibility gate (replaces the literal "
                             "category tables). Off by default; the literal dicts "
                             "remain the fallback when ConceptNet is silent.")
    parser.add_argument("--register", type=str, default="default",
                        choices=["default", "confident", "cautious", "verbose", "terse"],
                        help="P6: epistemic register (roadmap #12) — one knob for "
                             "confidence/verbosity/curiosity. 'terse' suppresses "
                             "on-demand retrieval + sourced-evidence clauses.")
    parser.add_argument("--mode", type=str, default="stochastic", choices=["stochastic", "deterministic", "exploratory"],
                        help="Reasoning mode: stochastic (default), deterministic (reproducible), exploratory (high-temp)")
    parser.add_argument("--debug", action="store_true", help="Print debug tracebacks for exceptions")

    parser.add_argument("--user", type=str, default=None,
                        help="User name for multi-user isolation (creates user-specific save files)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory for weights and GloVe cache")
    parser.add_argument("--export-graph", type=str, default=None,
                        help="Export graph to JSON file (all concepts + edges)")
    parser.add_argument("--import-graph", type=str, default=None,
                        help="Import graph from JSON file (merge into existing)")
    parser.add_argument("--stats", action="store_true",
                        help="Print graph statistics")
    parser.add_argument("--concept", type=str, default=None,
                        help="Show what RAVANA knows about a concept")
    args = parser.parse_args()

    # Handle --reset
    reset_suffix = args.user or ""
    save_path = os.path.join(_proj_root, "data", f"ravana_weights{reset_suffix}.pkl")
    if args.reset:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"  [Reset] Deleted {os.path.basename(save_path)}, starting fresh!")
        else:
            print(f"  [Reset] No saved weights found, starting fresh!")

    data_dir = args.data_dir
    user_suffix = args.user or ""
    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed, baby_mode=True, data_dir=data_dir, user_suffix=user_suffix)
    engine.start_background_learning()
    if args.trace:
        engine._trace_enabled = True
    if args.trace_monitors:
        # M10: observability. Reuse _trace_enabled so guard fires also print,
        # and dump the structured monitor log at exit.
        engine._trace_enabled = True
    if args.no_vad:
        engine.use_vad = False
        print("  [Config] VAD modulation disabled")
    if args.no_rlm:
        engine.use_rlm = False
        print("  [Config] RLMv2 triple verification disabled")
    if args.no_beliefs:
        engine.use_beliefs = False
        print("  [Config] Belief store disabled")
    if args.no_curiosity:
        engine._curiosity_drive_enabled = False
        print('  [Curiosity] Autonomous learning disabled')
    if args.snippet_pe:
        engine.use_cerebellar_snippet = True
        print('  [Snippets] Track B Phase 2 learned structural-PE gate ENABLED')
    if args.source_trust:
        engine.use_source_trust = True
        print('  [Sources] Track B Phase 3 learned per-domain source-trust ENABLED')
    if args.learned_pos:
        engine.use_learned_pos = True
        print('  [POS] Track B Phase 5 learned distributional POS ENABLED')
    if args.conceptnet_primary:
        engine.use_conceptnet_primary = True
        print('  [Ontology] Track B Phase 6 ConceptNet-primary gate ENABLED')

    # P6: epistemic register (roadmap #12) — single knob for confidence/
    # verbosity/curiosity. Recompute the register multipliers after init.
    if args.register != "default":
        _REG = {
            "default":  {"curiosity": 1.0, "verbosity": 1.0, "confidence": 1.0},
            "confident": {"curiosity": 1.0, "verbosity": 1.0, "confidence": 1.3},
            "cautious":  {"curiosity": 1.0, "verbosity": 1.0, "confidence": 0.7},
            "verbose":   {"curiosity": 1.0, "verbosity": 1.0, "confidence": 1.0},
            "terse":     {"curiosity": 0.3, "verbosity": 0.2, "confidence": 1.0},
        }
        if args.register in _REG:
            engine.epistemic_register = args.register
            _r = _REG[args.register]
            engine._reg_curiosity = _r["curiosity"]
            engine._reg_verbosity = _r["verbosity"]
            engine._reg_confidence = _r["confidence"]
            print(f"  [Register] Epistemic register set to '{args.register}'")

    # Solution #2: Apply reasoning mode
    if args.mode != "stochastic":
        engine.reasoning_mode = args.mode
        print(f"  [Mode] Reasoning mode set to '{args.mode}'")


    # ── BATCH MODE (--chat) ──
    # ── Phase 6 CLI Actions ──
    if args.export_graph:
        try:
            import json
            g = engine.graph
            data = {
                "nodes": [{"id": n.id, "label": n.label} for n in g.nodes.values()],
                "edges": [{"source": src, "target": tgt,
                           "relation": e.relation_type, "weight": e.weight}
                          for (src, tgt), e in g.edges.items()],
            }
            with open(args.export_graph, "w") as f:
                json.dump(data, f, indent=2)
            ec = len(data["edges"])
            nc = len(data["nodes"])
            print(f"  [Export] Exported {nc} nodes + {ec} edges to {args.export_graph}")
        except Exception as e:
            print(f"  [Export] Failed: {e}")
        engine.save()
        return
    if args.import_graph:
        try:
            import json
            with open(args.import_graph, "r") as f:
                data = json.load(f)
            g = engine.graph
            count = 0
            for node_data in data.get("nodes", []):
                nid = node_data.get("id")
                label = node_data.get("label", "")
                if label:
                    # Use GloVe vector if available, otherwise let graph create random vector
                    vec = engine._glove_vector(label) if engine._glove_vecs is not None else None
                    added_node = g.add_node(vector=vec, label=label)
                    count += 1
            for edge_data in data.get("edges", []):
                src = edge_data.get("source")
                tgt = edge_data.get("target")
                rel = edge_data.get("relation", "related")
                w = edge_data.get("weight", 0.5)
                if src is not None and tgt is not None:
                    g.add_edge(src, tgt, relation_type=rel, weight=w)
            print(f"  [Import] Imported {len(data.get('nodes', []))} nodes + {len(data.get('edges', []))} edges")
        except Exception as e:
            print(f"  [Import] Failed: {e}")
        engine.save()
        return
    if args.stats:
        print(f"  [Stats] Graph has {len(engine.graph.nodes)} nodes and {len(engine.graph.edges)} edges")
        print(f"  [Stats] Turn count: {engine.turn_count}")
        return
    if args.concept:
        nids = engine._concept_keywords.get(args.concept.lower(), [])
        if nids:
            node = engine.graph.get_node(nids[0])
            if node:
                outgoing = engine.graph.get_outgoing(nids[0])
                print(f"  [Concept] '{args.concept}': {len(outgoing)} edges, vector dim={len(node.vector) if node.vector is not None else 0}")
                for tgt_node, e in outgoing[:10]:
                    tgt_label = engine.graph.get_node(tgt_node).label if engine.graph.get_node(tgt_node) else "?"
                    print(f"    -> {tgt_label} [{e.relation_type}] w={e.weight:.3f}")
            else:
                print(f"  [Concept] '{args.concept}' found but no node data")
        else:
            print(f"  [Concept] '{args.concept}' not found in graph")
        return

    if args.chat is not None:
        queries = [q.strip() for q in args.chat.split("|") if q.strip()]
        if not queries:
            return
        results = []
        for i, q in enumerate(queries):
            t0 = time.time()
            try:
                resp = engine.process_turn(q)
            except Exception as e:
                resp = f"[error: {e}]"
            elapsed = time.time() - t0
            strategy = engine._last_strategy if args.strategy else ""
            strat_tag = f" [{strategy}]" if strategy else ""
            results.append((q, resp, elapsed))
            print(f"Q{i+1}: {q}")
            print(f"A{i+1}: {resp}{strat_tag}")
            if elapsed > 0.5:
                print(f"     [...{elapsed:.1f}s]")
            if args.trace:
                engine.print_traces(f"Q{i+1}")
            print()
        # Stop background learning first so any web-learned definitions are
        # flushed into memory before we persist state (otherwise they're lost).
        engine.stop_background_learning()
        result = engine.save()
        print(f"  [{result}]")
        print(f"  [Stats] Turns: {engine.turn_count}, Words: {len(engine.graph.nodes)}, Sleeps: {engine.sleep_cycles_completed}")
        if args.trace_monitors:
            _rep = engine.monitor_report()
            print("  [Monitors] self-monitor log summary:")
            print(f"    total_fires: {_rep['total_fires']}")
            print(f"    by_monitor: {_rep['by_monitor']}")
            print(f"    by_reason: {_rep['by_reason']}")
            for _e in _rep['recent']:
                print(f"    - {_e['monitor']} | {_e['reason']} | {_e['dropped_clause'][:80]}")
        return

    # ── INTERACTIVE MODE ──
    print()
    print("  ============================================")
    print("   RAVANA - teenage mind, learning from the web...")
    print("  ============================================")
    print()

    if engine.turn_count == 0:
        print()
        print("  Hey! I'm a teenage mind — I know some things but")
        print("  I'm always curious to learn more. I can think about")
        print("  causes, patterns, and different perspectives.")
        print("  Talk to me about anything!")
    else:
        print(f"  Welcome back! I now know {len(engine.graph.nodes)} words across {len(engine.graph.edges)} connections.")
        print(f"  I've slept {engine.sleep_cycles_completed} times to consolidate my learning.")
    print()

    try:
        while True:
            try:
                user_input = input("  You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_input:
                continue

            # Detect quit naturally
            if user_input.lower() in ("bye", "goodbye", "see you", "good night"):
                print(f"\n  RAVANA: Bye bye! I'll remember what you taught me!")
                return

            try:
                t0 = time.time()
                response = engine.process_turn(user_input)
                elapsed = time.time() - t0
                print(f"\n  RAVANA: {response}")

                # Show learning stats every 5 turns
                if engine._learning_count > 0 and engine.turn_count % 5 == 0:
                    print(f"  [I've learned {engine._learning_count} times from the web and know "
                          f"{len(engine.graph.nodes)} words now!]")

                if elapsed > 0.5:
                    print(f"  [...took a moment to think...]")

            except Exception as e:
                print(f"\n  RAVANA: Hmm, I got confused. Let me try again!")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()
    finally:
        # Stop background learning before saving
        engine.stop_background_learning()
        # Auto-save on any exit
        result = engine.save()
        print(f"  [{result}]")


if __name__ == "__main__":
    main()