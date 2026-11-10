"""
Chat Interface - RAVANA's public chat entry point.

A thin facade over :class:`ravana.chat.engine.CognitiveChatEngine`, which is
the canonical generation stack (see ``docs/GENERATION_STACK_DECISION.md`` and
``docs/WRAPPER_CONTRACT_DECISION.md``).

What this module keeps:
  * ``ChatConfig`` - the public configuration dataclass.
  * ``CognitiveResponseContext`` - the shared context type.
  * ``ChatInterface`` - the public facade the CLI and tests construct.
  * ``main()`` - the argument-parsing CLI.

Everything cognitive is delegated to the engine. The only local logic is the
attribute bridges the CLI and the integration tests reach for by the old
monolith's names.
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import CognitiveResponseContext  # noqa: F401  (re-exported)
from .engine import CognitiveChatEngine


@dataclass
class ChatConfig:
    """Configuration for the chat facade."""
    dim: int = 64
    seed: int = 42
    baby_mode: bool = True
    data_dir: Optional[str] = None
    user_suffix: str = ""
    use_vad: bool = True
    use_rlm: bool = True
    use_beliefs: bool = True
    trace_enabled: bool = False
    reasoning_mode: str = "stochastic"


class _GraphEngineShim:
    """Present the engine's ``ConceptGraph`` under the old ``GraphEngine`` name.

    The old monolith exposed a ``GraphEngine`` wrapper as ``graph_engine``; the
    engine exposes the raw ``ConceptGraph`` as ``graph``. The CLI's
    ``--stats`` / ``--concept`` / ``--export-graph`` / ``--import-graph``
    paths read ``graph_engine.graph`` (then ``.nodes`` / ``.edges`` /
    ``.get_node``), ``.glove_ready``, ``._glove_vector`` and
    ``._concept_keywords``.
    """

    def __init__(self, graph, engine):
        self.graph = graph
        self._engine = engine

    @property
    def glove_ready(self) -> bool:
        return bool(getattr(self._engine, "glove_ready", False))

    def _glove_vector(self, word: str):
        return self._engine._glove_vector(word)

    @property
    def _concept_keywords(self) -> Dict[str, List[int]]:
        return getattr(self._engine, "_concept_keywords", {}) or {}

    @property
    def _all_labels(self) -> Dict[str, int]:
        """label -> first node id (the shape WebToGraph expects)."""
        kw = self._concept_keywords
        return {lab: nids[0] for lab, nids in kw.items() if nids}

    @property
    def _promote_min_sources(self) -> int:
        return int(getattr(self._engine, "_promote_min_sources", 2) or 2)

    @_promote_min_sources.setter
    def _promote_min_sources(self, value: int) -> None:
        self._engine._promote_min_sources = int(value)


class _WebLearnerShim:
    """Bridge the CLI's ``web_learner`` name onto the engine's own state.

    ``--no-curiosity`` sets ``engine.web_learner._curiosity_drive_enabled =
    False``; the engine holds that flag directly as
    ``_curiosity_drive_enabled``.

    The old monolith's ``WebLearner`` exposed ``get_web_to_graph()``;
    ``CognitiveChatEngine`` spells it ``_get_web_to_graph()`` (the same
    ``WebToGraph`` object, built over the engine's own graph). The C-lite/E
    helpers (``curiosity_e_step``, ``knowledge_gap``, ``last_fact_count``) are
    already on the engine via ``WebLearningMixin``. Only the spelling differs,
    so the public name is aliased rather than re-implemented.
    """

    def __init__(self, engine):
        self._engine = engine

    def get_web_to_graph(self):
        """Alias for the engine's ``_get_web_to_graph`` (the old WebLearner name)."""
        return self._engine._get_web_to_graph()

    def last_fact_count(self) -> int:
        """Number of C-lite facts written this session.

        The old ``WebLearner.last_fact_count`` delegated to
        ``WebToGraph.fact_count``; the engine exposes the same object through
        ``_get_web_to_graph`` but never the accessor, so bridge it here.
        """
        w2g = self._engine._get_web_to_graph()
        return int(w2g.fact_count()) if w2g is not None else 0

    @property
    def _curiosity_drive_enabled(self) -> bool:
        return bool(getattr(self._engine, "_curiosity_drive_enabled", True))

    @_curiosity_drive_enabled.setter
    def _curiosity_drive_enabled(self, value: bool) -> None:
        self._engine._curiosity_drive_enabled = bool(value)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._engine, name)


class _DecoderEngineShim:
    """Bridge ``decoder_engine`` onto the engine's own ``NeuralDecoder``.

    The old monolith owned a ``DecoderEngine`` wrapper (``build_vocab``,
    ``generate``, ``training_count``, ``web_training_count``). The engine owns
    the ``NeuralDecoder`` directly (``engine.py`` ``self.neural_decoder``,
    built by ``response_gen._build_decoder_vocab``). The integration test only
    asserts the attribute is present and non-None; exposing the real decoder
    satisfies it without fabricating a dead object.
    """

    def __init__(self, engine):
        self._engine = engine

    @property
    def neural_decoder(self):
        return getattr(self._engine, "neural_decoder", None)

    @property
    def training_count(self) -> int:
        return int(getattr(self._engine, "_decoder_training_count", 0) or 0)

    @property
    def web_training_count(self) -> int:
        return int(getattr(self._engine, "_decoder_web_training_count", 0) or 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.neural_decoder, name)


class ChatInterface:
    """Public facade over :class:`CognitiveChatEngine`.

    HISTORY. This class used to be a self-contained 2,007-line monolith: it
    built its own graph, user_model, emotion, pfc_workspace,
    syntactic_assembly, cerebellar_ngram and mirror_engine, and carried its
    own copies of ``_generate_response``, ``_reasoning_loop``,
    ``_graph_fallback_response`` and ~27 other methods.

    ``CognitiveChatEngine`` is the canonical stack -- the CLI
    (``scripts/ravana_chat.py:44``), the architecture docs
    (``docs/ARCHITECTURE.md:13,179``) and 294 test files use it. The duplicate
    pipeline was already named a defect
    (``docs/LANGUAGE_PRODUCTION_PIVOT.md`` G13), and the interface lacked every
    modern response path (no ``_human_like_uncertainty``,
    ``_reflective_response``, ``_topic_grounded``, ``_web_direct_answer``,
    ``build_style_context``, dialogue memory or the agentic tool layer).

    What remains here is what the engine does not provide: the ``ChatConfig``
    dataclass, the attribute bridges the CLI/tests reach for, and the CLI.
    """

    def __init__(self, config: Optional[ChatConfig] = None):
        self.config = config or ChatConfig()

        # Path conventions preserved from the previous implementation so an
        # existing weights file is still found.
        self._proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if self.config.data_dir:
            self._save_path = os.path.join(
                self.config.data_dir,
                f"ravana_weights{self.config.user_suffix}.pkl")
            self._glove_cache_path = os.path.join(
                self.config.data_dir, "ravana_glove_cache.npz")
        else:
            os.makedirs(os.path.join(self._proj_root, "weights"), exist_ok=True)
            os.makedirs(os.path.join(self._proj_root, "data"), exist_ok=True)
            self._save_path = os.path.join(
                self._proj_root, "weights",
                f"ravana_weights{self.config.user_suffix}.pkl")
            self._glove_cache_path = os.path.join(
                self._proj_root, "data", "ravana_glove_cache.npz")

        self._engine = CognitiveChatEngine(
            dim=self.config.dim,
            seed=self.config.seed,
            baby_mode=self.config.baby_mode,
            data_dir=self.config.data_dir,
            user_suffix=self.config.user_suffix,
        )

        # ── Save-path contract ─────────────────────────────────────────────
        # The engine derives its OWN ``_save_path`` from ``data_dir`` +
        # ``user_suffix`` (engine.py:1080-1090), which ignores the explicit path
        # this facade was constructed with. The old monolith stored the caller's
        # path verbatim, and the CLI/tests set ``engine._save_path`` after
        # construction -- so without re-applying it here, ``save()`` writes to
        # ``<data_dir>/ravana_weights<suffix>.pkl`` while the caller believes it
        # wrote to ``_save_path``. ``save()`` even reports the file it *did*
        # write, so the mismatch is silent. Re-apply the facade's path AFTER
        # construction (the engine's own assignment above runs during it).
        self._apply_save_path()

        # Integration toggles. The engine defines all of these with the same
        # defaults; they are copied across so `--no-vad` / `--no-rlm` /
        # `--no-beliefs` / `--trace` keep working.
        for _attr in ("use_vad", "use_rlm", "use_beliefs",
                      "reasoning_mode", "_trace_enabled"):
            if hasattr(self.config, _attr):
                try:
                    setattr(self._engine, _attr, getattr(self.config, _attr))
                except Exception:
                    pass

        # ── Attribute bridges ──────────────────────────────────────────────
        # These are the names the CLI and the integration tests reach for by
        # the old monolith's spelling. Each bridges to the engine's real
        # object rather than fabricating a stand-in.
        self.graph_engine = _GraphEngineShim(self._engine.graph, self._engine)
        self.web_learner = _WebLearnerShim(self._engine)
        self.decoder_engine = _DecoderEngineShim(self._engine)
        # NOTE: `bootstrap_manager`, `relation_predictor` and `propagation`
        # are deliberately NOT bridged. The engine has none of them by design
        # (see docs/WRAPPER_CONTRACT_DECISION.md): the engine owns the
        # NeuralDecoder directly, has no BootstrapManager, and does not import
        # ravana.nn.rlm at all. Adding dead attributes to satisfy an
        # assertion is the anti-pattern the goal doc forbids.

    # ── public surface ────────────────────────────────────────────────────
    @property
    def _save_path(self) -> str:
        """The facade's save path; assignment propagates to the engine.

        The old monolith exposed a plain attribute that callers (the CLI, the
        integration tests) overwrote directly: ``engine._save_path = p``. The
        engine computes its own path during ``__init__``, so a plain attribute
        here would be shadowed by ``__getattr__`` delegation and the assignment
        would land on the engine while ``save()`` read the engine's own value.
        A property makes both directions agree.
        """
        return self.__dict__.get("_save_path_value", "") or \
            getattr(self._engine, "_save_path", "")

    @_save_path.setter
    def _save_path(self, value: str) -> None:
        self.__dict__["_save_path_value"] = value
        # During ``__init__`` the engine does not exist yet; ``_apply_save_path``
        # is called explicitly once construction finishes.
        if "_engine" in self.__dict__:
            self._apply_save_path()

    def _apply_save_path(self) -> None:
        """Push the facade's save path onto the engine and its SQLite mirror."""
        path = self.__dict__.get("_save_path_value")
        if not path:
            return
        try:
            self._engine._save_path = path
            # The engine derives its DB path from ``_save_path`` at boot; keep
            # the mirror co-located so a load finds its ACID graph.
            try:
                self._engine._db_path = path.replace(".pkl", ".db")
            except Exception:
                pass
        except Exception:
            pass

    @property
    def rng(self):
        return self._engine.rng

    def process_turn(self, user_input: str) -> str:
        return self._engine.process_turn(user_input)

    def save(self) -> str:
        return self._engine.save()

    def _load(self) -> bool:
        return self._engine.load()

    def load(self) -> bool:
        return self._engine.load()

    def start_background_learning(self):
        return self._engine.start_background_learning()

    def stop_background_learning(self):
        return self._engine.stop_background_learning()

    def __getattr__(self, name):
        # Delegate anything the wrapper does not define explicitly, so
        # callers that reached into the old monolith keep working.
        return getattr(self._engine, name)


def main():
    parser = argparse.ArgumentParser(description="RAVANA - teenage mind on the web")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reset", action="store_true", help="Delete saved weights and start fresh")
    parser.add_argument("--chat", type=str, default=None, help='Send queries in batch mode')
    parser.add_argument("--strategy", action="store_true", help="Include strategy name in --chat output")
    parser.add_argument("--trace", action="store_true", help="Print edge-level chain traces")
    parser.add_argument("--no-vad", action="store_true", help="Disable VAD emotion modulation")
    parser.add_argument("--no-rlm", action="store_true", help="Disable RLMv2 triple verification")
    parser.add_argument("--no-beliefs", action="store_true", help="Disable belief store")
    parser.add_argument("--no-curiosity", action="store_true", help="Disable autonomous curiosity-driven learning")
    parser.add_argument("--mode", type=str, default="stochastic", choices=["stochastic", "deterministic", "exploratory"])
    parser.add_argument("--debug", action="store_true", help="Print debug tracebacks for exceptions")
    parser.add_argument("--user", type=str, default=None, help="User name for multi-user isolation")
    parser.add_argument("--data-dir", type=str, default=None, help="Custom data directory")
    parser.add_argument("--export-graph", type=str, default=None, help="Export graph to JSON")
    parser.add_argument("--import-graph", type=str, default=None, help="Import graph from JSON")
    parser.add_argument("--stats", action="store_true", help="Print graph statistics")
    parser.add_argument("--concept", type=str, default=None, help="Show what RAVANA knows about a concept")
    args = parser.parse_args()

    reset_suffix = args.user or ""
    save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             f"ravana_weights{reset_suffix}.pkl")
    if args.reset:
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"  [Reset] Deleted {os.path.basename(save_path)}, starting fresh!")
        else:
            print("  [Reset] No saved weights found, starting fresh!")

    config = ChatConfig(
        dim=args.dim, seed=args.seed, baby_mode=True,
        data_dir=args.data_dir, user_suffix=reset_suffix,
        trace_enabled=args.trace
    )
    engine = ChatInterface(config)

    if args.no_vad:
        engine.config.use_vad = False
        print("  [Config] VAD modulation disabled")
    if args.no_rlm:
        engine.config.use_rlm = False
        print("  [Config] RLMv2 triple verification disabled")
    if args.no_beliefs:
        engine.config.use_beliefs = False
        print("  [Config] Belief store disabled")
    if args.no_curiosity:
        engine.web_learner._curiosity_drive_enabled = False
        print("  [Curiosity] Autonomous learning disabled")

    if args.mode != "stochastic":
        engine.reasoning_mode = args.mode
        print(f"  [Mode] Reasoning mode set to '{args.mode}'")

    if args.export_graph:
        import json
        g = engine.graph_engine.graph
        data = {"nodes": [{"id": n.id, "label": n.label} for n in list(g.nodes.values())],
                "edges": [{"source": src, "target": tgt, "relation": e.relation_type, "weight": e.weight}
                          for (src, tgt), e in list(g.edges.items())]}
        with open(args.export_graph, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  [Export] Exported {len(data['nodes'])} nodes + {len(data['edges'])} edges to {args.export_graph}")
        engine.save()
        return

    if args.import_graph:
        import json
        with open(args.import_graph, "r") as f:
            data = json.load(f)
        g = engine.graph_engine.graph
        for node_data in data.get("nodes", []):
            label = node_data.get("label", "")
            if label:
                vec = engine.graph_engine._glove_vector(label) if engine.graph_engine.glove_ready else None
                g.add_node(vector=vec, label=label)
        for edge_data in data.get("edges", []):
            src = edge_data.get("source")
            tgt = edge_data.get("target")
            rel = edge_data.get("relation", "related")
            w = edge_data.get("weight", 0.5)
            if src is not None and tgt is not None:
                g.add_edge(src, tgt, relation_type=rel, weight=w)
        engine.save()
        return

    if args.stats:
        print(f"  [Stats] Graph has {len(engine.graph_engine.graph.nodes)} nodes and {len(engine.graph_engine.graph.edges)} edges")
        print(f"  [Stats] Turn count: {engine.turn_count}")
        return

    if args.concept:
        nids = engine.graph_engine._concept_keywords.get(args.concept.lower(), [])
        if nids:
            node = engine.graph_engine.graph.get_node(nids[0])
            if node:
                outgoing = engine.graph_engine.graph.get_outgoing(nids[0])
                print(f"  [Concept] '{args.concept}': {len(outgoing)} edges, vector dim={len(node.vector) if node.vector is not None else 0}")
                for tgt_node, e in outgoing[:10]:
                    tgt_label = engine.graph_engine.graph.get_node(tgt_node).label if engine.graph_engine.graph.get_node(tgt_node) else "?"
                    print(f"    -> {tgt_label} [{e.relation_type}] w={e.weight:.3f}")
            else:
                print(f"  [Concept] '{args.concept}' found but no node data")
        else:
            print(f"  [Concept] '{args.concept}' not found in graph")
        return

    if args.chat is not None:
        queries = [q.strip() for q in args.chat.split("|") if q.strip()]
        for i, q in enumerate(queries):
            t0 = time.time()
            try:
                resp = engine.process_turn(q)
            except Exception as e:
                resp = f"[error: {e}]"
            elapsed = time.time() - t0
            strategy = engine._last_strategy if args.strategy else ""
            print(f"Q{i+1}: {q}")
            print(f"A{i+1}: {resp}{' [' + strategy + ']' if strategy else ''}")
            if elapsed > 0.5:
                print(f"     [...{elapsed:.1f}s]")
            if args.trace:
                print(f"  [trace] Q{i+1}")
            print()
        print(f"  [{engine.save()}]")
        print(f"  [Stats] Turns: {engine.turn_count}, Words: {len(engine.graph_engine.graph.nodes)}, "
              f"Sleeps: {engine.sleep_engine.metrics.get('total_sleep_cycles', 0)}")
        return

    # Interactive mode
    print()
    print("  ============================================")
    print("   RAVANA - teenage mind, learning from the web...")
    print("  ============================================")
    print()
    if engine.turn_count == 0:
        print("  Hey! I'm a teenage mind — I know some things but")
        print("  I'm always curious to learn more. I can think about")
        print("  causes, patterns, and different perspectives.")
        print("  Talk to me about anything!")
    else:
        print(f"  Welcome back! I now know {len(engine.graph_engine.graph.nodes)} words across {len(engine.graph_engine.graph.edges)} connections.")
        print(f"  I've slept {engine.sleep_engine.metrics.get('total_sleep_cycles', 0)} times to consolidate my learning.")
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
            if user_input.lower() in ("bye", "goodbye", "see you", "good night"):
                print("\n  RAVANA: Bye bye! I'll remember what you taught me!")
                break
            try:
                t0 = time.time()
                response = engine.process_turn(user_input)
                elapsed = time.time() - t0
                print(f"\n  RAVANA: {response}")
                if engine._learning_count > 0 and engine.turn_count % 5 == 0:
                    print(f"  [I've learned {engine._learning_count} times from the web and know {len(engine.graph_engine.graph.nodes)} words now!]")
                if elapsed > 0.5:
                    print("  [...took a moment to think...]")
            except Exception:
                print("\n  RAVANA: Hmm, I got confused. Let me try again!")
                if "--debug" in sys.argv:
                    import traceback
                    traceback.print_exc()
    finally:
        engine.stop_background_learning()
        print(f"  [{engine.save()}]")


if __name__ == "__main__":
    main()
