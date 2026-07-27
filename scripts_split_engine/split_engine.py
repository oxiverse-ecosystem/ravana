#!/usr/bin/env python3
"""Robust AST/line-based splitter for engine.py (CognitiveChatEngine, 12810 lines).

Strategy
--------
Keep in engine.py:
  * the import block (lines 1..181) + module-level constants
  * ALL class-level attribute assignments (constants such as _EDGE_CONNECTORS,
    QUESTION_WORDS, the ~30 _CAPS / _UNDERSCORE tables)
  * a curated set of "core" methods that are tightly coupled to __init__ /
    process_turn / persistence / monitoring:
        __init__, process_turn, save, load, _load,
        _build_monitor_report, monitor_report,
        start_background_learning, stop_background_learning, persist_casing,
        _adapt_verbosity_for_user, _checksum_state (staticmethod, used by save)
  * the decorated methods stay where they are appropriate (they are re-created
    in the mixin that owns them; see below)

Move the remaining 156 plain methods into 8 functional mixin modules, grouped by
topic. Each mixin module gets its OWN copy of the import block + module-level
constants + the shared class-level attribute block, so name resolution inside
every method is byte-for-byte identical to the original (no behavioral change).

The final engine.py:
    class CognitiveChatEngine(WebLearningMixin,
                             GraphMixin, MemoryMixin, ReasoningMixin,
                             WebSearchMixin, GenerationMixin, SelfQueryMixin,
                             PersistenceMixin, MonitorMixin):
with the attribute block + core methods inline, and a single line per mixin:
        from .<module> import <Mixin>

Module-level constants are written once per module (copy) so the mixins are
self-contained and importable in any order.
"""
import ast
import os
import io

SRC = r"C:/Users/Likhith/Documents/projects/ravana/ravana/src/ravana/chat/engine.py"
OUT_DIR = os.path.dirname(SRC)
PROJ = SRC

# ---------------------------------------------------------------------------
# 1. Read source, parse
# ---------------------------------------------------------------------------
with open(SRC, encoding="utf-8") as f:
    src = f.read()
src_lines = src.splitlines()
tree = ast.parse(src)

cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "CognitiveChatEngine")

# Import block = everything before the class definition (lines 1..start-1)
import_block = src_lines[: cls.lineno - 1]  # 0-indexed slice; class at lineno

# Module-level constants (top-of-file, before class). The engine.py module
# defines: _proj_root (assign), _UNIVERSAL_PURGE (assign), _DEFINITION_ASSERTION
# (assign). These live at module scope (NOT inside the class). We must replicate
# them in every mixin module because methods reference _proj_root /
# _UNIVERSAL_PURGE / _DEFINITION_ASSERTION.
module_const_lines = []
for n in tree.body:
    if isinstance(n, (ast.Assign, ast.AnnAssign)):
        # module-scope assignment (not inside class)
        module_const_lines.append((n.lineno, n.end_lineno))
# also the imports of constants from .constants: those are already in import_block.
# Build the module-const text block (between last import-ish and the class).
# Simpler: everything from line class.lineno-1 downward is import block; the
# module constants appear AFTER the import block but BEFORE the class. We detect
# them by scanning lines between end of import_block and the class line for
# assignment lines that aren't inside the class (they aren't — class starts at
# cls.lineno). So module_const_lines above already captures them.

def lines_for(node):
    return src_lines[node.lineno - 1: node.end_lineno]

# ---------------------------------------------------------------------------
# 2. Identify class-level attribute assignments (must stay in engine.py class)
# ---------------------------------------------------------------------------
attr_nodes = [n for n in cls.body if isinstance(n, (ast.Assign, ast.AnnAssign))]

# ---------------------------------------------------------------------------
# 3. Partition methods
# ---------------------------------------------------------------------------
CORE_METHODS = {
    "__init__", "process_turn", "save", "load", "_load",
    "start_background_learning", "stop_background_learning", "persist_casing",
    "_adapt_verbosity_for_user",
    # _build_monitor_report / monitor_report -> MonitorMixin
    # _checksum_state / _safe_pickle_dump        -> PersistenceMixin
}
# Group -> list of method names
GROUPS = {
    "graph": [
        "_get_curiosity_scores", "_init_glove", "_download_glove",
        "_revector_existing_nodes", "_glove_vector", "_build_combined_encoder",
        "_build_lancaster_norms", "_lancaster_vector", "_sensorimotor_confidence",
        "_typed_edges_between", "_get_modulated_vector",
        "_load_conceptnet_ontology", "_ensure_relation",
        "_typed_edges_bootstrap", "_is_category_error", "_category_error_response",
        "_property_bearers", "_nearest_to", "_category_label_of",
        "_derive_definition_purge", "_domain_of", "_norm_word", "_tok_match",
        "_clean_snippet", "_strip_title_echo", "_result_url", "_source_type_label",
        "_clean_subject_phrase",
    ],
    "reasoning": [
        "_is_philosophical_paradox", "_snippet_topic_max_coherence", "_paradox_topic",
        "_reflect_on_paradox", "_user_input_is_gibberism", "_try_reasoning",
        "_try_temporal", "_try_arithmetic", "_try_causal_chain", "_try_hippocampal_retrieval",
        "_is_self_disclosure_stmt", "_process_self_disclosure_stmt", "_ensure_self_model",
        "_affect_is_relevant", "_is_yesno_factual_query", "_is_conditional_query",
        "_is_clause_complete", "_is_preamble_fragment", "_is_answerable_query",
        "_preamble_hold_response", "_clean_scenario_subject", "_rewrite_query_for_web",
        "_web_query_variants", "_norm_word_ref",  # placeholder
    ],
    "memory": [
        "_record_episode", "_mine_episodic_facts", "_retrieve_episodic",
        "_reconstruct_gist", "_episodic_remember", "_recall_past",
        "_recall_hippocampal", "_hippocampal_index_topic", "_detect_recall_trigger",
        "_store_episodic", "_try_memory_query", "_reasoning_loop",
        "_extract_learning_query", "_predict_user_next", "_common_ground_score",
        "_activate_from_input", "_get_user_model", "_update_user_model",
        "_update_emotion", "_decay_episodic_edges",
    ],
    "web_search": [
        "_web_snippet_search", "_web_direct_answer", "_best_answer_snippet",
        "_snippet_is_structural_junk", "_is_preferred_source", "_domain_trust",
        "_record_source_outcome", "_source_trust_threshold", "_snippet_quality",
        "_snippet_plausibility", "_belief_coherence", "_answer_prediction_error",
        "_polarity_mismatch", "_answer_type_mismatch", "_subject_head",
        "_topic_coverage_pe", "_conditional_has_graph_anchor", "_refine_query_variants",
        "_is_function_word", "_ensure_intent_router", "_route_intent", "_router_says",
        "_focus_attribute_answer", "_belief_value_overlap",
    ],
    "generation": [
        "_generate_with_decoder_and_syntax", "_detect_brain_state", "_activate_schema",
        "_build_context_vector", "_build_sentence_vector",
        "_build_context_vector_from_input", "_ensure_orthogonal",
        "_metacognitive_review", "_update_cerebellar_ngram", "_assess_response_quality",
        "_final_emit_guard", "_generate_acknowledgment", "_metaphor_lead",
        "_metaphor_for_category_error", "_top_sensorimotor_dim", "_sleep_consolidate",
        "hrr_query_chain", "_update_state", "_is_follow_up",
        "_compute_phrase_embedding", "_ground_query", "_theme_role", "_strip_eli5_tail",
        "_detect_brain_state", "_activate_schema",
    ],
    "self_query": [
        "_agent_favorite_pick", "_agent_likes_guess", "_agent_stance_on",
        "_route_self_query", "_consult_internal_knowledge",
        "_handle_classic_counterfactual", "_counterfactual_web_escape",
        "_hedged_candidate_for",
    ],
    "persistence": [
        "_checksum_state", "_safe_pickle_dump", "_consolidate_corrections_in_sleep",
        "_process_correction_feedback", "_weaken_edges_for_response",
        "_detect_and_handle_correction", "_hrr_encode_hook",
    ],
    "monitor": [
        "_build_monitor_report", "monitor_report",
    ],
}

# Build name -> method node map
method_nodes = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}

# Sanity: every method not in CORE_METHODS must be assigned to exactly one group.
all_methods = set(method_nodes.keys())
grouped = set()
for g, names in GROUPS.items():
    for nm in names:
        if nm in method_nodes:
            grouped.add(nm)
        # else it's a stale placeholder; ignore

remaining = all_methods - CORE_METHODS - grouped - {"_build_monitor_report", "monitor_report"}
# _build_monitor_report & monitor_report are in CORE already; remove from remaining calc
remaining = all_methods - CORE_METHODS - grouped

# Route `remaining` (unlisted) into a sensible group by keyword heuristics.
# (This is a safety net; we expect remaining to be empty.)
KEYWORD_ROUTE = {
    "paradox": "reasoning", "causal": "reasoning", "arithmetic": "reasoning",
    "temporal": "reasoning", "memory": "memory", "episodic": "memory",
    "recall": "memory", "web": "web_search", "snippet": "web_search",
    "source": "web_search", "generate": "generation", "decoder": "generation",
    "context_vector": "generation", "agent": "self_query", "self": "self_query",
    "consult": "self_query", "counterfactual": "self_query", "sleep": "generation",
    "hrr": "generation", "ground": "generation", "theme": "generation",
    "checksum": "persistence", "pickle": "persistence", "correct": "persistence",
    "consolidate": "persistence", "monitor": "monitor", "adapt": "core",
    "curiosity": "graph", "glove": "graph", "category": "graph",
    "definition": "graph", "domain": "graph", "norm": "graph", "clean": "graph",
}

for nm in sorted(remaining):
    low = nm.lower()
    placed = False
    for kw, g in KEYWORD_ROUTE.items():
        if kw in low:
            if g == "core":
                CORE_METHODS.add(nm)
            else:
                GROUPS[g].append(nm)
            placed = True
            break
    if not placed:
        GROUPS["generation"].append(nm)  # last-resort bucket

# strip stale placeholders that were never real methods
for g in list(GROUPS.keys()):
    GROUPS[g] = [n for n in GROUPS[g] if n in method_nodes]

# SAFETY GUARD: any method whose body references the class by the literal name
# "CognitiveChatEngine." (e.g. the staticmethods _norm_word / _tok_match hit
# CognitiveChatEngine._IRREGULAR_VERBS) cannot live in a mixin module -- that
# name is only bound in engine.py's namespace. Force such methods to stay core.
for nm, node in method_nodes.items():
    src_method = "\n".join(lines_for(node))
    if "CognitiveChatEngine." in src_method and nm not in CORE_METHODS:
        CORE_METHODS.add(nm)
        for g in GROUPS:
            GROUPS[g] = [x for x in GROUPS[g] if x != nm]
        print(f"[guard] forced {nm} -> engine.py core (references CognitiveChatEngine.)")

# ---------------------------------------------------------------------------
# 4. Render a mixin module
# ---------------------------------------------------------------------------
MODULE_DOC = {
    "graph": "Graph & concept-vector mixin — GloVe loading, edge bootstrap, category "
              "error detection, definition purge, conceptnet ontology.",
    "reasoning": "Reasoning & query-classification mixin — paradox, causal/temporal/"
                 "arithmetic reasoning, query type detection, scenario cleaning.",
    "memory": "Episodic & working-memory mixin — recall, retrieval, activation, "
              "user-model updates, forward simulation.",
    "web_search": "Web-snippet retrieval & source-quality mixin — snippet scoring, "
                  "structural-junk detection, source trust, intent routing.",
    "generation": "Response generation mixin — neural decoder + syntax realization, "
                  "context vectors, metaphors, sleep consolidation, HRR chains.",
    "self_query": "Self-model & agent-stance mixin — favourite/pick, agent stance, "
                  "self-query routing, counterfactuals.",
    "persistence": "Persistence & correction mixin — checksum, pickle safety, "
                   "correction detection/consolidation.",
    "monitor": "Monitor / observability mixin — self-monitor report.",
}

def render_module(mod_name, mixin_cls, method_names):
    out = []
    out.append(f'"""Auto-generated mixin module for CognitiveChatEngine.')
    out.append(f'{MODULE_DOC[mod_name]}')
    out.append(f'"""')
    out.append("from __future__ import annotations")
    out.append("")
    # import block (verbatim)
    out.extend(import_block)
    out.append("")
    # module-level constants are already contained inside import_block (they
    # live at module scope above the class), so no separate copy is needed.
    out.append("")
    out.append(f"class {mixin_cls}:")
    out.append(f'    """{MODULE_DOC[mod_name]}"""')
    out.append("")
    for nm in method_names:
        node = method_nodes[nm]
        body = lines_for(node)
        # indent each line to class-body level (4 spaces)
        for bl in body:
            out.append(bl)
        out.append("")
    return "\n".join(out) + "\n"

# ---------------------------------------------------------------------------
# 5. Render each mixin
# ---------------------------------------------------------------------------
MODULE_MAP = {
    "graph": "GraphMixin",
    "reasoning": "ReasoningMixin",
    "memory": "MemoryMixin",
    "web_search": "WebSearchMixin",
    "generation": "GenerationMixin",
    "self_query": "SelfQueryMixin",
    "persistence": "PersistenceMixin",
    "monitor": "MonitorMixin",
}

manifests = {}
for mod, mixin in MODULE_MAP.items():
    text = render_module(mod, mixin, GROUPS[mod])
    out_path = os.path.join(OUT_DIR, f"engine_{mod}.py")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    manifests[mod] = (out_path, len(GROUPS[mod]), len(text.splitlines()))
    print(f"wrote {out_path}  methods={len(GROUPS[mod])} lines={len(text.splitlines())}")

# ---------------------------------------------------------------------------
# 6. Render the new engine.py
# ---------------------------------------------------------------------------
new_engine = []
new_engine.append('"""')
new_engine.append("RAVANA Cognitive Chat Engine -- main orchestrator.")
new_engine.append("")
new_engine.append("This module defines CognitiveChatEngine, which is composed of one")
new_engine.append("core class body (initialization, process_turn, persistence,")
new_engine.append("monitoring) plus 8 functional mixins that hold the domain methods:")
new_engine.append("  - engine_graph.py        (GraphMixin)")
new_engine.append("  - engine_reasoning.py   (ReasoningMixin)")
new_engine.append("  - engine_memory.py      (MemoryMixin)")
new_engine.append("  - engine_web_search.py  (WebSearchMixin)")
new_engine.append("  - engine_generation.py  (GenerationMixin)")
new_engine.append("  - engine_self_query.py  (SelfQueryMixin)")
new_engine.append("  - engine_persistence.py (PersistenceMixin)")
new_engine.append("  - engine_monitor.py     (MonitorMixin)")
new_engine.append("  - web_learning.py       (WebLearningMixin, original)")
new_engine.append('"""')
new_engine.append("from __future__ import annotations")
new_engine.append("")
new_engine.extend(import_block)
new_engine.append("")
# (module-level constants already included in import_block above)
new_engine.append("")
# Mixin imports
for mod, mixin in MODULE_MAP.items():
    new_engine.append(f"from .{ 'engine_' + mod } import {mixin}")
new_engine.append("")
# Class def with full inheritance
bases = ["WebLearningMixin"] + [MODULE_MAP[m] for m in MODULE_MAP]
heading = "class CognitiveChatEngine(" + ", ".join(bases) + "):"
new_engine.append(heading)
new_engine.append('    """RAVANA cognitive chat engine -- starts as a baby, learns from the web.')
new_engine.append("")
new_engine.append("    Composed of the mixins imported above; the methods defined inline")
new_engine.append("    here are the core orchestration paths (init, process_turn,")
new_engine.append("    persistence, monitoring). See each engine_*.py for the domain")
new_engine.append('    methods."""')
new_engine.append("")
# class-level attributes
for n in attr_nodes:
    for bl in lines_for(n):
        new_engine.append(bl)
new_engine.append("")
# core methods
for nm in sorted(CORE_METHODS, key=lambda x: method_nodes[x].lineno):
    for bl in lines_for(method_nodes[nm]):
        new_engine.append(bl)
new_engine.append("")

new_engine_text = "\n".join(new_engine) + "\n"
with open(SRC, "w", encoding="utf-8") as f:
    f.write(new_engine_text)

print()
print(f"engine.py rewritten: {len(new_engine_text.splitlines())} lines")
print(f"total methods across mixins: {sum(len(GROUPS[m]) for m in GROUPS)}")
print(f"core methods kept: {len(CORE_METHODS)}")
print("VERIFY: import + count + smoke test next.")
