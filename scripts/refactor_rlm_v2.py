"""
RAVANA rlm_v2.py modularization script.

Reads the monolithic rlm_v2.py, identifies all methods, and extracts them into
separate submodule files. The main rlm_v2.py is rewritten as a thin orchestrator
that inherits from mixin classes.

Created submodules:
  rlm_v2_common.py  — imports, constants, _build_glove_embedding_matrix
  rlm_v2_graph.py   — GraphMixin: concept graph operations, prototypes, adapters
  rlm_v2_encoder.py — EncoderMixin: autoencoder, embedding initialization
  rlm_v2_rp.py      — RPMixin: relation predictor forward/backward
  rlm_v2_verb.py    — VerbMixin: verb offset predictor
  rlm_v2_sleep.py   — SleepMixin: sleep consolidation, episodic memory, alignment

Usage: python scripts/refactor_rlm_v2.py
"""

import os
import re
import sys
from typing import Dict, List, Tuple


# ── Method-to-module mapping: method_name -> (module_name, mixin_class) ──
# Methods not listed here stay in rlm_v2.py
METHOD_MAP: Dict[str, Tuple[str, str]] = {
    # ── GraphMixin ──
    "_init_structured_concepts": ("rlm_v2_graph", "GraphMixin"),
    "_project_to_concept": ("rlm_v2_graph", "GraphMixin"),
    "_project_to_embed": ("rlm_v2_graph", "GraphMixin"),
    "_cached_norm": ("rlm_v2_graph", "GraphMixin"),
    "_get_node_matrix": ("rlm_v2_graph", "GraphMixin"),
    "_invalidate_caches": ("rlm_v2_graph", "GraphMixin"),
    "decompose_triple": ("rlm_v2_graph", "GraphMixin"),
    "classify_relation": ("rlm_v2_graph", "GraphMixin"),
    "_classify_relation_learned": ("rlm_v2_graph", "GraphMixin"),
    "_decode_token": ("rlm_v2_graph", "GraphMixin"),
    "_nearest_concept": ("rlm_v2_graph", "GraphMixin"),
    "_get_or_create_concept": ("rlm_v2_graph", "GraphMixin"),
    "_init_entity_adapter": ("rlm_v2_graph", "GraphMixin"),
    "_get_or_adapt_entity_adapter": ("rlm_v2_graph", "GraphMixin"),
    "_adapt_entity_adapter_at_test_time": ("rlm_v2_graph", "GraphMixin"),
    "get_robust_embedding": ("rlm_v2_graph", "GraphMixin"),
    "_get_anchor_regularized_latent": ("rlm_v2_graph", "GraphMixin"),
    "get_query_confidence": ("rlm_v2_graph", "GraphMixin"),
    "_find_nearest_prototype": ("rlm_v2_graph", "GraphMixin"),
    "_register_prototype": ("rlm_v2_graph", "GraphMixin"),
    "_inherit_from_prototype": ("rlm_v2_graph", "GraphMixin"),
    "_init_default_prototypes": ("rlm_v2_graph", "GraphMixin"),
    "traverse": ("rlm_v2_graph", "GraphMixin"),
    "_validate_edge_bindings": ("rlm_v2_graph", "GraphMixin"),
    "_inject_direct_edges_if_needed": ("rlm_v2_graph", "GraphMixin"),
    "_anti_hebbian_prune_polluted_edges": ("rlm_v2_graph", "GraphMixin"),
    "_normalize_outgoing_weights": ("rlm_v2_graph", "GraphMixin"),
    "_prune_weak_edges": ("rlm_v2_graph", "GraphMixin"),
    "_inject_cross_domain_edge": ("rlm_v2_graph", "GraphMixin"),

    # ── EncoderMixin ──
    "_initialize_token_embeddings_from_tokenizer": ("rlm_v2_encoder", "EncoderMixin"),
    "_pretrain_encoder_autoencoder": ("rlm_v2_encoder", "EncoderMixin"),
    "_encoder_forward_full": ("rlm_v2_encoder", "EncoderMixin"),
    "_encoder_backward": ("rlm_v2_encoder", "EncoderMixin"),
    "_compute_contrastive_gradients": ("rlm_v2_encoder", "EncoderMixin"),

    # ── RPMixin ──
    "_rp_forward": ("rlm_v2_rp", "RPMixin"),
    "_rp_backward": ("rlm_v2_rp", "RPMixin"),

    # ── VerbMixin ──
    "_verb_stem": ("rlm_v2_verb", "VerbMixin"),
    "_accumulate_verb_offset": ("rlm_v2_verb", "VerbMixin"),
    "_compute_verb_offsets": ("rlm_v2_verb", "VerbMixin"),
    "_rp_forward_verb_offset": ("rlm_v2_verb", "VerbMixin"),
    "_rp_forward_verb_offset_from_adapted": ("rlm_v2_verb", "VerbMixin"),

    # ── SleepMixin ──
    "_store_episode": ("rlm_v2_sleep", "SleepMixin"),
    "_evict_lowest_salience": ("rlm_v2_sleep", "SleepMixin"),
    "_consolidate_episodic_to_semantic": ("rlm_v2_sleep", "SleepMixin"),
    "_decay_semantic_memories": ("rlm_v2_sleep", "SleepMixin"),
    "_bridge_memories_to_graph": ("rlm_v2_sleep", "SleepMixin"),
    "_regulate_cognitive_state": ("rlm_v2_sleep", "SleepMixin"),
    "buffer_experience": ("rlm_v2_sleep", "SleepMixin"),
    "_replay_old_memories": ("rlm_v2_sleep", "SleepMixin"),
    "sleep_cycle": ("rlm_v2_sleep", "SleepMixin"),
    "end_wake_epoch": ("rlm_v2_sleep", "SleepMixin"),
    "mark_alignment_needed": ("rlm_v2_sleep", "SleepMixin"),
    "_prune_phantom_nodes": ("rlm_v2_sleep", "SleepMixin"),
    "compute_neighbor_recall_at_5": ("rlm_v2_sleep", "SleepMixin"),
    "align_encoder_to_graph": ("rlm_v2_sleep", "SleepMixin"),
}

# Module-level symbols to extract to common
MODULE_SYMBOLS = [
    "_build_glove_embedding_matrix",
    "RELATION_TYPES",
    "_KEYWORD_MAP",
]

# Imports to move to common
IMPORT_LINES = [
    "import numpy as np",
    "import time",
    "import pickle",
    "import os",
    "from typing import Optional, List, Tuple, Dict, Set, Any",
    "from collections import defaultdict",
    "from .module import Module, Linear, Embedding",
    "from ..graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap",
    "from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity",
    "from ..propagation import PropagationEngine",
    "from ..currencies import CognitiveCurrencies",
    "from ..currency import create_rlm_currency",
    "import zipfile",
    "import json",
]

# Import for neural_decoder (used by some methods that stay in rlm_v2.py)
ADDITIONAL_IMPORTS = {
    "rlm_v2_rp": ["from ..embedder import LearnedEmbedder"],
    "rlm_v2_graph": ["from ..embedder import LearnedEmbedder"],
    "rlm_v2_sleep": ["from ..embedder import LearnedEmbedder"],
    "rlm_v2_encoder": ["from ..embedder import LearnedEmbedder"],
}



def find_method_boundaries(lines: List[str], start_line: int) -> Tuple[int, int]:
    """Find the start and end line of a method given its def line index.
    
    Returns (start_line_inclusive, end_line_exclusive).
    The method starts at the decorator (if any) or def line.
    """
    # Walk back to find decorators
    s = start_line
    while s > 0 and (lines[s-1].strip().startswith("@") or lines[s-1].strip() == ""):
        s -= 1
    
    # Walk forward to find the end of the method (next line with same or less indentation)
    if s >= len(lines):
        return s, s
    
    # Get the indentation of the def line
    def_line = lines[start_line]
    base_indent = len(def_line) - len(def_line.lstrip())
    
    # If it's a module-level function (no indent), find end differently
    if base_indent == 0:
        end = start_line + 1
        while end < len(lines):
            line = lines[end]
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and not stripped.startswith("\"\"\"") and not stripped.startswith("'") and not stripped.startswith("@"):
                # Check if this is a new module-level definition
                if (stripped.startswith("def ") or stripped.startswith("class ") 
                    or stripped.startswith("import ") or stripped.startswith("from ")):
                    break
                # Check if this is a module-level assignment
                if "=" in stripped and not stripped.startswith(" "):
                    break
            end += 1
        return s, end
    
    # Class method: find next line with same or less indentation (for non-blank, non-decorator lines)
    end = start_line + 1
    while end < len(lines):
        line = lines[end]
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and not line.startswith("@") and not stripped.startswith("\"\"\"") and not stripped.startswith("'"):
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= base_indent and not stripped.startswith("(") and not stripped.startswith(")") and not stripped.startswith(","):
                break
        end += 1
    
    return s, end


def get_method_name(def_line: str) -> str:
    """Extract method name from a def line."""
    m = re.match(r'\s*def\s+(\w+)', def_line)
    if m:
        return m.group(1)
    return ""


def is_method_extracted(def_line: str) -> bool:
    """Check if a method should be extracted."""
    name = get_method_name(def_line)
    return name in METHOD_MAP


def build_submodule(module_name: str, mixin_class: str, methods: Dict[str, List[str]]) -> str:
    """Build the content of a mixin submodule file."""
    lines = []
    lines.append(f'"""')
    lines.append(f'RAVANA {module_name} — {mixin_class} for RLMv2.')
    lines.append(f'')
    lines.append(f'Auto-generated from rlm_v2.py. Edits should be made in the source file')
    lines.append(f'and the extraction script (scripts/refactor_rlm_v2.py) re-run, OR made directly')
    lines.append(f'in this file and tested. Choose whichever is easier.')
    lines.append(f'"""')
    lines.append(f'')
    
    # Add imports needed by this module
    lines.append(f'import numpy as np')
    lines.append(f'import time')
    if module_name in ["rlm_v2_sleep"]:
        lines.append(f'from typing import Optional, List, Tuple, Dict, Set, Any')
    else:
        lines.append(f'from typing import Optional, List, Tuple, Dict')

    if module_name in ["rlm_v2_verb"]:
        lines.append(f'import numpy as np')
    if module_name in ["rlm_v2_rp"]:
        lines.append(f'import numpy as np')
    if module_name in ["rlm_v2_graph"]:
        lines.append(f'import numpy as np')
        lines.append(f'from typing import Optional, List, Tuple, Dict')
    if module_name in ["rlm_v2_sleep"]:
        lines.append(f'from collections import defaultdict')
    if module_name in ["rlm_v2_encoder", "rlm_v2_graph", "rlm_v2_sleep"]:
        lines.append(f'from ..embedder import LearnedEmbedder')
    if module_name in ["rlm_v2_rp"]:
        lines.append(f'from ..embedder import LearnedEmbedder')
    
    lines.append(f'')
    lines.append(f'')
    lines.append(f'class {mixin_class}:')
    lines.append(f'    """Mixin providing {module_name} methods for RLMv2."""')
    lines.append(f'    pass  # methods added below')
    lines.append(f'')
    
    # Track which methods were added
    added_count = 0
    for method_name, method_lines in sorted(methods.items()):
        for line in method_lines:
            lines.append(line)
        lines.append(f'')
        added_count += 1
    
    # Create the final content with proper indent
    content = '\n'.join(lines)
    
    # Replace "    pass  # methods added below\n\n" with proper method definitions
    # The methods were added after the class with their original indentation (4 spaces for class methods)
    # But since we wrote them after "    pass", we need to make sure they're properly indented
    
    return content


def extract():
    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    source_path = os.path.join(project_root, "ravana_ml", "src", "ravana_ml", "nn", "rlm_v2.py")
    output_dir = os.path.join(project_root, "ravana_ml", "src", "ravana_ml", "nn")
    
    print(f"Reading {source_path}...")
    with open(source_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"File has {len(lines)} lines.")
    
    # ── Phase 1: Parse the file into sections ──
    
    # Module-level section (before class definition)
    class_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("class RLMv2("):
            class_start = i
            break
    
    if class_start < 0:
        print("ERROR: Could not find RLMv2 class definition")
        sys.exit(1)
    
    print(f"  Class definition at line {class_start+1}")
    
    # Find class end (last line that's indented relative to class)
    class_indent = len(lines[class_start]) - len(lines[class_start].lstrip())
    class_end = class_start + 1
    # Find the last line that's part of the class
    last_class_line = class_start
    in_triple_quote = False
    for i in range(class_start + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        
        # Handle triple quotes (docstrings spanning multiple lines)
        if '"""' in line:
            in_triple_quote = not in_triple_quote
        
        if not stripped or in_triple_quote:
            last_class_line = i
            continue
        
        current_indent = len(line) - len(line.lstrip())
        if current_indent < class_indent + 4:
            # We've left the class
            break
        last_class_line = i
    
    print(f"  Class body: lines {class_start+1} to {last_class_line+1}")
    
    # ── Phase 2: Extract module-level items to common ──
    
    # Module-level items: imports + _build_glove_embedding_matrix + RELATION_TYPES + _KEYWORD_MAP
    module_level_end = class_start
    
    # Collect all module-level lines
    module_lines = lines[:module_level_end]
    
    # ── Phase 3: Parse class body to identify each method ──
    
    # Find all method definitions in the class body
    in_docstring = False
    
    # Collect methods organized by (start_line, end_line, method_name, def_line_content)
    extracted_methods: Dict[str, Dict[str, List[Tuple[int, int, str]]]] = {
        "rlm_v2_graph": {},
        "rlm_v2_encoder": {},
        "rlm_v2_rp": {},
        "rlm_v2_verb": {},
        "rlm_v2_sleep": {},
    }
    
    # Track which lines to remove from the main file
    lines_to_remove = set()
    
    i = class_start + 1
    while i < last_class_line + 1:
        line = lines[i]
        stripped = line.strip()
        
        # Handle docstrings (class-level or method-level)
        if '"""' in stripped and stripped.count('"') >= 3:
            if not in_docstring:
                in_docstring = True
            else:
                in_docstring = False
            i += 1
            continue
        
        if in_docstring:
            i += 1
            continue
        
        # Check for method definitions
        if stripped.startswith("def "):
            method_name = get_method_name(stripped)
            if method_name:
                start, end = find_method_boundaries(lines, i)
                
                if method_name in METHOD_MAP:
                    module_name, mixin_class = METHOD_MAP[method_name]
                    # Store the method lines
                    method_lines = lines[start:end]
                    if module_name not in extracted_methods:
                        extracted_methods[module_name] = {}
                    extracted_methods[module_name][method_name] = method_lines
                    
                    # Mark lines for removal
                    for r in range(start, end):
                        lines_to_remove.add(r)
                    
                    print(f"  Extracted: {module_name}.{mixin_class}.{method_name} (lines {start+1}-{end})")
                
                i = end
                continue
        
        # Skip property decorators that belong to extracted methods
        if stripped.startswith("@") and not stripped.startswith("@staticmethod") and not stripped.startswith("@classmethod"):
            # Check if the next def is extracted
            for j in range(i+1, min(i+3, len(lines))):
                if lines[j].strip().startswith("def "):
                    next_name = get_method_name(lines[j])
                    if next_name in METHOD_MAP:
                        # Find the full decorator block
                        dec_start = i
                        while dec_start > class_start + 1 and lines[dec_start-1].strip().startswith("@"):
                            dec_start -= 1
                        for r in range(dec_start, j):
                            lines_to_remove.add(r)
                        break
        
        i += 1
    
    print(f"\nMarked {len(lines_to_remove)} lines for removal from rlm_v2.py")
    
    # ── Phase 4: Write submodule files ──
    
    module_order = ["rlm_v2_graph", "rlm_v2_encoder", "rlm_v2_rp", "rlm_v2_verb", "rlm_v2_sleep"]
    mixin_names = {
        "rlm_v2_graph": "GraphMixin",
        "rlm_v2_encoder": "EncoderMixin",
        "rlm_v2_rp": "RPMixin",
        "rlm_v2_verb": "VerbMixin",
        "rlm_v2_sleep": "SleepMixin",
    }
    
    for module_name in module_order:
        methods = extracted_methods.get(module_name, {})
        mixin_class = mixin_names[module_name]
        
        content_lines = [
            f'"""',
            f'Mixin: {mixin_class} — {module_name} methods for RLMv2.',
            f'',
            f'Auto-extracted from rlm_v2.py. Edit in the source or directly here.',
            f'"""',
            f'import numpy as np',
            f'from typing import Optional, List, Tuple, Dict, Set, Any',
        ]
        
        # Add module-specific imports
        if module_name in ["rlm_v2_graph", "rlm_v2_encoder", "rlm_v2_sleep"]:
            content_lines.append(f'from ..embedder import LearnedEmbedder')
        if module_name in ["rlm_v2_sleep"]:
            content_lines.append(f'from collections import defaultdict')
        
        content_lines.append(f'')
        content_lines.append(f'')
        content_lines.append(f'class {mixin_class}:')
        content_lines.append(f'    """Mixin providing {module_name} methods for RLMv2."""')
        content_lines.append(f'')
        
        # Sort methods by their original line order
        # We need to track original line numbers - store them when we extract
        # Actually, let me rebuild the method->line mapping
        # For now, just add methods in alphabetical order
        for method_name in sorted(methods.keys()):
            method_lines = methods[method_name]
            for line in method_lines:
                content_lines.append(line)
            content_lines.append(f'')
        
        content = '\n'.join(content_lines)
        
        output_path = os.path.join(output_dir, f"{module_name}.py")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Written: {output_path} ({len(methods)} methods)")
    
    # ── Phase 5: Write rlm_v2_common.py ──
    
    # Extract module-level items from the original file
    common_lines = [
        f'"""',
        f'Shared constants and utilities for RLMv2 submodules.',
        f'',
        f'Auto-extracted from rlm_v2.py.',
        f'"""',
        f'',
        f'import numpy as np',
        f'import time',
        f'import pickle',
        f'import os',
        f'import zipfile',
        f'import json',
        f'from typing import Optional, List, Tuple, Dict, Set, Any',
        f'from collections import defaultdict',
        f'',
        f'from .module import Module, Linear, Embedding',
        f'from ..graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap',
        f'from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity',
        f'from ..propagation import PropagationEngine',
        f'from ..currencies import CognitiveCurrencies',
        f'from ..currency import create_rlm_currency',
        f'',
    ]
    
    # Add RELATION_TYPES, _KEYWORD_MAP, _build_glove_embedding_matrix from module-level
    in_rel_types = False
    in_keyword_map = False
    in_glove = False
    brace_depth = 0
    
    for line in module_lines:
        stripped = line.strip()
        
        # Track _build_glove_embedding_matrix 
        if stripped.startswith("def _build_glove_embedding_matrix"):
            in_glove = True
            common_lines.append(line)
            continue
        
        if in_glove:
            # Module-level functions end when we hit a non-indented, non-blank, non-comment line
            if stripped and not stripped.startswith("#") and not stripped.startswith("\"\"\"") and not stripped.startswith("'") and not line.startswith(" "):
                in_glove = False
                common_lines.append(f'\n')
            else:
                common_lines.append(line)
            continue
        
        # Track RELATION_TYPES block
        if stripped.startswith("RELATION_TYPES"):
            in_rel_types = True
            common_lines.append(line)
            continue
        
        if in_rel_types:
            common_lines.append(line)
            # Check for end of list
            if stripped == "]":
                in_rel_types = False
                common_lines.append(f'\n')
            continue
        
        # Track _KEYWORD_MAP block
        if stripped.startswith("_KEYWORD_MAP"):
            in_keyword_map = True
            common_lines.append(line)
            brace_depth = 0
            continue
        
        if in_keyword_map:
            common_lines.append(line)
            # Count braces to find end
            brace_depth += line.count("{") - line.count("}")
            if brace_depth == 0 and stripped.startswith("}"):
                in_keyword_map = False
                common_lines.append(f'\n')
            continue
    
    # Remove exit() calls from common
    common_content = '\n'.join(common_lines)
    common_content = common_content.replace('\nsys.exit(1)', '')
    
    common_path = os.path.join(output_dir, "rlm_v2_common.py")
    with open(common_path, 'w', encoding='utf-8') as f:
        f.write(common_content)
    print(f"  Written: {common_path}")
    
    # ── Phase 6: Rewrite rlm_v2.py as orchestrator ──
    
    # Build the new rlm_v2.py content
    new_lines = [
        f'"""',
        f'RLM v2 — Triple-Based Cognitive Architecture (orchestrator)',
        f'',
        f'This module is the main entry point for RLMv2. The class RLMv2 inherits',
        f'from mixin classes that provide specialized functionality:',
        f'  - GraphMixin: concept graph operations, prototypes, entity adapters',
        f'  - EncoderMixin: autoencoder, embedding projection',
        f'  - RPMixin: relation predictor forward/backward',
        f'  - VerbMixin: verb offset predictor',
        f'  - SleepMixin: sleep consolidation, episodic memory, alignment',
        f'',
        f'Mixins are defined in sibling modules (rlm_v2_graph.py, etc.).',
        f'"""',
        f'',
        f'import numpy as np',
        f'import time',
        f'import pickle',
        f'import os',
        f'import zipfile',
        f'import json',
        f'from typing import Optional, List, Tuple, Dict, Set, Any',
        f'from collections import defaultdict',
        f'',
        f'from .module import Module, Linear, Embedding',
        f'from ..graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap',
        f'from ..plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity',
        f'from ..propagation import PropagationEngine',
        f'from ..currencies import CognitiveCurrencies',
        f'from ..currency import create_rlm_currency',
        f'from ..embedder import LearnedEmbedder',
        f'',
        f'from .rlm_v2_common import (',
        f'    RELATION_TYPES, _KEYWORD_MAP, _build_glove_embedding_matrix',
        f')',
        f'from .rlm_v2_graph import GraphMixin',
        f'from .rlm_v2_encoder import EncoderMixin',
        f'from .rlm_v2_rp import RPMixin',
        f'from .rlm_v2_verb import VerbMixin',
        f'from .rlm_v2_sleep import SleepMixin',
        f'',
        f'',
    ]
    
    # Build the class definition with mixin inheritance
    new_lines.append(f'class RLMv2(GraphMixin, EncoderMixin, RPMixin, VerbMixin, SleepMixin, Module):')
    new_lines.append(f'    """')
    new_lines.append(f'    Triple-based cognitive architecture with spreading activation inference.')
    new_lines.append(f'')
    new_lines.append(f'    Inherits from mixins:')
    new_lines.append(f'      GraphMixin   — Concept graph operations, prototypes, entity adapters')
    new_lines.append(f'      EncoderMixin — Autoencoder, embedding projection')
    new_lines.append(f'      RPMixin      — Relation predictor forward/backward')
    new_lines.append(f'      VerbMixin    — Verb offset predictor')
    new_lines.append(f'      SleepMixin   — Sleep consolidation, episodic memory, alignment')
    new_lines.append(f'    """')
    new_lines.append(f'')
    
    # Copy the class body, skipping extracted methods
    # Find the __init__ method and copy everything
    in_class = False
    current_method_name = None
    skip_lines = False
    
    i = class_start + 1
    while i <= last_class_line:
        line = lines[i]
        stripped = line.strip()
        
        # Check if we're at a method that was extracted
        if stripped.startswith("def "):
            method_name = get_method_name(stripped)
            if method_name in METHOD_MAP:
                # Find the full method block (including decorators) and skip it
                start, end = find_method_boundaries(lines, i)
                i = end
                continue
        
        # Skip property decorators for extracted methods
        if stripped.startswith("@") and not stripped.startswith("@staticmethod") and not stripped.startswith("@classmethod"):
            # Check next def
            for j in range(i+1, min(i+3, len(lines))):
                if lines[j].strip().startswith("def "):
                    next_name = get_method_name(lines[j])
                    if next_name in METHOD_MAP:
                        # Find the decorator start and skip the whole block
                        dec_start = i
                        while dec_start > class_start + 1 and lines[dec_start-1].strip().startswith("@"):
                            dec_start -= 1
                        start, end = find_method_boundaries(lines, j)
                        i = end
                        break
                    else:
                        new_lines.append(line)
                    break
            else:
                new_lines.append(line)
            i += 1
            continue
        
        # Keep the line
        new_lines.append(line)
        i += 1
    
    # Add remaining class body (after last_class_line, there may be class-level code)
    for i in range(last_class_line + 1, len(lines)):
        new_lines.append(lines[i])
    
    # Write the new rlm_v2.py
    new_content = ''.join(new_lines)
    with open(source_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"\n  Written: {source_path} ({len(new_lines)} lines)")
    
    print(f"\n{'='*60}")
    print(f"Refactoring complete!")
    print(f"{'='*60}")
    print(f"\nCreated submodules:")
    for module_name in module_order:
        count = len(extracted_methods.get(module_name, {}))
        print(f"  {module_name}.py — {count} methods")
    print(f"  rlm_v2_common.py — shared constants and utilities")
    print(f"\nUpdated: rlm_v2.py (now inherits from mixins, {len(new_lines)} lines)")
    print(f"\nMethods extracted: {sum(len(m) for m in extracted_methods.values())}")
    print(f"Methods remaining in rlm_v2.py: substantial (__init__, forward, learn, etc.)")


if __name__ == "__main__":
    extract()
