"""
Patch RLMv2 to implement Phase 1 P0:
1. Expand verb offsets to ALL verbs (remove whitelist)
2. Confidence-weighted blending with W_rel output
3. Hierarchical semantic prototype system for novel entities

Run: python scripts/patch_verb_offset_blending.py
"""
import re
import sys

def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    changes_made = []
    
    # ================================================================
    # CHANGE 1: Rewrite _rp_forward_verb_offset to support ALL verbs
    # ================================================================
    old_verb_offset = '''    def _rp_forward_verb_offset(self, subject_tid: int, verb_word: str) -> Optional[np.ndarray]:
        """Predict using verb-stem offset arithmetic.

        Uses DOMAIN-SPECIFIC verb offsets: offset(verb) = avg(target - subject) 
        for that verb in the current domain.

        Only applies to highly functional verbs where target is predictable from subject+verb.
        For non-functional verbs (causes, produces, etc.), returns None to fall back to W_rel.

        predicted_embed = subject_embed + offset(verb, domain)
        logits_k = predicted_embed @ token_embed_k / (temperature)

        Returns logits over vocab, or None if verb is unknown or non-functional.
        """
        if not verb_word or not self.use_verb_offset:
            return None
        stem = self._verb_stem(verb_word)
        
        # Whitelist of highly functional verbs where verb offset works well
        # These verbs have consistent target patterns across subjects
        functional_verbs = {
            'melt', 'freez', 'is', 'are', 'was', 'were', 'be',  # phase changes, copula
            'enabl', 'prevent', 'caus', 'produ', 'creat', 'drives', 'shap',  # keep for now
        }
        # Actually, only use for truly functional verbs
        # 'caus' and 'produc' are NOT functional - target varies by subject
        strictly_functional = {'melt', 'freez', 'is', 'are', 'was', 'were', 'be'}
        if stem not in strictly_functional:
            return None  # Fall back to bilinear W_rel
        
        # Get domain-specific offsets
        domain_id = self.current_domain_id if self.current_domain_id is not None else 0
        if domain_id not in self._verb_offsets or stem not in self._verb_offsets[domain_id]:
            return None
        
        source_embed = self.token_embed.weight.data[subject_tid]
        offset = self._verb_offsets[domain_id][stem]
        predicted = source_embed + offset
        
        # Cosine similarity against all token embeddings
        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)
        token_norms = np.linalg.norm(token_embeds, axis=1)
        pred_norm = np.linalg.norm(predicted)
        
        if pred_norm > 0 and np.any(token_norms > 0):
            valid_tok = token_norms > 0
            normed_tok = token_embeds.copy()
            normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
            logits = (predicted / pred_norm) @ normed_tok.T  # cosine similarities
            # Suppress subject token (self-prediction) - strong negation
            if 0 <= subject_tid < len(logits):
                logits[subject_tid] = np.min(logits) - 10.0
            # Scale up to make softmax meaningful
            logits *= 10.0
            return logits
        return None'''

    new_verb_offset = '''    def _rp_forward_verb_offset(self, subject_tid: int, verb_word: str, return_count: bool = False):
        """Predict using verb-stem offset arithmetic with confidence-weighted blending.

        Uses DOMAIN-SPECIFIC verb offsets: offset(verb) = avg(target - subject) 
        for that verb in the current domain.

        NOW supports ALL verbs (not just a whitelist). For rare verbs with few
        training examples, the offset is less reliable; caller blends with W_rel.

        predicted_embed = subject_embed + offset(verb, domain)
        logits_k = predicted_embed @ token_embed_k / (temperature)

        Args:
            subject_tid: subject token ID
            verb_word: the verb string
            return_count: if True, returns (logits, count) tuple for blending

        Returns:
            logits over vocab, or None if verb is unknown.
            If return_count=True, returns (logits, count) or (None, 0).
        """
        if not verb_word or not self.use_verb_offset:
            return (None, 0) if return_count else None
        stem = self._verb_stem(verb_word)
        
        # Get domain-specific offsets
        domain_id = self.current_domain_id if self.current_domain_id is not None else 0
        if domain_id not in self._verb_offsets or stem not in self._verb_offsets[domain_id]:
            return (None, 0) if return_count else None
        
        # Get the training example count for blending weight computation
        count = 0
        if domain_id in self._verb_offset_count and stem in self._verb_offset_count[domain_id]:
            count = self._verb_offset_count[domain_id][stem]
        
        source_embed = self.token_embed.weight.data[subject_tid]
        offset = self._verb_offsets[domain_id][stem]
        predicted = source_embed + offset
        
        # Cosine similarity against all token embeddings
        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)
        token_norms = np.linalg.norm(token_embeds, axis=1)
        pred_norm = np.linalg.norm(predicted)
        
        if pred_norm > 0 and np.any(token_norms > 0):
            valid_tok = token_norms > 0
            normed_tok = token_embeds.copy()
            normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
            logits = (predicted / pred_norm) @ normed_tok.T  # cosine similarities
            # Suppress subject token (self-prediction) - strong negation
            if 0 <= subject_tid < len(logits):
                logits[subject_tid] = np.min(logits) - 10.0
            # Scale up to make softmax meaningful
            logits *= 10.0
            return (logits, count) if return_count else logits
        return (None, 0) if return_count else None'''

    if old_verb_offset in content:
        content = content.replace(old_verb_offset, new_verb_offset)
        changes_made.append("_rp_forward_verb_offset: expanded to ALL verbs + count tracking")
        print("  [OK] Patched _rp_forward_verb_offset")
    else:
        print("  ⚠ Could not find old _rp_forward_verb_offset (may already be patched)")

    # ================================================================
    # CHANGE 2: Rewrite _rp_forward to blend verb offset with W_rel
    # ================================================================
    old_rp_forward_verb_section = '''        # ── Verb-Stem Offset Path (primary) ──
        # When use_verb_offset is True and the verb is known, use offset arithmetic
        # instead of bilinear W_rel. This enables same-subject different-target
        # predictions (e.g., "cold causes" -> shivering, "cold freezes" -> water).
        # It also enables cross-domain transfer via shared verb semantics.
        # IMPORTANT: Verb offsets are computed on RAW embeddings (target - subject),
        # so we must use the raw subject embedding here, NOT the entity-adapted one.
        if verb_word and self.use_verb_offset:
            verb_logits = self._rp_forward_verb_offset(subject_tid, verb_word)
            if verb_logits is not None:
                self._rp_cache = None  # prevent stale cache from corrupting W_rel
                return verb_logits'''

    new_rp_forward_verb_section = '''        # ── Verb-Stem Offset Path (primary) ──
        # When use_verb_offset is True, compute verb offset logits and blend
        # with W_rel output using confidence-weighted interpolation.
        # For well-trained verbs (count >= 10), verb offset dominates.
        # For rare verbs, fall back more heavily on W_rel.
        # IMPORTANT: Verb offsets are computed on RAW embeddings (target - subject),
        # so we use the raw subject embedding here, NOT the entity-adapted one.
        verb_blend_logits = None
        verb_blend_weight = 0.0
        if verb_word and self.use_verb_offset:
            verb_result = self._rp_forward_verb_offset(subject_tid, verb_word, return_count=True)
            if verb_result is not None and verb_result[0] is not None:
                verb_logits, verb_count = verb_result
                verb_blend_weight = min(1.0, verb_count / 10.0)
                verb_blend_logits = verb_logits
                if verb_blend_weight >= 1.0:
                    # Fully confident in verb offset - return directly (no W_rel needed)
                    self._rp_cache = None
                    return verb_logits'''

    if old_rp_forward_verb_section in content:
        content = content.replace(old_rp_forward_verb_section, new_rp_forward_verb_section)
        changes_made.append("_rp_forward: added verb offset + W_rel blending")
        print("  [OK] Patched _rp_forward verb offset section")
    else:
        print("  ⚠ Could not find old _rp_forward verb section (may already be patched)")

    # ================================================================
    # CHANGE 3: Add blending at the end of _rp_forward (before return)
    # ================================================================
    # Find the cache + return at the end of _rp_forward and add blending
    old_rp_forward_end = '''        # Cache for backprop
        self._rp_cache = (
            subject_tid, rel_type_idx, domain_id,
            source_embed, source_latent,
            token_embeds, target_latents,
            W_rel, logits
        )
        return logits'''

    new_rp_forward_end = '''        # ── Blend verb offset with W_rel (confidence-weighted) ──
        # When verb offset is available (even for rare verbs), blend it with
        # the bilinear W_rel output. High-frequency verbs dominate; rare verbs
        # contribute proportionally.
        if verb_blend_logits is not None and verb_blend_weight > 0.0:
            # Blend: final = blend_weight * verb_logits + (1 - blend_weight) * w_rel_logits
            blended = verb_blend_weight * verb_blend_logits + (1.0 - verb_blend_weight) * logits
            # Cache W_rel path for backprop (verb offsets are not trained via backprop)
            self._rp_cache = (
                subject_tid, rel_type_idx, domain_id,
                source_embed, source_latent,
                token_embeds, target_latents,
                W_rel, blended
            )
            return blended

        # Cache for backprop
        self._rp_cache = (
            subject_tid, rel_type_idx, domain_id,
            source_embed, source_latent,
            token_embeds, target_latents,
            W_rel, logits
        )
        return logits'''

    if old_rp_forward_end in content:
        content = content.replace(old_rp_forward_end, new_rp_forward_end)
        changes_made.append("_rp_forward: added blending at end of method")
        print("  [OK] Patched _rp_forward blending at end")
    else:
        print("  ⚠ Could not find old _rp_forward end (may already be patched)")

    # ================================================================
    # CHANGE 4: Add hierarchical prototype initialization in __init__
    # ================================================================
    # Add prototype fields after the verb offset section
    old_proto_init = '''        self._verb_offsets: Dict[str, np.ndarray] = {}          # verb_stem -> offset vector
        self._verb_offset_count: Dict[str, int] = {}             # verb_stem -> count
        self._verb_accum_buffer: List[Tuple[str, np.ndarray]] = []  # (verb_stem, offset_vec) pairs
        self.use_verb_offset = False                              # enabled by experiment'''

    new_proto_init = '''        self._verb_offsets: Dict[str, np.ndarray] = {}          # verb_stem -> offset vector
        self._verb_offset_count: Dict[str, int] = {}             # verb_stem -> count
        self._verb_accum_buffer: List[Tuple[str, np.ndarray]] = []  # (verb_stem, offset_vec) pairs
        self.use_verb_offset = False                              # enabled by experiment

        # ── HIERARCHICAL SEMANTIC PROTOTYPE SYSTEM ──
        # For novel entities, find nearest prototype node and inherit its edges.
        # prototype_label -> [concept_ids that belong to this prototype]
        self._prototype_hierarchy: Dict[str, List[int]] = {}
        # prototype_label -> prototype concept vector (centroid of members)
        self._prototype_vectors: Dict[str, np.ndarray] = {}
        # Track which concept IDs were created as "novel entities" (inherited from prototype)
        self._novel_entity_concepts: Dict[int, float] = {}  # concept_id -> confidence
        # Prototype similarity threshold for inheritance
        self._prototype_similarity_threshold: float = 0.55
        # Enable/disable prototype inheritance
        self.use_prototype_inheritance: bool = True'''

    if old_proto_init in content:
        content = content.replace(old_proto_init, new_proto_init)
        changes_made.append("Added _prototype_hierarchy, _prototype_vectors, _novel_entity_concepts")
        print("  [OK] Added prototype hierarchy fields to __init__")
    else:
        print("  ⚠ Could not find verb offset init section (may already be patched)")

    # ================================================================
    # CHANGE 5: Add prototype methods after _verb_stem
    # ================================================================
    old_after_verb_stem = '''    def _accumulate_verb_offset(self, subject_tid: int, target_tid: int, verb_word: str, domain_id: int = None):
        """Accumulate offset = target_embed - subject_embed for a verb stem.

        Called during learn() for each training triple.
        Offsets are finalized into _verb_offsets by _compute_verb_offsets().
        Now tracks domain_id to compute domain-specific verb offsets.
        """'''

    new_after_verb_stem_with_proto = '''    def _find_nearest_prototype(self, embed_vec: np.ndarray) -> Tuple[Optional[str], float]:
        """Find the nearest prototype for an embedding vector.

        Computes cosine similarity between the projected concept vector
        and all prototype vectors. Returns (prototype_label, similarity).

        Used when a novel token arrives to inherit edges from its closest prototype.
        """
        if not self._prototype_vectors:
            return None, 0.0
        
        concept_vec = self._project_to_concept(embed_vec)
        cv_norm = np.linalg.norm(concept_vec)
        if cv_norm == 0:
            return None, 0.0
        
        best_label = None
        best_sim = 0.0
        for label, proto_vec in self._prototype_vectors.items():
            sim = float(np.dot(concept_vec, proto_vec) / (cv_norm * np.linalg.norm(proto_vec) + 1e-10))
            if sim > best_sim:
                best_sim = sim
                best_label = label
        return best_label, best_sim

    def _register_prototype(self, label: str, concept_ids: List[int]):
        """Register a prototype node by label and list of concept IDs.

        Computes the prototype vector as the centroid of member concept vectors.
        """
        vectors = []
        for cid in concept_ids:
            node = self.graph.get_node(cid)
            if node is not None and node.vector is not None:
                vectors.append(node.vector)
        if not vectors:
            return
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid /= norm
        self._prototype_hierarchy[label] = concept_ids
        self._prototype_vectors[label] = centroid

    def _inherit_from_prototype(self, new_concept_id: int, prototype_label: str, similarity: float):
        """Inherit edges from a prototype to a novel entity concept.

        Copies the most confident edges from the prototype's outgoing edges,
        weighted by similarity. Novel entities start as a weaker version of
        their prototype, which gets refined through actual use.
        """
        proto_ids = self._prototype_hierarchy.get(prototype_label, [])
        if not proto_ids:
            return
        
        confidence_multiplier = 0.5 * similarity  # discount prototype confidence
        
        for proto_cid in proto_ids:
            for tgt_id, edge in self.graph.get_outgoing(proto_cid):
                if tgt_id == proto_cid:
                    continue
                inherited_weight = edge.weight * confidence_multiplier
                if inherited_weight > 0.05:
                    new_edge = self.graph.add_edge(
                        new_concept_id, tgt_id,
                        weight=inherited_weight,
                        relation_type=edge.relation_type
                    )
                    new_edge.confidence = edge.confidence * 0.5  # lower confidence for inherited
                    new_edge.predicate_token_id = edge.predicate_token_id

    def _init_default_prototypes(self):
        """Initialize default prototypes from the concept graph.

        Builds prototypes for common semantic categories based on embedding
        clusters in the concept space. Called once during first use.
        """
        if len(self.graph.nodes) < 5:
            return
        
        # Cluster concepts by similarity and register as prototypes
        from collections import defaultdict
        node_ids = sorted(self.graph.nodes.keys())
        
        # Use a simple greedy clustering approach
        assigned = set()
        for nid in node_ids[:50]:  # Process first 50 nodes
            if nid in assigned:
                continue
            node = self.graph.get_node(nid)
            if node is None or node.vector is None:
                continue
            
            # Find all similar nodes (cosine > 0.6)
            cluster = [nid]
            assigned.add(nid)
            for other_id in node_ids:
                if other_id in assigned or other_id == nid:
                    continue
                other = self.graph.get_node(other_id)
                if other is None or other.vector is None:
                    continue
                sim = float(np.dot(node.vector, other.vector) / (
                    np.linalg.norm(node.vector) * np.linalg.norm(other.vector) + 1e-10
                ))
                if sim > 0.6:
                    cluster.append(other_id)
                    assigned.add(other_id)
            
            if len(cluster) >= 2:
                label = f"prototype_{len(self._prototype_hierarchy)}"
                self._register_prototype(label, cluster)

    def _accumulate_verb_offset(self, subject_tid: int, target_tid: int, verb_word: str, domain_id: int = None):
        """Accumulate offset = target_embed - subject_embed for a verb stem.

        Called during learn() for each training triple.
        Offsets are finalized into _verb_offsets by _compute_verb_offsets().
        Now tracks domain_id to compute domain-specific verb offsets.
        """'''

    if old_after_verb_stem in content:
        content = content.replace(old_after_verb_stem, new_after_verb_stem_with_proto)
        changes_made.append("Added prototype methods: _find_nearest_prototype, _register_prototype, _inherit_from_prototype, _init_default_prototypes")
        print("  [OK] Added prototype methods")
    else:
        print("  ⚠ Could not find _accumulate_verb_offset anchor (may already be patched)")

    # ================================================================
    # CHANGE 6: Modify _get_or_create_concept for prototype inheritance
    # ================================================================
    old_get_or_create = '''    def _get_or_create_concept(self, token_id: int, embed_vec: np.ndarray) -> int:
        """Get existing concept for a token, or create one if needed.

        1-to-1 mapping: each token gets exactly one concept. No merging.
        This prevents unrelated concepts from being collapsed together.
        """
        # Check binding map first - reuse existing concept for this token
        bindings = self.binding_map.get_concepts(token_id, min_confidence=0.1)
        if bindings:
            cid = bindings[0].concept_id
            # Validate that the node still exists in the graph
            # (binding may be stale if node was pruned)
            if self.graph.get_node(cid) is not None:
                return cid
            # Stale binding - fall through to create a fresh concept

        # Always create a new concept for this token (no merging)
        concept_vec = self._project_to_concept(embed_vec)
        if len(self.graph.nodes) < self._max_concepts:
            node = self.graph.add_node(vector=concept_vec, label=self._decode_token(token_id))
            nid = node.id
        else:
            # At capacity - find nearest existing concept to reuse
            # (better than a phantom ID that doesn't exist in the graph)
            nid, sim = self._nearest_concept(embed_vec)
            if nid < 0:
                # No concepts at all - force create
                node = self.graph.add_node(vector=concept_vec, label=self._decode_token(token_id))
                nid = node.id

        # Bind token to concept
        self.binding_map.bind(token_id, nid, confidence=0.9)
        
        # Initialize entity adapter for this token (Fix #3)
        if token_id not in self._entity_adapters:
            self._init_entity_adapter(token_id)
        
        return nid'''

    new_get_or_create = '''    def _get_or_create_concept(self, token_id: int, embed_vec: np.ndarray) -> int:
        """Get existing concept for a token, or create one if needed.

        1-to-1 mapping: each token gets exactly one concept. No merging.
        For novel tokens, inherits edges from the nearest semantic prototype
        when prototype inheritance is enabled (Hierarchical Semantic Prototype System).
        """
        # Check binding map first - reuse existing concept for this token
        bindings = self.binding_map.get_concepts(token_id, min_confidence=0.1)
        if bindings:
            cid = bindings[0].concept_id
            # Validate that the node still exists in the graph
            # (binding may be stale if node was pruned)
            if self.graph.get_node(cid) is not None:
                return cid
            # Stale binding - fall through to create a fresh concept

        # Always create a new concept for this token (no merging)
        concept_vec = self._project_to_concept(embed_vec)
        if len(self.graph.nodes) < self._max_concepts:
            node = self.graph.add_node(vector=concept_vec, label=self._decode_token(token_id))
            nid = node.id
        else:
            # At capacity - find nearest existing concept to reuse
            # (better than a phantom ID that doesn't exist in the graph)
            nid, sim = self._nearest_concept(embed_vec)
            if nid < 0:
                # No concepts at all - force create
                node = self.graph.add_node(vector=concept_vec, label=self._decode_token(token_id))
                nid = node.id

        # ── Prototype inheritance for novel entities ──
        # If this is a new concept (just created), find nearest prototype
        # and inherit its edges with discounted confidence.
        if self.use_prototype_inheritance and self._prototype_vectors:
            proto_label, proto_sim = self._find_nearest_prototype(embed_vec)
            if proto_label is not None and proto_sim >= self._prototype_similarity_threshold:
                self._inherit_from_prototype(nid, proto_label, proto_sim)
                self._novel_entity_concepts[nid] = proto_sim * 0.5  # initial confidence

        # Bind token to concept
        self.binding_map.bind(token_id, nid, confidence=0.9)
        
        # Initialize entity adapter for this token (Fix #3)
        if token_id not in self._entity_adapters:
            self._init_entity_adapter(token_id)
        
        return nid'''

    if old_get_or_create in content:
        content = content.replace(old_get_or_create, new_get_or_create)
        changes_made.append("_get_or_create_concept: added prototype inheritance for novel entities")
        print("  [OK] Patched _get_or_create_concept with prototype inheritance")
    else:
        print("  ⚠ Could not find old _get_or_create_concept")

    # ================================================================
    # Write the patched file
    # ================================================================
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nPatching complete. {len(changes_made)} changes made:")
    for c in changes_made:
        print(f"  - {c}")
    return changes_made


if __name__ == "__main__":
    import os
    filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "ravana_ml", "src", "ravana_ml", "nn", "rlm_v2.py")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)
    print(f"Patching {filepath}...")
    patch_file(filepath)
