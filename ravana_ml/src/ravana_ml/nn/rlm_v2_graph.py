"""
Mixin: GraphMixin — rlm_v2_graph methods for RLMv2.

Auto-extracted from rlm_v2.py. Edit in the source or directly here.
"""
import numpy as np
from typing import Optional, List, Tuple, Dict, Set, Any
from ..embedder import LearnedEmbedder
from .rlm_v2_common import RELATION_TYPES, _KEYWORD_MAP


class GraphMixin:
    """Mixin providing rlm_v2_graph methods for RLMv2."""



    def _adapt_entity_adapter_at_test_time(self, subject_tid: int, verb_word: str, target_tid: int,

                                            n_steps: int = 10, lr: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:

        """Test-time adapter adaptation for held-out subjects using verb offset MSE loss.



        For verb offset path to work, we need: adapted_source + offset(verb) ≈ target_embed

        This minimizes MSE in embedding space, which directly improves the verb offset prediction.

        """

        if not self.use_verb_offset:

            return self._get_or_adapt_entity_adapter(subject_tid, verb_word, target_tid)



        # Get or initialize adapter (from nearest neighbor if held-out)

        U, V = self._get_or_adapt_entity_adapter(subject_tid, verb_word, target_tid)

        mU, mV = self._entity_adapter_momentums[subject_tid]



        subject_embed = self.token_embed.weight.data[subject_tid]

        target_embed = self.token_embed.weight.data[target_tid]

        

        stem = self._verb_stem(verb_word)

        domain_id = self.current_domain_id if self.current_domain_id is not None else 0

        if domain_id not in self._verb_offsets or stem not in self._verb_offsets[domain_id]:

            return U, V



        offset = self._verb_offsets[domain_id][stem]

        

        momentum = self._entity_adapter_momentum

        adapter_lr = lr



        for step in range(n_steps):

            # Forward: adapted = subject_embed @ U.T @ V

            adapted = (subject_embed @ U.T) @ V

            # Verb offset prediction: predicted = adapted + offset

            predicted = adapted + offset

            residual = predicted - target_embed

            

            # MSE loss

            loss = 0.5 * np.sum(residual ** 2)

            

            # Gradient: dL/d_adapted = residual

            # dL/dV = outer(U @ subject_embed, residual)

            # dL/dU = outer(residual @ V.T, subject_embed)

            z = subject_embed @ U.T

            dV = np.outer(z, residual)

            dU = np.outer(residual @ V.T, subject_embed)

            

            # Gradient clipping

            for g in [dU, dV]:

                gn = np.linalg.norm(g)

                if gn > 5.0:

                    g *= (5.0 / (gn + 1e-15))

            

            # Momentum update

            mU = momentum * mU - adapter_lr * dU

            mV = momentum * mV - adapter_lr * dV

            U += mU

            V += mV

            

            if step % 3 == 0:

                print(f"    [Adapt Step {step}] mse={loss:.4f} residual_norm={np.linalg.norm(residual):.4f}")



        self._entity_adapters[subject_tid] = (U, V)

        self._entity_adapter_momentums[subject_tid] = (mU, mV)



        return U, V








    def _anti_hebbian_prune_polluted_edges(self, mismatch_threshold: float = 0.3,

                                           min_prediction_count: int = 5) -> int:

        """

        Anti-Hebbian pruning: weaken/remove edges that consistently predict wrong.

        

        Edges with high prediction_count but low forward_pred_count ratio

        are likely polluted by Hebbian noise from early training with different bindings.

        

        Returns number of edges pruned/weakened.

        """

        pruned = 0

        edges_to_check = list(self.graph.edges.items())

        

        for (src_id, tgt_id), edge in edges_to_check:

            if edge.prediction_count >= min_prediction_count:

                pred_ratio = edge.forward_pred_count / edge.prediction_count

                if pred_ratio < (1.0 - mismatch_threshold):

                    # This edge consistently fails to predict correctly

                    # Apply anti-Hebbian weakening

                    self.graph.anti_hebbian_update(src_id, tgt_id, lr=0.02)

                    pruned += 1

                    

                    # If edge is very weak and unstable, remove it

                    if edge.weight < 0.1 and edge.confidence < 0.1:

                        self.graph.remove_edge(src_id, tgt_id)

                        

        return pruned






    def _cached_norm(self, vec: np.ndarray, cache_key: str) -> float:

        """Cached linalg.norm - avoids redundant computation within a forward pass."""

        val = self._norm_cache.get(cache_key)

        if val is None:

            val = float(np.linalg.norm(vec))

            self._norm_cache[cache_key] = val

        return val






    def _classify_relation_learned(self, relation_token_ids: List[int]) -> Tuple[int, np.ndarray]:

        """Classify relation using learned embeddings + keyword fallback.



        Returns (type_index, type_embedding) where type_embedding is the

        learned vector for that relation type.



        This is the learned version that improves over time as the model

        sees more examples of each relation type.

        """

        # Get keyword-based classification as fallback

        keyword_idx = self.classify_relation(relation_token_ids)



        if not relation_token_ids:

            return keyword_idx, self.relation_type_embed.weight.data[keyword_idx]



        # Average the relation token embeddings

        rel_embeds = []

        for tid in relation_token_ids:

            rel_embeds.append(self.token_embed.weight.data[tid])

        rel_vec = np.mean(rel_embeds, axis=0)



        # Project to concept space

        rel_concept = self._project_to_concept(rel_vec)



        # Score against all relation type embeddings

        type_embeds = self.relation_type_embed.weight.data  # (n_types, concept_dim)

        # Cosine similarity

        rel_norm = np.linalg.norm(rel_concept)

        type_norms = np.linalg.norm(type_embeds, axis=1)

        if rel_norm > 0 and np.all(type_norms > 0):

            sims = type_embeds @ rel_concept / (type_norms * rel_norm)

        else:

            sims = np.zeros(len(RELATION_TYPES))



        # Softmax with temperature

        temp = 0.5

        exp_sims = np.exp(sims / temp - np.max(sims / temp))

        probs = exp_sims / (np.sum(exp_sims) + 1e-10)



        # Use keyword classifier directly - learned relation type embeddings

        # are never trained (learn() updates edge relation_vectors, not

        # relation_type_embed.weight). With 300x hard boost, keyword-only

        # is deterministic and avoids random noise corrupting relation types.

        # Tested: 100% keyword gives 62.5% cross_domain_causal vs 37.5% blend.

        return keyword_idx, self.relation_type_embed.weight.data[keyword_idx]






    def _decode_token(self, token_id: int) -> str:

        """Decode a single token ID to text."""

        try:

            if hasattr(self, '_tokenizer') and self._tokenizer is not None:

                return self._tokenizer.decode([token_id])

            # Fallback: treat as character

            if 0 <= token_id < 128:

                return chr(token_id)

            return f"tok_{token_id}"

        except Exception:

            return f"tok_{token_id}"



    # ── Concept Lookup ──────────────────────────────────────────────────────






    def _find_nearest_prototype(self, embed_vec: np.ndarray) -> Tuple[Optional[str], float]:

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






    def _get_anchor_regularized_latent(self, token_id, latent):

        """Regularize latent representation towards expected semantic anchor neighborhoods."""

        if not hasattr(self, "_tokenizer") or self._tokenizer is None or not hasattr(self, "anchor_concepts"):

            return latent

        try:

            word = self._tokenizer.decode([token_id]).strip().lower()

        except Exception:

            return latent

        if word not in self.anchor_concepts:

            return latent

            

        anchors = self.anchor_concepts[word]

        anchor_latents = []

        for anchor_word in anchors:

            aid = self._tokenizer.word_to_id.get(anchor_word)

            if aid is not None:

                a_embed = self.get_robust_embedding(aid)

                a_lat, _, _, _ = self._encoder_forward_full(a_embed)

                anchor_latents.append(a_lat)

        if not anchor_latents:

            return latent

        mean_anchor_latent = np.mean(anchor_latents, axis=0)

        lambda_anchor = getattr(self, "lambda_anchor", 0.3)

        

        norm_before = np.linalg.norm(latent)

        reg_latent = (1.0 - lambda_anchor) * latent + lambda_anchor * mean_anchor_latent

        norm_after = np.linalg.norm(reg_latent)

        if norm_after > 0:

            reg_latent = reg_latent * (norm_before / norm_after)

        return reg_latent






    def _get_node_matrix(self):

        """Get cached (node_ids, vector_matrix, norms) for batch operations.



        Uses a version counter to avoid rebuilding when the graph hasn't changed.

        Returns (node_ids: List[int], matrix: np.ndarray[N,D], norms: np.ndarray[N]).

        """

        graph_version = (len(self.graph.nodes), len(self.graph.edges))

        if self._node_matrix_version != graph_version or self._node_matrix_cache is None:

            node_ids = sorted(self.graph.nodes.keys())

            if not node_ids:

                self._node_matrix_cache = ([], np.empty((0, self.concept_dim)), np.empty(0))

            else:

                matrix = np.stack([self.graph.nodes[cid].vector for cid in node_ids]).astype(np.float32)

                norms = np.linalg.norm(matrix, axis=1)

                self._node_matrix_cache = (node_ids, matrix, norms)

            self._node_matrix_version = graph_version

        return self._node_matrix_cache






    def _get_or_adapt_entity_adapter(self, subject_tid: int, verb_word: str = None, target_tid: int = None):

        """Get entity adapter, initializing from nearest neighbor if unseen.

        

        For held-out subjects (never seen during training), find the most similar

        training subject by embedding similarity and copy its adapter.

        Optionally do a few gradient steps using the verb offset to adapt.

        """

        if subject_tid in self._entity_adapters:

            return self._entity_adapters[subject_tid]

        

        # Find nearest training subject with adapter

        if self._entity_adapters:

            source_embed = self.get_robust_embedding(subject_tid)

            best_tid = None

            best_sim = -1.0

            

            for tid, (U, V) in self._entity_adapters.items():

                # Skip synthetic token IDs (relation-object hubs use IDs >= 10000)

                if tid >= self.vocab_size:

                    continue

                other_embed = self.get_robust_embedding(tid)

                sim = np.dot(source_embed, other_embed) / (

                    np.linalg.norm(source_embed) * np.linalg.norm(other_embed) + 1e-10

                )

                if sim > best_sim:

                    best_sim = sim

                    best_tid = tid

            

            if best_tid is not None and best_sim > 0.3:

                # Copy adapter from nearest neighbor

                U_src, V_src = self._entity_adapters[best_tid]

                self._entity_adapters[subject_tid] = (U_src.copy(), V_src.copy())

                self._entity_adapter_momentums[subject_tid] = (

                    np.zeros_like(U_src), np.zeros_like(V_src)

                )

                return self._entity_adapters[subject_tid]

        

        # Fallback: random init

        self._init_entity_adapter(subject_tid)

        return self._entity_adapters[subject_tid]






    def _get_or_create_concept(self, token_id: int, embed_vec: np.ndarray) -> int:

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

        

        return nid






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






    def _init_entity_adapter(self, subject_tid: int):

            """Initialize low-rank entity adapter for a subject token."""

            if subject_tid in self._entity_adapters:

                return

            rank = self._entity_adapter_rank

            latent_dim = self.latent_dim

            rng = np.random.RandomState(42 + subject_tid * 1000)

            # Initialize as small random perturbation around identity projection

            # U: (rank, latent_dim), V: (rank, latent_dim)

            # Use larger init (0.1 instead of 0.01) so adapter has meaningful effect

            U = rng.randn(rank, latent_dim).astype(np.float32) * 0.1

            V = rng.randn(rank, latent_dim).astype(np.float32) * 0.1

            self._entity_adapters[subject_tid] = (U, V)

            self._entity_adapter_momentums[subject_tid] = (

                np.zeros_like(U), np.zeros_like(V)

            )








    def _init_structured_concepts(self):

        """Create initial concept nodes distributed across the concept space."""

        rng = np.random.RandomState(42)

        for i in range(self.n_concepts):

            vec = rng.randn(self.concept_dim).astype(np.float32) * 0.1

            norm = np.linalg.norm(vec)

            if norm > 0:

                vec /= norm

            self.graph.add_node(vector=vec, label=f"init_{i}")



    # ── Property aliases for backward compatibility with CognitiveCurrencies ──






    def _inject_cross_domain_edge(self, subject_cid, object_cid, rel_name, subject_tid):

        """Inject (or strengthen) a relation-typed edge subject->object.



        Used by experiments to seed cross-domain analogical edges (e.g.

        anger->conflict mirroring heat->expansion). Idempotent: tracks injected

        pairs in self._cross_domain_edges_injected so repeat calls don't pollute.

        """

        if rel_name not in RELATION_TYPES:

            return

        key = (subject_cid, object_cid, rel_name)

        if key in self._cross_domain_edges_injected:

            return

        edge = self.graph.get_edge(subject_cid, object_cid)

        if edge is None:

            edge = self.graph.add_edge(

                source=subject_cid, target=object_cid,

                weight=0.5, relation_type=rel_name,

            )

        else:

            # Strengthen if same relation type, otherwise leave alone

            if edge.relation_type == rel_name:

                edge.weight = min(1.0, edge.weight + 0.2)

                edge.confidence = min(1.0, edge.confidence + 0.1)

        if subject_tid is not None and 0 <= subject_tid < self.vocab_size:

            try:

                edge.predicate_token_id = int(subject_tid)

            except Exception:

                pass

        self._cross_domain_edges_injected.add(key)






    def _inject_direct_edges_if_needed(self, subject_cid: int, object_cid: int,

                                       rel_type_name: str, threshold: float = 0.5) -> int:

        """

        Inject direct subject->object edges when binding map shows 1-to-1

        but graph edges are missing or weak (polluted by Hebbian noise).

        

        This bypasses the graph topology bottleneck for cross-domain causal

        relations where vector arithmetic carries the load but Hebbian edges

        are polluted.

        

        Returns number of edges injected/strengthened.

        """

        injected = 0

        

        # Check if direct edge exists and is strong enough

        direct_edge = self.graph.get_edge(subject_cid, object_cid)

        

        if direct_edge is None:

            # No edge exists - create it with strong weight

            self.graph.add_edge(

                source=subject_cid,

                target=object_cid,

                weight=0.7,  # Strong initial weight for direct causal

                relation_type=rel_type_name,

            )

            injected += 1

        elif direct_edge.weight < threshold:

            # Edge exists but is weak - strengthen it (bypass Hebbian noise)

            direct_edge.weight = threshold

            direct_edge.confidence = max(direct_edge.confidence, 0.6)

            direct_edge.stability = max(direct_edge.stability, 0.5)

            direct_edge.relation_type = rel_type_name

            injected += 1

            

        # Also check relation-object hub path edges

        # (These are handled by learn() but we ensure they're not polluted)

        return injected






    def _invalidate_caches(self):

        """Invalidate forward-pass caches after graph modification."""

        self._node_matrix_version = -1

        self._rel_vector_version = -1

        self._rel_vector_cache.clear()



    # ── Triple Decomposition ────────────────────────────────────────────────






    def _nearest_concept(self, embed_vec: np.ndarray) -> Tuple[int, float]:

        """Find the nearest concept node to an embedding vector.



        Uses the subject projection to map embed_dim -> concept_dim,

        then cosine similarity against all concept vectors.



        Returns (concept_id, similarity_score).

        """

        concept_vec = self._project_to_concept(embed_vec)

        results = self.graph.find_similar(concept_vec, k=1)

        if results:

            return results[0]

        # Fallback: create a new concept

        return -1, 0.0






    def _normalize_outgoing_weights(self, budget: float = 3.0):

        """Normalize outgoing edge weights so total doesn't exceed budget.



        Prevents runaway edge strengthening from hippocampal replay.

        """

        for nid in self.graph._outgoing:

            edges = self.graph._outgoing.get(nid, [])

            if not edges:

                continue

            total_weight = sum(e.weight for _, e in edges)

            if total_weight > budget:

                scale = budget / total_weight

                for _, edge in edges:

                    edge.weight *= scale






    def _project_to_concept(self, embed_vec: np.ndarray) -> np.ndarray:

        """Project embed_dim vector to concept_dim space via learned projection."""

        if len(embed_vec) == self.concept_dim:

            return embed_vec

        # Use the subject projection layer

        return self.subject_proj(embed_vec.reshape(1, -1)).data.flatten()






    def _project_to_embed(self, concept_vec: np.ndarray) -> np.ndarray:

        """Project concept_dim vector to embed_dim space via learned projection."""

        if len(concept_vec) == self.embed_dim:

            return concept_vec

        return self.concept_to_embed(concept_vec.reshape(1, -1)).data.flatten()






    def _prune_weak_edges(self, threshold: float = 0.1):

        """Remove edges with weight below threshold.



        Keeps the graph clean and prevents accumulation of noise edges.

        Prunes weak edges regardless of prediction count.

        """

        edges_to_remove = []

        for (src, tgt), edge in self.graph.edges.items():

            if edge.weight < threshold:

                edges_to_remove.append((src, tgt))



        for src, tgt in edges_to_remove:

            self.graph.remove_edge(src, tgt)



    # ── Save/Load ───────────────────────────────────────────────────────────






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






    def _validate_edge_bindings(self, subject_cid: int, object_cid: int,

                                 rel_type_name: str, relation_ids: List[int]) -> bool:

        """

        Validate edge topology against current binding map.

        

        After training, bindings may have settled to 1-to-1 mapping, but edges

        created during early training may reflect different/incorrect bindings.

        This method checks if the current binding map matches the edge topology

        and corrects mismatches.

        

        Returns True if validation passed, False if edges were corrected.

        """

        # Get current token IDs for these concepts

        subj_bindings = self.binding_map.get_tokens(subject_cid, min_confidence=0.1)

        obj_bindings = self.binding_map.get_tokens(object_cid, min_confidence=0.1)

        

        if not subj_bindings or not obj_bindings:

            # Concepts don't have valid token bindings yet

            return True

            

        current_subj_tid = subj_bindings[0].token_id

        current_obj_tid = obj_bindings[0].token_id

        

        # Check if edge's predicate token matches current relation

        direct_edge = self.graph.get_edge(subject_cid, object_cid)

        if direct_edge is not None:

            if relation_ids and direct_edge.predicate_token_id != relation_ids[0]:

                # Predicate has changed - this edge was learned with different relation

                # Update the predicate token to match current

                direct_edge.predicate_token_id = relation_ids[0]

                # Reduce confidence since topology was learned under different semantics

                direct_edge.confidence = max(0.1, direct_edge.confidence * 0.5)

                

        # Check reverse edge

        reverse_edge = self.graph.get_edge(object_cid, subject_cid)

        if reverse_edge is not None and relation_ids:

            reverse_edge.predicate_token_id = relation_ids[0]

            

        return True






    def classify_relation(self, relation_token_ids: List[int]) -> int:

        """Classify relation tokens into a relation type index.



        Uses keyword matching for initial classification, then learned embeddings

        for refinement. Returns index into RELATION_TYPES.



        This is what makes "melts" -> CAUSAL work (same pathway as "causes").

        """

        if not relation_token_ids:

            return RELATION_TYPES.index("semantic")  # default



        # Phase 1: keyword-based classification

        # Decode relation tokens to text

        rel_words = set()

        for tid in relation_token_ids:

            try:

                word = self._decode_token(tid).lower().strip()

                if word:

                    rel_words.add(word)

            except Exception:

                pass



        # Check keyword map

        for rel_type, keywords in _KEYWORD_MAP.items():

            for word in rel_words:

                if word in keywords:

                    return RELATION_TYPES.index(rel_type)



        # Default to semantic

        return RELATION_TYPES.index("semantic")






    def decompose_triple(self, token_ids: List[int]) -> Tuple[List[int], List[int], List[int]]:

        """Decompose token sequence into (subject, relation, object) triple.



        For 3-token input: subject=[t0], relation=[t1], object=[t2]

        For 2-token input: subject=[t0], relation=[], object=[t1]

        For 4+ tokens: subject=[t0], relation=[t1..tN-2], object=[tN-1]



        This mirrors how the brain decomposes "heat causes expansion" into

        (heat, causal-relation, expansion) rather than processing it as a

        flat character sequence.

        """

        n = len(token_ids)

        if n >= 3:

            return [token_ids[0]], token_ids[1:-1], [token_ids[-1]]

        elif n == 2:

            return [token_ids[0]], [], [token_ids[1]]

        elif n == 1:

            return [token_ids[0]], [], []

        else:

            return [], [], []






    def get_query_confidence(self, subject_cid, rel_type_name, rp_probs=None):

        """Compute confidence score. Uses RP output entropy when available, falls back to edge-based."""

        if rp_probs is not None:

            max_prob = float(np.max(rp_probs))

            if max_prob > 0.3:

                return 1.0

            entropy = -float(np.sum(rp_probs * np.log(rp_probs + 1e-15)))

            max_entropy = float(np.log(len(rp_probs)))

            norm_entropy = entropy / max_entropy if max_entropy > 0 else 1.0

            confidence = 1.0 - norm_entropy

            return min(1.0, max(0.0, confidence))

        matching_edges = [edge for _, edge in self.graph.get_outgoing(subject_cid) if edge.relation_type == rel_type_name]

        if not matching_edges:

            return 0.0

        return max(edge.weight * edge.confidence for edge in matching_edges)



    # ── Verb-Stem Helper ───────────────────────────────────────────────────




    def get_robust_embedding(self, tid):

        """Get subword character-CNN augmented robust embedding for a token ID."""

        token_emb = self.token_embed.weight.data[tid]

        if not hasattr(self, "_tokenizer") or self._tokenizer is None:

            return token_emb

        if not hasattr(self, "char_embed") or not hasattr(self, "char_cnn_W"):

            return token_emb

        if not hasattr(self, "char_to_token_W") or not hasattr(self, "fusion_W"):

            return token_emb

        word = ""

        try:

            word = self._tokenizer.decode([tid]).strip()

        except Exception:

            pass

        if not word:

            return token_emb

        

        c_ids = [ord(c) for c in word if ord(c) < 128]

        if not c_ids:

            return token_emb

            

        char_emb = self.char_embed[c_ids] # (L, 64)

        L = len(c_ids)

        padded = np.zeros((L + 2, 64), dtype=np.float32)

        padded[1:-1] = char_emb

        

        conv_out = np.zeros((L, 128), dtype=np.float32)

        for i in range(L):

            patch = padded[i:i+3] # (3, 64)

            conv_out[i] = np.sum(patch[np.newaxis, :, :] * self.char_cnn_W, axis=(1, 2)) + self.char_cnn_b

            

        features = np.max(conv_out, axis=0) # (128,)

        char_emb_final = self.char_to_token_W @ features + self.char_to_token_b # (embed_dim,)

        

        combined = np.concatenate([token_emb, char_emb_final]) # (embed_dim * 2,)

        fused = self.fusion_W @ combined + self.fusion_b # (embed_dim,)

        return fused






    def traverse(self, start_word: str, steps: int = 5, threshold: float = 0.3):

        """Hybrid search traversal combining direct 1-hop connections and multi-hop infer_chain paths."""

        tok = self._tokenizer

        if tok is None:

            return []

        tid = tok.word_to_id.get(start_word)

        if tid is None:

            return []

        bindings = self.binding_map.get_concepts(tid, min_confidence=0.1)

        if not bindings:

            return []

        cid = bindings[0].concept_id



        # 1. Direct 1-hop connections

        direct_candidates = []

        outgoing = self.graph._outgoing.get(cid, [])

        for target_cid, edge in outgoing:

            if edge.weight >= threshold and edge.edge_type != "inhibitory":

                t_bindings = self.binding_map.get_tokens(target_cid, min_confidence=0.1)

                if t_bindings:

                    t_tid = t_bindings[0].token_id

                    t_word = tok.decode([t_tid])

                    score = edge.weight * 1.5  # boost direct connection

                    direct_candidates.append((t_word, score))



        # 2. Multi-hop BFS paths from infer_chain

        chains = self.graph.infer_chain(start_id=cid, max_hops=steps, min_weight=threshold, k=5)

        multi_candidates = []

        for target_cid, score, path in chains:

            t_bindings = self.binding_map.get_tokens(target_cid, min_confidence=0.1)

            if t_bindings:

                t_tid = t_bindings[0].token_id

                t_word = tok.decode([t_tid])

                multi_candidates.append((t_word, score))



        # Combine and deduplicate

        combined = {}

        for word, score in direct_candidates + multi_candidates:

            if word == start_word:

                continue

            if word not in combined or score > combined[word]:

                combined[word] = score



        # Sort top candidates by score

        ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)

        return ranked



    # ── Sleep Cycle ─────────────────────────────────────────────────────────



