"""
Mixin: SleepMixin — rlm_v2_sleep methods for RLMv2.

Auto-extracted from rlm_v2.py. Edit in the source or directly here.
"""
import numpy as np
import time
from typing import Optional, List, Tuple, Dict, Set, Any
from ..embedder import LearnedEmbedder
from collections import defaultdict


class SleepMixin:
    """Mixin providing rlm_v2_sleep methods for RLMv2."""



    def _bridge_memories_to_graph(self):

        for cid, mem in self._semantic_memories.items():

            node = self.graph.get_node(cid)

            if not node or mem['strength'] <= 0.2:

                continue

            try:

                similar = self.graph.find_similar(node.vector, k=10)

            except Exception:

                continue

            for target_cid, sim in similar:

                if target_cid == cid or sim < 0.2:

                    continue

                mem2 = self._semantic_memories.get(target_cid)

                if not mem2 or mem2['strength'] <= 0.2:

                    continue

                edge = self.graph.get_edge(cid, target_cid)

                if edge:

                    edge.weight += 0.01 * mem['strength'] * mem2['strength'] * sim

                    edge.confidence = min(1.0, edge.confidence + 0.005 * sim)

                elif mem['strength'] > 0.5 and mem2['strength'] > 0.5 and sim > 0.4:

                    edge = self.graph.add_edge(cid, target_cid, weight=0.1 * sim)

                    edge.confidence = 0.3






    def _consolidate_episodic_to_semantic(self):

        for ep in self._episodic_buffer:

            if ep['correct'] and ep['error'] < 0.3:

                importance = ep.get('importance', 0.5)

                is_novel = ep.get('consolidation_state', 'fresh') == 'fresh'

                strength_delta = 0.05 * importance * (1.5 if is_novel else 1.0)

                for cid in ep['concepts']:

                    if cid in self._semantic_memories:

                        self._semantic_memories[cid]['strength'] = min(1.0,

                            self._semantic_memories[cid]['strength'] + strength_delta)

                        self._semantic_memories[cid]['access_count'] += 1

                    else:

                        if len(self._semantic_memories) < self._semantic_memory_max:

                            self._semantic_memories[cid] = {

                                'strength': 0.3 * importance,

                                'access_count': 1,

                                'last_access': time.time(),

                            }






    def _decay_semantic_memories(self):

        now = time.time()

        to_remove = []

        for cid, mem in self._semantic_memories.items():

            dt = now - mem['last_access']

            access_factor = 0.5 + mem['access_count'] * 0.1

            retention = mem['strength'] * np.exp(-0.001 * dt / access_factor)

            mem['strength'] = retention

            if retention < 0.05:

                to_remove.append(cid)

        for cid in to_remove:

            del self._semantic_memories[cid]






    def _evict_lowest_salience(self, buffer: List[Dict]) -> List[Dict]:

        now = time.time()

        min_score = float('inf')

        min_idx = 0

        for i, ep in enumerate(buffer):

            age = now - ep.get('timestamp', now)

            recency = 1.0 / (1.0 + age * 0.01)

            importance = ep.get('importance', 0.5)

            error_signal = min(1.0, ep.get('error', 0.5))

            score = importance * 0.4 + recency * 0.3 + error_signal * 0.3

            if score < min_score:

                min_score = score

                min_idx = i

        buffer.pop(min_idx)

        return buffer






    def _prune_phantom_nodes(self, min_degree: int = 2):

        """Remove concept nodes without token bindings that are structural artifacts.



        Phantom nodes (token_id == None) created during tokenizer vocabulary expansion

        but never bound to actual tokens act as noise hubs during multi-seed traversal.

        Only remove if degree < min_degree to preserve legitimate hub concepts.

        """

        nodes_to_remove = []

        for nid, node in self.graph.nodes.items():

            bindings = self.binding_map.get_tokens(nid, min_confidence=0.0)

            has_token_binding = len(bindings) > 0

            if not has_token_binding:

                # Count edges

                out_degree = len(self.graph._outgoing.get(nid, []))

                in_degree = len(self.graph._incoming.get(nid, []))

                total_degree = out_degree + in_degree

                if total_degree < min_degree:

                    nodes_to_remove.append(nid)



        for nid in nodes_to_remove:

            # Remove all edges connected to this node

            for _, edge in self.graph._outgoing.get(nid, []):

                self.graph.remove_edge(nid, edge.target)

            for edge, _ in self.graph._incoming.get(nid, []):

                self.graph.remove_edge(edge.source, nid)

            # Remove the node

            if nid in self.graph.nodes:

                del self.graph.nodes[nid]

            if nid in self.graph._outgoing:

                del self.graph._outgoing[nid]

            if nid in self.graph._incoming:

                del self.graph._incoming[nid]



        if nodes_to_remove:

            self._invalidate_caches()

            print(f"[Sleep] Pruned {len(nodes_to_remove)} phantom nodes (degree < {min_degree})")






    def _regulate_cognitive_state(self):

        self.currencies.regulate()






    def _replay_old_memories(self, n_samples: int = 20):

        if not self._replay_buffer:

            return

        n = min(n_samples, len(self._replay_buffer))

        rng = np.random.RandomState()

        indices = rng.choice(len(self._replay_buffer), size=n, replace=False)

        for idx in indices:

            input_ids, target_ids = self._replay_buffer[idx]

            self.learn(input_ids, target_ids)



    # ── Dimension Bridging ──────────────────────────────────────────────────






    def _store_episode(self, error: float, is_correct: bool, domain: Optional[str] = None):

        importance = 1.0 - min(1.0, error)

        if is_correct:

            importance *= 0.8

        else:

            importance = min(1.0, importance + 0.3)



        episode = {

            'vector': self._last_hidden_state.copy() if self._last_hidden_state is not None else None,

            'concepts': list(self._last_predicted_concepts),

            'error': error,

            'correct': is_correct,

            'valence': self.valence,

            'arousal': self.arousal,

            'timestamp': time.time(),

            'importance': importance,

            'consolidation_state': 'fresh',

            'access_count': 0,

            'domain': domain,

        }

        self._episodic_buffer.append(episode)

        if len(self._episodic_buffer) > self._episodic_buffer_max:

            self._evict_lowest_salience(self._episodic_buffer)






    def align_encoder_to_graph(self, validation_queries: Optional[List[Dict[str, Any]]] = None):

        """Perform offline graph-aware contrastive alignment to sync embeddings with graph topology.

        

        Implements Bridge Alignment: combines graph-topology pairs with cross-domain semantic pairs

        and validation query analogy pairs to provide training signal for hard/out-of-distribution cases.

        """

        tok = self._tokenizer

        if tok is None or not self.graph.nodes:

            return

           

        # Save initial checkpoint

        checkpoint_W1 = self._enc_W1.copy()

        checkpoint_b1 = self._enc_b1.copy()

        checkpoint_W2 = self._enc_W2.copy()

        checkpoint_b2 = self._enc_b2.copy()

        

        # Track peak performance starting from initial checkpoint

        peak_encoder_state = (checkpoint_W1.copy(), checkpoint_b1.copy(), checkpoint_W2.copy(), checkpoint_b2.copy())

        

        # Calculate pre-alignment validation accuracy

        pre_acc = 0.0

        if validation_queries:

            successes = 0

            for tc in validation_queries:

                q = tc["query"]

                expected = tc["expected"]

                res, _ = self.retrieval_v2_multi_seed(q, k_neighbors=5, gate_mode="margin_multi")

                rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), 99)

                if rank <= 10:

                    successes += 1

            pre_acc = successes / len(validation_queries)

        peak_validation_acc = pre_acc

        

        # ── Pair Extraction ──

        # 1. Graph topology pairs: strong edges (weight >= threshold) across all relation types

        positive_pairs = []

        seen_pairs = set()  # for deduplication

        

        for u in self.graph.nodes:

            tokens_u = self.binding_map.get_tokens(u, 0.0)

            if not tokens_u:

                continue

            tid_u = tokens_u[0].token_id

            if tid_u >= self.token_embed.weight.data.shape[0]:

                continue

            word_u = tok.decode([tid_u])

            

            for v, edge in self.graph.get_outgoing(u):

                if edge.weight >= self.alignment_edge_threshold:

                    tokens_v = self.binding_map.get_tokens(v, 0.0)

                    if not tokens_v:

                        continue

                    tid_v = tokens_v[0].token_id

                    if tid_v >= self.token_embed.weight.data.shape[0]:

                        continue

                    word_v = tok.decode([tid_v])

                    pair_key = (word_u, word_v)

                    if pair_key not in seen_pairs:

                        seen_pairs.add(pair_key)

                        positive_pairs.append((word_u, word_v, u, v))

        

        # 2. Bridge Alignment: cross-domain semantic/analogy pairs from self.semantic_pairs

        for word_a, word_b in getattr(self, "semantic_pairs", []):

            tid_a = tok.word_to_id.get(word_a)

            tid_b = tok.word_to_id.get(word_b)

            if tid_a is None or tid_b is None:

                continue

            if tid_a >= self.token_embed.weight.data.shape[0] or tid_b >= self.token_embed.weight.data.shape[0]:

                continue

            # Get concept IDs if they exist

            bindings_a = self.binding_map.get_concepts(tid_a, min_confidence=0.1)

            bindings_b = self.binding_map.get_concepts(tid_b, min_confidence=0.1)

            cid_a = bindings_a[0].concept_id if bindings_a else None

            cid_b = bindings_b[0].concept_id if bindings_b else None

            pair_key = (word_a, word_b)

            if pair_key not in seen_pairs:

                seen_pairs.add(pair_key)

                positive_pairs.append((word_a, word_b, cid_a, cid_b))

        

        # 3. Bridge Alignment: validation query analogy pairs (query_word -> expected_seed)

        # These are USED ONLY for validation/early stopping, NOT for training.

        # This prevents overfitting to specific validation queries.

        validation_pairs = []

        if validation_queries:

            for tc in validation_queries:

                q = tc["query"]

                expected_seed = tc.get("expected_seed")

                if not expected_seed:

                    continue

                query_word = q.split()[0] if q else None

                if not query_word:

                    continue

                tid_a = tok.word_to_id.get(query_word)

                tid_b = tok.word_to_id.get(expected_seed)

                if tid_a is None or tid_b is None:

                    continue

                if tid_a >= self.token_embed.weight.data.shape[0] or tid_b >= self.token_embed.weight.data.shape[0]:

                    continue

                bindings_a = self.binding_map.get_concepts(tid_a, min_confidence=0.1)

                bindings_b = self.binding_map.get_concepts(tid_b, min_confidence=0.1)

                cid_a = bindings_a[0].concept_id if bindings_a else None

                cid_b = bindings_b[0].concept_id if bindings_b else None

                pair_key = (query_word, expected_seed)

                if pair_key not in seen_pairs:

                    seen_pairs.add(pair_key)

                    validation_pairs.append((query_word, expected_seed, cid_a, cid_b))

                    

        if not positive_pairs:

            print(f"[Debug Alignment] No positive pairs found! Graph has {len(self.graph.nodes)} nodes and {len(self.graph.edges)} edges.")

            if self.graph.edges:

                max_w = max(e.weight for e in self.graph.edges.values())

                print(f"[Debug Alignment] Max edge weight in graph = {max_w:.4f}")

            return

            

        vocab_words = list(tok.word_to_id.keys())

        

        # Manifold Regularization: Cache original projections before alignment starts

        original_latents = {}

        lambda_recon = getattr(self, "lambda_recon", 0.0)

        if lambda_recon > 0.0:

            for word, tid in tok.word_to_id.items():

                if tid < self.token_embed.weight.data.shape[0]:

                    embed = self.token_embed.weight.data[tid]

                    lat, *_ = self._encoder_forward_full(embed)

                    original_latents[tid] = lat.copy()



        # ── Alignment Loop ──

        lr = getattr(self, "alignment_lr", 0.005)

        margin = getattr(self, "alignment_margin", 0.15)

        lambda_anchor = getattr(self, "lambda_anchor", 0.05)

        max_epochs = getattr(self, "max_alignment_epochs", 10)



        patience_counter = 0



        min_epochs = 5  # minimum epochs before early stopping can trigger



        for epoch in range(1, max_epochs + 1):

            # Accumulators for this epoch

            d_con_W1 = np.zeros_like(self._enc_W1)

            d_con_b1 = np.zeros_like(self._enc_b1)

            d_con_W2 = np.zeros_like(self._enc_W2)

            d_con_b2 = np.zeros_like(self._enc_b2)

            

            total_loss = 0.0

            

            for word_a, word_b, cid_a, cid_b in positive_pairs:

                tid_a = tok.word_to_id.get(word_a)

                tid_b = tok.word_to_id.get(word_b)

                if tid_a is None or tid_b is None:

                    continue

                    

                embed_a = self.token_embed.weight.data[tid_a]

                embed_b = self.token_embed.weight.data[tid_b]

                

                lat_a, z1_a, h1_a, z2_a = self._encoder_forward_full(embed_a)

                lat_b, z1_b, h1_b, z2_b = self._encoder_forward_full(embed_b)

                

                norm_a = np.linalg.norm(lat_a)

                norm_b = np.linalg.norm(lat_b)

                unit_a = lat_a / (norm_a + 1e-15) if norm_a > 0 else lat_a

                unit_b = lat_b / (norm_b + 1e-15) if norm_b > 0 else lat_b

                

                s_p = float(np.dot(unit_a, unit_b))

                

                # Dynamic Stratified Negative Sampling:

                # 3 Random + 2 Hard Negatives

                neg_candidates = []

                scored_all = []

                for word_neg, tid_neg in tok.word_to_id.items():

                    if word_neg == word_a or word_neg == word_b:

                        continue

                    if tid_neg >= self.token_embed.weight.data.shape[0]:

                        continue

                    bindings = self.binding_map.get_concepts(tid_neg, min_confidence=0.1)

                    if not bindings:

                        continue

                    cid_neg = bindings[0].concept_id

                    # Robust negative sampling: skip graph edge check if cid_a is None (e.g., for query words without concept node)

                    if cid_a is not None and self.graph.get_edge(cid_a, cid_neg) is not None:

                        continue

                    embed_neg = self.token_embed.weight.data[tid_neg]

                    lat_neg, *_ = self._encoder_forward_full(embed_neg)

                    norm_neg = np.linalg.norm(lat_neg)

                    unit_neg = lat_neg / (norm_neg + 1e-15) if norm_neg > 0 else lat_neg

                    sim_neg = float(np.dot(unit_a, unit_neg))

                    scored_all.append((word_neg, sim_neg, tid_neg))

                    

                scored_all.sort(key=lambda x: x[1], reverse=True)

                for item in scored_all[:2]:

                    neg_candidates.append(item[2])

                    

                if vocab_words:

                    random_choices = np.random.choice(vocab_words, size=min(10, len(vocab_words)), replace=False)

                    for r_word in random_choices:

                        if len(neg_candidates) >= 5:

                            break

                        if r_word == word_a or r_word == word_b:

                            continue

                        tid_neg = tok.word_to_id[r_word]

                        if tid_neg >= self.token_embed.weight.data.shape[0]:

                            continue

                        bindings = self.binding_map.get_concepts(tid_neg, min_confidence=0.1)

                        if bindings:

                            cid_neg = bindings[0].concept_id

                            # Robust negative sampling: skip graph edge check if cid_a is None

                            if cid_a is not None and self.graph.get_edge(cid_a, cid_neg) is not None:

                                continue

                        if tid_neg not in neg_candidates:

                            neg_candidates.append(tid_neg)

                            

                for tid_neg in neg_candidates:

                    embed_neg = self.token_embed.weight.data[tid_neg]

                    lat_neg, z1_neg, h1_neg, z2_neg = self._encoder_forward_full(embed_neg)

                    norm_neg = np.linalg.norm(lat_neg)

                    unit_neg = lat_neg / (norm_neg + 1e-15) if norm_neg > 0 else lat_neg

                    

                    s_n = float(np.dot(unit_a, unit_neg))

                    loss_val = max(0.0, s_n - s_p + margin)

                    if loss_val <= 0.0:

                        continue

                        

                    total_loss += loss_val

                    

                    d_sp_d_lata = (unit_b - s_p * unit_a) / (norm_a + 1e-15)

                    d_sp_d_latb = (unit_a - s_p * unit_b) / (norm_b + 1e-15)

                    d_sn_d_lata = (unit_neg - s_n * unit_a) / (norm_a + 1e-15)

                    d_sn_d_latneg = (unit_a - s_n * unit_neg) / (norm_neg + 1e-15)

                    

                    d_lat_a = -1.0 * d_sp_d_lata + 1.0 * d_sn_d_lata

                    d_lat_b = -1.0 * d_sp_d_latb

                    d_lat_neg = 1.0 * d_sn_d_latneg

                    

                    dW1_a, db1_a, dW2_a, db2_a = self._encoder_backward(

                        embed_a[np.newaxis, :], z1_a[np.newaxis, :], h1_a[np.newaxis, :], z2_a[np.newaxis, :],

                        lat_a[np.newaxis, :], d_lat_a[np.newaxis, :]

                    )

                    dW1_b, db1_b, dW2_b, db2_b = self._encoder_backward(

                        embed_b[np.newaxis, :], z1_b[np.newaxis, :], h1_b[np.newaxis, :], z2_b[np.newaxis, :],

                        lat_b[np.newaxis, :], d_lat_b[np.newaxis, :]

                    )

                    dW1_neg, db1_neg, dW2_neg, db2_neg = self._encoder_backward(

                        embed_neg[np.newaxis, :], z1_neg[np.newaxis, :], h1_neg[np.newaxis, :], z2_neg[np.newaxis, :],

                        lat_neg[np.newaxis, :], d_lat_neg[np.newaxis, :]

                    )

                    

                    d_con_W1 += dW1_a + dW1_b + dW1_neg

                    d_con_b1 += db1_a + db1_b + db1_neg

                    d_con_W2 += dW2_a + dW2_b + dW2_neg

                    d_con_b2 += db2_a + db2_b + db2_neg

            

            # Calculate and accumulate reconstruction penalty (Manifold Regularization)

            if lambda_recon > 0.0 and original_latents:

                d_recon_W1 = np.zeros_like(self._enc_W1)

                d_recon_b1 = np.zeros_like(self._enc_b1)

                d_recon_W2 = np.zeros_like(self._enc_W2)

                d_recon_b2 = np.zeros_like(self._enc_b2)

                

                for tid, lat_orig in original_latents.items():

                    embed = self.token_embed.weight.data[tid]

                    lat_curr, z1_curr, h1_curr, z2_curr = self._encoder_forward_full(embed)

                    

                    # MSE gradient: d_loss / d_lat = 2 * (lat_curr - lat_orig) / N

                    d_lat = (2.0 * (lat_curr - lat_orig)) / len(original_latents)

                    

                    dW1_r, db1_r, dW2_r, db2_r = self._encoder_backward(

                        embed[np.newaxis, :], z1_curr[np.newaxis, :], h1_curr[np.newaxis, :], z2_curr[np.newaxis, :],

                        lat_curr[np.newaxis, :], d_lat[np.newaxis, :]

                    )

                    d_recon_W1 += dW1_r

                    d_recon_b1 += db1_r

                    d_recon_W2 += dW2_r

                    d_recon_b2 += db2_r

                

                # Add scaled reconstruction gradients to accumulators

                d_con_W1 += lambda_recon * d_recon_W1

                d_con_b1 += lambda_recon * d_recon_b1

                d_con_W2 += lambda_recon * d_recon_W2

                d_con_b2 += lambda_recon * d_recon_b2



            d_anchor_W1 = 2.0 * lambda_anchor * (self._enc_W1 - checkpoint_W1)

            d_anchor_b1 = 2.0 * lambda_anchor * (self._enc_b1 - checkpoint_b1)

            d_anchor_W2 = 2.0 * lambda_anchor * (self._enc_W2 - checkpoint_W2)

            d_anchor_b2 = 2.0 * lambda_anchor * (self._enc_b2 - checkpoint_b2)

            

            d_total_W1 = d_con_W1 + d_anchor_W1

            d_total_b1 = d_con_b1 + d_anchor_b1

            d_total_W2 = d_con_W2 + d_anchor_W2

            d_total_b2 = d_con_b2 + d_anchor_b2

            

            self._enc_mW1 = self._rp_momentum * self._enc_mW1 - lr * d_total_W1

            self._enc_mb1 = self._rp_momentum * self._enc_mb1 - lr * d_total_b1

            self._enc_mW2 = self._rp_momentum * self._enc_mW2 - lr * d_total_W2

            self._enc_mb2 = self._rp_momentum * self._enc_mb2 - lr * d_total_b2

            

            self._enc_W1 += self._enc_mW1

            self._enc_b1 += self._enc_mb1

            self._enc_W2 += self._enc_mW2

            self._enc_b2 += self._enc_mb2

            

            self._token_embed_norms = None

            

            recall_5 = self.compute_neighbor_recall_at_5()



            if validation_queries:

                successes = 0

                for tc in validation_queries:

                    q = tc["query"]

                    expected = tc["expected"]

                    res, _ = self.retrieval_v2_multi_seed(q, k_neighbors=5, gate_mode="margin_multi")

                    rank = next((i+1 for i, x in enumerate(res) if x[0] == expected), 99)

                    if rank <= 10:

                        successes += 1

                validation_acc = successes / len(validation_queries)



                if validation_acc > peak_validation_acc:

                    peak_validation_acc = validation_acc

                    peak_encoder_state = (self._enc_W1.copy(), self._enc_b1.copy(), self._enc_W2.copy(), self._enc_b2.copy())

                    patience_counter = 0  # reset patience on improvement

                else:

                    patience_counter += 1

                    if epoch >= min_epochs and patience_counter >= 3:  # patience of 3 epochs after min_epochs

                        print(f"[Align] Early stopping at epoch {epoch} (patience exhausted, peak_acc={peak_validation_acc:.3f})")

                        break

                    

        if peak_encoder_state is not None:

            self._enc_W1, self._enc_b1, self._enc_W2, self._enc_b2 = peak_encoder_state

        else:

            self._enc_W1 = checkpoint_W1

            self._enc_b1 = checkpoint_b1

            self._enc_W2 = checkpoint_W2

            self._enc_b2 = checkpoint_b2

        self._token_embed_norms = None

        self._last_aligned_version = getattr(self.graph, 'version', 0)






    def buffer_experience(self, input_ids: np.ndarray, target_ids: np.ndarray,

                          domain: Optional[str] = None):

        entry = (input_ids.copy(), target_ids.copy())

        self._replay_buffer.append(entry)

        if domain is not None:

            if domain not in self._domain_memories:

                self._domain_memories[domain] = []

            self._domain_memories[domain].append(entry)

        if len(self._replay_buffer) > self._replay_buffer_max:

            self._replay_buffer = self._replay_buffer[-self._replay_buffer_max:]










    def compute_neighbor_recall_at_5(self) -> float:

        """Calculate the fraction of strong topological neighbors in the top-5 candidate seeds."""

        tok = self._tokenizer

        if tok is None or not self.graph.nodes:

            return 0.0

            

        recalls = []

        node_ids = list(self.graph.nodes.keys())

        if not node_ids:

            return 0.0

            

        # Pre-compute latent representations

        latents = {}

        for nid in node_ids:

            tokens = self.binding_map.get_tokens(nid, 0.0)

            if tokens:

                tid = tokens[0].token_id

                if tid < self.token_embed.weight.data.shape[0]:

                    emb = self.token_embed.weight.data[tid]

                    lat, *_ = self._encoder_forward_full(emb)

                    latents[nid] = lat



        for u in node_ids:

            strong_neighbors = [

                v for v, edge in self.graph.get_outgoing(u)

                if edge.weight >= self.alignment_edge_threshold

            ]

            if not strong_neighbors:

                continue

                

            if u not in latents:

                continue

            lat_u = latents[u]

            

            scores = []

            for v in node_ids:

                if v == u or v not in latents:

                    continue

                lat_v = latents[v]

                na = np.linalg.norm(lat_u)

                nb = np.linalg.norm(lat_v)

                sim = np.dot(lat_u, lat_v) / (na * nb + 1e-15) if na > 0 and nb > 0 else 0.0

                scores.append((v, sim))

                

            scores.sort(key=lambda x: x[1], reverse=True)

            top_5_cids = {item[0] for item in scores[:5]}

            

            hits = sum(1 for v in strong_neighbors if v in top_5_cids)

            recalls.append(hits / len(strong_neighbors))

            

        return float(np.mean(recalls)) if recalls else 1.0






    def end_wake_epoch(self, validation_queries: Optional[List[Dict[str, Any]]] = None):

        """Call at the end of each wake epoch to track sleep cadence.



        Increments wake_epochs_since_sleep and triggers sleep if

        sleep_every_n_wake_epochs threshold reached.

        """

        self.wake_epochs_since_sleep += 1

        if self.wake_epochs_since_sleep >= self.sleep_every_n_wake_epochs:

            self.sleep_cycle(validation_queries=validation_queries)






    def mark_alignment_needed(self):

        """Mark that encoder has changed and alignment is needed on next sleep."""

        self.alignment_needed = True










    def sleep_cycle(self, validation_queries: Optional[List[Dict[str, Any]]] = None,

                    force_alignment: bool = False):

        """Sleep cycle: consolidate triples, prune weak edges, replay important memories.



        This is the brain's "offline consolidation" - during sleep, the model:

        1. Replays important episodic triples (hippocampal replay)

        2. Consolidates edge weights (homeostatic downscaling)

        3. Prunes weak/unstable edges

        4. Dynamically aligns the encoder to the graph topology (if needed)

        5. Updates concept vectors (drift defense)

        6. Prunes phantom nodes (unbound concepts)



        Args:

            validation_queries: Held-out queries for validation during alignment

            force_alignment: If True, run alignment even if encoder hasn't changed

        """

        # Prevent recursive sleep cycle calls during replay or model training inside sleep

        if getattr(self, '_in_sleep_cycle', False):

            return

        self._in_sleep_cycle = True

        self._last_sleep_step = self._step_counter



        try:

            # ── Hippocampal replay ──

            # Replay the most recent/important triples

            if self._episodic_triples:

                n_replay = min(10, len(self._episodic_triples))

                replay_triples = self._episodic_triples[-n_replay:]



                for subj_cid, rel_idx, obj_cid, ts in replay_triples:

                    src = self.graph.get_node(subj_cid)

                    tgt = self.graph.get_node(obj_cid)

                    if src is None or tgt is None:

                        continue



                    edge = self.graph.get_edge(subj_cid, obj_cid)

                    if edge is None:

                        continue



                    # Strengthen edge through replay

                    edge.weight = min(1.0, edge.weight + 0.02)

                    edge.confidence = min(1.0, edge.confidence + 0.01)



                    # Hebbian co-activation replay

                    self.graph.hebbian_update(subj_cid, obj_cid, coactivation=0.5, lr=0.005)



            # ── Homeostatic downscaling ──

            # Normalize edge weights to prevent runaway strengthening

            # Budget increased from 3.0 to 5.0 - less aggressive, preserves learned edges

            self._normalize_outgoing_weights(budget=5.0)



            # ── Prune weak edges ──

            # Prune edges with weight below 0.1 to clear out spurious semantic edges.

            self._prune_weak_edges(threshold=0.1)



            # ── Anti-Hebbian Pruning: Remove polluted edges ──

            # Edges created during early training with different bindings that now

            # consistently predict incorrectly are weakened/removed.

            n_pruned = self._anti_hebbian_prune_polluted_edges()

            if n_pruned:

                print(f"[Sleep] Anti-Hebbian pruned {n_pruned} polluted edges")



            # ── Prune phantom nodes ──

            # Remove concept nodes that have no token binding (token_id == None)

            # and degree < 2 (isolated or single-edge artifacts from tokenizer expansion)

            self._prune_phantom_nodes(min_degree=2)



            # ── Representation Alignment ──

            # Align encoder representations to graph topology if needed or forced or graph changed

            graph_version = getattr(self.graph, 'version', 0)

            last_aligned = getattr(self, '_last_aligned_version', -1)

            if self.alignment_needed or force_alignment or graph_version > last_aligned:

                self.align_encoder_to_graph(validation_queries=validation_queries)

                self.alignment_needed = False  # reset flag after alignment



            # ── Drift defense ──

            # Pull concept vectors back toward their core vectors if they've drifted too far.

            # Threshold increased from 0.4 to 0.7 - allow more movement before correction.

            # Pull strength reduced from 0.1 to 0.05 - gentler correction.

            for nid, node in self.graph.nodes.items():

                drift = node.drift_magnitude

                if drift > 0.7:

                    # Pull back toward core vector (gentler)

                    pull = 0.05 * (node.core_vector - node.vector)

                    node.vector += pull

                    norm = np.linalg.norm(node.vector)

                    if norm > 0:

                        node.vector /= norm



            # ── Episodic -> semantic consolidation ──

            self._consolidate_episodic_to_semantic()



            # ── Semantic memory decay ──

            self._decay_semantic_memories()



            # ── Memory -> weights bridge ──

            self._bridge_memories_to_graph()



            # ── Interleaved named domain memories replay ──

            self._replay_old_memories(n_samples=self._replay_n_samples)



            # ── Cognitive currency consolidation (emotion, identity, meaning, sleep) ──

            self.currencies.consolidate_on_sleep()



            # ── Reset sleep pressure ──

            self._sleep_pressure = 0.0

            self.wake_epochs_since_sleep = 0

            self.sleep_cycles_completed += 1



            # ── Final self-regulation ──

            self._regulate_cognitive_state()



            # ── Reset activations ──

            for node in self.graph.nodes.values():

                node.activation = 0.0

                node.fatigue = 0.0

        finally:

            self._in_sleep_cycle = False



