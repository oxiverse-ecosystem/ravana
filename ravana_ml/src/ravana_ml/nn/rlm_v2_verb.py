"""
Mixin: VerbMixin — rlm_v2_verb methods for RLMv2.

Auto-extracted from rlm_v2.py. Edit in the source or directly here.
"""
import numpy as np
from typing import Optional, List, Tuple, Dict, Set, Any


class VerbMixin:
    """Mixin providing rlm_v2_verb methods for RLMv2."""


    def _accumulate_verb_offset(self, subject_tid: int, target_tid: int, verb_word: str, domain_id: int = None):

        """Accumulate offset = target_embed - subject_embed for a verb stem.


        Called during learn() for each training triple.
        Offsets are finalized into _verb_offsets by _compute_verb_offsets().
        Now tracks domain_id to compute domain-specific verb offsets.
        Also stores individual offsets for variance computation.
        """

        if not verb_word or not self.use_verb_offset:

            return
        stem = self._verb_stem(verb_word)
        subject_embed = self.token_embed.weight.data[subject_tid]
        target_embed = self.token_embed.weight.data[target_tid]
        offset = target_embed - subject_embed
        # Use current domain if not provided

        if domain_id is None:

            domain_id = self.current_domain_id if self.current_domain_id is not None else 0
        self._verb_accum_buffer.append((stem, offset.copy(), domain_id))
    



    def _compute_verb_offsets(self):
        """Finalize verb offsets by averaging accumulated (target - subject) vectors.


        Now computes DOMAIN-SPECIFIC verb offsets. 
        Shape: _verb_offsets[domain_id][verb_stem] = offset_vector
        Also computes variance for each verb stem.
        Shape: _verb_offset_variance[domain_id][verb_stem] = variance_vector


        Should be called after training is complete (before evaluation).
        Unseen verbs at inference fall back to bilinear W_rel.
        """

        if not self.use_verb_offset:

            return
        # Group by (domain_id, verb stem) - collect all offset vectors for variance
        offset_lists = {}
        for stem, offset, domain_id in self._verb_accum_buffer:

            key = (domain_id, stem)
            if key not in offset_lists:

                offset_lists[key] = []
            offset_lists[key].append(offset)
        # Compute averages and variances
        self._verb_offsets = {}       # domain_id -> {stem: offset}
        self._verb_offset_count = {}  # domain_id -> {stem: count}
        self._verb_offset_variance = {}  # domain_id -> {stem: variance_vector}

        for (domain_id, stem), offsets_list in offset_lists.items():

            if len(offsets_list) > 0:
                offsets_array = np.stack(offsets_list)  # (n_samples, embed_dim)
                # Mean offset
                mean_offset = np.mean(offsets_array, axis=0)
                # Variance (per dimension, then average)
                var_per_dim = np.var(offsets_array, axis=0)
                mean_variance = float(np.mean(var_per_dim))
                # Normalize to prevent embedding-space drift
                norm = np.linalg.norm(mean_offset)
                if norm > 0:

                    mean_offset = mean_offset / norm * min(norm, 5.0)  # cap magnitude at 5
                if domain_id not in self._verb_offsets:

                    self._verb_offsets[domain_id] = {}
                    self._verb_offset_count[domain_id] = {}
                    self._verb_offset_variance[domain_id] = {}
                self._verb_offsets[domain_id][stem] = mean_offset
                self._verb_offset_count[domain_id][stem] = len(offsets_list)
                self._verb_offset_variance[domain_id][stem] = mean_variance

        # Print summary
        total_offsets = sum(len(v) for v in self._verb_offsets.values())
        print(f"  [Verb Offset] Computed {total_offsets} verb offsets from {len(self._verb_accum_buffer)} training pairs")

        for d in sorted(self._verb_offsets.keys()):

            for stem, cnt in sorted(self._verb_offset_count[d].items(), key=lambda x: -x[1])[:3]:
                var = self._verb_offset_variance[d].get(stem, 0.0)
                print(f"    Domain {d}: '{stem}': {cnt} examples, var={var:.4f}")


    def _cluster_verb_offsets(self, similarity_threshold: float = 0.85):
        """Cluster semantically similar verb stems by offset cosine similarity.
        
        During sleep, merge near-identical verb offsets (e.g., 'causes' and 'produces')
        to enable cross-verb generalization.
        
        Args:
            similarity_threshold: Minimum cosine similarity to merge verb stems (default 0.85)
        """
        if not self._verb_offsets:
            return
        
        for domain_id in list(self._verb_offsets.keys()):
            stems = list(self._verb_offsets[domain_id].keys())
            if len(stems) < 2:
                continue
            
            # Compute pairwise cosine similarities
            merged = set()
            for i, stem_i in enumerate(stems):
                if stem_i in merged:
                    continue
                offset_i = self._verb_offsets[domain_id][stem_i]
                norm_i = np.linalg.norm(offset_i)
                if norm_i == 0:
                    continue
                offset_i_norm = offset_i / norm_i
                
                cluster = [stem_i]
                total_count = self._verb_offset_count[domain_id][stem_i]
                weighted_sum = offset_i * total_count
                total_variance = self._verb_offset_variance[domain_id][stem_i] * total_count
                
                for j, stem_j in enumerate(stems):
                    if i == j or stem_j in merged:
                        continue
                    offset_j = self._verb_offsets[domain_id][stem_j]
                    norm_j = np.linalg.norm(offset_j)
                    if norm_j == 0:
                        continue
                    offset_j_norm = offset_j / norm_j
                    
                    sim = float(np.dot(offset_i_norm, offset_j_norm))
                    if sim > similarity_threshold:
                        cluster.append(stem_j)
                        cnt_j = self._verb_offset_count[domain_id][stem_j]
                        total_count += cnt_j
                        weighted_sum += offset_j * cnt_j
                        total_variance += self._verb_offset_variance[domain_id][stem_j] * cnt_j
                
                if len(cluster) > 1:
                    # Merge cluster into first stem
                    merged_offset = weighted_sum / total_count
                    merged_var = total_variance / total_count
                    
                    # Normalize
                    norm = np.linalg.norm(merged_offset)
                    if norm > 0:
                        merged_offset = merged_offset / norm * min(norm, 5.0)
                    
                    # Keep the most frequent stem as canonical
                    canonical = max(cluster, key=lambda s: self._verb_offset_count[domain_id][s])
                    
                    self._verb_offsets[domain_id][canonical] = merged_offset
                    self._verb_offset_count[domain_id][canonical] = total_count
                    self._verb_offset_variance[domain_id][canonical] = merged_var
                    
                    # Remove others
                    for s in cluster:
                        if s != canonical:
                            merged.add(s)
                            if s in self._verb_offsets[domain_id]:
                                del self._verb_offsets[domain_id][s]
                                del self._verb_offset_count[domain_id][s]
                                del self._verb_offset_variance[domain_id][s]
                    
                    print(f"  [Verb Offset] Merged cluster in domain {domain_id}: {cluster} -> '{canonical}' (sim={sim:.3f})")


    def _rp_forward_verb_offset(self, subject_tid: int, verb_word: str, return_count: bool = False):

        """Predict using verb-stem offset arithmetic with confidence-weighted blending.


        Uses DOMAIN-SPECIFIC verb offsets: offset(verb) = avg(target - subject) 
        for that verb in the current domain.


        NOW supports ALL verbs (not just a whitelist). For rare verbs with few
        training examples, the offset is less reliable; caller blends with W_rel.
        
        Blending now considers both count (logistic) AND variance (inverse reliability):
        - High count + low variance -> strong offset weight
        - Low count OR high variance -> more W_rel weight

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
        
        # Get variance for uncertainty-aware blending
        variance = 0.0
        if domain_id in self._verb_offset_variance and stem in self._verb_offset_variance[domain_id]:
            variance = self._verb_offset_variance[domain_id][stem]
        

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
            return (logits, count, variance) if return_count else logits
        return (None, 0, variance) if return_count else None


    def _rp_forward_verb_offset_from_adapted(self, adapted_source_embed: np.ndarray, verb_word: str) -> Optional[np.ndarray]:

        """Predict using verb-stem offset arithmetic from pre-adapted source embeded.
        
        
        used when entity adapter has already been applied to source_embed.
        """

        if not verb_word or not self.use_verb_offset:

            return None
        stem = self._verb_stem(verb_word)
        if stem not in self._verb_offsets:

            return None
        

        offset = self._verb_offsets[stem]
        predicted = adapted_source_embed + offset
        


        # Cosine similarity against all token embeddings

        token_embeds = self.token_embed.weight.data  # (vocab_size, embed_dim)
        token_norms = np.linalg.norm(token_embeds, axis=1)
        pred_norm = np.linalg.norm(predicted)
        


        if pred_norm > 0 and np.any(token_norms > 0):

            valid_tok = token_norms > 0
            normed_tok = token_embeds.copy()
            normed_tok[valid_tok] /= token_norms[valid_tok, np.newaxis]
            logits = (predicted / pred_norm) @ normed_tok.T  # cosine similarities
            # Note: we don't suppress subject token here since we don't have subject_tid

            # Scale up to make softmax meaningful

            logits *= 10.0
            return logits
        return None



    @staticmethod

    def _verb_stem(word: str) -> str:

        """Extract verb stem by stripping common suffixes.
        


        'causes' -> 'caus', 'freezes' -> 'freez', 'produces' -> 'produc',
        'makes' -> 'make', 'melts' -> 'melt', 'is' -> 'is'
        Stemming preserves enough information to distinguish verbs that
        map to the same relation type but have different semantics.
        """

        w = word.lower().strip()
        # Handle short words (2+ chars for stem)

        if len(w) <= 3:

            return w
        # Strip common suffixes

        for suffix in ['ing', 'ed', 'es', 's', 'd']:

            if w.endswith(suffix) and len(w) > len(suffix) + 1:

                w = w[:-len(suffix)]
                break
        return w

