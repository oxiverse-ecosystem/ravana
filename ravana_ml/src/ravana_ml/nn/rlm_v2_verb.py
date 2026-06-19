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

        self._verb_accum_buffer.append((stem, offset, domain_id))

    


    

    def _compute_verb_offsets(self):

        """Finalize verb offsets by averaging accumulated (target - subject) vectors.



        Now computes DOMAIN-SPECIFIC verb offsets. 

        Shape: _verb_offsets[domain_id][verb_stem] = offset_vector



        Should be called after training is complete (before evaluation).

        Unseen verbs at inference fall back to bilinear W_rel.

        """

        if not self.use_verb_offset:

            return

        # Group by (domain_id, verb stem)

        sums = {}

        counts = {}

        for stem, offset, domain_id in self._verb_accum_buffer:

            key = (domain_id, stem)

            if key not in sums:

                sums[key] = np.zeros_like(offset)

                counts[key] = 0

            sums[key] += offset

            counts[key] += 1

        # Compute averages

        self._verb_offsets = {}  # domain_id -> {stem: offset}

        self._verb_offset_count = {}  # domain_id -> {stem: count}

        for (domain_id, stem), total in sums.items():

            if counts[(domain_id, stem)] > 0:

                offset = total / counts[(domain_id, stem)]

                # Normalize to prevent embedding-space drift

                norm = np.linalg.norm(offset)

                if norm > 0:

                    offset = offset / norm * min(norm, 5.0)  # cap magnitude at 5

                if domain_id not in self._verb_offsets:

                    self._verb_offsets[domain_id] = {}

                    self._verb_offset_count[domain_id] = {}

                self._verb_offsets[domain_id][stem] = offset

                self._verb_offset_count[domain_id][stem] = counts[(domain_id, stem)]

        # Print summary

        total_offsets = sum(len(v) for v in self._verb_offsets.values())

        print(f"  [Verb Offset] Computed {total_offsets} verb offsets from {len(self._verb_accum_buffer)} training pairs")

        for d in sorted(self._verb_offsets.keys()):

            for stem, cnt in sorted(self._verb_offset_count[d].items(), key=lambda x: -x[1])[:3]:

                print(f"    Domain {d}: '{stem}': {cnt} examples")






    def _rp_forward_verb_offset(self, subject_tid: int, verb_word: str, return_count: bool = False):

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

        return (None, 0) if return_count else None






    def _rp_forward_verb_offset_from_adapted(self, adapted_source_embed: np.ndarray, verb_word: str) -> Optional[np.ndarray]:

        """Predict using verb-stem offset arithmetic from pre-adapted source embedding.

        

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



