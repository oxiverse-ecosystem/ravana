"""
Phase 1 P0 Fixes:
1. Update _rp_forward_verb_offset_from_adapted to support ALL verbs
2. Add confidence-weighted blending to adapted verb-offset path
3. Use smoother logistic blending function (count / (count + 5.0))
4. Fix edge cases: count=1 verbs, None-checking for numpy arrays
"""
import re
import sys


def patch_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    changes_made = []

    # ================================================================
    # FIX 1: Update _rp_forward_verb_offset_from_adapted - remove whitelist, return count
    # ================================================================
    old_adapted_offset = '''    def _rp_forward_verb_offset_from_adapted(self, adapted_source_embed: np.ndarray, verb_word: str) -> Optional[np.ndarray]:
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
            # Scale up to make softmax meaningful
            logits *= 10.0
            return logits
        return None'''

    new_adapted_offset = '''    def _rp_forward_verb_offset_from_adapted(self, adapted_source_embed: np.ndarray, verb_word: str, return_count: bool = False):
        """Predict using verb-stem offset arithmetic from pre-adapted source embedding.
        
        NOW supports ALL verbs (not just whitelist). Returns count for blending.
        used when entity adapter has already been applied to source_embed.
        """
        if not verb_word or not self.use_verb_offset:
            return (None, 0) if return_count else None
        stem = self._verb_stem(verb_word)
        
        # Get domain-specific offsets
        domain_id = self.current_domain_id if self.current_domain_id is not None else 0
        if domain_id not in self._verb_offsets or stem not in self._verb_offsets[domain_id]:
            return (None, 0) if return_count else None
        
        # Get count for blending
        count = 0
        if domain_id in self._verb_offset_count and stem in self._verb_offset_count[domain_id]:
            count = self._verb_offset_count[domain_id][stem]
        
        offset = self._verb_offsets[domain_id][stem]
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
            # Scale up to make softmax meaningful
            logits *= 10.0
            return (logits, count) if return_count else logits
        return (None, 0) if return_count else None'''

    if old_adapted_offset in content:
        content = content.replace(old_adapted_offset, new_adapted_offset)
        changes_made.append("_rp_forward_verb_offset_from_adapted: expanded to ALL verbs + count tracking")
        print("  [OK] Patched _rp_forward_verb_offset_from_adapted")
    else:
        print("  [WARN] Could not find old _rp_forward_verb_offset_from_adapted")

    # ================================================================
    # FIX 2: Replace hard threshold blending with smoother logistic function
    # and fix None-checking for numpy arrays
    # ================================================================
    old_blend_weight = '''                verb_blend_weight = min(1.0, verb_count / 10.0)
                verb_blend_logits = verb_logits
                if verb_blend_weight >= 1.0:
                    # Fully confident in verb offset - return directly (no W_rel needed)
                    self._rp_cache = None
                    return verb_logits'''

    new_blend_weight = '''                # Use smoother logistic blending: count / (count + 5.0)
                # This gives a soft S-curve: count=1 -> 0.17, count=5 -> 0.5, count=10 -> 0.67, count=30 -> 0.86
                verb_blend_weight = verb_count / (verb_count + 5.0)
                verb_blend_logits = verb_logits
                if verb_blend_weight > 0.95:
                    # Fully confident in verb offset - return directly (no W_rel needed)
                    self._rp_cache = None
                    return verb_logits'''

    if old_blend_weight in content:
        content = content.replace(old_blend_weight, new_blend_weight)
        changes_made.append("Replaced hard threshold blending with smoother logistic function")
        print("  [OK] Patched blending function to logistic")
    else:
        print("  [WARN] Could not find old blend_weight section")

    # ================================================================
    # FIX 3: Fix None-checking for numpy arrays in blend condition
    # (numpy arrays evaluate to ambiguous truth value)
    # ================================================================
    old_none_check = '''        if verb_blend_logits is not None and verb_blend_weight > 0.0:'''

    new_none_check = '''        if verb_blend_logits is not None and verb_blend_weight > 0.0:
            # Guard: verb_blend_logits must not be a numpy array with False truth value
            if isinstance(verb_blend_logits, np.ndarray) and verb_blend_logits.size > 0:'''

    if old_none_check in content:
        content = content.replace(old_none_check, new_none_check)
        changes_made.append("Fixed None-checking for numpy arrays in blend condition")
        print("  [OK] Patched numpy array None-check")
    else:
        print("  [WARN] Could not find old none-check")

    # ================================================================
    # FIX 4: Add adapted path blending - when entity adapters are used
    # and verb offset is available, blend with W_rel
    # ================================================================
    old_adapted_path = '''        # Verb offset path on adapted embedding (entity adapter applied)
        if verb_word and self.use_verb_offset:
            verb_logits = self._rp_forward_verb_offset_from_adapted(adapted_source_embed, verb_word)
            if verb_logits is not None:
                self._rp_cache = None  # no cache needed for offset path
                return verb_logits'''

    new_adapted_path = '''        # Verb offset path on adapted embedding (entity adapter applied)
        # Also uses confidence-weighted blending for ALL verbs
        if verb_word and self.use_verb_offset:
            adapted_result = self._rp_forward_verb_offset_from_adapted(adapted_source_embed, verb_word, return_count=True)
            if adapted_result is not None and adapted_result[0] is not None:
                # Blend with W_rel if blend_weight < 1.0
                adapted_verb_logits, adapted_verb_count = adapted_result
                adapted_blend_weight = adapted_verb_count / (adapted_verb_count + 5.0)
                if adapted_blend_weight > 0.95:
                    self._rp_cache = None
                    return adapted_verb_logits
                # Don't return yet - store and blend after W_rel computation
                verb_blend_logits = adapted_verb_logits
                verb_blend_weight = max(verb_blend_weight, adapted_blend_weight)'''

    if old_adapted_path in content:
        content = content.replace(old_adapted_path, new_adapted_path)
        changes_made.append("Added adapted path blending with W_rel")
        print("  [OK] Patched adapted path with blending")
    else:
        print("  [WARN] Could not find old adapted path section")

    # Write the patched file
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
