"""
Mixin: EncoderMixin — rlm_v2_encoder methods for RLMv2.

Auto-extracted from rlm_v2.py. Edit in the source or directly here.
"""
import numpy as np
from typing import Optional, List, Tuple, Dict, Set, Any
from ..embedder import LearnedEmbedder
from .rlm_v2_common import _build_glove_embedding_matrix


class EncoderMixin:
    """Mixin providing rlm_v2_encoder methods for RLMv2."""



    def _compute_contrastive_gradients(self):

        """Compute contrastive loss gradients w.r.t. encoder parameters."""

        d_con_W1 = np.zeros_like(self._enc_W1)

        d_con_b1 = np.zeros_like(self._enc_b1)

        d_con_W2 = np.zeros_like(self._enc_W2)

        d_con_b2 = np.zeros_like(self._enc_b2)

        

        pairs = getattr(self, "semantic_pairs", [])

        if not pairs:

            return d_con_W1, d_con_b1, d_con_W2, d_con_b2, 0.0

            

        tokenizer = getattr(self, "_tokenizer", None)

        if tokenizer is None:

            return d_con_W1, d_con_b1, d_con_W2, d_con_b2, 0.0

            

        # Draw negative samples

        vocab_words = list(tokenizer.word_to_id.keys())

        neg_size = getattr(self, "neg_sample_size", 5)

        

        total_loss = 0.0

        n_pairs_processed = 0

        

        for word_a, word_b in pairs:

            tid_a = tokenizer.word_to_id.get(word_a)

            tid_b = tokenizer.word_to_id.get(word_b)

            if tid_a is None or tid_b is None:

                continue

                

            embed_a = self.token_embed.weight.data[tid_a]

            embed_b = self.token_embed.weight.data[tid_b]

            

            lat_a, z1_a, h1_a, z2_a = self._encoder_forward_full(embed_a)

            lat_b, z1_b, h1_b, z2_b = self._encoder_forward_full(embed_b)

            

            norm_a = np.linalg.norm(lat_a)

            norm_b = np.linalg.norm(lat_b)

            

            unit_a = lat_a / (norm_a + 1e-15)

            unit_b = lat_b / (norm_b + 1e-15)

            

            s = np.dot(unit_a, unit_b)

            sig_s = 1.0 / (1.0 + np.exp(-s) + 1e-15)

            

            # Positive loss: -log(sigmoid(s))

            total_loss -= np.log(sig_s + 1e-15)

            n_pairs_processed += 1

            

            # Gradients of positive loss w.r.t lat_a and lat_b

            d_s_d_lat_a = (unit_b - s * unit_a) / (norm_a + 1e-15)

            d_s_d_lat_b = (unit_a - s * unit_b) / (norm_b + 1e-15)

            

            d_lat_a = (sig_s - 1.0) * d_s_d_lat_a

            d_lat_b = (sig_s - 1.0) * d_s_d_lat_b

            

            # Backprop for word_a and word_b

            dW1_a, db1_a, dW2_a, db2_a = self._encoder_backward(

                embed_a[np.newaxis, :], z1_a[np.newaxis, :], h1_a[np.newaxis, :], z2_a[np.newaxis, :],

                lat_a[np.newaxis, :], d_lat_a[np.newaxis, :]

            )

            dW1_b, db1_b, dW2_b, db2_b = self._encoder_backward(

                embed_b[np.newaxis, :], z1_b[np.newaxis, :], h1_b[np.newaxis, :], z2_b[np.newaxis, :],

                lat_b[np.newaxis, :], d_lat_b[np.newaxis, :]

            )

            

            d_con_W1 += dW1_a + dW1_b

            d_con_b1 += db1_a + db1_b

            d_con_W2 += dW2_a + dW2_b

            d_con_b2 += db2_a + db2_b

            

            # Negative sampling

            if vocab_words:

                neg_words = np.random.choice(vocab_words, size=min(neg_size, len(vocab_words)), replace=False)

                for word_neg in neg_words:

                    if word_neg == word_a or word_neg == word_b:

                        continue

                    tid_neg = tokenizer.word_to_id.get(word_neg)

                    if tid_neg is None:

                        continue

                    embed_neg = self.token_embed.weight.data[tid_neg]

                    lat_neg, z1_neg, h1_neg, z2_neg = self._encoder_forward_full(embed_neg)

                    

                    norm_neg = np.linalg.norm(lat_neg)

                    unit_neg = lat_neg / (norm_neg + 1e-15)

                    

                    s_neg = np.dot(unit_a, unit_neg)

                    sig_s_neg = 1.0 / (1.0 + np.exp(-s_neg) + 1e-15)

                    

                    # Negative loss: log(sigmoid(s_neg))

                    total_loss += np.log(sig_s_neg + 1e-15)

                    

                    # Gradients of negative loss w.r.t lat_a and lat_neg

                    d_s_d_lat_a_neg = (unit_neg - s_neg * unit_a) / (norm_a + 1e-15)

                    d_s_d_lat_neg = (unit_a - s_neg * unit_neg) / (norm_neg + 1e-15)

                    

                    d_lat_a_neg = (1.0 - sig_s_neg) * d_s_d_lat_a_neg

                    d_lat_neg = (1.0 - sig_s_neg) * d_s_d_lat_neg

                    

                    # Backprop

                    dW1_a_neg, db1_a_neg, dW2_a_neg, db2_a_neg = self._encoder_backward(

                        embed_a[np.newaxis, :], z1_a[np.newaxis, :], h1_a[np.newaxis, :], z2_a[np.newaxis, :],

                        lat_a[np.newaxis, :], d_lat_a_neg[np.newaxis, :]

                    )

                    dW1_neg, db1_neg, dW2_neg, db2_neg = self._encoder_backward(

                        embed_neg[np.newaxis, :], z1_neg[np.newaxis, :], h1_neg[np.newaxis, :], z2_neg[np.newaxis, :],

                        lat_neg[np.newaxis, :], d_lat_neg[np.newaxis, :]

                    )

                    

                    d_con_W1 += dW1_a_neg + dW1_neg

                    d_con_b1 += db1_a_neg + db1_neg

                    d_con_W2 += dW2_a_neg + dW2_neg

                    d_con_b2 += db2_a_neg + db2_neg

                    

        if n_pairs_processed > 0:

            scale = 1.0 / n_pairs_processed

            d_con_W1 *= scale

            d_con_b1 *= scale

            d_con_W2 *= scale

            d_con_b2 *= scale

            total_loss *= scale

            

        return d_con_W1, d_con_b1, d_con_W2, d_con_b2, total_loss






    def _encoder_backward(self, X, z1, h1, z2, h2, d_h2):

        """Compute encoder parameter gradients.

        X: (B, embed_dim)

        z1, h1: (B, hidden_dim) pre/post activation of first layer

        z2, h2: (B, latent_dim) pre/post activation of second layer

        d_h2: (B, latent_dim) gradient w.r.t. h2

        Returns (d_W1, d_b1, d_W2, d_b2).

        """

        d_z2 = d_h2 * (1.0 - h2 * h2)

        d_enc_W2 = d_z2.T @ h1

        d_enc_b2 = np.sum(d_z2, axis=0)



        d_h1 = d_z2 @ self._enc_W2

        d_z1 = d_h1 * (1.0 - h1 * h1)

        d_enc_W1 = d_z1.T @ X

        d_enc_b1 = np.sum(d_z1, axis=0)



        return d_enc_W1, d_enc_b1, d_enc_W2, d_enc_b2




    def _encoder_forward_full(self, X):

        """Pass inputs through the encoder, returning all activations for backpropagation.

        X: (B, embed_dim) or (embed_dim,)

        """

        is_flat = X.ndim == 1

        if is_flat:

            X_batch = X[np.newaxis, :]

        else:

            X_batch = X

        z1 = X_batch @ self._enc_W1.T + self._enc_b1       # (B, hidden_dim)

        h1 = np.tanh(z1)                            # (B, hidden_dim)

        z2 = h1 @ self._enc_W2.T + self._enc_b2     # (B, latent_dim)

        latent = np.tanh(z2)                        # (B, latent_dim)

        if is_flat:

            return latent[0], z1[0], h1[0], z2[0]

        return latent, z1, h1, z2






    def _initialize_token_embeddings_from_tokenizer(self):

        """Initialize token embeddings using pre-trained GloVe vectors.

        

        Uses glove.6B.100d.txt (cached in data/glove/) projected to embed_dim.

        GloVe vectors capture genuine semantic relationships, making the

        verb-stem offset predictor work:

          offset("causes") = avg(expansion - heat, vision - light, ...)

        Character n-gram embeddings (previously used) cannot capture this.

        """

        import numpy as np

        tokenizer = self._tokenizer_val

        

        matrix = _build_glove_embedding_matrix(

            tokenizer, target_dim=self.embed_dim, glove_dim=100

        )

        

        if matrix is not None:

            n_found = np.count_nonzero(matrix.any(axis=1))

            coverage = n_found / max(1, self.vocab_size)

            print(f"  [Embeddings] GloVe 100D -> {self.embed_dim}D: {n_found}/{self.vocab_size} tokens ({coverage:.1%})")

            self.token_embed.weight.data[:matrix.shape[0]] = matrix

        else:

            # Fallback: random orthogonal init

            print(f"  [Embeddings] GloVe unavailable. Using random init.")

            rng = np.random.RandomState(42)

            max_dim = max(self.vocab_size, self.embed_dim)

            full_q, _ = np.linalg.qr(rng.randn(max_dim, max_dim).astype(np.float32))

            self.token_embed.weight.data[:] = full_q[:self.vocab_size, :self.embed_dim] * 0.1

                

        # Clear/invalidate cached norms

        self._token_embed_norms = None






    def _pretrain_encoder_autoencoder(self, epochs=300, lr=0.01):

        """Pre-train the encoder as an autoencoder over all vocabulary tokens."""

        X = self.token_embed.weight.data  # (vocab_size, embed_dim)

        

        # Momentum buffers for autoencoder weights

        dec_mW1 = np.zeros_like(self._dec_W1)

        dec_mb1 = np.zeros_like(self._dec_b1)

        dec_mW2 = np.zeros_like(self._dec_W2)

        dec_mb2 = np.zeros_like(self._dec_b2)

        

        for epoch in range(epochs):

            # Forward pass: Encoder

            z1 = X @ self._enc_W1.T + self._enc_b1      # (V, hidden_dim)

            h1 = np.tanh(z1)                           # (V, hidden_dim)

            z2 = h1 @ self._enc_W2.T + self._enc_b2    # (V, latent_dim)

            latent = np.tanh(z2)                       # (V, latent_dim)

            

            # Forward pass: Decoder

            dec_z1 = latent @ self._dec_W1.T + self._dec_b1  # (V, hidden_dim)

            dec_h1 = np.tanh(dec_z1)                        # (V, hidden_dim)

            dec_z2 = dec_h1 @ self._dec_W2.T + self._dec_b2  # (V, embed_dim)

            recon = dec_z2                                  # (V, embed_dim)

            

            # Loss: Mean Squared Error

            loss = np.mean((recon - X) ** 2)

            if epoch % 50 == 0 or epoch == epochs - 1:

                print(f"  [Autoencoder] Epoch {epoch:3d} Loss: {loss:.6f}")

            

            # Backward pass: Decoder

            d_recon = 2.0 * (recon - X) / len(X)       # (V, embed_dim)

            

            d_dec_W2 = d_recon.T @ dec_h1              # (embed_dim, hidden_dim)

            d_dec_b2 = np.sum(d_recon, axis=0)         # (embed_dim,)

            

            d_dec_h1 = d_recon @ self._dec_W2          # (V, hidden_dim)

            d_dec_z1 = d_dec_h1 * (1.0 - dec_h1 * dec_h1) # (V, hidden_dim)

            

            d_dec_W1 = d_dec_z1.T @ latent             # (hidden_dim, latent_dim)

            d_dec_b1 = np.sum(d_dec_z1, axis=0)        # (hidden_dim,)

            

            d_latent = d_dec_z1 @ self._dec_W1         # (V, latent_dim)

            

            # Backward pass: Encoder

            d_z2 = d_latent * (1.0 - latent * latent)  # (V, latent_dim)

            d_enc_W2 = d_z2.T @ h1                     # (latent_dim, hidden_dim)

            d_enc_b2 = np.sum(d_z2, axis=0)            # (latent_dim,)

            

            d_h1 = d_z2 @ self._enc_W2                 # (V, hidden_dim)

            d_z1 = d_h1 * (1.0 - h1 * h1)              # (V, hidden_dim)

            d_enc_W1 = d_z1.T @ X                      # (hidden_dim, embed_dim)

            d_enc_b1 = np.sum(d_z1, axis=0)            # (hidden_dim,)

            

            # Update Decoder

            dec_mW2 = self._rp_momentum * dec_mW2 - lr * d_dec_W2

            dec_mb2 = self._rp_momentum * dec_mb2 - lr * d_dec_b2

            dec_mW1 = self._rp_momentum * dec_mW1 - lr * d_dec_W1

            dec_mb1 = self._rp_momentum * dec_mb1 - lr * d_dec_b1

            

            self._dec_W2 += dec_mW2

            self._dec_b2 += dec_mb2

            self._dec_W1 += dec_mW1

            self._dec_b1 += dec_mb1

            

            # Update Encoder

            self._enc_mW2 = self._rp_momentum * self._enc_mW2 - lr * d_enc_W2

            self._enc_mb2 = self._rp_momentum * self._enc_mb2 - lr * d_enc_b2

            self._enc_mW1 = self._rp_momentum * self._enc_mW1 - lr * d_enc_W1

            self._enc_mb1 = self._rp_momentum * self._enc_mb1 - lr * d_enc_b1

            

            self._enc_W2 += self._enc_mW2

            self._enc_b2 += self._enc_mb2

            self._enc_W1 += self._enc_mW1

            self._enc_b1 += self._enc_mb1

            # Encoder changed - need re-alignment on next sleep

            self.mark_alignment_needed()



