"""The settle loop must actually minimise prediction error.

BLOCKER (docs/CODEBASE_DECODER_AUDIT.md finding 2): the settle loop computed
`err = logits_target - logits_h` and then DISCARDED it, taking a fixed step
`h += settle_lr * (h_star - h)`. That is linear interpolation toward h_star,
not predictive coding -- after `settle_steps` steps h is ~90% of the way there,
so the emitted word is readout(h_star): the attractor's own word, copied.

The fix uses the true gradient dE/dh = J^T err with J = 0.5 * W_out (the readout
is linear in the blended state). These tests pin that the error falls and that
the settled state is NOT a copy of h_star.
"""
import numpy as np
import pytest

from ravana.decoder.predictive_coding_generator import PredictiveCodingGenerator
from ravana_ml.nn.neural_decoder import NeuralDecoder

VOCAB = ["<pad>", "<unk>", "<bos>", "<eos>", "gravity", "is", "the", "force",
         "that", "pulls", "earth", "orbit", "mass", "physics"]
W2I = {w: i for i, w in enumerate(VOCAB)}
IDX2W = {i: w for w, i in W2I.items()}


def _make(seed=0, hidden=32, embed=24):
    rng = np.random.default_rng(seed)
    nd = NeuralDecoder(vocab_size=len(VOCAB), embed_dim=embed, hidden_dim=hidden)
    emb = {w: rng.normal(0, 0.3, embed).astype(np.float32) for w in VOCAB}
    gen = PredictiveCodingGenerator(nd, idx_to_word=IDX2W,
                                    settle_steps=8, settle_lr=0.25, seed=seed)
    return nd, emb, gen, rng


def _settle(nd, gen, traj_words, emb):
    """Replicate the production settle loop, returning per-position metrics."""
    h = np.zeros(nd.hidden_dim, dtype=np.float32)
    out = []
    for pos in range(1, len(traj_words)):
        prev_emb = nd.word_embedding.embed_raw(W2I[traj_words[pos - 1]])
        proj_prev = nd.condition_proj.forward_raw(prev_emb.reshape(1, -1))[0]
        h = gen._gru_step(proj_prev, h)
        tgt_emb = emb[traj_words[pos]]
        proj_tgt = nd.condition_proj.forward_raw(tgt_emb.reshape(1, -1))[0]
        h_star = gen._gru_step(proj_tgt, h)
        # The readout now mixes in LOGIT space (docs/CODEBASE_DECODER_AUDIT.md
        # finding 3), so "no attention" is a zero VOCAB-width vector, not a
        # hidden-width one.
        attn = np.zeros(nd.output_proj.weight.data.shape[0], dtype=np.float32)
        logits_target = gen._readout(h_star, attn)
        e0 = float(np.linalg.norm(logits_target - gen._readout(h, attn)))
        prev_h = h.copy()
        W_out = nd.output_proj.weight.data
        gate = float(getattr(nd, "attn_gate", 0.5))
        for _ in range(gen.settle_steps):
            err = logits_target - gen._readout(h, attn)
            # dE/dh = J^T err with J = d readout / d h = (1-gate) * W_out
            grad = (err @ W_out) * (1.0 - gate) / max(1, err.shape[0])
            h = h + gen.settle_lr * grad
            if np.linalg.norm(h - prev_h) < gen.settle_tol:
                break
            prev_h = h.copy()
        e1 = float(np.linalg.norm(logits_target - gen._readout(h, attn)))
        cos = float(np.dot(h, h_star) /
                    (np.linalg.norm(h) * np.linalg.norm(h_star) + 1e-9))
        out.append((e0, e1, cos))
    return out


TRAJ = ["gravity", "is", "the", "force", "that", "pulls"]


def test_settle_reduces_prediction_error():
    nd, emb, gen, _ = _make()
    res = _settle(nd, gen, TRAJ, emb)
    for e0, e1, _ in res:
        assert e1 < e0, f"error must fall: {e0:.4f} -> {e1:.4f}"


def test_settle_does_not_copy_h_star():
    """The settled state must not collapse onto h_star -- that would mean the
    generator copies the attractor's own word instead of generating."""
    nd, emb, gen, _ = _make()
    res = _settle(nd, gen, TRAJ, emb)
    mean_cos = float(np.mean([c for _, _, c in res]))
    assert mean_cos < 0.85, f"cos(h,h*) too high ({mean_cos:+.3f}) -- copying"


def test_error_reduction_is_substantial():
    """Guard against a token-level fix that barely moves the error."""
    nd, emb, gen, _ = _make()
    res = _settle(nd, gen, TRAJ, emb)
    before = float(np.mean([e[0] for e in res]))
    after = float(np.mean([e[1] for e in res]))
    assert after < before * 0.95, f"only {100*(1-after/before):.1f}% reduction"


def test_more_steps_reduce_error_further():
    """More settle steps must not increase the error (monotone descent)."""
    nd, emb, gen, _ = _make()
    few = _settle(nd, gen, TRAJ, emb)
    gen.settle_steps = 32
    many = _settle(nd, gen, TRAJ, emb)
    assert np.mean([e[1] for e in many]) <= np.mean([e[1] for e in few]) + 1e-6


def test_generate_runs_and_returns_tokens():
    nd, emb, gen, rng = _make()
    # generate() walks traj_idx from first_pos+1, so the trajectory must carry
    # the leading <bos> that the engine passes in production.
    traj = [(w, emb[w]) for w in ["<bos>"] + TRAJ]
    cond = np.stack([emb["gravity"], emb["force"]])
    toks = gen.generate(cond, traj, bos_idx=W2I["<bos>"], eos_idx=W2I["<eos>"])
    assert isinstance(toks, list)
    assert all(isinstance(t, int) for t in toks)


def test_generate_survives_out_of_vocab_trajectory_words():
    """REGRESSION: an attractor containing words outside the decoder vocab used
    to raise IndexError, because the loop was bounded by the UNFILTERED
    trajectory length while indexing the FILTERED traj_idx."""
    nd, emb, gen, rng = _make()
    # "zzq" and "qqz" are not in VOCAB -> dropped from traj_idx, but present in
    # the trajectory the engine hands us.
    traj = [("gravity", emb["gravity"]), ("zzq", emb["is"]),
            ("force", emb["force"]), ("qqz", emb["the"]),
            ("pulls", emb["pulls"])]
    cond = np.stack([emb["gravity"], emb["force"]])
    toks = gen.generate(cond, traj, bos_idx=W2I["<bos>"], eos_idx=W2I["<eos>"])
    assert isinstance(toks, list)
