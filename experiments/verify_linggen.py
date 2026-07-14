"""Hermetic offline verification for LingGen P6 (no network, no LLM).

Tests the BRAIN-INSPIRED pieces that are new:
  B2  W_sm (65->75 ridge) learns the Binder->dual-code mapping and reproduces
      it on warm-start (save -> load -> identical conditioning vector).
  B5  Fail-closed gate (should_use_freeform): ood_abstain=True forces the
      realize_dim fallback; a low adaptive confidence floor also forces fallback.
  B4  NeuralDecoder.train_on_sentence(sensorimotor_conditioning=...) overrides
      the blended conditioning WITHOUT crashing (the angular-gyrus binding
      reaches the GRU). We use a tiny standalone decoder (no engine, no web).

Real grounded descriptions + multi-pass decoder training happen in
scripts/train.py (network step). This file proves the mechanism is correct
and fail-closed using synthetic oracle data so CI needs no network.
"""
import os
import sys
import numpy as np

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ, os.path.join(_PROJ, "ravana", "src"),
          os.path.join(_PROJ, "ravana_ml", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ravana.ontology.linggen import (
    LingGenConditioner, adaptive_floor, should_use_freeform)
from ravana_ml.nn.neural_decoder import NeuralDecoder


def _synth_pairs(n=60, dim_b=65, dim_e=75, seed=0):
    """Synthetic oracle: embed75 = W_true @ binder + small noise.

    This mimics the real supervision (concept Binder vec -> its 75-D dual-code
    embedding) so we can verify W_sm recovers the mapping.
    """
    rng = np.random.RandomState(seed)
    W_true = rng.randn(dim_e, dim_b).astype(np.float64) * 0.3
    binder = rng.randn(n, dim_b).astype(np.float64)
    embed = (W_true @ binder.T).T + rng.randn(n, dim_e).astype(np.float64) * 0.05
    # unit-norm the targets like the real embed75 path does
    embed = embed / np.linalg.norm(embed, axis=1, keepdims=True)
    return binder, embed


def test_B2_fit_and_warmstart():
    binder, embed = _synth_pairs(n=60)
    c = LingGenConditioner.fit(binder, embed, lam=1.0)
    assert c.trained, "W_sm should be trained on 60 pairs"
    # Reproduces the mapping direction (cosine) on held-out-ish points.
    test_b = binder[:5]
    cos = []
    for i in range(5):
        out = c.condition(test_b[i])
        assert out is not None and out.shape[0] == 75
        tgt = embed[i] / np.linalg.norm(embed[i])
        cos.append(float(np.dot(out, tgt)))
    mean_cos = float(np.mean(cos))
    print(f"[B2] W_sm recovered mapping mean-cos={mean_cos:.3f} (expect > 0)")
    assert mean_cos > 0.0, "W_sm must capture the Binder->embed direction"

    # Warm-start: save -> load -> identical conditioning vector.
    path = os.path.join(_PROJ, "data", "_verify_linggen_wsm.npz")
    c.save(path)
    c2 = LingGenConditioner.load(path)
    v1 = c.condition(binder[0])
    v2 = c2.condition(binder[0])
    assert np.allclose(v1, v2, atol=1e-5), "warm-start must reproduce conditioning"
    print("[B2] warm-start reproduces identical conditioning vector OK")
    os.remove(path)
    return True


def test_B5_fail_closed_gate():
    # On-manifold + confident history -> use free-form.
    assert should_use_freeform(False, 0.85, [0.82, 0.84, 0.83, 0.86, 0.85]) is True
    # OOD (off_manifold) -> ALWAYS fall back (realize_dim), regardless of conf.
    assert should_use_freeform(True, 0.99, [0.99] * 5) is False
    # Low confidence vs its own history -> fall back.
    assert should_use_freeform(False, 0.40, [0.80, 0.82, 0.81, 0.79, 0.83]) is False
    # adaptive_floor is data-derived (mean - 2*std), not a constant.
    floor = adaptive_floor([0.80, 0.82, 0.81, 0.79, 0.83])
    assert 0.0 <= floor < 0.80, f"adaptive floor should be below mean, got {floor}"
    print(f"[B5] fail-closed gate OK (adaptive_floor={floor:.3f})")
    return True


def test_B4_decoder_override():
    # Tiny standalone decoder with a vocab LARGER than the top-k sampled-softmax
    # bound (>=10) so the existing argpartition(-10) path works. Real decoders
    # have thousands of words; this just needs to be big enough to exercise the
    # sensorimotor_conditioning override.
    n_vocab = 64
    vocab = ["<bos>", "<eos>", "<unk>"] + [f"w{i}" for i in range(n_vocab - 3)]
    idx = {w: i for i, w in enumerate(vocab)}
    rng0 = np.random.RandomState(0)
    embed = {w: rng0.randn(75).astype(np.float32) for w in vocab}
    nd = NeuralDecoder(vocab_size=n_vocab, embed_dim=75, hidden_dim=32)
    nd._vocab_embed_cache = None
    W = rng0.randn(75, 65).astype(np.float32) * 0.2
    nd.set_linggen_projection(W)
    assert nd._linggen_W_sm is not None

    sm = rng0.randn(65).astype(np.float32)
    ce = nd.train_on_sentence(
        ["w1", "w2", "w3", "w4"], embed, idx,
        word_indices=[idx[w] for w in ["w1", "w2", "w3", "w4"]],
        freeze_core=False, sensorimotor_conditioning=sm)
    print(f"[B4] decoder train_on_sentence(sensorimotor_conditioning) CE={ce:.3f} OK")
    assert ce >= 0.0

    sents = nd.prepare_sentences(
        "w1 w2 w3 w4. w5 w6 w7 w8.", embed, idx,
        min_sentence_len=3, sensorimotor_conditioning=sm)
    assert len(sents) >= 1
    c0 = sents[0]["conditioning_embs"]
    exp = (W @ sm).astype(np.float32)
    nn = np.linalg.norm(exp)
    if nn > 0:
        exp = exp / nn
    row0 = c0[0] / np.linalg.norm(c0[0])
    sim = float(np.dot(row0, exp / np.linalg.norm(exp)))
    print(f"[B4] prepare_sentences override aligned sim={sim:.3f} (expect ~1.0)")
    assert sim > 0.9, "sensorimotor override must reach the conditioning rows"
    return True


def main():
    ok = True
    for name, fn in [("B2 fit+warmstart", test_B2_fit_and_warmstart),
                     ("B5 fail-closed gate", test_B5_fail_closed_gate),
                     ("B4 decoder override", test_B4_decoder_override)]:
        try:
            fn()
            print(f"  VERDICT: {name} CONFIRMED")
        except AssertionError as e:
            ok = False
            print(f"  VERDICT: {name} FAILED -> {e}")
        except Exception as e:
            ok = False
            print(f"  VERDICT: {name} ERROR -> {e}")
    print("\n" + ("ALL CONFIRMED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
