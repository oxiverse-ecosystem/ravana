"""Does the conditioning signal change the decoder's output at all?

If the decoder is truly subject-conditioned, feeding two DIFFERENT subjects'
conditioning must produce different text. If the outputs are near-identical,
the conditioning is inert and the model is emitting a fixed register.

Usage:
    uv run python scripts/measure_conditioning_effect.py [subject_a] [subject_b]
"""
import os
import sys

import numpy as np

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src"),
           os.path.join(PROJ, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    a = sys.argv[1] if len(sys.argv) > 1 else "gravity"
    b = sys.argv[2] if len(sys.argv) > 2 else "ocean"

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine

    eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                              user_suffix="")
    nd = eng.neural_decoder
    W_sm = getattr(nd, "_linggen_W_sm", None)
    enc = getattr(eng, "_combined_attr_encoder", None)
    gv = getattr(eng, "_glove_vector", None)
    print(f"[cond] W_sm={'yes' if W_sm is not None else 'NO'} "
          f"encoder={'yes' if enc is not None else 'NO'} "
          f"glove={'yes' if gv is not None else 'NO'}")

    def cond_for(subject):
        if enc is None or gv is None or W_sm is None:
            return None
        v = gv(subject)
        if v is None:
            return None
        sig = np.asarray(enc.attribute_vector(np.asarray(v, dtype=np.float64)),
                         dtype=np.float32).reshape(-1)[:65]
        if sig.shape[0] != 65:
            return None
        proj = np.asarray(W_sm, dtype=np.float32) @ sig
        n = np.linalg.norm(proj)
        return (proj / n).astype(np.float32) if n > 0 else None

    ca, cb = cond_for(a), cond_for(b)
    print(f"[cond] conditioning for {a!r}: {'yes' if ca is not None else 'NO'}")
    print(f"[cond] conditioning for {b!r}: {'yes' if cb is not None else 'NO'}")
    if ca is not None and cb is not None:
        print(f"[cond] cosine(ca, cb) = {float(np.dot(ca, cb)):.4f} "
              f"(different subjects should be well below 1.0)")

    bos = eng._decoder_word_to_idx.get("<bos>", 0)
    eos = eng._decoder_word_to_idx.get("<eos>", 2)

    def gen(persist, n=8):
        out = []
        for _ in range(n):
            toks = nd.generate(
                conditioning_embs=np.zeros((1, nd.embed_dim), dtype=np.float32),
                max_steps=14, bos_idx=bos, eos_idx=eos, temperature=0.5,
                persistent_emb=persist)
            words = [eng._decoder_idx_to_word.get(t, "") for t in toks]
            out.append(" ".join(w for w in words if not w.startswith("<")))
        return out

    print(f"\n[cond] generations conditioned on {a!r}:")
    for s in gen(ca):
        print("   ", s)
    print(f"\n[cond] generations conditioned on {b!r}:")
    for s in gen(cb):
        print("   ", s)
    print("\n[cond] generations with NO conditioning:")
    for s in gen(None):
        print("   ", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
