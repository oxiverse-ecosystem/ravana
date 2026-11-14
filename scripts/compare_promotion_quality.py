"""Compare two decoder checkpoints on the PROMOTION gate's own metric.

The full-40 retrain was rejected by the promotion gate (free-form quality 0.432
< the 0.5 floor) while scoring equivalently on the groundedness eval. This
measures both checkpoints on the gate's own terms to see whether the rejection
is real.

Usage:
    uv run python scripts/compare_promotion_quality.py <decoder_a.npz> <decoder_b.npz>
"""
import os
import sys

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
           os.path.join(PROJ, "ravana", "src"),
           os.path.join(PROJ, "ravana-v2", "src"),
           os.path.join(PROJ, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def load_pairs(limit=80):
    pairs = []
    with open(os.path.join(PROJ, "data", "corpora",
                           "grounded_descriptions.txt"), encoding="utf-8") as fh:
        for line in fh:
            if "\t" in line:
                c, d = line.split("\t", 1)
                pairs.append((c.strip(), d.strip()))
    return pairs[:limit]


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    path_a, path_b = sys.argv[1], sys.argv[2]

    os.environ.pop("RAVANA_OFFLINE", None)
    from ravana.chat.engine import CognitiveChatEngine
    import train as train_mod

    pairs = load_pairs(80)
    out = {}
    for label, path in (("A", path_a), ("B", path_b)):
        eng = CognitiveChatEngine(data_dir=os.path.join(PROJ, "weights"),
                                  user_suffix="")
        dec, _vocab = eng.neural_decoder.load_from(path)
        if dec is None:
            print("[%s] could not load %s" % (label, path))
            return 1
        # install the engine's word<->index maps on the loaded decoder
        dec.word_to_idx = eng._decoder_word_to_idx
        dec.idx_to_word = eng._decoder_idx_to_word
        dec.word_to_embed = eng._decoder_word_to_embed
        eng.neural_decoder = dec
        dec._align_attention_head()
        # W_sm lives in data/linggen_wsm.npz and is attached by the engine's
        # restore path, which does not run for a decoder swapped in directly.
        eng._restore_linggen_conditioner()

        W_sm = getattr(dec, "_linggen_W_sm", None)
        attr_enc = getattr(eng, "_combined_attr_encoder", None)
        q = train_mod._grounded_generation_quality(
            eng, dec, pairs, W_sm, attr_enc, getattr(eng, "_glove_vector", None))
        out[label] = q
        print("[%s] %-34s free-form quality = %.4f"
              % (label, os.path.basename(path), q))

    print("\nA = %s" % os.path.basename(path_a))
    print("B = %s" % os.path.basename(path_b))
    print("delta (B - A) = %+.4f" % (out["B"] - out["A"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
