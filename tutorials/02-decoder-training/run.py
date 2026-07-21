"""Tutorial 02: Neural Decoder Training — load engine, train decoder, measure accuracy.

Builds on Tutorial 01 (loads the saved engine state).
After this, run Tutorial 03 to inspect the graph.

Usage:
    python tutorials/02-decoder-training/run.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "ravana", "src"))
sys.path.insert(0, os.path.join(ROOT, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(ROOT, "ravana-v2", "src"))

from ravana.chat.engine import CognitiveChatEngine


def main() -> None:
    # 1. Load engine — uses the state saved by Tutorial 01
    import os.path as _path
    save_path = os.path.join(ROOT, "data", "ravana_weights.pkl")
    if not _path.exists(save_path):
        print(f"  ❌ Save file not found at {save_path}")
        print("  Run Tutorial 01 first: python tutorials/01-chat-basics/run.py")
        return
    print(f"Loading engine from {save_path} ...")
    engine = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    nd = engine.neural_decoder

    # 2. Check initial state
    initial_examples = getattr(nd, "_total_training_examples", 0)
    initial_ce = getattr(nd, "_avg_cross_entropy", "N/A")
    initial_top1 = getattr(nd, "_avg_top1_acc", "N/A")
    print(f"  Initial: {initial_examples} examples, CE={initial_ce}, top1={initial_top1}")

    # 3. Reset plasticity for a clean measure
    if initial_examples == 0:
        nd.reset_plasticity(stability=0.5)
        print("  [reset] Decoder plasticity reset")

    # 4. Train on a few sample sentences
    sample_sentences = [
        "trust is the basis of learning.",
        "courage means moving forward despite fear.",
        "memory connects past experience to future choice.",
    ]
    for text in sample_sentences:
        nd.train_on_sentence(
            text.split(),
            engine._decoder_word_to_embed,
            engine._decoder_word_to_idx,
        )
    engine._decoder_training_count += sum(len(t.split()) for t in sample_sentences)

    # 5. Read updated metrics
    print(
        "  trained_on_examples=" + str(getattr(nd, "_total_training_examples", "N/A"))
        + " avg_ce=" + str(getattr(nd, "_avg_cross_entropy", "N/A"))
        + " top1=" + str(getattr(nd, "_avg_top1_acc", "N/A"))
        + " top5=" + str(getattr(nd, "_avg_top5_acc", "N/A"))
    )

    # 6. Save for Tutorial 03
    engine.save()
    print("  [OK] State saved - ready for Tutorial 03")


if __name__ == "__main__":
    main()
