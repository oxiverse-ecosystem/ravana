"""Background driver: larger LingGen P6 harvest + longer training to EARN use_linggen.

Runs:
  1) harvest_grounded_corpus(local_books=True, max_concepts=2000)  -> clean
     sentences from the seeded Gutenberg novels (offline, no LLM).
  2) train_decoder_grounded(n_passes=40, pp=200) -> fit W_sm on all pairs,
     train the decoder on the grounded descriptions with sensorimotor
     conditioning, and promote use_linggen only if held-out CE <= baseline.

Writes a small report to data/linggen_train_report.txt and prints a summary.
"""

import sys, os, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (".", os.path.join(ROOT, "ravana_ml", "src"),
          os.path.join(ROOT, "ravana", "src"), os.path.join(ROOT, "ravana-v2", "src"),
          os.path.join(ROOT, "scripts")):
    sys.path.insert(0, p)

from scripts.ravana_chat import CognitiveChatEngine
from scripts.train import train_decoder_grounded


def main():
    t0 = time.time()
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True)
    nd = eng.neural_decoder

    # 1) Larger grounded harvest from local books (clean, deterministic).
    n_harvest = eng.harvest_grounded_corpus(max_concepts=2000, local_books=True, force=True)
    report = []
    report.append(f"[harvest] local_books concepts written: {n_harvest}")

    # 2) Longer grounded training. freeze_core=True so ONLY the new grounding
    #    parameters (W_h_bias persistent concept-bias + av_head load-bearing
    #    head) learn -- the seed-language core (GRU / condition_proj /
    #    attention) is protected from the drift that previously degraded
    #    coherence. Scheduled sampling (eps) closes the teacher-forcing ->
    #    free-run exposure-bias gap; aux_lambda adds the "Cosine Misleads"
    #    load-bearing reconstruction loss so the thin concept pointer actually
    #    flows through the network instead of being bypassed.
    SCHEDULED_EPS = 0.25   # probability of feeding own prediction during training
    AUX_LAMBDA = 0.5       # weight of av-reconstruction auxiliary loss
    trained, use_ling = train_decoder_grounded(
        eng, nd, n_passes=40, pp=200, si=4, freeze_core=True,
        scheduled_eps=SCHEDULED_EPS, aux_lambda=AUX_LAMBDA)
    report.append(f"[train] trained={trained} use_linggen={use_ling}")
    report.append(f"[train] W_sm saved={os.path.exists(os.path.join(ROOT, 'data', 'linggen_wsm.npz'))}")
    report.append(f"[elapsed] {time.time()-t0:.1f}s")

    out = os.path.join(ROOT, "data", "linggen_train_report.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(report) + "\n")
    print("\n".join(report))
    print(f"REPORT_WRITTEN={out}")


if __name__ == "__main__":
    main()
