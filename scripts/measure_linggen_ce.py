"""Measure decoder-CE vs template-CE on a fixed probe set (LingGen P6 gate).

Usage:
  RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe scripts/measure_linggen_ce.py

Both generation paths produce a reply for every probe; each reply is scored by
the SAME decoder (`NeuralDecoder.score_sequence` -- a read-only, teacher-forced
sampled-softmax NLL). The comparison is therefore a genuine distribution-fit
test: which path's surface forms does the learned language model find more
probable?

A path-independent coherence check is reported alongside, because a model will
always score its own samples higher. Promotion requires the decoder to win on
BOTH signals, not just on the one that favours it.

Writes docs/LINGGEN_CE_COMPARISON.md.
"""
import os, sys, json, re, math
os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (PROJ, os.path.join(PROJ, "ravana", "src"),
          os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana-v2", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

PROBES = [
    "what is gravity?",
    "how do you process the world?",
    "tell me about love",
    "what is your name?",
    "describe the ocean",
    "what makes you happy?",
    "explain time",
    "what is memory?",
    "how do you learn?",
    "what is identity?",
]


def _to_ids(eng, text):
    """Map a surface form onto decoder token ids, OOV -> <unk>."""
    w2i = getattr(eng, "_decoder_word_to_idx", None) or {}
    if not w2i:
        return []
    unk = w2i.get("<unk>", 1)
    return [w2i.get(w, unk) for w in re.findall(r"[a-z']+", (text or "").lower())]


def _coherence(eng, query, reply):
    """Path-independent quality: topical coherence with the query subject,
    penalised when the learned salad classifier calls the reply degenerate."""
    if not reply:
        return 0.0
    try:
        g = eng._ground_query(query)
        subj = (g[0] if g else "") or ""
    except Exception:
        subj = ""
    score = 0.0
    try:
        if subj and hasattr(eng, "_snippet_topic_max_coherence"):
            score = float(eng._snippet_topic_max_coherence(subj, reply))
    except Exception:
        score = 0.0
    try:
        from ravana.chat.salad_classifier import is_salad_learned
        if is_salad_learned(reply, subj) is True:
            score *= 0.25
    except Exception:
        pass
    return score


def main():
    eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                              user_suffix="linggenprobe")
    dec = eng.neural_decoder
    if dec is None:
        print(json.dumps({"error": "no neural decoder"}, indent=2))
        return

    # Warm up (both paths) so cached state does not skew the first probe.
    for q in PROBES:
        eng.process_turn(q)

    rows = []
    for q in PROBES:
        eng.use_linggen = False
        template_reply = eng.process_turn(q) or ""
        eng.use_linggen = True
        decoder_reply = eng.process_turn(q) or ""

        t_ids, d_ids = _to_ids(eng, template_reply), _to_ids(eng, decoder_reply)
        rows.append({
            "query": q,
            "template_reply": template_reply,
            "decoder_reply": decoder_reply,
            "decoder_ce": (dec.score_sequence(d_ids)
                           if d_ids else None),
            "template_ce": (dec.score_sequence(t_ids)
                            if t_ids else None),
            "decoder_coherence": _coherence(eng, q, decoder_reply),
            "template_coherence": _coherence(eng, q, template_reply),
        })

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    decoder_ce = _mean([r["decoder_ce"] for r in rows])
    template_ce = _mean([r["template_ce"] for r in rows])
    decoder_coh = round(sum(r["decoder_coherence"] for r in rows) / len(rows), 4)
    template_coh = round(sum(r["template_coherence"] for r in rows) / len(rows), 4)

    decoder_produced = sum(1 for r in rows
                           if r["decoder_reply"] and r["decoder_ce"] is not None)
    template_produced = sum(1 for r in rows
                            if r["template_reply"] and r["template_ce"] is not None)

    # A sampled-softmax model that has learned nothing sits at the uniform
    # baseline ln(K+1) for K negatives. Reporting it turns "both paths score
    # ~3.93, a near-tie" into the readable fact it is: the decoder has learned
    # nothing, so the CE comparison carries no evidence either way.
    uniform_ce = round(math.log(50 + 1), 4)

    ce_ok = (decoder_ce is not None and template_ce is not None
             and decoder_ce <= template_ce)
    coherence_ok = decoder_coh >= template_coh
    trained = int(getattr(dec, "_total_training_examples", 0) or 0)
    promote = bool(ce_ok and coherence_ok and trained > 0)

    result = {
        "N": len(PROBES),
        "method": ("per-path surface forms scored by the same decoder "
                   "(read-only sampled-softmax NLL); coherence is a "
                   "path-independent check"),
        "decoder_ce": decoder_ce,
        "template_ce": template_ce,
        "decoder_le_template": ce_ok,
        "decoder_coherence": decoder_coh,
        "template_coherence": template_coh,
        "decoder_not_worse_coherence": coherence_ok,
        "decoder_training_examples": trained,
        "decoder_replies_produced": decoder_produced,
        "template_replies_produced": template_produced,
        "uniform_baseline_ce": uniform_ce,
        "promote_use_linggen": promote,
    }

    out = os.path.join(PROJ, "docs", "LINGGEN_CE_COMPARISON.md")
    with open(out, "w") as f:
        f.write("# LingGen CE comparison\n\n")
        f.write(f"N = {result['N']}\n\n")
        f.write(f"Method: {result['method']}\n\n")
        f.write(f"decoder-CE = {decoder_ce}\n\n")
        f.write(f"template-CE = {template_ce}\n\n")
        f.write(f"decoder-CE <= template-CE: {ce_ok}\n\n")
        f.write(f"decoder coherence = {decoder_coh}\n\n")
        f.write(f"template coherence = {template_coh}\n\n")
        f.write(f"decoder not worse on coherence: {coherence_ok}\n\n")
        f.write(f"decoder training examples = {trained}\n\n")
        f.write(f"uniform baseline CE (ln 51) = {uniform_ce}\n\n")
        f.write(f"replies produced: decoder {decoder_produced}/{len(PROBES)}, "
                f"template {template_produced}/{len(PROBES)}\n\n")
        f.write(f"Decision: use_linggen = {promote}. ")
        if promote:
            f.write("Decoder wins on both distribution fit and coherence.\n")
        elif trained == 0:
            f.write("Flag stays False. The decoder has never been trained "
                    "(0 examples), so there is no distribution to fit -- the "
                    "measurement is valid and the answer is genuinely no. "
                    "Run scripts/train.py first.\n")
        else:
            f.write("Flag stays False: the decoder does not beat the template "
                    "path on both signals.\n")
    print(json.dumps(result, indent=2))
    eng.save()


if __name__ == "__main__":
    main()
