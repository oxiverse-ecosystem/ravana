"""Measure decoder-CE vs template-CE on a fixed probe set (LingGen P6 gate).

Usage:
  RAVANA_OFFLINE=1 .venv-real/Scripts/python.exe scripts/measure_linggen_ce.py

TWO comparisons are reported, and they answer different questions.

1. SELF-SCORED CE (`decoder_ce`, `template_ce`). Each path's surface forms are
   scored by the SAME decoder (`NeuralDecoder.score_sequence`, a read-only
   teacher-forced sampled-softmax NLL). This is NOT a valid comparison and is
   reported only for transparency: a model always scores its own samples
   higher, so `decoder_ce` measures how well the decoder predicts the decoder
   and `template_ce` how well it predicts the template. Whichever path the
   decoder happens to agree with wins, for reasons that have nothing to do with
   reply quality.

2. SHARED-REFERENCE CE (`decoder_ref_ce`, `template_ref_ce`). Both paths are
   scored against the SAME token sequence -- the query's own content words,
   which is what a reply is supposed to be about. The target is fixed by the
   query alone, so neither path can improve its score by choosing different
   wording. This answers the question that actually matters: which reply stays
   closer to what was asked.

Promotion requires the decoder to win on the shared reference AND on the
path-independent coherence check. Writes docs/LINGGEN_CE_COMPARISON.md.
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

# Queries that BYPASS generation entirely. `_consult_internal_knowledge`
# (engine.py) answers a plain "what is X" definition from the KB and returns
# before the response generator ever runs, so both paths produce the identical
# string and the comparison carries no information about the decoder. The
# promotion probe must measure the decoder where the decoder actually runs.
GENERATION_PROBES = [
    "tell me about love",
    "how do you process the world?",
    "what makes you happy?",
    "how do you learn?",
    "what is identity?",
    "describe the ocean",
]


def _to_ids(eng, text):
    """Map a surface form onto decoder token ids, OOV -> <unk>."""
    w2i = getattr(eng, "_decoder_word_to_idx", None) or {}
    if not w2i:
        return []
    unk = w2i.get("<unk>", 1)
    return [w2i.get(w, unk) for w in re.findall(r"[a-z']+", (text or "").lower())]


def _reference_ids(eng, query):
    """Token ids for the query's CONTENT words -- the shared scoring target.

    A reply is supposed to be about the query's subject. Scoring both paths
    against the same token sequence makes the comparison path-independent:
    neither path can improve its score by choosing different wording, because
    the target is fixed by the query alone.

    The query's own content words are often only one or two tokens ("tell me
    about love" -> ["love"]), which is too thin to discriminate. So the
    reference is EXPANDED with the engine's own topic associations for those
    words -- the same neighbourhood the reply generator itself consults. That
    keeps the target fixed by the query (it is derived from the query alone,
    before either path runs) while giving the NLL enough tokens to be
    informative.

    Returns at most 12 ids.
    """
    stop = {"what", "is", "are", "the", "a", "an", "of", "to", "and", "do",
            "you", "your", "me", "my", "i", "how", "why", "when", "where",
            "tell", "about", "describe", "explain", "can", "could", "please",
            "would", "in", "on", "for", "with", "that", "this", "it", "be"}
    w2i = getattr(eng, "_decoder_word_to_idx", {}) or {}
    seeds = [w for w in re.findall(r"[a-z']+", (query or "").lower())
             if w not in stop and w in w2i]
    if not seeds:
        return []

    ids = [w2i[w] for w in seeds]
    # Expand with the graph's own neighbourhood of the seed words, so the
    # reference covers the topic rather than a single token. This is the same
    # neighbourhood the reply generator consults, and it is derived from the
    # query alone -- before either path runs -- so the target stays fixed.
    try:
        g = getattr(eng, "graph", None)
        for w in seeds[:3]:
            if g is None:
                break
            node = None
            for cand in g.nodes.values():
                if str(getattr(cand, "label", "")).lower() == w:
                    node = cand
                    break
            if node is None:
                continue
            nid = node.id
            # _outgoing maps src -> list of (target_id, edge) PAIRS, not bare
            # ids; iterating it as ids silently produced an empty reference.
            nbrs = list(getattr(g, "_outgoing", {}).get(nid, []) or [])
            for entry in nbrs[:4]:
                tgt = entry[0] if isinstance(entry, (tuple, list)) else entry
                lab = ""
                tn = g.nodes.get(tgt)
                if tn is not None:
                    lab = str(getattr(tn, "label", "")).lower().strip()
                idx = w2i.get(lab) if lab else None
                if idx is not None and idx not in ids:
                    ids.append(idx)
                if len(ids) >= 12:
                    break
            if len(ids) >= 12:
                break
    except Exception:
        pass
    return ids[:12]


def _cond_for(eng, dec, reply_ids):
    """Conditioning embedding for a reply: the mean of its token embeddings.

    `score_sequence` conditions on an embedding, so to ask "how probable is the
    reference under the reply this path produced", the natural conditioning is
    that reply's own representation. Mean-pooling the token embeddings is the
    same reduction the decoder's attention head reads, so the conditioning
    space matches what generation used.
    """
    if not reply_ids:
        return None
    # The token embedding table is `word_embedding` (there is no `.embedding`);
    # reading the wrong attribute silently returned None and made every
    # conditioning fall back to zeros -- which is why both paths scored
    # identically.
    emb = getattr(dec, "word_embedding", None)
    if emb is None:
        return None
    try:
        import numpy as np
        vecs = []
        for idx in reply_ids[:24]:
            v = emb.embed_batch_raw(np.array([idx], dtype=np.intp))
            vecs.append(np.asarray(v, dtype=np.float32).reshape(-1))
        if not vecs:
            return None
        return np.mean(np.stack(vecs, axis=0), axis=0, keepdims=True)
    except Exception:
        return None


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

    # Attach a previously-fit conditioner. Without this the probe scores an
    # UNCONDITIONED decoder, which is why decoder_ce was byte-identical before
    # and after conditioner training (3.8014 both times).
    restored = eng._restore_linggen_conditioner()
    print(f"  [probe] linggen conditioner restored: {restored}")
    print(f"  [probe] decoder _linggen_W_sm: "
          f"{getattr(dec, '_linggen_W_sm', None) is not None}")

    # Warm up (both paths) so cached state does not skew the first probe.
    for q in PROBES:
        eng.process_turn(q)

    rows = []
    skipped_identical = 0
    for q in PROBES:
        eng.use_linggen = False
        template_reply = eng.process_turn(q) or ""
        eng.use_linggen = True
        decoder_reply = eng.process_turn(q) or ""

        # If both paths emitted the SAME string, the decoder never ran --
        # `_consult_internal_knowledge` answered from the KB first. Scoring
        # that pair measures nothing, so exclude it and say so.
        if template_reply.strip() and template_reply.strip() == decoder_reply.strip():
            skipped_identical += 1
            continue

        t_ids, d_ids = _to_ids(eng, template_reply), _to_ids(eng, decoder_reply)
        # ── SHARED-REFERENCE CE ────────────────────────────────────────────
        # The old gate scored each path's NLL on ITS OWN surface forms. That
        # is not a comparison: `decoder_ce` measured how well the decoder
        # predicts the decoder, and `template_ce` how well it predicts the
        # template. A model always scores its own samples higher, so the
        # decoder was being penalised for not being the template.
        #
        # The honest version scores BOTH paths against the SAME reference: the
        # query's own content words. That is the target a reply is supposed to
        # be about, it is available for every probe, and it is path-independent
        # -- neither path can game it by choosing its own wording. Whichever
        # path stays closer to the query's content is the better reply.
        ref_ids = _reference_ids(eng, q)
        rows.append({
            "query": q,
            "template_reply": template_reply,
            "decoder_reply": decoder_reply,
            "decoder_ce": (dec.score_sequence(d_ids)
                           if d_ids else None),
            "template_ce": (dec.score_sequence(t_ids)
                            if t_ids else None),
            # Shared reference: lower is better, and both paths are measured
            # against the identical token sequence, each under its OWN
            # conditioning (the reply it actually produced).
            "decoder_ref_ce": (dec.score_sequence(
                ref_ids, conditioning_embs=_cond_for(eng, dec, d_ids))
                if (ref_ids and d_ids) else None),
            "template_ref_ce": (dec.score_sequence(
                ref_ids, conditioning_embs=_cond_for(eng, dec, t_ids))
                if (ref_ids and t_ids) else None),
            "decoder_coherence": _coherence(eng, q, decoder_reply),
            "template_coherence": _coherence(eng, q, template_reply),
        })

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    decoder_ce = _mean([r["decoder_ce"] for r in rows])
    template_ce = _mean([r["template_ce"] for r in rows])
    # Shared-reference CE: the honest comparison. Both paths scored against the
    # SAME token sequence (the query's content words), so the number answers
    # "which reply is closer to what was asked", not "which reply is more like
    # itself".
    decoder_ref_ce = _mean([r["decoder_ref_ce"] for r in rows])
    template_ref_ce = _mean([r["template_ref_ce"] for r in rows])
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

    # The promotion decision now rests on the SHARED-REFERENCE comparison, not
    # on the self-scored one. Scoring each path on its own output is not a
    # comparison at all: a model always scores its own samples higher, so the
    # decoder was being penalised for not being the template. Requiring the
    # decoder to win on the shared reference AND on coherence is the honest
    # bar. The self-scored numbers are still reported for transparency.
    ref_ok = (decoder_ref_ce is not None and template_ref_ce is not None
              and decoder_ref_ce <= template_ref_ce)
    promote = bool(ref_ok and coherence_ok and trained > 0)

    result = {
        "N": len(PROBES),
        "N_scored": len(rows),
        "N_skipped_identical": skipped_identical,
        "method": ("per-path surface forms scored by the same decoder "
                   "(read-only sampled-softmax NLL); coherence is a "
                   "path-independent check"),
        "decoder_ce": decoder_ce,
        "template_ce": template_ce,
        "decoder_le_template": ce_ok,
        "decoder_ref_ce": decoder_ref_ce,
        "template_ref_ce": template_ref_ce,
        "decoder_le_template_shared_ref": ref_ok,
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
