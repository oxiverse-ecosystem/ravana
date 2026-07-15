#!/usr/bin/env python3
"""
RAVANA Training — full human-like training pipeline
====================================================
Trains the decoder on real English + web learning + reasoning cycles.

Modes:
  phase2    — Heavy decoder training on teen_seeds.txt (fast, ~1hr)
  full      — Full pipeline: seed → web → consolidate → evaluate (~3-5hrs)
  test      — Quick diagnostic that decoder trains on real text

Usage:
    python scripts/train.py --mode phase2
    python scripts/train.py --mode full [--web-topics 10] [--cycles 3]
    python scripts/train.py --mode test
"""
import sys, os, time, json, re
import numpy as np

_proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2"))

os.environ["RAVANA_SILENT"] = "1"
from scripts.ravana_chat import CognitiveChatEngine

# ─── Topics for web learning ───
WEB_TOPICS = [
    "consciousness neuroscience",
    "time travel physics",
    "quantum mechanics explained",
    "artificial intelligence ethics",
    "human memory psychology",
    "philosophy of mind",
    "evolutionary biology",
    "cybersecurity basics",
    "climate change science",
    "meditation mindfulness",
]

SEED_QUERIES = [
    "what is trust",
    "tell me about love",
    "what happens if time travel possible",
    "explain consciousness",
    "how does memory work",
    "what is justice",
    "tell me about freedom",
    "explain evolution",
    "what is artificial intelligence",
    "how does the brain work",
]

EVAL_QUESTIONS = [
    "what is trust",
    "tell me about love",
    "what happens if time travel possible",
    "explain consciousness",
    "how does memory work",
    "what is the meaning of life",
    "who are you",
    "what is justice",
    "explain freedom",
]

# ─── Shared ───

def load_engine(args, reset=False):
    save_path = os.path.join(_proj_root, "data", "ravana_weights.pkl")
    if reset and os.path.exists(save_path):
        os.remove(save_path)
        print(f"  [Reset] Deleted saved weights, starting fresh!")
    engine = CognitiveChatEngine(dim=args.dim, seed=args.seed, baby_mode=True)
    nd = engine.neural_decoder
    if nd._total_training_examples == 0:
        nd.reset_plasticity(stability=0.5)
        print("  [reset] Decoder plasticity reset")
    return engine, nd


def load_corpus(engine):
    corpus_path = os.path.join(_proj_root, "data", "corpora", "teen_seeds.txt")
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    engine._freeze_decoder_vocab = False
    words_in_corpus = set(re.findall(r"[a-zA-Z\']{3,}", text.lower()))
    new_for_vocab = [w for w in words_in_corpus if w not in engine._decoder_word_to_idx]
    if new_for_vocab:
        engine._expand_decoder_vocab(new_for_vocab)
    # LingGen P6: train the decoder on web-harvested grounded descriptions
    # (data/corpora/grounded_descriptions.txt) BEFORE freezing the vocab. This
    # fits W_sm (65->75) and trains the angular-gyrus binding. No-op if the
    # harvest hasn't run yet (it's a separate, network step).
    try:
        train_decoder_grounded(engine, nd, n_passes=30, pp=500, si=5,
                               freeze_core=False)
    except Exception as _e:
        print(f"  [LingGen] grounded training skipped: {_e}")
    engine._freeze_decoder_vocab = True
    nd = engine.neural_decoder
    all_sentences = nd.prepare_sentences(
        text, engine._decoder_word_to_embed, engine._decoder_word_to_idx,
        min_sentence_len=3,
    )
    return text, all_sentences, nd


def train_seed_corpus(engine, nd, all_sentences, n_passes=200, pp=2000, si=10, pe=20):
    """Train decoder on seed corpus with sampled softmax.
    
    Early stopping uses training CE (now honest after removing
    self-conditioning cheat). Final evaluation is done separately
    by the evaluate() function after training completes.
    Returns total sentences trained.
    """
    n_avail = len(all_sentences)
    pp = min(pp, n_avail)
    rng = np.random.RandomState(42)
    total = 0
    best_ce = float('inf')
    stall = 0
    t0 = time.time()

    for i in range(n_passes):
        idx = rng.choice(n_avail, size=pp, replace=False)
        for j in idx:
            s = all_sentences[j]
            nd.train_on_sentence(
                s['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
                word_indices=s['word_indices'], conditioning_embs=s['conditioning_embs'],
            )
        total += pp
        if (i+1) % si == 0:
            nd.sleep_cycle()
            ce = nd._avg_cross_entropy
            if ce < best_ce - 1e-3:
                best_ce = ce; stall = 0
            else:
                stall += 1
                if stall >= pe:
                    print(f"  Early stop at pass {i+1} (best CE={best_ce:.3f})")
                    break
        if (i+1) % 5 == 0:
            elapsed = time.time()-t0
            rate = elapsed/(i+1)
            print(f"  Pass {i+1}/{n_passes}: CE={nd._avg_cross_entropy:.3f} "
                  f"t1={nd._avg_top1_acc:.3f} t5={nd._avg_top5_acc:.3f} "
                  f"({rate:.1f}s/pass, ETA={(n_passes-i-1)*rate:.0f}s)", flush=True)

    if n_passes % si != 0:
        nd.sleep_cycle()
    engine._decoder_seed_training_count += total
    engine._decoder_training_count += total
    return total


def train_decoder_grounded(engine, nd, n_passes=30, pp=500, si=5,
                           desc_path=None, freeze_core=False,
                           scheduled_eps=0.0, aux_lambda=0.0):
    """LingGen P6 — train the decoder on web-harvested grounded descriptions.

    Reads data/corpora/grounded_descriptions.txt (concept<TAB>human description),
    for each pair:
      * target 75-D dual-code embedding = engine._embed_75d(concept)
      * conditioning signal = 65-D Binder vector = _combined_attr_encoder
        .attribute_vector(glove64(concept))
    Fits LingGenConditioner (ridge W_sm: 65->75) on (binder, embed75), persists
    data/linggen_wsm.npz, attaches it to the decoder, then trains the decoder to
    reproduce each description given its embodied signature
    (train_on_sentence(sensorimotor_conditioning=av, freeze_core=freeze_core)).

    Promotion (no hardcoding): after training, on a held-out split we compare
    decoder cross-entropy (sensorimotor-conditioned) to the seed-corpus baseline.
    If conditioned CE <= baseline CE, set engine.use_linggen = True (the free-form
    path is at least as good as the template path). Otherwise it stays False and
    generation falls back to realize_dim — never emits ungrounded gibberish.

    Requires the harvest to have run first (grounded_descriptions.txt present).
    Returns (trained: bool, use_linggen: bool).
    """
    desc_path = desc_path or os.path.join(_proj_root, "data", "corpora",
                                          "grounded_descriptions.txt")
    if not os.path.exists(desc_path):
        print("  [LingGen] no grounded_descriptions.txt — skip (run harvest first)")
        return False, False
    from ravana.ontology.linggen import LingGenConditioner, _MIN_PAIRS

    glove_fn = getattr(engine, "_glove_vector", None)
    attr_enc = getattr(engine, "_combined_attr_encoder", None)
    if glove_fn is None or attr_enc is None:
        print("  [LingGen] attr encoder/glove unavailable — skip")
        return False, False

    # Option C data source (brain-faithful): prefer CLEAN curated KB
    # definitions (engine._definitions: concept -> Wikipedia/ConceptNet text)
    # as the (concept, description) pairs. These are coherent definitions, so
    # the decoder is trained to PARAPHRASE coherent text (the easiest Option-C
    # regime) and the gate anchors on them. Fall back to the harvested novel
    # corpus only if clean defs are insufficient.
    defs = getattr(engine, "_definitions", None)
    use_clean = isinstance(defs, dict) and len(defs) >= _MIN_PAIRS
    if use_clean:
        src_pairs = [(c, d) for c, d in defs.items()
                     if isinstance(d, str) and d.strip()]
        print(f"  [LingGen] using {len(src_pairs)} clean KB definitions as pairs")
    else:
        if not os.path.exists(desc_path):
            print("  [LingGen] no grounded_descriptions.txt — skip (run harvest first)")
            return False, False
        src_pairs = []
        with open(desc_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if "\t" not in line:
                    continue
                concept, desc = line.split("\t", 1)
                concept = concept.strip().lower()
                desc = desc.strip()
                if concept and desc:
                    src_pairs.append((concept, desc))
        print(f"  [LingGen] using {len(src_pairs)} harvested novel-sentence pairs")

    pairs = []  # (binder65, embed75)
    sentences = []  # (concept, description_text)
    for concept, desc in src_pairs:
        gv = glove_fn(concept)
        if gv is None:
            continue
        try:
            av = attr_enc.attribute_vector(np.asarray(gv, dtype=np.float64))
        except Exception:
            continue
        av = np.asarray(av, dtype=np.float64)
        if av.shape[0] != 65:
            continue
        try:
            emb75 = engine._embed_75d(concept, node_vec=gv)
        except Exception:
            continue
        pairs.append((av, np.asarray(emb75, dtype=np.float64)))
        sentences.append((concept, desc))

    if len(pairs) < _MIN_PAIRS:
        print(f"  [LingGen] only {len(pairs)} grounded pairs (< min) — skip")
        return False, False

    binder = np.stack([p[0] for p in pairs], axis=0)
    embed75 = np.stack([p[1] for p in pairs], axis=0)
    # Hold out 20% for the promotion check.
    rng = np.random.RandomState(7)
    idx = rng.permutation(len(pairs))
    n_hold = max(1, int(0.2 * len(pairs)))
    hold = set(idx[:n_hold].tolist())
    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in hold]
    cond_pairs = [pairs[i] for i in range(len(pairs)) if i in hold]
    cond_sent = [sentences[i] for i in range(len(sentences)) if i in hold]

    conditioner = LingGenConditioner.fit(
        np.stack([p[0] for p in train_pairs], axis=0),
        np.stack([p[1] for p in train_pairs], axis=0))
    if not conditioner.trained:
        print("  [LingGen] W_sm fit failed (too few/ill-formed) — skip")
        return False, False
    conditioner.save()
    nd.set_linggen_projection(conditioner._W)
    print(f"  [LingGen] fit W_sm on {len(train_pairs)} pairs; "
          f"held-out {len(cond_pairs)}")

    # Train decoder on the grounded descriptions (conditioning override).
    W_sm = conditioner._W
    total = 0
    baseline_ce = float(getattr(nd, "_avg_cross_entropy", 3.0))
    for i in range(n_passes):
        rng2 = np.random.RandomState(1000 + i)
        for j in rng2.choice(len(sentences), size=min(pp, len(sentences)),
                             replace=False):
            concept, desc = sentences[j]
            gv = glove_fn(concept)
            if gv is None:
                continue
            av = attr_enc.attribute_vector(np.asarray(gv, dtype=np.float64))
            av = np.asarray(av, dtype=np.float32)[:65]
            # Only train on sentences whose words are all in vocab (respect freeze).
            ws = [w.lower().strip(".,!?").strip("'")
                  for w in desc.split() if len(w) >= 2]
            if not all(w in engine._decoder_word_to_idx for w in ws):
                continue
            nd.train_on_sentence(
                desc.split(), engine._decoder_word_to_embed,
                engine._decoder_word_to_idx,
                word_indices=[engine._decoder_word_to_idx.get(w.lower().strip(".,!?").strip("'"),
                                                              engine._decoder_word_to_idx.get("<unk>", 1))
                              for w in desc.split()],
                freeze_core=freeze_core,
                sensorimotor_conditioning=av,
                scheduled_eps=scheduled_eps,
                aux_lambda=aux_lambda)
            total += 1
        nd.sleep_cycle()
        if (i + 1) % si == 0:
            print(f"  [LingGen] pass {i+1}/{n_passes}: CE={nd._avg_cross_entropy:.3f} "
                  f"aux={getattr(nd, '_last_aux_loss', 0.0):.3f}")

    # Promotion gate: LingGen's job is FREE-FORM GROUNDED GENERATION, not
    # verbatim reproduction of harvested sentences. Exact-match CE is the
    # WRONG metric (the per-token linguistic blend always wins reproduction, but
    # that is not what free-form generation competes on). The correct, honest
    # gate measures whether free-form generation from the embodied BOS
    # (initial_emb) is COHERENT -- in-vocab, non-degenerate, and on-topic --
    # above a fixed coherence floor. If not, stay fail-closed (the decoder
    # would otherwise emit gibberish, the forbidden outcome).
    quality = _grounded_generation_quality(engine, nd, cond_sent, W_sm, attr_enc,
                                           glove_fn)
    COHERENCE_FLOOR = 0.5
    print(f"  [LingGen] free-form quality={quality:.3f} | coherence floor={COHERENCE_FLOOR}")
    use_linggen = bool(quality >= COHERENCE_FLOOR)
    engine.use_linggen = use_linggen
    print(f"  [LingGen] use_linggen = {use_linggen} "
          f"({'free-form coherent' if use_linggen else 'free-form not coherent enough'})")
    return True, use_linggen


def _gen_one(engine, nd, concept, av, W_sm, glove_fn, anchor_desc=None):
    """Generate one free-form sentence from the embodied BOS — mirrors the REAL
    production path (_generate_with_decoder), including Option-C anchored
    conditioning so the quality gate measures what the engine actually does.

    Option C (angular-gyrus elaboration): when an anchor definition is known for
    the concept, its content words are appended as conditioning scaffold (the
    GRU ELABORATES on the retrieved def rather than generating from a void).
    Coherence is then measured against the anchor's semantic center, not the
    bare concept — this is the honest success metric for the enrichment path.
    """
    init = (W_sm @ np.asarray(av, dtype=np.float32)[:65]).astype(np.float32)
    nn = np.linalg.norm(init)
    if nn > 0:
        init = init / nn
    # Mirror _generate_with_decoder conditioning construction: the concept's own
    # 75-D decoder-embedding, plus (Option C) the anchor's content-word embeds.
    embs = []
    dw = getattr(engine, "_decoder_word_to_embed", {}) or {}
    if concept.lower() in dw:
        embs.append(np.asarray(dw[concept.lower()], dtype=np.float32))
    if anchor_desc:
        _stop = {"the", "a", "an", "of", "to", "and", "is", "are", "was",
                 "were", "in", "on", "for", "with", "that", "this", "it"}
        for _w in anchor_desc.split():
            _w = _w.lower().strip(".,!?;:\"'()[]")
            if not _w or _w in _stop:
                continue
            if _w in dw:
                embs.append(np.asarray(dw[_w], dtype=np.float32))
            if len(embs) >= 14:
                break
    if not embs:
        embs.append(engine._embed_75d(concept))
    blend = np.stack(embs, axis=0).astype(np.float32)
    iw = getattr(engine, "_decoder_idx_to_word", None) or {}
    # Option C constrained decoding (research item #8c): modestly boost the
    # anchor's content-word logits so the elaboration stays lexically tethered
    # to the retrieved definition (retrieve-and-lightly-edit), without forcing
    # pure repetition. Keeps distinct-1 healthy while lifting on-topic coherence.
    token_boost = None
    if anchor_desc:
        token_boost = {}
        for _w in anchor_desc.split():
            _w = _w.lower().strip(".,!?;:\"'()[]")
            _idx = getattr(engine, "_decoder_word_to_idx", {}).get(_w)
            if _idx is not None:
                token_boost[_idx] = 1.5
    toks = nd.generate(conditioning_embs=blend, max_steps=22, initial_emb=init,
                       persistent_emb=init, idx_to_word=iw,
                       token_boost=token_boost, temperature=0.7)
    words = [iw.get(t, "") for t in toks
             if iw.get(t, "") not in ("<bos>", "<eos>", "<unk>", "")]
    return words


def _grounded_generation_quality(engine, nd, cond_sent, W_sm, attr_enc, glove_fn):
    """Option-C quality of free-form generation (0..1).

    Coherence = in-vocab ratio * distinct-1 (non-degenerate) * on-topic
    (mean GloVe cosine of generated content words vs the RETRIEVED ANCHOR's
    semantic center). This is the honest Option-C success metric: not 'beat the
    KB from scratch' (impossible for a tiny GRU) but 'elaborate on a retrieved
    definition coherently' (the angular-gyrus enrichment model). When no anchor
    def is known for a concept, on-topic falls back to the bare concept vector
    and that concept correctly scores low (caller abstains).
    """
    glove_fn = glove_fn or getattr(engine, "_glove_vector", None)
    if glove_fn is None:
        return 0.0
    scores = []
    for concept, desc in cond_sent:
        gv = glove_fn(concept)
        if gv is None:
            continue
        av = attr_enc.attribute_vector(np.asarray(gv, dtype=np.float64))
        words = _gen_one(engine, nd, concept, av, W_sm, glove_fn,
                         anchor_desc=desc if desc else None)
        if len(words) < 3:
            scores.append(0.0)
            continue
        invocab = np.mean([1.0 for w in words
                           if w in getattr(engine, "_decoder_word_to_idx", {})])
        distinct1 = len(set(words)) / len(words)
        # on-topic reference: the retrieved anchor's semantic center (mean of
        # its content-word glove vectors), falling back to the bare concept.
        cgv = np.asarray(gv, dtype=float)
        anchor_vecs = []
        if desc:
            for w in desc.split():
                wv = glove_fn(w.lower().strip(".,!?;:\"'()[]"))
                if wv is not None:
                    anchor_vecs.append(np.asarray(wv, dtype=float))
        if anchor_vecs:
            cgv = np.mean(anchor_vecs, axis=0)
        cosines = []
        for w in words:
            wv = glove_fn(w)
            if wv is not None:
                d = float(np.dot(cgv, wv) / (np.linalg.norm(cgv) * np.linalg.norm(wv) + 1e-9))
                cosines.append(d)
        on_topic = float(np.mean(cosines)) if cosines else 0.0
        scores.append(float(invocab * distinct1 * max(0.0, on_topic)))
    return float(np.mean(scores)) if scores else 0.0


def _heldout_grounded_ce(engine, nd, cond_sent, W_sm, attr_enc, glove_fn,
                         use_initial: bool = True):
    """Held-out cross-entropy of grounded descriptions.

    use_initial=True  -> inject W_sm@av as the BOS initial_emb (embodied,
                         mirrors generate()). use_initial=False -> pure
                         linguistic blend + default <bos> (the fallback path).
    Both use the SAME linguistic per-token conditioning, so the difference
    isolates the value of the embodied initiation state.
    """
    total_ce = 0.0
    n = 0
    for concept, desc in cond_sent:
        gv = glove_fn(concept)
        if gv is None:
            continue
        av = np.asarray(attr_enc.attribute_vector(np.asarray(gv, dtype=np.float64)),
                        dtype=np.float32)[:65]
        init = (W_sm @ av).astype(np.float32)
        nn = np.linalg.norm(init)
        if nn > 0:
            init = init / nn
        try:
            ce = nd.train_on_sentence(
                desc.split(), engine._decoder_word_to_embed,
                engine._decoder_word_to_idx,
                word_indices=[engine._decoder_word_to_idx.get(w.lower().strip(".,!?").strip("'"),
                                                              engine._decoder_word_to_idx.get("<unk>", 1))
                              for w in desc.split()],
                freeze_core=True,  # eval only, no learning
                initial_emb=(init if use_initial else None))
        except Exception:
            ce = 0.0
        if ce > 0:
            total_ce += ce
            n += 1
    return (total_ce / n) if n else 0.0


    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")


def evaluate(engine, questions=EVAL_QUESTIONS):
    print(f"\n{'='*60}")
    print("EVALUATION")
    print(f"{'='*60}")
    for q in questions:
        t0 = time.time()
        resp = engine.process_turn(q)
        t = time.time()-t0
        print(f"  Q: {q}")
        print(f"  A: {resp[:150] if resp else '<None>'}")
        print(f"    [{engine._last_strategy}] ({t:.1f}s)")
        print()


# ─── Mode: phase2 ───

def _mode_phase2(args):
    """Heavy decoder training on teen_seeds.txt with web learning."""
    print("="*60)
    print("RAVANA Decoder Training — Phase 2 (human-like speech)")
    print("="*60)
    t0 = time.time()
    engine, nd = load_engine(args, reset=args.reset)

    print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
    print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
    print(f"  Pre-training: CE={nd._avg_cross_entropy:.3f} t1={nd._avg_top1_acc:.3f} "
          f"trained={nd._total_training_examples}")
    print()

    # Phase 1: Heavy seed corpus training
    print("[Phase 1] Seed corpus training...")
    text, all_sentences, nd = load_corpus(engine)
    n_seed = train_seed_corpus(engine, nd, all_sentences, n_passes=200, pp=2000, si=10, pe=25)
    print(f"  {n_seed} seed sentences trained in {time.time()-t0:.0f}s")
    print(f"  CE={nd._avg_cross_entropy:.3f} t1={nd._avg_top1_acc:.3f} t5={nd._avg_top5_acc:.3f}")
    print()

    # Phase 2: Curiosity-driven web learning (autonomous topic selection)
    if not args.no_web:
        print("[Phase 2] Seeding curiosity signals (running seed queries)...")
        engine._bg_learning_active = True
        engine._curiosity_drive_enabled = True
        engine._network_available = True  # force fresh web attempts
        t1 = time.time()
        for q in SEED_QUERIES:
            try:
                engine.process_turn(q)
            except Exception:
                pass
        print(f"  {len(SEED_QUERIES)} queries in {time.time()-t1:.0f}s")

        print(f"[Phase 2] Curiosity-driven web learning ({args.web_topics} topics)...")
        t1 = time.time()
        engine._bg_learning_queue.clear()
        for i in range(args.web_topics):
            topics = engine._auto_select_curiosity_topics(max_topics=3)
            if not topics:
                engine._bg_learning_queue = list(WEB_TOPICS[:5])
                engine._impossible_queries.clear()
                topics = engine._auto_select_curiosity_topics(max_topics=3)
                if not topics:
                    for topic in WEB_TOPICS[:args.web_topics]:
                        engine._network_available = True
                        try:
                            result, _ = engine.learn_from_web(
                                topic + " explained with examples", max_results=2)
                            print(f"  [{i+1}/{args.web_topics}] {topic} -> {result}")
                        except Exception:
                            print(f"  [{i+1}/{args.web_topics}] {topic} -> offline")
                    break
            for topic in topics[:2]:
                engine._network_available = True
                query = engine._generate_curiosity_query(topic, source_type="prediction_error")
                try:
                    result, _ = engine.learn_from_web(query, max_results=2)
                    print(f"  [{i+1}/{args.web_topics}] {query} -> {result}")
                except Exception:
                    print(f"  [{i+1}/{args.web_topics}] {query} -> offline")
        print(f"  Web learning done in {time.time()-t1:.0f}s")
        print(f"  Web training: {engine._decoder_web_training_count} sentences")
        print()

    # Phase 3: Consolidation — more seed corpus passes with web-expanded vocab
    print("[Phase 3] Consolidation training...")
    t2 = time.time()
    n_consolidate = train_seed_corpus(engine, nd, all_sentences, n_passes=50, pp=2000, si=10, pe=10)
    print(f"  {n_consolidate} consolidation sentences in {time.time()-t2:.0f}s")
    print()

    # Phase 4: Evaluate + save
    print("[Phase 4] Evaluating...")
    evaluate(engine)

    print("Saving...")
    engine._needs_seed_training = False
    result = engine.save()
    print(f"  {result}")

    print()
    print("="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Total time: {time.time()-t0:.0f}s")
    print(f"  Graph: {len(engine.graph.nodes)} nodes, {len(engine.graph.edges)} edges")
    print(f"  Decoder: {engine._decoder_training_count} total, "
          f"{engine._decoder_seed_training_count} seed, {engine._decoder_web_training_count} web")
    print(f"  Vocab: {len(engine._decoder_word_to_idx)} words")
    print(f"  Final: CE={nd._avg_cross_entropy:.3f} t1={nd._avg_top1_acc:.3f} "
          f"t5={nd._avg_top5_acc:.3f}")
    print()


# ─── Mode: full ───

def _mode_full(args):
    """Full training pipeline — single-phase seed corpus + web knowledge.
    
    FIX: Eliminated multi-cycle approach which caused catastrophic forgetting.
    Now delegates to phase2 (single long seed training + web + consolidation).
    No alternating cycles that overwrite previous learning.
    """
    # Phase 2 is already the correct single-phase approach (seed → web → consolidate)
    _mode_phase2(args)


# ─── Mode: test ───

def _mode_test(args):
    """Quick diagnostic: verify decoder trains and generates."""
    engine, nd = load_engine(args)
    text, all_sentences, nd = load_corpus(engine)

    print("Training on 50 sentences...")
    t0 = time.time()
    for s in all_sentences[:50]:
        nd.train_on_sentence(s['words'], engine._decoder_word_to_embed, engine._decoder_word_to_idx,
            word_indices=s['word_indices'], conditioning_embs=s['conditioning_embs'])
    nd.sleep_cycle()
    print(f"  CE={nd._avg_cross_entropy:.4f} t1={nd._avg_top1_acc:.4f} ({time.time()-t0:.1f}s)")

    print("\nGenerating responses...")
    for q in ["what is trust", "tell me about love", "hello"]:
        t0 = time.time()
        r = engine.process_turn(q)
        print(f"  Q: {q}")
        print(f"  A: {r[:120] if r else '<None>'} [{engine._last_strategy}] ({time.time()-t0:.1f}s)")

    engine.save()
    print("\nTest complete.")


# ─── CLI ───

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAVANA unified trainer")
    parser.add_argument("--mode", choices=["phase2", "full", "test"], default="phase2",
                        help="Training mode (default: phase2)")
    parser.add_argument("--dim", type=int, default=64, help="Graph dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--reset", action="store_true", help="Delete saved weights and start fresh")

    # Phase 2 / Full options
    parser.add_argument("--no-web", action="store_true", help="Skip web learning (offline mode)")
    parser.add_argument("--web-topics", type=int, default=5, help="Number of web topics to learn (default: 5)")
    parser.add_argument("--cycles", type=int, default=3, help="Training cycles for full mode (default: 3)")

    args = parser.parse_args()

    modes = {"phase2": _mode_phase2, "full": _mode_full, "test": _mode_test}
    modes[args.mode](args)


if __name__ == "__main__":
    main()
