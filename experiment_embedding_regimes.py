1|"""
2|Embedding Regime Experiment
3|============================
4|Tests whether RLMv2's graph topology can USE semantic structure when it exists.
5|
6|Three regimes + nearest-neighbor baseline:
7|  A. Random embeddings (current state — expected: 0% transfer)
8|  B. Hand-crafted semantic embeddings (2D — proves topology works)
9|  C. Co-occurrence learned embeddings (data-driven — realistic)
10|  D. Nearest-neighbor baseline (no graph — isolates graph contribution)
11|
12|For every transfer query, we compare:
13|  - Graph prediction (activation spreading + edge traversal)
14|  - Nearest-neighbor retrieval (cosine similarity in embedding space)
15|
16|If graph ≈ NN, the graph isn't adding reasoning.
17|If graph > NN, the graph is genuinely composing knowledge.
18|"""
19|
20|import sys
21|import io
22|import numpy as np
23|from collections import defaultdict
24|
25|sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
26|
27|from ravana_ml.nn.rlm_v2 import RLMv2
28|from ravana_ml.tokenizer import WordTokenizer
29|
30|
31|def build_facts():
32|    """Structured facts with clear semantic categories."""
33|    facts = [
34|        # Animals → attributes
35|        ("cat", "has", "tail"),
36|        ("cat", "has", "whiskers"),
37|        ("dog", "has", "tail"),
38|        ("dog", "has", "nose"),
39|        ("bird", "has", "wing"),
40|        ("bird", "has", "beak"),
41|        ("fish", "has", "fin"),
42|        ("fish", "has", "scale"),
43|        ("horse", "has", "tail"),
44|        ("horse", "has", "hoof"),
45|        # Animals → actions
46|        ("cat", "can", "purr"),
47|        ("dog", "can", "bark"),
48|        ("bird", "can", "fly"),
49|        ("fish", "can", "swim"),
50|        ("horse", "can", "gallop"),
51|        # Science causal
52|        ("heat", "causes", "expansion"),
53|        ("cold", "causes", "contraction"),
54|        ("friction", "causes", "wear"),
55|        ("pressure", "causes", "deformation"),
56|        ("voltage", "causes", "current"),
57|        # Social causal
58|        ("kindness", "causes", "trust"),
59|        ("anger", "causes", "conflict"),
60|        ("patience", "causes", "understanding"),
61|        ("honesty", "causes", "respect"),
62|        ("generosity", "causes", "gratitude"),
63|    ]
64|    return facts
65|
66|
67|def build_transfer_tests():
68|    """Transfer queries with expected answers."""
69|    tests = [
70|        # Same-relation novel subject (the critical test)
71|        {"query": "tiger has", "expected": "tail", "category": "animal_attr",
72|         "reasoning": "tiger≈cat,dog,horse → has tail"},
73|        {"query": "eagle has", "expected": "wing", "category": "animal_attr",
74|         "reasoning": "eagle≈bird → has wing"},
75|        {"query": "shark has", "expected": "fin", "category": "animal_attr",
76|         "reasoning": "shark≈fish → has fin"},
77|        {"query": "wolf has", "expected": "tail", "category": "animal_attr",
78|         "reasoning": "wolf≈dog → has tail"},
79|        {"query": "parrot can", "expected": "fly", "category": "animal_action",
80|         "reasoning": "parrot≈bird → can fly"},
81|        {"query": "whale can", "expected": "swim", "category": "animal_action",
82|         "reasoning": "whale≈fish → can swim"},
83|        # Cross-domain causal
84|        {"query": "warmth causes", "expected": "trust", "category": "cross_causal",
85|         "reasoning": "warmth≈kindness → causes trust"},
86|        {"query": "rudeness causes", "expected": "conflict", "category": "cross_causal",
87|         "reasoning": "rudeness≈anger → causes conflict"},
88|        {"query": "loyalty causes", "expected": "respect", "category": "cross_causal",
89|         "reasoning": "loyalty≈honesty → causes respect"},
90|        {"query": "voltage causes", "expected": "current", "category": "in_domain_causal",
91|         "reasoning": "voltage is in training, just checking memorization"},
92|    ]
93|    return tests
94|
95|
96|# ─── Regime A: Random Embeddings ───────────────────────────────────────────
97|
98|def build_word_tokenizer(facts, transfer_tests):
99|    """Build vocab from all texts."""
100|    tok = WordTokenizer()
101|    for s, r, o in facts:
102|        tok.encode(f"{s} {r} {o}")
103|    for t in transfer_tests:
104|        tok.encode(t["query"])
105|    return tok
106|
107|
108|def run_regime_a(facts, transfer_tests):
109|    """Baseline: random embeddings. Expected: ~0% transfer."""
110|    print("\n" + "="*70)
111|    print("REGIME A: RANDOM EMBEDDINGS")
112|    print("="*70)
113|
114|    tok = build_word_tokenizer(facts, transfer_tests)
115|
116|    model = RLMv2(
117|        vocab_size=tok.vocab_size,
118|        embed_dim=32,
119|        concept_dim=32,
120|        n_concepts=500,
121|        sleep_interval=200,
122|        gate_concept_creation=False,
model._tokenizer = tok
123|    )
124|
125|    # Train
126|    for epoch in range(3):
127|        for s, r, o in facts:
128|            text = f"{s} {r} {o}"
129|            ids = tok.encode(text)
130|            if len(ids) < 2:
131|                continue
132|            ctx = np.array([ids[:-1]], dtype=np.int64)
133|            tgt = np.array([[ids[-1]]], dtype=np.int64)
134|            model.learn(ctx, tgt)
135|
136|    # Evaluate
137|    results = evaluate_transfer(model, tok, transfer_tests, "A")
138|    nn_results = evaluate_nn_baseline(model, tok, facts, transfer_tests)
139|    return results, nn_results
140|
141|
142|# ─── Regime B: Hand-Crafted Semantic Embeddings ────────────────────────────
143|
144|def run_regime_b(facts, transfer_tests):
145|    """Hand-crafted 2D embeddings encoding semantic categories."""
146|    print("\n" + "="*70)
147|    print("REGIME B: HAND-CRAFTED SEMANTIC EMBEDDINGS")
148|    print("="*70)
149|
150|    tok = build_word_tokenizer(facts, transfer_tests)
151|
152|    model = RLMv2(
153|        vocab_size=tok.vocab_size,
154|        embed_dim=32,
155|        concept_dim=32,
156|        n_concepts=500,
157|        sleep_interval=200,
158|        gate_concept_creation=False,
model._tokenizer = tok
159|    )
160|
161|    # Inject hand-crafted embeddings
162|    # Dimension 0: animal vs object vs abstract
163|    # Dimension 1: size / intensity
164|    # Dimensions 2-31: noise (to fill 32D)
165|    semantic_vectors = {
166|        # Animals (dim0=1.0, dim1=varies by size)
167|        "cat":     [1.0, 0.4, 0.8, 0.2],
168|        "dog":     [1.0, 0.5, 0.7, 0.3],
169|        "bird":    [1.0, 0.2, 0.6, 0.1],
170|        "fish":    [1.0, 0.3, 0.5, 0.4],
171|        "horse":   [1.0, 0.8, 0.9, 0.1],
172|        "tiger":   [1.0, 0.6, 0.85, 0.25],  # close to cat
173|        "eagle":   [1.0, 0.3, 0.65, 0.15],  # close to bird
174|        "shark":   [1.0, 0.7, 0.55, 0.45],  # close to fish
175|        "wolf":    [1.0, 0.55, 0.72, 0.32], # close to dog
176|        "parrot":  [1.0, 0.15, 0.62, 0.12], # close to bird
177|        "whale":   [1.0, 0.9, 0.52, 0.42],  # close to fish (aquatic)
178|
179|        # Attributes (dim0=0.0, dim1=body part type)
180|        "tail":    [0.0, 0.8, 0.3, 0.1],
181|        "whiskers":[0.0, 0.3, 0.35, 0.15],
182|        "nose":    [0.0, 0.4, 0.4, 0.2],
183|        "wing":    [0.0, 0.6, 0.2, 0.7],
184|        "beak":    [0.0, 0.5, 0.25, 0.65],
185|        "fin":     [0.0, 0.7, 0.15, 0.8],
186|        "scale":   [0.0, 0.2, 0.1, 0.9],
187|        "hoof":    [0.0, 0.9, 0.45, 0.05],
188|
189|        # Actions (dim0=-0.5, dim1=type)
190|        "purr":    [-0.5, 0.3, 0.1, 0.2],
191|        "bark":    [-0.5, 0.5, 0.15, 0.25],
192|        "fly":     [-0.5, 0.8, 0.2, 0.7],
193|        "swim":    [-0.5, 0.6, 0.1, 0.8],
194|        "gallop":  [-0.5, 0.9, 0.3, 0.1],
195|
196|        # Relations
197|        "has":     [0.5, 0.0, 0.0, 0.0],
198|        "can":     [0.5, 0.0, 0.0, 0.0],
199|        "causes":  [0.5, 0.0, 0.0, 0.0],
200|
201|        # Science (dim0=-1.0, dim1=intensity)
202|        "heat":        [-1.0, 0.8, 0.9, 0.1],
203|        "cold":        [-1.0, 0.2, 0.85, 0.15],
204|        "friction":    [-1.0, 0.6, 0.7, 0.3],
205|        "pressure":    [-1.0, 0.7, 0.75, 0.25],
206|        "voltage":     [-1.0, 0.5, 0.8, 0.2],
207|        "expansion":   [-1.0, 0.3, 0.6, 0.4],
208|        "contraction": [-1.0, 0.2, 0.55, 0.45],
209|        "wear":        [-1.0, 0.4, 0.5, 0.5],
210|        "deformation": [-1.0, 0.5, 0.45, 0.55],
211|        "current":     [-1.0, 0.6, 0.4, 0.6],
212|
213|        # Social (dim0=-1.0, dim1=intensity — close to science for cross-domain)
214|        "kindness":     [-1.0, 0.75, 0.88, 0.12],  # close to heat
215|        "anger":        [-1.0, 0.65, 0.82, 0.28],  # close to friction
216|        "patience":     [-1.0, 0.55, 0.73, 0.33],  # close to pressure
217|        "honesty":      [-1.0, 0.45, 0.78, 0.22],  # close to voltage
218|        "generosity":   [-1.0, 0.7, 0.65, 0.35],   # close to heat
219|        "trust":        [-1.0, 0.35, 0.58, 0.42],   # close to expansion
220|        "conflict":     [-1.0, 0.25, 0.53, 0.47],   # close to contraction
221|        "understanding":[-1.0, 0.45, 0.48, 0.52],   # close to wear
222|        "respect":      [-1.0, 0.55, 0.43, 0.57],   # close to deformation
223|        "gratitude":    [-1.0, 0.65, 0.38, 0.62],   # close to current
224|
225|        # Novel transfer words
226|        "warmth":   [-1.0, 0.73, 0.87, 0.13],  # very close to kindness
227|        "rudeness": [-1.0, 0.63, 0.83, 0.27],  # very close to anger
228|        "loyalty":  [-1.0, 0.47, 0.77, 0.23],  # very close to honesty
229|    }
230|
231|    # Inject into embedding table
232|    for word, vec4 in semantic_vectors.items():
233|        tid = tok.word_to_id.get(word)
234|        if tid is None:
235|            continue
236|        # Expand 4D to 32D by repeating + small noise
237|        full_vec = np.zeros(32, dtype=np.float32)
238|        for i in range(32):
239|            full_vec[i] = vec4[i % 4] + np.random.randn() * 0.01
240|        full_vec /= np.linalg.norm(full_vec)
241|        model.token_embed.weight.data[tid] = full_vec
242|
243|    # Train
244|    for epoch in range(3):
245|        for s, r, o in facts:
246|            text = f"{s} {r} {o}"
247|            ids = tok.encode(text)
248|            if len(ids) < 2:
249|                continue
250|            ctx = np.array([ids[:-1]], dtype=np.int64)
251|            tgt = np.array([[ids[-1]]], dtype=np.int64)
252|            model.learn(ctx, tgt)
253|
254|    # Evaluate
255|    results = evaluate_transfer(model, tok, transfer_tests, "B")
256|    nn_results = evaluate_nn_baseline(model, tok, facts, transfer_tests)
257|    return results, nn_results
258|
259|
260|# ─── Regime C: Co-occurrence Learned Embeddings ────────────────────────────
261|
262|def run_regime_c(facts, transfer_tests):
263|    """Co-occurrence skip-gram style pre-training."""
264|    print("\n" + "="*70)
265|    print("REGIME C: CO-OCCURRENCE LEARNED EMBEDDINGS")
266|    print("="*70)
267|
268|    tok = build_word_tokenizer(facts, transfer_tests)
269|
270|    model = RLMv2(
271|        vocab_size=tok.vocab_size,
272|        embed_dim=32,
273|        concept_dim=32,
274|        n_concepts=500,
275|        sleep_interval=200,
276|        gate_concept_creation=False,
model._tokenizer = tok
277|    )
278|
279|    # Pre-train embeddings via co-occurrence
280|    # Build co-occurrence matrix from facts
281|    word_ids = {}
282|    for s, r, o in facts:
283|        for w in [s, r, o]:
284|            tid = tok.word_to_id.get(w)
285|            if tid is not None:
286|                word_ids[w] = tid
287|
288|    # Co-occurrence counting (window=2)
289|    cooc = defaultdict(lambda: defaultdict(int))
290|    for s, r, o in facts:
291|        words = [s, r, o]
292|        for i, w1 in enumerate(words):
293|            for j, w2 in enumerate(words):
294|                if i != j:
295|                    t1 = tok.word_to_id.get(w1)
296|                    t2 = tok.word_to_id.get(w2)
297|                    if t1 is not None and t2 is not None:
298|                        cooc[t1][t2] += 1
299|
300|    # Skip-gram training: update embeddings based on co-occurrence
301|    lr = 0.1
302|    for epoch in range(50):
303|        total_loss = 0
304|        for t1, neighbors in cooc.items():
305|            for t2, count in neighbors.items():
306|                # Pull co-occurring words closer
307|                v1 = model.token_embed.weight.data[t1]
308|                v2 = model.token_embed.weight.data[t2]
309|                diff = v1 - v2
310|                loss = np.dot(diff, diff) * count
311|                total_loss += loss
312|                grad = 2 * diff * count * lr / (epoch + 1)
313|                model.token_embed.weight.data[t1] -= grad
314|                model.token_embed.weight.data[t2] += grad
315|
316|        # Normalize
317|        norms = np.linalg.norm(model.token_embed.weight.data, axis=1, keepdims=True)
318|        norms = np.maximum(norms, 1e-8)
319|        model.token_embed.weight.data /= norms
320|
321|    # Show learned similarities
322|    print("\nLearned embedding similarities (top pairs):")
323|    embeds = model.token_embed.weight.data
324|    sims = []
325|    word_list = list(word_ids.items())
326|    for i, (w1, t1) in enumerate(word_list):
327|        for j, (w2, t2) in enumerate(word_list):
328|            if i < j:
329|                cos = float(np.dot(embeds[t1], embeds[t2]))
330|                sims.append((cos, w1, w2))
331|    sims.sort(reverse=True)
332|    for cos, w1, w2 in sims[:15]:
333|        print(f"  {w1:12s} <-> {w2:12s}: {cos:.3f}")
334|
335|    # Train
336|    for epoch in range(3):
337|        for s, r, o in facts:
338|            text = f"{s} {r} {o}"
339|            ids = tok.encode(text)
340|            if len(ids) < 2:
341|                continue
342|            ctx = np.array([ids[:-1]], dtype=np.int64)
343|            tgt = np.array([[ids[-1]]], dtype=np.int64)
344|            model.learn(ctx, tgt)
345|
346|    # Evaluate
347|    results = evaluate_transfer(model, tok, transfer_tests, "C")
348|    nn_results = evaluate_nn_baseline(model, tok, facts, transfer_tests)
349|    return results, nn_results
350|
351|
352|# ─── Evaluation Helpers ────────────────────────────────────────────────────
353|
354|def evaluate_transfer(model, tok, tests, regime_label):
355|    """Evaluate transfer using graph-based prediction."""
356|    print(f"\n--- Regime {regime_label}: Graph-Based Transfer ---")
357|    results = []
358|    for test in tests:
359|        query = test["query"]
360|        expected = test["expected"]
361|        category = test["category"]
362|
363|        ids = tok.encode(query)
364|        if len(ids) == 0:
365|            results.append({"test": test, "hit": False, "top5": []})
366|            continue
367|
368|        ctx = np.array([ids], dtype=np.int64)
369|        logits = np.asarray(model.forward(ctx).data).flatten()
370|        top5_ids = list(np.argsort(logits)[::-1][:5])
371|        top5_words = [tok.decode([tid]) for tid in top5_ids]
372|
373|        hit = expected in top5_words
374|        exp_tid = tok.word_to_id.get(expected, -1)
375|        top10 = set(np.argsort(logits)[::-1][:10])
376|        hit10 = exp_tid in top10
377|
378|        marker = "✓" if hit10 else "✗"
379|        print(f"  {marker} {query:20s} → expected: {expected:12s} | top5: {top5_words}")
380|
381|        results.append({
382|            "test": test,
383|            "hit10": hit10,
384|            "hit5": hit,
385|            "top5": top5_words,
386|            "top1": top5_words[0] if top5_words else "",
387|        })
388|
389|    hit10_count = sum(1 for r in results if r["hit10"])
390|    hit5_count = sum(1 for r in results if r["hit5"])
391|    print(f"\n  Regime {regime_label} Graph: {hit10_count}/{len(results)} top-10, "
392|          f"{hit5_count}/{len(results)} top-5")
393|    return results
394|
395|
396|def evaluate_nn_baseline(model, tok, facts, tests):
397|    """Nearest-neighbor baseline: no graph, just embedding similarity."""
398|    print(f"\n--- Nearest-Neighbor Baseline (no graph) ---")
399|
400|    # Build a simple mapping: for each relation, map subject → object
401|    # Then for novel subjects, find nearest trained subject and return its object
402|    rel_to_pairs = defaultdict(list)
403|    for s, r, o in facts:
404|        rel_to_pairs[r].append((s, o))
405|
406|    embeds = model.token_embed.weight.data
407|    results = []
408|
409|    for test in tests:
410|        query = test["query"]
411|        expected = test["expected"]
412|
413|        # Parse query: "tiger has" → subject=tiger, relation=has
414|        parts = query.split()
415|        if len(parts) < 2:
416|            results.append({"test": test, "hit10": False, "hit5": False})
417|            continue
418|        subj_word = parts[0]
419|        rel_word = parts[1]
420|
421|        subj_tid = tok.word_to_id.get(subj_word)
422|        if subj_tid is None:
423|            results.append({"test": test, "hit10": False, "hit5": False})
424|            continue
425|
426|        subj_vec = embeds[subj_tid]
427|
428|        # Find nearest trained subjects for this relation
429|        pairs = rel_to_pairs.get(rel_word, [])
430|        if not pairs:
431|            results.append({"test": test, "hit10": False, "hit5": False})
432|            continue
433|
434|        scored = []
435|        for s, o in pairs:
436|            s_tid = tok.word_to_id.get(s)
437|            o_tid = tok.word_to_id.get(o)
438|            if s_tid is None or o_tid is None:
439|                continue
440|            sim = float(np.dot(subj_vec, embeds[s_tid]))
441|            scored.append((sim, o, o_tid))
442|
443|        scored.sort(reverse=True)
444|
445|        # Top-1 NN prediction
446|        if scored:
447|            nn_top1 = scored[0][1]
448|            # Top-5 NN predictions (unique objects)
449|            seen = set()
450|            nn_top5 = []
451|            for sim, o, o_tid in scored:
452|                if o not in seen:
453|                    seen.add(o)
454|                    nn_top5.append(o)
455|                if len(nn_top5) >= 5:
456|                    break
457|        else:
458|            nn_top1 = "?"
459|            nn_top5 = []
460|
461|        exp_tid = tok.word_to_id.get(expected, -1)
462|        nn_top5_tids = [tok.word_to_id.get(w, -1) for w in nn_top5]
463|        hit5 = expected in nn_top5
464|        hit10 = expected in nn_top5  # NN only returns up to 5 unique objects
465|
466|        marker = "✓" if hit5 else "✗"
467|        neighbors_str = ", ".join(f"{s}({sim:.2f})" for sim, s, _ in scored[:3])
468|        print(f"  {marker} {test['query']:20s} → expected: {expected:12s} | "
469|              f"NN top1: {nn_top1:12s} | neighbors: {neighbors_str}")
470|
471|        results.append({
472|            "test": test,
473|            "hit10": hit10,
474|            "hit5": hit5,
475|            "nn_top1": nn_top1,
476|            "nn_top5": nn_top5,
477|        })
478|
479|    hit5_count = sum(1 for r in results if r["hit5"])
480|    print(f"\n  NN Baseline: {hit5_count}/{len(results)} top-5")
481|    return results
482|
483|
484|# ─── Main ──────────────────────────────────────────────────────────────────
485|
486|def main():
487|    print("="*70)
488|    print("EMBEDDING REGIME EXPERIMENT")
489|    print("Testing whether graph topology can USE semantic structure")
490|    print("="*70)
491|
492|    facts = build_facts()
493|    transfer_tests = build_transfer_tests()
494|
495|    print(f"\nTraining facts: {len(facts)}")
496|    print(f"Transfer tests: {len(transfer_tests)}")
497|
498|    # Run all three regimes
499|    a_graph, a_nn = run_regime_a(facts, transfer_tests)
500|    b_graph, b_nn = run_regime_b(facts, transfer_tests)
501|