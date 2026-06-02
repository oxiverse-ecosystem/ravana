1|"""
2|Task Taxonomy Experiment
3|=========================
4|Identifies which cognitive task the graph is actually good at.
5|
6|Three task types:
7|  RETRIEVAL:  "cat has tail, tiger has ?"  → NN should win
8|  MULTI-HOP:  "A causes B, B causes C, A causes ?"  → Graph should win
9|  HYBRID:     "warmth ≈ kindness, kindness causes trust, warmth causes ?"  → Both needed
10|
11|For each task type, we measure:
12|  - Embedding-only (NN baseline)
13|  - Graph-only (no embedding similarity, only edges/activation)
14|  - Combined (current forward() blend)
15|
16|The critical number: "Graph right, Embedding wrong" count.
17|If zero for all task types, the graph has no niche.
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
28|from ravana_ml.word_tokenizer import WordTokenizer
29|
30|
31|# ─── Task Definitions ──────────────────────────────────────────────────────
32|
33|def build_training_facts():
34|    """Core training facts — common across all task types."""
35|    return [
36|        # Animal attributes (for retrieval tests)
37|        ("cat", "has", "tail"),
38|        ("cat", "has", "whiskers"),
39|        ("dog", "has", "tail"),
40|        ("dog", "has", "nose"),
41|        ("bird", "has", "wing"),
42|        ("bird", "has", "beak"),
43|        ("fish", "has", "fin"),
44|        ("fish", "has", "scale"),
45|        ("horse", "has", "hoof"),
46|
47|        # Animal actions
48|        ("cat", "can", "purr"),
49|        ("dog", "can", "bark"),
50|        ("bird", "can", "fly"),
51|        ("fish", "can", "swim"),
52|
53|        # 2-hop causal chains (for multi-hop tests)
54|        ("virus", "causes", "illness"),
55|        ("illness", "causes", "absence"),
56|        ("bug", "causes", "crash"),
57|        ("crash", "causes", "outage"),
58|        ("spark", "causes", "fire"),
59|        ("fire", "causes", "damage"),
60|        ("lie", "causes", "distrust"),
61|        ("distrust", "causes", "isolation"),
62|
63|        # Simple causal (for hybrid tests)
64|        ("kindness", "causes", "trust"),
65|        ("anger", "causes", "conflict"),
66|        ("honesty", "causes", "respect"),
67|        ("heat", "causes", "expansion"),
68|        ("cold", "causes", "contraction"),
69|    ]
70|
71|
72|def build_retrieval_tests():
73|    """Simple nearest-neighbor retrieval. Embeddings should win."""
74|    return [
75|        {"query": "tiger has",    "expected": "tail",  "note": "tiger≈cat,dog,horse"},
76|        {"query": "eagle has",    "expected": "wing",  "note": "eagle≈bird"},
77|        {"query": "shark has",    "expected": "fin",   "note": "shark≈fish"},
78|        {"query": "wolf has",     "expected": "tail",  "note": "wolf≈dog"},
79|        {"query": "parrot can",   "expected": "fly",   "note": "parrot≈bird"},
80|        {"query": "whale can",    "expected": "swim",  "note": "whale≈fish"},
81|    ]
82|
83|
84|def build_multihop_tests():
85|    """2-hop chain inference. Graph should win (embeddings can't chain)."""
86|    return [
87|        {"query": "virus causes",  "expected": "absence",  "chain": "virus→illness→absence"},
88|        {"query": "bug causes",    "expected": "outage",   "chain": "bug→crash→outage"},
89|        {"query": "spark causes",  "expected": "damage",   "chain": "spark→fire→damage"},
90|        {"query": "lie causes",    "expected": "isolation", "chain": "lie→distrust→isolation"},
91|    ]
92|
93|
94|def build_hybrid_tests():
95|    """Need both: embedding similarity + graph edge traversal."""
96|    return [
97|        {"query": "warmth causes",    "expected": "trust",
98|         "note": "warmth≈kindness, kindness→trust"},
99|        {"query": "rudeness causes",  "expected": "conflict",
100|         "note": "rudeness≈anger, anger→conflict"},
101|        {"query": "loyalty causes",   "expected": "respect",
102|         "note": "loyalty≈honesty, honesty→respect"},
103|        {"query": "frigidity causes", "expected": "contraction",
104|         "note": "frigidity≈cold, cold→contraction"},
105|    ]
106|
107|
108|# ─── Semantic Embeddings (Regime B-style, hand-crafted) ───────────────────
109|
110|def inject_semantic_embeddings(model, tok):
111|    """Inject hand-crafted embeddings for meaningful experiments."""
112|    semantic = {
113|        # Animals
114|        "cat":     [1.0, 0.4, 0.8],   "dog":     [1.0, 0.5, 0.7],
115|        "bird":    [1.0, 0.2, 0.6],   "fish":    [1.0, 0.3, 0.5],
116|        "horse":   [1.0, 0.8, 0.9],
117|        "tiger":   [1.0, 0.45, 0.82],  # ≈ cat
118|        "eagle":   [1.0, 0.22, 0.62],  # ≈ bird
119|        "shark":   [1.0, 0.35, 0.52],  # ≈ fish
120|        "wolf":    [1.0, 0.52, 0.72],  # ≈ dog
121|        "parrot":  [1.0, 0.18, 0.58],  # ≈ bird
122|        "whale":   [1.0, 0.75, 0.48],  # ≈ fish (aquatic)
123|
124|        # Attributes
125|        "tail":    [0.0, 0.8, 0.3],   "whiskers":[0.0, 0.3, 0.35],
126|        "nose":    [0.0, 0.4, 0.4],   "wing":    [0.0, 0.6, 0.2],
127|        "beak":    [0.0, 0.5, 0.25],  "fin":     [0.0, 0.7, 0.15],
128|        "scale":   [0.0, 0.2, 0.1],   "hoof":    [0.0, 0.9, 0.45],
129|
130|        # Actions
131|        "purr":    [-0.5, 0.3, 0.1],  "bark":    [-0.5, 0.5, 0.15],
132|        "fly":     [-0.5, 0.8, 0.2],  "swim":    [-0.5, 0.6, 0.1],
133|
134|        # Relations
135|        "has":     [0.5, 0.0, 0.0],   "can":     [0.5, 0.0, 0.0],
136|        "causes":  [0.5, 0.0, 0.0],
137|
138|        # Causal chain agents (grouped by domain, with chain proximity)
139|        "virus":   [-1.0, 0.8, 0.2],  "illness": [-1.0, 0.5, 0.3],
140|        "absence": [-1.0, 0.3, 0.4],  # virus→illness→absence
141|        "bug":     [-1.0, 0.75, 0.15],"crash":   [-1.0, 0.5, 0.25],
142|        "outage":  [-1.0, 0.3, 0.35], # bug→crash→outage
143|        "spark":   [-1.0, 0.7, 0.18], "fire":    [-1.0, 0.5, 0.28],
144|        "damage":  [-1.0, 0.3, 0.38], # spark→fire→damage
145|        "lie":     [-1.0, 0.72, 0.12],"distrust":[-1.0, 0.48, 0.22],
146|        "isolation":[0.3, 0.28, 0.32],# lie→distrust→isolation
147|
148|        # Social causal (for hybrid tests)
149|        "kindness":  [-1.0, 0.85, 0.15],
150|        "anger":     [-1.0, 0.65, 0.35],
151|        "honesty":   [-1.0, 0.75, 0.25],
152|        "heat":      [-1.0, 0.9, 0.1],
153|        "cold":      [-1.0, 0.1, 0.9],
154|        "trust":     [-1.0, 0.4, 0.2],
155|        "conflict":  [-1.0, 0.3, 0.4],
156|        "respect":   [-1.0, 0.35, 0.3],
157|        "expansion": [-1.0, 0.5, 0.15],
158|        "contraction":[0.1, 0.15, 0.85],
159|
160|        # Novel hybrid words (similar to trained agents)
161|        "warmth":    [-1.0, 0.83, 0.17],  # ≈ kindness
162|        "rudeness":  [-1.0, 0.63, 0.37],  # ≈ anger
163|        "loyalty":   [-1.0, 0.73, 0.27],  # ≈ honesty
164|        "frigidity": [-1.0, 0.12, 0.88],  # ≈ cold
165|    }
166|
167|    dim = model.embed_dim
168|    for word, vec3 in semantic.items():
169|        tid = tok.word_to_id.get(word)
170|        if tid is None:
171|            continue
172|        full = np.zeros(dim, dtype=np.float32)
173|        for i in range(dim):
174|            full[i] = vec3[i % 3] + np.random.randn() * 0.005
175|        full /= np.linalg.norm(full)
176|        model.token_embed.weight.data[tid] = full
177|
178|
179|# ─── Three Evaluation Modes ────────────────────────────────────────────────
180|
181|def eval_embedding_only(model, tok, query, expected, facts):
182|    """Pure nearest-neighbor: find nearest trained subject, return its object."""
183|    parts = query.split()
184|    if len(parts) < 2:
185|        return False, "?", []
186|    subj_word, rel_word = parts[0], parts[1]
187|
188|    subj_tid = tok.word_to_id.get(subj_word)
189|    if subj_tid is None:
190|        return False, "?", []
191|
192|    embeds = model.token_embed.weight.data
193|    subj_vec = embeds[subj_tid]
194|
195|    # Gather all (subject, object) pairs for this relation
196|    pairs = [(s, o) for s, r, o in facts if r == rel_word]
197|    scored = []
198|    for s, o in pairs:
199|        s_tid = tok.word_to_id.get(s)
200|        if s_tid is None:
201|            continue
202|        sim = float(np.dot(subj_vec, embeds[s_tid]))
203|        scored.append((sim, o))
204|    scored.sort(reverse=True)
205|
206|    top5 = [o for _, o in scored[:5]]
207|    return expected in top5, top5[0] if top5 else "?", top5
208|
209|
210|def eval_graph_only(model, tok, query, expected):
211|    """Graph-based prediction (current forward())."""
212|    ids = tok.encode(query)
213|    if not ids:
214|        return False, "?", []
215|    ctx = np.array([ids], dtype=np.int64)
216|    logits = np.asarray(model.forward(ctx).data).flatten()
217|    top5_ids = list(np.argsort(logits)[::-1][:5])
218|    top5 = [tok.decode([tid]) for tid in top5_ids]
219|    return expected in top5, top5[0] if top5 else "?", top5
220|
221|
222|def eval_combined(model, tok, query, expected, facts, blend_alpha=0.5):
223|    """Blend embedding similarity + graph logits."""
224|    ids = tok.encode(query)
225|    if not ids:
226|        return False, "?", []
227|
228|    # Embedding signal: for each vocab word, score by NN similarity to subject
229|    parts = query.split()
230|    subj_word = parts[0] if parts else ""
231|    rel_word = parts[1] if len(parts) > 1 else ""
232|
233|    subj_tid = tok.word_to_id.get(subj_word)
234|    embeds = model.token_embed.weight.data
235|
236|    nn_logits = np.zeros(model.vocab_size, dtype=np.float32)
237|    if subj_tid is not None:
238|        subj_vec = embeds[subj_tid]
239|        pairs = [(s, o) for s, r, o in facts if r == rel_word]
240|        for s, o in pairs:
241|            s_tid = tok.word_to_id.get(s)
242|            o_tid = tok.word_to_id.get(o)
243|            if s_tid is None or o_tid is None:
244|                continue
245|            sim = float(np.dot(subj_vec, embeds[s_tid]))
246|            nn_logits[o_tid] += sim  # accumulate similarity-weighted object scores
247|
248|    # Graph signal
249|    ctx = np.array([ids], dtype=np.int64)
250|    graph_logits = np.asarray(model.forward(ctx).data).flatten()
251|
252|    # Blend
253|    combined = (1 - blend_alpha) * nn_logits + blend_alpha * graph_logits
254|
255|    top5_ids = list(np.argsort(combined)[::-1][:5])
256|    top5 = [tok.decode([tid]) for tid in top5_ids]
257|    return expected in top5, top5[0] if top5 else "?", top5
258|
259|
260|# ─── Diagnostic Table ──────────────────────────────────────────────────────
261|
262|def run_task_set(model, tok, task_name, tests, facts):
263|    """Run all three evaluation modes on a task set. Returns detailed log."""
264|    print(f"\n{'='*70}")
265|    print(f"TASK: {task_name}")
266|    print(f"{'='*70}")
267|
268|    results = []
269|    for test in tests:
270|        query = test["query"]
271|        expected = test["expected"]
272|
273|        nn_hit, nn_top1, nn_top5 = eval_embedding_only(model, tok, query, expected, facts)
274|        g_hit, g_top1, g_top5 = eval_graph_only(model, tok, query, expected)
275|        c_hit, c_top1, c_top5 = eval_combined(model, tok, query, expected, facts)
276|
277|        # Classification
278|        if nn_hit and g_hit:
279|            verdict = "BOTH RIGHT"
280|        elif nn_hit and not g_hit:
281|            verdict = "NN RIGHT, GRAPH WRONG"
282|        elif not nn_hit and g_hit:
283|            verdict = "GRAPH RIGHT, NN WRONG"  # THE CRITICAL CASE
284|        else:
285|            verdict = "BOTH WRONG"
286|
287|        marker = {"BOTH RIGHT": "✓✓", "NN RIGHT, GRAPH WRONG": "✓✗",
288|                   "GRAPH RIGHT, NN WRONG": "✗✓", "BOTH WRONG": "✗✗"}[verdict]
289|
290|        print(f"\n  {query:20s} → {expected:12s}  [{marker}]")
291|        print(f"    NN:     {nn_top1:12s}  top5: {nn_top5}")
292|        print(f"    Graph:  {g_top1:12s}  top5: {g_top5}")
293|        print(f"    Blend:  {c_top1:12s}  top5: {c_top5}")
294|        print(f"    Verdict: {verdict}")
295|
296|        if "note" in test:
297|            print(f"    Note: {test['note']}")
298|        if "chain" in test:
299|            print(f"    Chain: {test['chain']}")
300|
301|        results.append({
302|            "query": query, "expected": expected,
303|            "nn_hit": nn_hit, "g_hit": g_hit, "c_hit": c_hit,
304|            "verdict": verdict,
305|        })
306|
307|    # Summary
308|    n = len(results)
309|    nn_hits = sum(1 for r in results if r["nn_hit"])
310|    g_hits = sum(1 for r in results if r["g_hit"])
311|    c_hits = sum(1 for r in results if r["c_hit"])
312|    both_right = sum(1 for r in results if r["verdict"] == "BOTH RIGHT")
313|    nn_right_g_wrong = sum(1 for r in results if r["verdict"] == "NN RIGHT, GRAPH WRONG")
314|    g_right_nn_wrong = sum(1 for r in results if r["verdict"] == "GRAPH RIGHT, NN WRONG")
315|    both_wrong = sum(1 for r in results if r["verdict"] == "BOTH WRONG")
316|
317|    print(f"\n  --- {task_name} Summary ---")
318|    print(f"  NN:    {nn_hits}/{n}")
319|    print(f"  Graph: {g_hits}/{n}")
320|    print(f"  Blend: {c_hits}/{n}")
321|    print(f"  Both right:        {both_right}")
322|    print(f"  NN right, Graph X: {nn_right_g_wrong}")
323|    print(f"  Graph right, NN X: {g_right_nn_wrong}  ← GRAPH'S UNIQUE CONTRIBUTION")
324|    print(f"  Both wrong:        {both_wrong}")
325|
326|    return results
327|
328|
329|# ─── Main ──────────────────────────────────────────────────────────────────
330|
331|def main():
332|    print("="*70)
333|    print("TASK TAXONOMY EXPERIMENT")
334|    print("Finding the graph's genuine niche")
335|    print("="*70)
336|
337|    # Build all data
338|    facts = build_training_facts()
339|    retrieval_tests = build_retrieval_tests()
340|    multihop_tests = build_multihop_tests()
341|    hybrid_tests = build_hybrid_tests()
342|
343|    # Collect all words for vocabulary
344|    all_queries = ([t["query"] for t in retrieval_tests] +
345|                   [t["query"] for t in multihop_tests] +
346|                   [t["query"] for t in hybrid_tests])
347|
348|    tok = WordTokenizer()
349|    for s, r, o in facts:
350|        tok.encode(f"{s} {r} {o}")
351|    for q in all_queries:
352|        tok.encode(q)
353|
354|    print(f"\nVocabulary: {tok.vocab_size} words")
355|    print(f"Training facts: {len(facts)}")
356|    print(f"Retrieval tests: {len(retrieval_tests)}")
357|    print(f"Multi-hop tests: {len(multihop_tests)}")
358|    print(f"Hybrid tests: {len(hybrid_tests)}")
359|
360|    # Create model with semantic embeddings
361|    model = RLMv2(
362|        vocab_size=tok.vocab_size,
363|        embed_dim=32,
364|        concept_dim=32,
365|        n_concepts=500,
366|        sleep_interval=200,
367|        gate_concept_creation=False,
model._tokenizer = tok
368|    )
369|    model._tokenizer = tok
370|    inject_semantic_embeddings(model, tok)
371|
372|    # Train on all facts
373|    print("\nTraining...")
374|    for epoch in range(5):
375|        for s, r, o in facts:
376|            ids = tok.encode(f"{s} {r} {o}")
377|            if len(ids) < 2:
378|                continue
379|            ctx = np.array([ids[:-1]], dtype=np.int64)
380|            tgt = np.array([[ids[-1]]], dtype=np.int64)
381|            model.learn(ctx, tgt)
382|
383|    print(f"Graph: {len(model.graph.nodes)} nodes, {len(model.graph.edges)} edges")
384|
385|    # Run three task sets
386|    r_results = run_task_set(model, tok, "RETRIEVAL (NN should win)", retrieval_tests, facts)
387|    m_results = run_task_set(model, tok, "MULTI-HOP (Graph should win)", multihop_tests, facts)
388|    h_results = run_task_set(model, tok, "HYBRID (Both needed)", hybrid_tests, facts)
389|
390|    # ─── Grand Summary ─────────────────────────────────────────────────────
391|    print("\n" + "="*70)
392|    print("GRAND SUMMARY")
393|    print("="*70)
394|
395|    print(f"\n{'Task':<15s} {'NN':<8s} {'Graph':<8s} {'Blend':<8s} {'G unique':<10s}")
396|    print("-" * 49)
397|    for name, results in [("Retrieval", r_results), ("Multi-hop", m_results), ("Hybrid", h_results)]:
398|        n = len(results)
399|        nn = sum(1 for r in results if r["nn_hit"])
400|        g = sum(1 for r in results if r["g_hit"])
401|        c = sum(1 for r in results if r["c_hit"])
402|        g_unique = sum(1 for r in results if r["verdict"] == "GRAPH RIGHT, NN WRONG")
403|        print(f"{name:<15s} {nn}/{n:<6d} {g}/{n:<6d} {c}/{n:<6d} {g_unique}/{n:<8d}")
404|
405|    print("\n'G unique' = cases where graph is right AND embeddings are wrong.")
406|    print("If zero across all tasks, the graph currently contributes nothing unique.")
407|    print("If non-zero for multi-hop, that's the graph's niche.")
408|
409|    # ─── Entropy Analysis ──────────────────────────────────────────────────
410|    print("\n" + "="*70)
411|    print("ACTIVATION ENTROPY ANALYSIS")
412|    print("Does entropy predict graph confidence?")
413|    print("="*70)
414|
415|    all_tests = ([(t, "R") for t in retrieval_tests] +
416|                 [(t, "M") for t in multihop_tests] +
417|                 [(t, "H") for t in hybrid_tests])
418|
419|    for test, category in all_tests:
420|        query = test["query"]
421|        ids = tok.encode(query)
422|        if not ids:
423|            continue
424|        ctx = np.array([ids], dtype=np.int64)
425|        logits = np.asarray(model.forward(ctx).data).flatten()
426|
427|        # Softmax entropy
428|        probs = np.exp(logits - np.max(logits))
429|        probs /= probs.sum()
430|        entropy = -np.sum(probs * np.log(probs + 1e-10))
431|        max_entropy = np.log(len(logits))
432|        normalized_entropy = entropy / max_entropy
433|
434|        # Top-1 probability (concentration)
435|        top1_prob = float(np.max(probs))
436|
437|        # Is the answer in top-5?
438|        top5 = list(np.argsort(logits)[::-1][:5])
439|        expected_tid = tok.word_to_id.get(test["expected"], -1)
440|        hit = expected_tid in top5
441|
442|        marker = "✓" if hit else "✗"
443|        print(f"  [{category}] {marker} {query:20s} entropy={normalized_entropy:.3f} "
444|              f"top1_prob={top1_prob:.4f} top1={tok.decode([top5[0]])}")
445|
446|
447|if __name__ == "__main__":
448|    main()
449|