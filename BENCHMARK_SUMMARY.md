# RAVANA vs nanoGPT Shakespeare Character-Level Benchmark

## Overview
Comparing graph-native n-gram co-occurrence models against nanoGPT (85K-parameter transformer) on Shakespeare character prediction.

**Dataset**: Tiny Shakespeare (1.1M chars), 90/5/5 split → 180K train, 10K val, 10K test (using first 200K for graph experiments)
**Vocabulary**: 62 characters (graph) vs 65 characters (nanoGPT, includes '$' and '3')

---

## Graph-Native Models (Option A)

### 1. Bigram Co-occurrence (Baseline)
- **File**: `cooccurrence_shakespeare.py`
- **Params**: 1,084 edges (non-zero bigrams)
- **Train time**: 0.18s
- **Val PPL**: 11.96
- **Test PPL**: 11.97
- **Generation**: Degenerates to "the the the..."

### 2. 4-gram Co-occurrence (v1)
- **File**: `cooccurrence_shakespeare_4gram.py`
- **Params**: 32,579 edges (bigram 1,084 + trigram 7,201 + 4-gram 24,294)
- **Train time**: 0.66s
- **Val PPL**: 5.89
- **Test PPL**: 6.34
- **Improvement**: 2.03x better than bigram
- **Generation**: Degenerates to "the with the with..."

### 3. Pruned 5-gram with Kneser-Ney Smoothing
- **File**: `cooccurrence_shakespeare_5gram.py`
- **Params**: 43,336
- **Train time**: 12.95s (too slow - continuation counting)
- **Val PPL**: 7.34 (worse - over-pruned/smoothing issues)
- **Status**: Failed targets

### 4. Spreading Activation Graph
- **File**: `spreading_activation_shakespeare.py`
- **Params**: 3,844 (bigram transitions only)
- **Train time**: 0.24s
- **Val PPL**: 28.09 (worse than bigram!)
- **Issue**: Personalized PageRank converges to stationary distribution, losing context signal

### 5. Optimized N-gram with Learned Interpolation (EM)
- **File**: `optimized_ngram_shakespeare.py`
- **Params**: 399,744 (dense arrays with smoothing = all non-zero)
- **Train time**: ~4-5s (EM iterations)
- **Val PPL**: 5.92
- **Status**: Over param budget, slow

### 6. Minimal N-gram (Sparse, Fast) ★ BEST GRAPH RESULT
- **File**: `minimal_ngram_shakespeare.py`
- **Best config**: max_order=4, min_count=3, lambdas={2:0.1, 3:0.3, 4:0.6}
- **Params**: 16,251 (under 17K budget ✓)
- **Train time**: 1.16s (slightly over 1s target)
- **Val PPL**: 5.85
- **Test PPL**: 6.46
- **Targets**: ✓ Params ≤17K, ✗ Train <1s, ✗ PPL <2.0

---

## Graph Approach Summary

| Model | Params | Train (s) | Val PPL | Meets Targets |
|-------|--------|-----------|---------|---------------|
| Bigram | 1,084 | 0.18 | 11.96 | ✗ PPL, ✓ others |
| 4-gram v1 | 32,579 | 0.66 | 5.89 | ✗ Params |
| Pruned 5-gram | 43,336 | 12.95 | 7.34 | ✗ All |
| Spreading Activation | 3,844 | 0.24 | 28.09 | ✗ PPL |
| Learned Interp (EM) | 399,744 | 4.7 | 5.92 | ✗ Params, Time |
| **Minimal N-gram** | **16,251** | **1.16** | **5.85** | **✓ Params only** |

### Key Findings for Graph Approach

1. **Data Sparsity Wall**: With 180K tokens and 62 vocab:
   - Bigrams: 3,844 possible, ~1K observed (28% coverage) → works well
   - Trigrams: 238K possible, ~7K observed (3% coverage) → helps but limited
   - 4-grams: 14.7M possible, ~24K observed (0.16% coverage) → severe sparsity
   - 5-grams: 916M possible, negligible coverage

2. **Interpolation Helps**: Learned weights favor 4-grams (~0.65) when available, but most test contexts fall back to lower orders.

3. **Generation Quality**: All n-gram models degenerate to loops ("the the the...", "the shall the shall...") because greedy decoding finds the highest-probability cycle.

4. **Fundamental Limit**: Pure n-gram counting cannot generalize to unseen contexts. Without embeddings/attention, PPL plateaus at ~5-6.

---

## nanoGPT Transformer (85K params)

### Configuration
- **Layers**: 4
- **Heads**: 2
- **Embedding dim**: 40
- **Block size**: 256
- **Total params**: ~89,640 (89.6K)
- **Training**: CPU, 5000 iterations planned
- **Status**: Training in progress (see train.log)

### Expected Performance
Based on nanoGPT scaling laws for character-level Shakespeare:
- 85K params → ~1.5-2.5 val PPL after full training
- 5000 iterations should converge
- Generalizes via embeddings + attention → unseen n-grams handled

### Current Checkpoint (iter=100)
- **Val loss**: 3.03 nats → PPL = exp(3.03) = 20.7
- **Status**: Early training, not converged

---

## Comparison (Projected)

| Metric | Bigram Graph | 4-gram Graph | nanoGPT (proj.) |
|--------|-------------|--------------|-----------------|
| Params | 1K | 16K | 85K |
| Train time | 0.2s | 1.2s | ~1-2 hours (CPU) |
| Val PPL | 12.0 | 5.85 | ~1.5-2.5 |
| Generalization | None | Limited | Full |
| Generation | Loops | Loops | Coherent Shakespeare |

---

## Conclusions for Option A

**The graph-native n-gram approach (Option A) cannot reach PPL < 2.0** on this dataset size due to fundamental sparsity limitations of character-level n-grams. The best achievable is ~5.85 PPL at 16K params.

**Recommendation**: 
1. ✓ Document that pure co-occurrence graphs hit a sparsity wall
2. → Proceed to **Option B (RAVANA adaptation)** if user wants to test whether RAVANA's graph architecture (concept nodes, spreading activation, Hebbian learning) can generalize better
3. nanoGPT training continues in background for final comparison

---

## Files Created

```
C:\Users\Likhith\Documents\projects\ravana\
├── cooccurrence_shakespeare.py          # Bigram baseline (PPL 11.96)
├── cooccurrence_shakespeare_4gram.py    # 4-gram v1 (PPL 5.89, 32K params)
├── cooccurrence_shakespeare_5gram.py    # Pruned 5-gram + Kneser-Ney (PPL 7.34)
├── spreading_activation_shakespeare.py  # Spreading activation (PPL 28.09)
├── optimized_ngram_shakespeare.py       # Learned interpolation EM (PPL 5.92)
├── minimal_ngram_shakespeare.py         # ★ Best graph result (PPL 5.85, 16K params)
└── BENCHMARK_SUMMARY.md                 # This file
```

---

## Next Steps
1. Wait for nanoGPT training to complete (or reach reasonable iteration count)
2. Evaluate nanoGPT final PPL
3. If user wants Option B: Implement RAVANA adaptation for next-token
   - Bind 65 chars to fixed concept nodes
   - K-token context → spreading activation from K nodes
   - Learn edges via Hebbian mechanism (not counting)
   - Disable relation predictor, alignment, sleep
   - Target: competitive PPL at ~85K params

---

## Cross-Domain Structural Transfer (RLMv2) — NEW BENCHMARK

### Task: Domain A verb + Domain B subject → Domain B target (structural transfer)

| Configuration | Top-1 Acc | Top-10 Acc | Inference/Query |
|--------------|-----------|------------|-----------------|
| Joint training (baseline) | **44.0%** | **67.0%** | 4.4ms (230 QPS) |
| + Abstract bridges (6 semantic primitives) | 60.0% | 72.0% | 4.4ms |
| + W_rel alignment (30 steps) | 64.0% | 76.0% | 4.4ms |
| + Sleep consolidation | **65.0%** | **84.0%** | 4.4ms |

**Model**: RLMv2 (RAVANA), 85K params, single shared domain head  
**Domains**: Science (10 causal facts) + Social (10 causal facts)  
**Mechanism**: Verb-offset arithmetic `pred = subject + offset(verb)` + graph spreading activation via injected semantic bridges  
**Semantic bridges**: `anger→intense_bridge→expansion`, `kindness→warm_bridge→warmth`, etc.

### Key Findings
1. **Verb offsets** are the primary cross-domain driver — "anger produces" → "conflict" works because `offset("produces")` transfers from Domain A
2. **Abstract bridges** provide analogical pathways when verb offsets alone are insufficient (e.g., "anger creates" → "conflict" via `fire_bridge`)
3. **W_rel alignment** improves relation matrix cosine similarity (causal: 0.35→0.49, semantic: 0.42→0.54)
4. **Sleep consolidation** prunes noisy edges, protects high-confidence bridges via anti-Hebbian plasticity
5. **Inference**: 4.4ms/query on CPU (228 QPS) — viable for real-time apps