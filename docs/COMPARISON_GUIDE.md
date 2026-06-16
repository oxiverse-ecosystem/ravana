# RAVANA vs nanoGPT: OpenWebText Training Comparison
## Complete Framework for Training and Comparison

---

## The Friend's Advice
> "train it on same sample data as andrej nanogpt was trained on, then we can compare since his nanogpt is very good at his param size and data. so we can get param/data size ratio. that way we will know if its good"

---

## nanoGPT Baseline (GPT-2 124M on OpenWebText)

| Metric | Value |
|--------|-------|
| **Model** | GPT-2 (124M parameters) |
| **Architecture** | Transformer decoder (12 layers, 12 heads, 768 dim) |
| **Training Data** | OpenWebText (~9B tokens, ~8M documents) |
| **Tokens Trained** | ~300B tokens (multiple epochs) |
| **Tokenizer** | GPT-2 BPE (50,257 vocab) |
| **Config** | batch=12, seq=1024, grad_acc=40, 8×A100 |
| **Val Loss** | ~2.85 (finetuned), ~3.11 (raw GPT-2) |
| **Params/Token Ratio** | **124M / 300B = 0.000413** |

---

## RAVANA NeuralDecoder Architecture

| Metric | Value |
|--------|-------|
| **Model** | Corticostriatal GRU decoder (Hebbian learning) |
| **Architecture** | GRU + Attention + BasalGangliaGate (no backprop) |
| **Default Config** | embed=64, hidden=256, heads=4 |
| **Learning** | Local predictive coding (online, Hebbian) |
| **Data Sources** | 1. Seed corpus (teen_seeds.txt, 20 passes)<br>2. Synthetic graph sentences<br>3. Web articles (curiosity-driven, continuous) |
| **Tokenizer** | Word-level (built from data) |

---

## Training RAVANA on OpenWebText

### Script: `train_openwebtext.py`

```bash
# Quick test (100K tokens, small vocab)
python train_openwebtext.py --max-tokens 100000 --vocab-size 5000 --vocab-sample 1000 --eval-every 50 --batch-size 8

# Full comparison scale (300M tokens to match nanoGPT proportionally)
python train_openwebtext.py --max-tokens 300000000 --vocab-size 15000 --vocab-sample 50000 --eval-every 5000 --batch-size 32

# Resume from checkpoint
python train_openwebtext.py --resume --max-tokens 300000000
```

### Key Features
- **Streaming dataset** - no full download needed
- **Resume capability** - checkpoint saves vocab, model, stats
- **Network resilience** - auto-retries on HuggingFace connection issues
- **Vocab building** - samples N documents to build word-level vocab
- **Real-time stats** - tokens/sec, ETA, error tracking

---

## Comparison Methodology

### 1. Param/Token Ratio Comparison
```
nanoGPT:     124M params / 300B tokens = 0.000413
RAVANA:      [your_params] / [your_tokens] = [your_ratio]
```

### 2. Data Efficiency
- Train both on **same token count** (e.g., 10M, 100M, 1B, 300B)
- Compare validation perplexity / loss at each checkpoint
- Plot: tokens vs loss for both models

### 3. Architecture Comparison
| Aspect | nanoGPT | RAVANA |
|--------|---------|--------|
| **Learning** | Backprop (SGD/Adam) | Hebbian (local) |
| **Architecture** | Transformer | GRU + Attention |
| **Sequence** | Fixed context (1024) | Variable (stateful GRU) |
| **Conditioning** | None (unconditional) | Graph concepts (semantic) |
| **Vocab** | Subword (BPE) | Word-level |

---

## Expected Outcomes

### If RAVANA is competitive:
- Similar or better loss at same param/token ratio
- Faster convergence (Hebbian is sample-efficient)
- Better few-shot (graph conditioning provides semantic priors)

### Key Metrics to Track:
1. **Training loss curve** (perplexity)
2. **Parameters / token ratio** at convergence
3. **Generation quality** (coherence, factuality)
4. **Web training contribution** (seed vs web vs synthetic)

---

## Files Created

| File | Purpose |
|------|---------|
| `train_openwebtext.py` | Main training script for OpenWebText |
| `COMPARISON_GUIDE.md` | This document |

---

## Running the Full Comparison

```bash
# 1. Install dependencies
pip install datasets tiktoken --only-binary=:all:

# 2. Quick validation (5 min)
python train_openwebtext.py --max-tokens 50000 --vocab-size 3000 --vocab-sample 500 --eval-every 50 --batch-size 4

# 3. Scaling test (1 hour)
python train_openwebtext.py --max-tokens 10000000 --vocab-size 10000 --vocab-sample 10000 --eval-every 1000 --batch-size 16

# 4. Full comparison (run overnight)
python train_openwebtext.py --max-tokens 300000000 --vocab-size 15000 --vocab-sample 50000 --eval-every 5000 --batch-size 32

# 5. Compare results
# Check openwebtext_checkpoint.json for final stats
```

---

## Notes on Architecture Differences

The comparison isn't perfectly apples-to-apples because:

1. **RAVANA uses graph conditioning** - semantic context from concept graph provides inductive bias
2. **Hebbian learning** - no gradient descent, different optimization dynamics
3. **Word-level vocab** - vs BPE subwords (different compression)
4. **Continuous learning** - RAVANA can keep learning; GPT-2 is static after training

But the **param/token ratio** is the key metric your friend suggested - it normalizes for these differences and shows raw data efficiency.

---

## Next Steps

1. ✅ Run quick validation (confirms pipeline works)
2. ⏳ Run scaling tests at 10M, 100M tokens
3. ⏳ Run full 300M token training
4. 📊 Plot comparison curves
5. 📝 Document findings