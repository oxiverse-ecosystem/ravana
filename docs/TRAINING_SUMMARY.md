# RAVANA vs nanoGPT: Training Framework Complete

## Summary

Created complete training pipeline to train RAVANA on OpenWebText (same data as nanoGPT) and compare param/data ratios.

### Files Created

| File | Purpose |
|------|---------|
| `train_openwebtext.py` | Main script - streams OpenWebText from HuggingFace, trains NeuralDecoder, outputs comparison |
| `train_localweb.py` | Alternative - uses local corpus + web fetching via DuckDuckGo, avoids HF streaming |
| `COMPARISON_GUIDE.md` | Documentation with methodology, run commands, expected outcomes |

### Verified Working

```
✓ NeuralDecoder training works (Hebbian, no backprop)
✓ Vocabulary building from text corpus
✓ Checkpoint saving/loading (JSON)
✓ Param counting for comparison
```

### Test Results (Local Environment)

| Config | Params | Tokens | Time | Params/Token |
|--------|--------|--------|------|--------------|
| vocab=800, hidden=128, heads=2, 2 passes | 310K | 18K | 56s | 16.9 |
| vocab=1000, hidden=128, heads=2, 3 passes | ~400K | 27K | 72s | ~14.8 |

**nanoGPT baseline**: 124M params / 300B tokens = **0.000413 params/token**

### To Match nanoGPT Scale

To achieve comparable param/token ratio on RAVANA:
- For ~400K param model → need **~1B tokens** (400K / 0.000413)
- For ~1M param model → need **~2.4B tokens**
- For fair comparison → train on **full OpenWebText (~9B tokens)** or multiple epochs to 300B tokens

### Run Commands (on machine with good network)

```bash
# Install deps
pip install datasets tiktoken --only-binary=:all:

# Quick validation (local corpus, 1 min)
python train_localweb.py --max-tokens 50000 --vocab-size 3000 --passes 3

# Full OpenWebText streaming (requires network, hours)
python train_openwebtext.py --max-tokens 10000000 --vocab-size 10000 --vocab-sample 10000

# Full comparison scale (overnight)
python train_openwebtext.py --max-tokens 300000000 --vocab-size 15000 --vocab-sample 50000 --batch-size 32
```

### Expected Comparison Output

The script automatically prints:
```
======================================================================
COMPARISON WITH NANOGPT
======================================================================
nanoGPT GPT-2 124M:
  - Parameters: 124,000,000
  - Training tokens: ~300,000,000,000 (300B)
  - Params/token ratio: 124M / 300B = 0.000413

RAVANA Decoder:
  - Parameters: ~[your_params]
  - Training tokens: [your_tokens]
  - Params/token ratio: [your_ratio]
  - Data scale vs nanoGPT: [x.xx%]
```

### Architecture Differences (Not Apples-to-Apples)

| Aspect | nanoGPT | RAVANA |
|--------|---------|--------|
| Learning | Backprop (Adam) | Hebbian (local predictive coding) |
| Architecture | Transformer | GRU + Attention + BasalGangliaGate |
| Conditioning | None | Graph concepts (semantic priors) |
| Vocab | BPE (50K) | Word-level (built from data) |
| Training | Static (done) | Continuous (curiosity-driven) |

**The param/token ratio normalizes these differences** - it's the key metric your friend suggested.

### Next Steps

1. Run on machine with stable network (HuggingFace streaming)
2. Scale to 10M, 100M, 1B tokens
3. Plot loss curves vs nanoGPT
4. Evaluate generation quality at each scale