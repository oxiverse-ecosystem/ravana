# RAVANA Benchmark Report

**Generated:** 2026-06-20 19:21:43

---

## 1. Triple Completion (Same-Domain + Held-Out)

| Model | Train Acc | Test Acc | Held-Out Acc | Cross-Domain Acc | Gen Gap | Params | Latency (ms) |
|-------|-----------|----------|--------------|-------------------|---------|--------|--------------|
| RLMv2 (RAVANA) | 0.0% | 0.0% | 0.0% | 66.7% | 0.0% | 153,218 | 0.61 |
| Linear Baseline | 0.0% | 1.0% | 0.0% | 0.0% | 0.0% | 3,296 | 0.00 |
| MLP Baseline (2-layer) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 10,308 | 0.00 |
| DistilGPT-2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 81,912,576 | 0.00 |

### Per-Relation-Type Breakdown (Test Set)

| Model | analogical | causal | contextual | possessive | semantic | temporal |
|-------|---|---|---|---|---|---|
| RLMv2 (RAVANA) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Linear Baseline | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| MLP Baseline (2-layer) | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| DistilGPT-2 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |

## 2. Catastrophic Forgetting (Sequential A -> B -> C)

| Domain | After A | After B | After C | Forgetting |
|--------|---------|---------|---------|------------|
| science | 80.0% | 93.3% | 93.3% | -13.3% |
| cooking | 0.0% | 100.0% | 100.0% | -100.0% |
| music | 0.0% | 0.0% | 100.0% | -100.0% |
| **Average** | | | | **-71.1%** |

**Forgetting curves show accuracy retention across 3 domains.**

## 3. Cross-Domain Transfer (Science -> Social -> Held-Out)

| Metric | Score |
|--------|-------|
| science_accuracy | 66.7% |
| social_accuracy | 41.7% |
| held_out_accuracy | 0.0% |
| cross_domain_gap | 66.7% |

### Held-Out Per-Relation Accuracy
- **causes**: 0.0%
- **produces**: 0.0%

## 4. Conversation Quality

| Model | Coherence | Diversity (1-gram) | Diversity (2-gram) | Diversity (3-gram) | Repetition Rate | Avg Length |
|-------|-----------|-------------------|-------------------|-------------------|-----------------|------------|
| RLMv2 (RAVANA) | 0.825 | 0.646 | 0.795 | 0.975 | 0.205 | 12.0 |

**Higher coherence & diversity = better. Lower repetition = better.**

## 5. Parameter Efficiency Comparison

| Model | Parameters | Test Accuracy | Params/Accuracy (↓ better) | Speed Score (↑ better) |
|-------|------------|---------------|---------------------------|------------------------|
| RLMv2 (RAVANA) | 153,218 | 0.0% | 153,218,000 | 0.0 |
| Linear Baseline | 3,296 | 1.0% | 329,600 | 100.0 |
| MLP Baseline (2-layer) | 10,308 | 0.0% | 10,308,000 | 0.0 |
| DistilGPT-2 | 81,912,576 | 0.0% | 81,912,576,000 | 0.0 |

### Theoretical Baselines for Comparison

| Model | Parameters | × RAVANA |
|-------|------------|----------|
| DistilGPT-2 | 82,000,000 | 535.2× |
| Tiny Transformer (4-layer) | 10,000,000 | 65.3× |
| Tiny LLaMA (1.1B) | 1,100,000,000 | 7179.3× |
| GPT-2 Small (124M) | 124,000,000 | 809.3× |
| GPT-2 Medium (355M) | 355,000,000 | 2317.0× |

## Summary

- **Best test accuracy**: Linear Baseline (1.0%)
- **Best param efficiency**: Linear Baseline (3,296 params)
- **Catastrophic forgetting**: avg -71.1% across domains
- **Cross-domain held-out**: 0.0%

---
*RAVANA: Forward-only, Hebbian, sleep-consolidating cognitive architecture.*
*No backprop. No templates. Continuous web learning. Curiosity-driven.*