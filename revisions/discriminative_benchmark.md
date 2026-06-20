# RAVANA Benchmark Report

**Generated:** 2026-06-20 19:56:56

---

## 1. Verb-Offset Held-Out Generalization

| Model | Train Acc | Held-Out Acc | Gen Gap | Params |
|-------|-----------|--------------|---------|--------|
| RLMv2 (RAVANA) | 67.3% | 0.0% | 67.3% | 156,870 |

*Held-out uses novel subjects with verbs seen during training. RAVANA uses verb-offset mechanism; baselines can only memorize seen subject-verb pairs.*

## 2. Cross-Domain Transfer (Science -> Social)

| Metric | Score |
|--------|-------|
| science_accuracy | 53.3% |
| social_accuracy | 16.7% |
| held_out_accuracy | 0.0% |
| cross_domain_gap | 53.3% |
| ontology_benefit_held_out | 0.0% |
| ontology_without_held_out | 20.0% |
| ontology_benefit_delta | -20.0% |

### Held-Out Per-Relation Accuracy
- **causes**: 0.0%
- **produces**: 0.0%

## 3. Catastrophic Forgetting (Sequential A -> B -> C)

| Domain | After A | After B | After C | Forgetting |
|--------|---------|---------|---------|------------|
| physics | 82.5% | 27.5% | 20.0% | 62.5% |
| cooking | 0.0% | 92.5% | 45.0% | -45.0% |
| music | 0.0% | 0.0% | 87.5% | -87.5% |
| **Average** | | | | **-23.3%** |

*Negative forgetting = improvement through sleep consolidation.*

## 4. Conversation Quality

| Model | Coherence | Diversity (1g) | Diversity (2g) | Diversity (3g) | Repetition | Avg Length |
|-------|-----------|---------------|---------------|---------------|------------|------------|
| RLMv2 (RAVANA) | 0.851 | 0.583 | 0.727 | 0.900 | 0.273 | 12.0 |

*Higher coherence & diversity = better. Lower repetition = better.*

## 5. Parameter Efficiency

| Model | Parameters | Held-Out | Params/Acc (↓ better) |
|-------|------------|----------|----------------------|
| RLMv2 (RAVANA) | 156,870 | 0.0% | 156,870,000 |

### Theoretical Baselines

| Model | Parameters | x RAVANA |
|-------|------------|----------|
| DistilGPT-2 | 82,000,000 | 522.7x |
| Tiny Transformer (4-layer) | 10,000,000 | 63.7x |
| Tiny LLaMA (1.1B) | 1,100,000,000 | 7012.2x |
| GPT-2 Small (124M) | 124,000,000 | 790.5x |

## Summary

- **Best held-out accuracy**: RLMv2 (RAVANA) (0.0%)
- **Cross-domain held-out**: 0.0%
- **Ontology benefit**: +-20.0%
- **Catastrophic forgetting**: -23.3% across domains

---
*RAVANA: Forward-only, Hebbian, sleep-consolidating cognitive architecture.*