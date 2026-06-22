# RAVANA Benchmark Report

**Generated:** 2026-06-20 20:54:08

---

## 1. Verb-Offset Held-Out Generalization

| Model | Train Acc | Held-Out Acc | Gen Gap | Params |
|-------|-----------|--------------|---------|--------|
| RLMv2 (RAVANA) | 50.9% | 0.0% | 50.9% | 156,870 |

*Held-out uses novel subjects with verbs seen during training. RAVANA uses verb-offset mechanism; baselines can only memorize seen subject-verb pairs.*

## 2. Cross-Domain Transfer (Science -> Social)

| Metric | Score |
|--------|-------|
| science_accuracy | 86.7% |
| social_accuracy | 91.7% |
| held_out_accuracy | 0.0% |
| cross_domain_gap | 86.7% |
| ontology_benefit_held_out | 40.0% |
| ontology_without_held_out | 0.0% |
| ontology_benefit_delta | 40.0% |

### Held-Out Per-Relation Accuracy
- **causes**: 0.0%
- **produces**: 0.0%

## 3. Catastrophic Forgetting (Sequential A -> B -> C)

| Domain | After A | After B | After C | Forgetting |
|--------|---------|---------|---------|------------|
| physics | 60.0% | 37.5% | 35.0% | 25.0% |
| cooking | 0.0% | 87.5% | 47.5% | -47.5% |
| music | 0.0% | 0.0% | 97.5% | -97.5% |
| **Average** | | | | **-40.0%** |

*Negative forgetting = improvement through sleep consolidation.*

## 4. Conversation Quality

| Model | Coherence | Diversity (1g) | Diversity (2g) | Diversity (3g) | Repetition | Avg Length |
|-------|-----------|---------------|---------------|---------------|------------|------------|
| RLMv2 (RAVANA) | 0.833 | 0.542 | 0.750 | 0.925 | 0.250 | 12.0 |

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
- **Ontology benefit**: +40.0%
- **Catastrophic forgetting**: -40.0% across domains

---
*RAVANA: Forward-only, Hebbian, sleep-consolidating cognitive architecture.*