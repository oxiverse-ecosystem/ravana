# Benchmarks & Diagnostics

Every benchmark/diagnostic entry point in the repo. Benchmarks mostly measure
the properties the architecture is built for: honest abstention, held-out
generalization, cross-domain transfer, catastrophic-forgetting resistance, and
graph latency.

All commands are run from the repo root. `scripts/` is on the path for
`scripts.X` imports; the `src` dirs are auto-prepended.

## Chat / end-to-end

| Script | Measures |
|--------|----------|
| `scripts/ravana_chat.py` | The live chatbot. Not a benchmark, but the reference runtime. |
| `scripts/benchmark_queries.py` | 6-class query taxonomy (chitchat, factual, hypothetical, conditional, identity, OOD). `OOD` items are **abstention probes** — a confident answer is a confabulation. Run `python -m experiments.benchmark_arc` to drive it. |

## Decoder / relation learner (RLMv2)

| Script | Measures |
|--------|----------|
| `scripts/benchmark_vs_transformers.py` | P3 harness: verb-offset held-out generalization, cross-domain transfer, ontology benefit, catastrophic forgetting, conversation quality, parameter efficiency. `-p scripts/benchmark_vs_transformers.py --quick`. |
| `scripts/external_benchmark.py` | PCX/NeuroBench-style text tasks, lifelong retention under task-switching, large-graph (~100–200K node) latency/memory profiling. `--quick` for a fast run. |
| `scripts/diagnose_transfer.py` | Cross-domain semantic transfer & held-out diagnostic (Science→Social) with verb-offset + test-time adapter adaptation. |
| `scripts/validate_held_out_generalization.py` | Phase-1 P0 checks: all-verb offset support, confidence-weighted blending, prototype inheritance, cross-domain transfer. |
| `scripts/triple_eval.py` | Per-triple diagnostics (relation type, PE, confidence, source, edge attributes). `--triples-file` or interactive. |
| `scripts/run_ablation.py` | Off-topic-snippet rate for paradox grounding, before vs after the gated retrieval fix. |
| `scripts/scaling_benchmark.py` | `ConceptGraph` latency at 1K→10K nodes. |

## Analysis / profiling helpers

| Script | Purpose |
|--------|----------|
| `scripts/profile_phase3.py` | `cProfile` of `RLM.learn()` + individual graph-op timings. |
| `scripts/plot_analysis.py` | Post-hoc plots (backward transfer, concept drift, cross-domain summary) from `experiment_results/`. |
| `scripts/analyze_longitudinal.py` | Reads `checkpoints/longitudinal/metrics.jsonl`; phase-transition / forgetting analysis. |
| `scripts/audit_coverage.py` | Compares source modules against test imports (coverage audit). |

## Corpus / artifact setup (one-shot)

| Script | Purpose |
|--------|----------|
| `scripts/gather_teen_seeds.py` | Fetch teen-level conversational English → `data/corpora/teen_seeds.txt`. |
| `scripts/expand_corpus.py` | Expand the corpus via the IntentForge gateway. |
| `scripts/train_lancaster_probe.py` | Train the wide-coverage Lancaster sensorimotor probe → `data/lancaster_encoder.npz`. |

## Experiments package (`experiments/`)

Research harnesses behind the numbers above: `experiment_cross_domain.py`
(`build_domain_a_science` / `build_domain_b_social`), `benchmark_arc.py`
(`ALL_QUERIES`), `experiment_ablation.py`, `experiment_sleep_memory.py`,
`experiment_chat_quality.py`, etc. Imported by the benchmark scripts; not run
directly in normal use.

## Interpreting results

- **Abstention probes** (OOD class) must score *below* honest "I don't know".
  Confident-wrong is treated as high free-energy.
- **Held-out / cross-domain** top-1 is expected to be modest; top-10 and
  verb-offset behavior are the meaningful signals.
- **Forgetting**: with sleep consolidation, Domain-A retention after Domain-B
  training should stay well above the ~14% random baseline.
