# Architecture

This document describes how a single user turn flows through RAVANA, and how the
three first-party packages relate. All paths are relative to the repo root.

## The three packages

RAVANA is one integrated system split across three source trees under `*/src`:

| Package | Path | Role |
|---------|------|------|
| `ravana_ml` | `ravana_ml/src/ravana_ml/` | CPU-native ML substrate: tensors, `ConceptGraph`, `RLM`/`RLMv2`, neural decoder, embedders, ontologies. |
| `ravana` | `ravana/src/ravana/` | Decoder-first chat engine: the live `CognitiveChatEngine`, brain-repair preprocessing, language generation, web learning. |
| `ravana_grace` | `ravana-v2/src/ravana_grace/` | GRACE 27-phase cognitive governor: identity, emotion, sleep, meaning, world model, theory-of-mind, metacognition, etc. |

They are imported together by the chat entrypoint
(`scripts/ravana_chat.py` inserts all three `src` dirs on `sys.path`). There is
no separate runtime dependency between them at the OS level — they are one repo.

## Turn-level data flow

```
user text
   │
   ▼
[1] brain_regions.py  ── BRAIN-REPAIR PREPASSES ──
      • SelfModel.from_graph        (who am I / who is the user)
      • EpisodicIndex               (FIRST/LAST/BY_ENTITY recall)
      • classify_cause              (GloVe-centroid cause classification)
      • select_empathy_frame        (tightly-gated empathy)
      • humor_is_coherent           (learned salad classifier ORed with
                                     rule-based _is_word_salad; fail-closed)
      • parse_number_phrase / mirror_deictic / consult_internal
   │
   ▼
[2] engine.py :: CognitiveChatEngine.process_turn
      • intent_router          (chitchat / factual / hypothetical / identity / OOD)
      • monitor_gate           (abstention / free-energy check)
      • coherence_gate         (GloVe-cosine coherence floor)
      • junk_scorer            (word-salad / degenerate-text rejection)
   │
   ├─ if unknown / low-confidence ─► honest abstention ("I don't know")
   │
   ▼
[3] graph walk  (chain_walker.py)  over the typed ConceptGraph
      edges: causal, contrastive, analogical, temporal, semantic, is_a
   │
   ▼
[4] neural decoder  (ravana_ml.nn.neural_decoder)
      conditions generation on the graph-walk embedding + sensorimotor
      (LingGen) signal when promoted
   │
   ▼
[5] surface_realizer + syntactic cell assembly  →  natural-language response
   │
   ▼
[6] web_learning.py  (background)  ── if a gap was detected, learn from the web
      and write new typed edges into the graph for next time
```

## Concept graph

`ravana_ml.graph.ConceptGraph` stores nodes (concepts, 64-D vectors) and typed
edges with weights, confidence, and prediction free-energy. At init it snapshots
**stable nodes** so that same-turn web-learning cannot defeat the connectivity /
anchor gates (a previously-fixed pitfall — see `brain_regions.py` notes).

Seed concepts (~180 teen-level words) and their relations are defined in
`scripts/ravana_chat.py` (`DOMAIN_CONCEPTS`). ConceptNet + GloVe (projected to
64-D) supply semantic priors.

## Decoder

`ravana_ml.nn.neural_decoder.NeuralDecoder` is a small GRU conditioned on a
concept embedding. It is trained online (no offline corpus required) on
`data/corpora/teen_seeds.txt` plus whatever the web-learning loop harvests.
`sleep_cycle()` performs consolidation (the free-energy "instead of
`optimizer.step()`" loop).

## GRACE governor

`ravana_grace.core.Governor` wraps the chat/cognitive stack with 27 phases of
regulation: identity, emotion (VAD), sleep, meaning, world model, belief
reasoning, active epistemology, metacognition, strategy selection, and more. It
is imported by `scripts/ravana_chat.py` and driven per turn.

## State & persistence

Runtime artifacts (`data/`, `checkpoints/`, `output/`) are gitignored. The
engine serializes weights to `data/ravana_weights.pkl` and a SQLite store
(`data/ravana_weights.db`). A fresh clone needs the corpus present (or run
`python scripts/gather_teen_seeds.py` to rebuild it).
