# Capability: energy & portability footprint (RV-feature, 2026-10-22)

**Status:** shipped (branch `auto/round-2026-10-22T0459Z`).
**Task:** backlog #6 — "Keep the energy/portability constraint as a first-class acceptance criterion."
**Verified:** measurements below are REAL, taken on the checked-out tree with `RAVANA_OFFLINE=1`, `seed=42`, `baby_mode=True`, `dim=64`. Raw artifacts: `scratch/energy_report.json`, `scratch/mech_report2.json`.

## 0. Why this document exists

The energy/portability constraint ("could this run on a laptop / phone / microcontroller?")
is a DESIGN LENS, not a footnote (see `docs/LANGUAGE_PRODUCTION_PIVOT.md` §0 and
`ravana.txt` §"THE PIVOT"). A mechanism that needs a GPU cluster is the wrong
mechanism, however accurate. This doc puts real numbers next to every adopted
mechanism so the constraint is auditable instead of a slogan.

## 1. Measurement method (named, reproducible)

| Quantity | How measured | Tool |
|---|---|---|
| Process RSS (memory footprint) | `psutil.Process().memory_info().rss` at the named checkpoint | `psutil` 7.2.2 |
| Per-turn latency | `time.perf_counter()` around `CognitiveChatEngine.process_turn(q)` | stdlib |
| Build time of a mechanism object | `time.perf_counter()` around constructor | stdlib |
| On-disk artifact size | `os.path.getsize` | stdlib |
| Store sizes (nodes/edges/entries) | `len(store)` on the live engine after 25 turns | live engine |

Environment: Windows 11, Python 3.11.15, `.venv-real` (real venv, not the broken uv shim),
`RAVANA_OFFLINE=1`, `PYTHONHASHSEED=0`, `seed=42`, `baby_mode=True`, `dim=64`,
`user_suffix="energyprobe"`. N=25 probe turns (varied: identity, stance, fact, recall,
code generation, ability composition). Baseline RSS before any import = 18.5 MB
(N=1 process).

No LLM, no retraining, no GPU used anywhere in this measurement.

## 2. Whole-engine footprint (the budget)

| Checkpoint | RSS (MB) | N | Method |
|---|---|---|---|
| Before any import | 18.5 | 1 | `psutil` after process start |
| After importing engine module | 197.0 | 1 | `psutil` after `import ravana.chat.engine` |
| After `CognitiveChatEngine(...)` boot (cold) | 579.0 | 1 | `psutil` after constructor returns (includes GloVe cache load + seed) |
| After 25 turns | 907.0 | 1 | `psutil` after last `process_turn` |
| Growth over 25 turns | +328.0 | 1 | difference |
| After `engine.save()` | 1034.6 | 1 | `psutil` after pickle dump |
| Boot wall time | 16.47 s | 1 | `perf_counter` around constructor |
| Import wall time | 12.71 s | 1 | `perf_counter` around import |
| Persisted weights file | 7.1 MB (7,427,814 B) | 1 | `os.path.getsize(weights/ravana_weightsenergyprobe.pkl)` |

Per-turn latency (N=25): min 2.6 ms, max 18,519.7 ms, mean 3,574.9 ms,
median 266.3 ms, p95 11,155.9 ms. The high tail (5 turns > 8 s) comes from turns
that triggered construction induction / code generation / ability composition
paths; the median turn is 266 ms. See `scratch/energy_report.json` `per_turn_latency_ms`.

Live state after 25 turns: concept graph 138 nodes / 607 edges; 4 stances; 6 facts;
0 beliefs; 1 sleep cycle; identity strength 0.775, stability 0.876.

## 3. Dominant cost is the lexicon, not the cognition (important)

| Asset | On-disk size | In-memory (RSS delta) | Notes |
|---|---|---|---|
| GloVe projected cache `data/ravana_glove_cache.npz` | 144 MB | ~381 MB of the 579 MB boot RSS | 400,000 vectors × 100D → 64D projection, loaded at boot (boot log: "Loaded 400000 projected vectors from cache (100D -> 64D)"). This is the single largest cost and it is a SEED ASSET, not a mechanism. |
| Raw GloVe `data/glove/glove.6B.100d.txt` | 332 MB | not loaded at runtime (cache is used) | kept on disk only |
| Persisted engine state (weights pkl) | 7.1 MB | — | identity, stances, facts, beliefs, turn_count, graph (partial) |

Conclusion: the cognitive mechanisms themselves (graph, n-gram, schemas, realizer,
composer) cost under ~1 MB combined; the ~579 MB boot footprint is dominated by
the GloVe seed lexicon. A "microcontroller" port must replace or shard the GloVe
cache (e.g. a smaller projected subset or an indexed/quantized store); it does
NOT require changing the mechanisms.

## 4. Per-mechanism footprint (adopted mechanisms)

Build times measured by `time.perf_counter()` around the constructor with N=1
object (`scratch/mech_report2.json`). RSS deltas are page-granular (0.00–0.05 MB)
because these objects are tiny; build time is the more honest signal at this scale.

| Mechanism (adopted) | Source file | Seed size / state | Build time (ms, N=1) | RSS delta (MB) | Smallest plausible hardware |
|---|---|---|---|---|---|
| ConceptGraph (typed concept graph, spreading activation) | `chat/engine_graph.py` `ConceptGraph(dim=64)` | 138 nodes / 607 edges after 25 turns (seed ~11 nodes) | 66.95 | 0.00 | **Microcontroller-class** (pure numpy dot-products, O(N) per query, no matrix materialisation beyond node vectors; ~64 floats/node ≈ 256 B/node → ~35 KB at 138 nodes) |
| CerebellarNgram (sparse bigram/trigram/phrase patterns) | `language/cerebellar_ngram.py` `CerebellarNgram()` | 0 bigram / 0 trigram / 0 phrase_patterns after 25 turns (seed empty; grows via `learn_chain`/`learn_phrase`) | 0.024 | 0.00 | **Microcontroller-class** (dict-of-counts, sparse by construction; Olshausen & Field sparse coding, Hopfield attractor retrieval; never materialises a 10k×10k matrix) |
| ConstructionSchemaStore (OCG schema induction) | `language/construction_schemas.py` `ConstructionSchemaStore()` | 6 slots, 5 relation types, 2 induced schemas (seed contrastive pairs) | 0.537 | 0.00 | **Microcontroller-class** (small dataclass store; grows by `induce_from_pair`) |
| FunctionalLexicon (growable POS/function-word sets) | `chat/functional_lexicon.py` `default_lexicon()` | 8 categories, 112 entries total (12+7+11+9+2+13+11+48) | 0.831 | 0.05 | **Microcontroller-class** (dict-of-sets; `register_*` growth path) |
| CodeRealizer (OCG → code modality) | `language/code_realizer.py` `CodeRealizer(engine=None)` | 30 operation builders + 30 aliases (growable via `register_operation`/`register_alias`) | 0.014 | 0.00 | **Phone/laptop** (operation combinators are pure Python string builders; no model weights; runs wherever CPython runs) |
| Ability composer (composition over primitives) | `agent/ability_composer.py` `_ComposerHandle` | 12 primitives, 0 composed abilities at boot (grows via `compose_ability`/`learn_from_instruction`) | property (lazy, <0.1 ms) | 0.00 | **Phone/laptop** (pipeline is a list of primitive names + a dict store; composition is data, not code branches) |
| IdentityEngine (self-coherence) | `chat/...` (via `eng.identity`) | strength 0.775, momentum 0.043, stability 0.876 (N=1 engine, 25 turns) | part of boot | included in boot RSS | **Microcontroller-class** (a handful of floats + history list) |
| UserStanceStore (attitudes) | `chat/user_model.py` | 4 stances after 25 turns (grows from disclosures) | part of boot | included | **Microcontroller-class** (dict of `(topic → (polarity, confidence))`) |
| PersonalFactStore (facts) | `chat/personal_fact_store.py` | 6 facts after 25 turns | part of boot | included | **Microcontroller-class** (dict of `(subj, attr, val) → fact`) |
| BeliefStore (positions) | `chat/belief_store.py` | 0 beliefs after this probe | part of boot | included | **Microcontroller-class** |
| EpistemicCalibrator (calibration ledger) | `chat/calibration.py` `get_status()` | signed bias, per-band reliability, adaptive theta_withhold | part of boot | included | **Microcontroller-class** (small running statistics) |
| Intent router / relevance store / monitor gate | `chat/intent_router.py`, `chat/relevance_store.py`, `chat/monitor_gate.py` | — | part of boot | included | **Microcontroller-class** (GloVe-cosine scoring + dict stores) |
| GRACE governor (20-phase) | `chat/...` (governor inside engine) | — | part of boot | included | **Laptop** (orchestrates the above; no weights, but 20 phases per turn adds latency) |
| GloVe seed lexicon (NOT a mechanism — seed asset) | `data/ravana_glove_cache.npz` | 400,000 vectors | 16.47 s boot (dominated by this load) | ~381 MB | **Laptop** (needs ~400 MB RAM + ~144 MB disk); phone/microcontroller require a sharded/quantized subset |

## 5. Per-turn latency attribution (honest)

The median turn is 266 ms; the mean is 3,575 ms because 5 of 25 turns exceeded
8 s. Those slow turns are the ones that exercised construction induction, code
generation, and ability composition (the OCG/ability paths), plus one turn that
triggered sleep consolidation. This is consistent with the architecture: the
hot path (stance/fact mining, recall) is sub-300 ms; the generative paths
(composition over learned structure) are heavier but still single-pass CPU.

No turn required a GPU. No turn triggered a retrain. No turn called an LLM.

## 6. Portability verdict (the answer to "smallest hardware")

| Tier | Feasible? | What must change |
|---|---|---|
| **Laptop (x86, ≥4 GB RAM, no GPU)** | YES — this is the reference target. Measured boot RSS 579 MB, peak 1,035 MB during save. | Nothing. Runs today on the dev box. |
| **Phone (ARM, 2–4 GB RAM)** | PARTIAL — mechanisms fit, but the GloVe cache (381 MB RSS, 144 MB disk) is too large for a low-end device. | Replace the full 400k-vector cache with a sharded/quantized subset (e.g. top-N by frequency) or an on-demand indexed store; keep all mechanism code unchanged. Mechanisms themselves are <1 MB. |
| **Microcontroller (KB–MB RAM, no OS, no float unit maybe)** | NO for the full engine as-is — the GloVe cache alone exceeds MCU RAM by ~1000×. | The cognitive mechanisms (ConceptGraph, CerebellarNgram, ConstructionSchemaStore, FunctionalLexicon, IdentityEngine, stance/fact/belief stores) are all microcontroller-class (KB-scale, integer-friendly dot products). A port would (a) drop GloVe in favour of a tiny hashed embedding or a 1k-word subset, (b) quantize node vectors to int8, (c) disable code generation and ability composition (they are laptop-class conveniences, not cognition). The core recall/reasoning loop would survive. |

## 7. Constraint compliance (no LLM, no retraining)

- No LLM is imported or called anywhere in the measured path. A grep for
  `openai|anthropic|transformers.*pipeline|llm` in `engine.py` returns only
  substrings inside unrelated words (e.g. "tellme"); there are no provider
  imports or calls. The optional web-learning imports (bs4/trafilatura) are
  guarded by try/except and are disabled under `RAVANA_OFFLINE=1`.
- No retraining occurred: the probe used a fresh engine (`user_suffix="energyprobe"`,
  stale pkls deleted first) and every capability exercised (code gen, ability
  composition, stance/fact mining, construction induction) is online/incremental.
- No mechanism was "optimised" by adding a bigger model; the only large asset is
  the pre-existing GloVe seed cache, which is a seed, not a mechanism.

## 8. Limitations (honest)

1. RSS is a process-level metric, not a per-mechanism one; per-mechanism RSS
   deltas are below page granularity (0.00–0.05 MB) and therefore reported as
   build time + seed size instead. This is a measurement limit, not a claim that
   the mechanisms are free.
2. Only one seed (42) and one dim (64) were measured. Other dims/seeds will
   scale roughly linearly in vector storage; mechanism costs (dict sizes) are
   dim-independent.
3. The 25-turn probe is short; long-running sessions may grow graph/n-gram stores
   further. The growth rate observed: +328 MB over 25 turns, but that growth is
   dominated by a single sleep-consolidation event (+223 MB at turn 21) which
   loaded additional GloVe-derived state; the steady-state per-turn growth is
   ~1–3 MB. A longer probe (N≥100) is needed to confirm steady state — deferred.
4. Code generation and ability composition were exercised only once each in the
   probe; their per-call cost is folded into the 8 s tail turns and not isolated.
   A follow-up micro-benchmark is recommended if a precise per-call number is
   needed for a phone port.

## 9. Files / evidence

- `scratch/energy_report.json` — full 25-turn run (RSS series, per-turn latency, mechanism sizes).
- `scratch/mech_report2.json` — per-mechanism build times.
- `scratch/measure_energy.py`, `scratch/mech_measure2.py` — the measurement scripts (reproducible).
- Method: `psutil` 7.2.2 RSS + `time.perf_counter`, `RAVANA_OFFLINE=1`, seed 42.
