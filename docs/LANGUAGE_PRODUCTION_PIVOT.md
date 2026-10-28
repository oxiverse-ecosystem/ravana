# Language Production Pivot

**Status:** research + design. No code changed.
**Scope:** replace template-shaped emission with genuine generation from learned structure, inside a ~20 W budget, with no LLM, no retraining-from-scratch, and no hardcoded reply strings.
**Deliverable:** this document. Task #1 of the pivot.
**Audience:** whoever implements tasks #2+.

---

## 0. TL;DR

The premise that started this task — *"RAVANA emits template-shaped text and has no generative machinery"* — is **half wrong**, and the half that is right is worse than the premise.

RAVANA already contains a complete generative stack:

| Component | File | Line |
|---|---|---|
| `NeuralDecoder` (GRU, 256 hidden, 4 heads, vocab 15 000) | `ravana_ml/src/ravana_ml/nn/neural_decoder.py` | `generate` 250, `train_on_sentence` 489; instantiated at `response_gen.py:270-278` |
| `PredictiveCodingGenerator` (settle-to-attractor, top-p `_pick`) | `ravana/src/ravana/decoder/predictive_coding_generator.py` | `generate` 114, `_pick` 261 |
| `AttractorMemory` | `ravana/src/ravana/core/attractor_memory.py` | 41 |
| `CerebellarNgram` (learned transitions) | `ravana/src/ravana/language/cerebellar_ngram.py` | `learn_chain` 87, `predict_next` 283 |
| `VerbLexicon` (Hebbian relation→verb) | `ravana/src/ravana/language/verb_lexicon.py` | `select_verb` 500, `reinforce` 704 |
| `SyntacticCellAssembly` (frames, feedback learning) | `ravana/src/ravana/language/syntactic_cell_assembly.py` | `bind_to_sentence` 200, `learn_from_feedback` 452 |
| `BasalGangliaGate` (Go/NoGo competitive selection) | `ravana/src/ravana/language/basal_ganglia.py` | `select_concept` 206 |
| `SurfaceRealizer` (morphology, agreement) | `ravana/src/ravana/language/surface_realizer.py` | 58 |

**Every one of these is orphaned, disabled, or reduced to a cosmetic garnish.** The emission decisions are still taken by `random.choice` over hand-written string pools.

Three pieces of evidence, each a single line you can grep:

1. `_generate_with_decoder` (`response_gen.py:852`, body 852–1182) — ~330 lines of decoder integration — has **zero call sites in the repository**. Only its own `def` line matches a repo-wide search. It is dead code.
2. `self.use_linggen = False` (`engine.py:1230`) explicitly disables the settle/attractor path, with the promotion condition written in the comment: *decoder-CE ≤ template-CE on held-out data*.
3. `_generate_response`'s own docstring (`response_gen.py:6839-6841`) records the removal: `"Neural decoder (bag-of-words, CE ~3.9, never produced coherent output)"`.

So the actual problem is not *"build a generator."* It is:

> **RAVANA has a generator, has already measured that it loses to templates, and therefore answers every turn by sampling from a table. The pivot is to make the generator win — on grounded, incremental, decoder-first emission — rather than to build one from nothing.**

That reframing changes the whole plan. The work is not "add a neural net." It is:

- **Repair the conditioning**, not the decoder. The decoder was fed bag-of-words. A GRU at CE 3.9 is not evidence that small decoders fail; it is evidence that *unconditioned* small decoders fail. The brain's production system is also small and also bad at unconditioned generation — it is good because it is conditioned on a structured message and on a learned forward model.
- **Move the decision** from `random.choice` to a competitive gate over graph-derived candidates (`BasalGangliaGate`, already implemented, currently only used for concept selection, `basal_ganglia.py:206`).
- **Make the emitter incremental**, one lemma at a time, with unification — which is what every production model in the literature actually does, and which is also what makes the search small enough for 20 W.
- **Keep the honesty guard.** `_forward_model_check` (`response_gen.py:8158`) and `_coherence_ok` (`response_gen.py:7770`) are good and should survive the pivot.

Counts at the end: **ADOPT 16, REJECT 4, DEFER 3** (23 mechanisms surveyed).

---

## 1. Method, and what "research first" meant here

1. Read the neuroscience and psycholinguistics literature on **production** specifically — Levelt and successors, the dual-stream literature, predictive processing, incremental sentence formulation, basal-ganglia selection, and the metabolic-cost literature. Sources are listed in §10.
2. Read the actual code path that emits a turn, end to end, and recorded **function name + line number** for every claim. No claim in Part II is adjective-only.
3. Mapped each mechanism onto the code. Where a component already exists, the verdict is about **wiring**, not construction.
4. Wrote implementation sketches only for ADOPT items, and constrained every sketch to the three hard requirements: **online-learnable**, **decoder-first**, **typed-concept-graph-grounded**.

### Evidence-strength grading

Neuroscience is not uniform in quality, and a design document that pretends otherwise is useless. I use three grades:

- **[S] Strong** — replicated across methods (fMRI + MEG/EEG + lesion + intracranial), convergent across labs.
- **[M] Moderate** — robust behavioral/model evidence, neuroanatomical mapping contested or coarse.
- **[W] Weak / inferential** — plausible, widely cited, but the neural evidence is indirect or the theory is underdetermined by data.

A **[W]** mechanism can still be ADOPTed if it is cheap and falsifiable, but it must not carry a load-bearing claim.

---

## 2. What the brain actually does

### 2.1 The functional layout: message → lemma → form, with a convergence hub

The consensus production architecture (Levelt 1989; Levelt, Roelofs & Meyer 1999; Indefrey & Levelt 2004) is a **feed-forward cascade with feedback**, in three macro-stages:

1. **Conceptual preparation** — a non-linguistic *message* is formed (who did what to whom, with what focus).
2. **Grammatical encoding** — the message is mapped onto lemmas (syntactic words: a lemma carries meaning + syntax, not phonology). Lemmas are ordered into a syntactic frame; **function words and inflections are computed here**, not retrieved.
3. **Morpho-phonological encoding** — the selected lemma activates its **lexeme** (form), which is spelled out.

The critical structural point, and the one RAVANA gets most wrong:

> **Function words and grammatical morphemes are not lexical items to be sampled. They are consequences of the frame.** "the", "of", "is", "-ed", "-s" are *outputs* of grammatical encoding, not entries in a word list.

In RAVANA today, the equivalent of step 2 is a `random.choice` over whole pre-agreed sentences. Steps 1–3 have collapsed into one table lookup.

The **convergence zone** account (Hagoort 2005, 2013; Binder & Desai 2011; lambda-sub>ATL) supplies the other half: semantic combination happens where modality-specific representations converge, and the result is a **distributed but structured** representation — a *graph*, not a vector. This is direct support for RAVANA's `typed concept graph` choice, and against a purely embedding-based message representation.

**Evidence: [S]** for the stage decomposition (Indefrey & Levelt 2004 meta-analysis of 82 studies gives consistent time windows: conceptual ~0–200 ms, lemma ~200–400 ms, form ~400–600 ms). **[M]** for the exact anatomical assignment of each stage — the temporal ordering is solid, the cortical map is not.

### 2.2 Comprehending and producing are not the same computation

This is the asymmetry that the original premise blurred, and it matters because it decides which parts of the literature transfer.

| | Comprehension | Production |
|---|---|---|
| Direction | signal → meaning | meaning → signal |
| Input | continuous, noisy, redundant | an intention, not a signal |
| Core problem | **infer** the hidden structure | **select and order** an action |
| Search space | given by the input | must be constructed |
| Predictive processing evidence | strong | weak-to-moderate |

Hickok & Poeppel (2007) split the **comprehension** system into a dorsal (sound→articulation) and ventral (sound→meaning) stream. Hickok (2012) and Matchin & Hickok (2020) extend a *separate* production-side circuit (left frontal + SMA + basal ganglia + cerebellum) that is **not** the comprehension stream run backwards. Pickering & Garrod (2013) argue the two systems are *integrated* through forward models and alignment, but they are still two systems.

Two consequences for RAVANA:

1. **A comprehension scorer cannot be repurposed as a generator.** Any design of the form "score candidate strings with the comprehension model and pick the best" is a category error. It also reintroduces brute-force search (§2.5). RAVANA's `_coherence_ok` and `_forward_model_check` are *guards* — they reject bad output — and that is the correct, limited role for a comprehension-side signal in production.
2. **The generator needs its own forward model.** Production is supervised by an efference copy: the speaker predicts what the utterance will sound like and compares. That is `_forward_model_check`'s stated job (`response_gen.py:8158-8175`) and it is currently only a degeneracy filter, not a predictive comparator.

**Evidence: [S]** for the double dissociation (aphasia: production-impaired vs comprehension-impaired patients are routinely doubly dissociated). **[M]** for "separate circuits"; **[W]** for any precise claim that the production stream contains a full hierarchical predictive hierarchy.

### 2.3 Hierarchical predictive processing — and the honest caveat

The predictive-processing framework (Rao & Ballard 1999; Friston 2010; Clark 2013) says a cortical hierarchy passes **predictions downward** and **prediction errors upward**; only the residual is transmitted, which is what makes the system cheap. Heilbron et al. (2022, *PNAS*) is the cleanest empirical demonstration in language: MEG during natural narrative listening shows **distinct prediction-error signals at lexical, syntactic, and semantic levels**, each with its own temporal signature, consistent with a multi-level hierarchy.

**But:** essentially all of this evidence is **comprehension**. Heilbron et al. is listening. Kuperberg & Jaeger (2016) is a review of *comprehension* prediction. There is no equally clean production-side demonstration of a multi-level prediction-error hierarchy, because production is harder to control and the "input" is internal.

The defensible position:

- Adopt hierarchical prediction as an **engineering principle** for the emitter: predict at the level of (a) next lemma, (b) next syntactic role, (c) next morpheme, and learn from the residual at each level.
- Do **not** claim the brain's production system is proven to be a predictive hierarchy. Claim only that the mechanism is cheap, has a working comprehension precedent, and is falsifiable in our setting (the per-level residuals are measurable).
- Do **not** adopt predictive coding as a *complete* theory of production. See §M19 (REJECT).

**Evidence: [S]** for predictive coding in comprehension; **[W]** for predictive coding as the organizing principle of production.

### 2.4 Incremental, left-to-right composition — and why it is *strategic*, not architectural

Kempen & Hoenkamp (1987) introduced **Incremental Procedural Grammar**: the formulator does not build a whole sentence then linearize it. It produces **fragments** (a phrase, a clause) as soon as they are grammaticalizable, and hands each to the articulator while the rest is still being planned. Vosse & Kempen (2000) formalize the parser side as **unification over a feature-structure tree with a small, fixed set of syntactic nodes** — a structure that is *assembled*, not *searched*.

This is the single most important engineering lesson for a 20 W budget:

> Incrementality is not a stylistic choice. **It is what keeps the search space small.** A whole-sentence generator must choose among exponentially many sentences. An incremental generator chooses among a handful of *local* continuations at each step, each already constrained by the frame built so far.

Ferreira & Swets (2002) add the necessary correction: **incrementality in production is strategic, not architectural.** Speakers *can* plan further ahead; they do so when the material is predictable and stop doing so when it is not. So the right design is **variable planning grain** — plan one phrase ahead by default, extend the window when the graph is dense and the frame is unambiguous, shrink to one lemma when it is not.

**Evidence: [S]** for incremental formulation; **[S]** for the strategic-grain qualifier.

### 2.5 How 20 W avoids brute-force search

The brain runs on ~20 W — about 20 % of a 100 W resting metabolic budget (Raichle & Gusnard 2002; Attwell & Laughlin 2001). It cannot afford to score thousands of candidate sentences. It avoids this with **five** distinct economies, and they compose:

1. **Sparsity.** At any moment only ~1–15 % of neurons are active (Attwell & Laughlin 2001; Olshausen & Field 1996, 2004). The code is sparse because sparse codes are metabolically cheap *and* because they reduce interference between representations. — **This is the load-bearing one.**
2. **Prediction.** Most of the signal is cancelled by a top-down prediction before it is ever transmitted upward (Rao & Ballard 1999; Friston 2010). You only pay for surprise.
3. **Energy-based settling, not search.** Hopfield (1982) networks and interactive-activation models (McClelland & Rumelhart 1981) settle into a low-energy state by local relaxation. There is no global enumeration; the state falls downhill. Cost is proportional to *depth of settling*, not to *size of the candidate set*.
4. **Competitive gating, not ranking.** The basal ganglia implement **focused selection with inhibition of competitors** (Mink 1996; Crosson et al. 2003, 2007 — the GODIVA model for language). A Go signal amplifies one candidate while NoGo suppresses the rest. You do not evaluate all options to completion; the first one that clears threshold wins and the rest are actively shut down. This is *early exit by inhibition*, and it is why the brain does not need to score everything.
5. **Cortical hierarchy as a prior.** Higher levels constrain lower levels so tightly that lower levels have almost nothing left to decide. The hierarchy is a *search-space compressor*: each level hands down a distribution narrow enough that the level below samples rather than searches (Heilbron et al. 2022).

The engineering translation for RAVANA:

- Emit **one token at a time**, conditioned on a small active set (`k ≤ 4` content nodes from the graph, plus the current frame state). Never enumerate sentences.
- Maintain a **sparse active set**, not a dense vocabulary distribution over everything.
- Use **settling** (`PredictiveCodingGenerator.generate`, line 114) rather than beam search.
- Use **Go/NoGo early exit** (`BasalGangliaGate.select_concept`, line 206) rather than top-k ranking.
- Add a **learned transition prior** (`CerebellarNgram.predict_next`, line 283) so most continuations cost nothing to decide.
- Measure **MACs per token** as the energy proxy, since we cannot measure watts.

**Evidence: [S]** for the metabolic numbers and for sparsity; **[M]** for GODIVA-style basal-ganglia language selection; **[S]** for energy-based settling as a computational principle.

### 2.6 The lexicon is a network, not a list

A "mental lexicon" of ~50 000 entries is not a dictionary. It is a weighted graph in which:
- lemmas are linked by semantic relations, by syntactic frame membership, and by co-occurrence;
- activation spreads (Dell 1986), so retrieving one node raises its neighbours;
- frequency and recency modulate edge weights (usage-based: Goldberg 2006; Tomasello 2003);
- **constructions** — form–meaning pairings larger than a word, e.g. the ditransitive — are themselves stored nodes with their own weights.

Consequence: word choice is a **settling process over a weighted graph**, and "which word" and "which frame" are the same decision, not two.

**Evidence: [S]** for network/spreading-activation structure; **[M]** for the strong constructionist claim.

### 2.7 Learning is error-driven and online; there is no offline epoch

The brain does not run gradient descent over a global corpus loss. It updates locally, on surprise, during use (Chang, Dell & Bock 2006 — the *dual-path* model learns syntax by predicting its own next input; Saffran, Aslin & Newport 1996 — statistical learning from ~2 minutes of exposure). Consolidation happens offline (sleep replay) but the *learning signal* is generated online by prediction error.

This is good news for the pivot: **online, per-turn, error-gated updates are the biologically correct design**, not a compromise. It means RAVANA does not need a training corpus or a training loop — it needs a prediction-error channel and small local updates at emit time.

**Evidence: [S]** for statistical learning; **[M]** for dual-path as a production model.

---

## 3. What RAVANA actually does today

### 3.1 The emit path, end to end

```
engine.py:10584          response, strategy = self._generate_response(ctx)
  └─ response_gen.py:6826  _generate_response(ctx)
       ├─ 6852            pfc_workspace.detect_question_type(...)      # routing only
       ├─ 6849-6858       social/chitchat -> _handle_chitchat         # def at 1959
       │                     └─ _compose_greeting   3090
       │                        _compose_wellbeing  3130
       │                        _compose_capability 3197
       │                        _compose_farewell   3225
       ├─ 3732            _ventral_path(ctx)      # fast, retrieval
       │                     └─ 3757 _generate_with_decoder_and_syntax  (engine_generation.py:186)
       ├─ 3798            _dorsal_path(ctx)       # slow, reasoning
       │                     └─ 3814 _generate_with_decoder_and_syntax
       └─ terminal        chain_walker.py:2011 _graph_fallback_response
  └─ engine_generation.py:841  _final_emit_guard(...)
  └─ response_gen.py:8158      _forward_model_check(...)
  └─ engine.py:10700-10706     feedback: syntactic_assembly.learn_from_feedback(...)
```

There *is* a structured pipeline: `_generate_with_decoder_and_syntax` (`engine_generation.py:186`) runs PFC → discourse plan → `SyntacticCellAssembly` → `BasalGangliaGate` → `CerebellarNgram` → `SurfaceRealizer`. That is, on paper, exactly the architecture §2 recommends.

It is defeated in practice by three facts:

1. It is only reached on the ventral/dorsal paths (`response_gen.py:3757`, `:3814`), which are the *non-social* paths; the social paths — the highest-volume turn type — bypass it entirely and go to the composed composers.
2. `use_linggen = False` (`engine.py:1230`) means the settle/attractor stage is skipped even when reached.
3. Even when the pipeline runs, it terminates in tables: `SurfaceRealizer` picks among authored variant strings with `random.choices` (`surface_realizer.py:776`).

### 3.2 The decision point is uniform RNG over authored strings

The user-specified pool sites, verified at their exact lines:

| Line | Construct | What it decides |
|---|---|---|
| `response_gen.py:2227` | `return random.choice([...])` | an entire reply |
| `response_gen.py:2275` | `return random.choice([...])` | an entire reply |
| `response_gen.py:2429` | `return random.choice([...])` | an entire reply |
| `response_gen.py:2490` | `return random.choice([...])` | an entire reply |
| `response_gen.py:3064` | `return random.choice(_warm)` | an entire reply |
| `response_gen.py:3088` | `return random.choice(leads) + " " + random.choice(follows)` | an entire reply, by two independent draws |
| `response_gen.py:3102` | `greeting_word = random.choice([...])` | one word of a greeting |

Line 3088 is the clearest illustration of the failure mode. Two independent uniform draws, concatenated with a space. There is no conditioning on the user's turn, no conditioning on the graph, no conditioning on the previous sentence, and no learned weight anywhere in the expression. The output distribution is fixed at authoring time and is identical for every input.

**Quantified template floor** (AST scan of the two emission files):

| File | `random.choice`/`choices` calls | Inline authored-list sites | Authored string literals in those lists |
|---|---|---|---|
| `chat/response_gen.py` | 41 | 31 | 104 |
| `language/surface_realizer.py` | 8 | 3 `random.choices` sites + class-level tables | — |

`_compose_greeting` (`response_gen.py:3090`) documents its own combinatorics at line 3098: `"Combinations: 5 x 4 x 5 = ~100 (vs 9 hardcoded strings)"`. This is presented as a victory over hardcoding. It is not: it is a Cartesian product of **authored** strings, every factor of which was written by a human, and none of which is conditioned on anything. 100 outputs is a bigger table, not a generator.

### 3.3 The generative machinery, and why it is not running

| Component | Implemented | Where it is wired | Why it does not decide emission |
|---|---|---|---|
| `NeuralDecoder` | yes, ~1 200 lines | `engine.py:508` sets `MAX_DECODER_VOCAB_SIZE = 15000`; `_DECODER_DIM = 64 + 11` (`response_gen.py:169`) | `generate` (250) is reached only from the dead `_generate_with_decoder` (852) and from the disabled linggen path |
| `PredictiveCodingGenerator` | yes, 292 lines | only invoked when `use_linggen` | `use_linggen = False` (`engine.py:1230`) |
| `AttractorMemory` | yes, 234 lines | retrieval for the graph path | feeds the disabled path |
| `CerebellarNgram` | yes, 428 lines | `predict_next` used inside the syntactic pipeline | pipeline is bypassed on social turns |
| `VerbLexicon` | yes, 983 lines | `select_verb` (500) / `reinforce` (704) | only consulted inside the bypassed pipeline |
| `SyntacticCellAssembly` | yes, 627 lines | `bind_to_sentence` (200) | same |
| `BasalGangliaGate` | yes, 320 lines | `select_concept` (206) | selects *concepts* only; never selects tokens |
| `SurfaceRealizer` | yes, 1 252 lines | end of the syntactic pipeline | its variant step is `random.choices` (`776`) over authored templates |

The pattern is consistent: **each component exists, is qualitatively good, and is downstream of a gate that is closed.** The architecture is not missing. It is unpowered.

### 3.4 The promotion gate is already written down

`engine.py:1230`:

```python
self.use_linggen = False
```

The accompanying comment states the condition for turning it on: **decoder cross-entropy ≤ template cross-entropy on held-out data.** That is a falsifiable, measurable gate, and it is the correct one. The pivot should adopt it verbatim rather than invent a new success criterion.

The docstring at `response_gen.py:6839-6841` records why it was closed:

```
Removed from response path:
- Neural decoder (bag-of-words, CE ~3.9, never produced coherent output)
```

"Bag-of-words" is the diagnosis. CE ~3.9 on an unconditioned bag-of-words input is roughly what you would expect from a model that has been given the *topic* but not the *message*. The decoder was not asked to produce a specific proposition; it was asked to produce language from a topic cloud, and it produced plausible word salad. That is a conditioning failure, not a capacity failure — and it is fixable without a bigger model.

### 3.5 There is no cross-sentence state

`DiscourseState` declares three fields (`surface_realizer.py:48-50`):

```python
previous_subject: Optional[str] = None
previous_object:  Optional[str] = None
previous_verb:    Optional[str] = None
```

A repo-wide read finds **no reader** of any of the three. They are written by nothing and consumed by nothing.

Consequence: RAVANA cannot do the thing that makes multi-turn text feel generated — carry a referent forward, elide a repeated subject, use a pronoun because the noun was just mentioned. Every turn is a fresh sample. This is also why the templates *must* be self-contained: a template cannot refer to context it cannot see.

### 3.6 The epistemic frames are a hash, not an estimate

`surface_realizer.py:137-147` defines `EPISTEMIC_FRAMES` keyed by confidence level. Selection at `:263`:

```python
idx = (len(subject) + sum(ord(c) for c in subject)) % len(frames)
```

The frame is a deterministic hash of the **subject string's characters**. Two turns with equal confidence but different subjects get different hedges; the same subject always gets the same hedge regardless of confidence. This is decoration imitating calibration. It also means `hedges.py`'s `modality_from_support` (`:161`) and `hedge_frame` (`:108`) — which are the real, evidence-based machinery — are not what actually drives the surface form.

### 3.7 There is a divergent duplicate

`_generate_with_decoder_and_syntax` exists twice:

- `ravana/src/ravana/chat/engine_generation.py:186` (2653-line module, the maintained one, called from `response_gen.py:3757/3814`, `engine_memory.py:3269`)
- `ravana/src/ravana/chat/interface.py:1602` (2142-line module, a copy with the same docstring, reachable from `interface.py:654/1177`)

Two copies of the same pipeline that can drift independently. Any pivot work on the pipeline must first collapse them, or the fix will only apply to half the call sites.

---

## 4. Structural gap

Every row is a real location. This is the section to cite when someone asks "what exactly is missing."

| # | Gap | Evidence (file:line) | Consequence |
|---|---|---|---|
| G1 | Decoder integration is dead code | `response_gen.py:852` `_generate_with_decoder` (body 852–1182); repo-wide grep matches only this `def` | ~330 lines of conditioning work unreachable |
| G2 | Settle/attractor path disabled | `engine.py:1230` `self.use_linggen = False` | `PredictiveCodingGenerator.generate` (114) never runs in production |
| G3 | Promotion gate never evaluated in CI | gate documented in comment at `engine.py:1230`; no test asserts it | the bit can be flipped without evidence, or never flipped |
| G4 | Social turns bypass the structured pipeline | `response_gen.py:6858` calls `_handle_chitchat` (def at 1959) *before* `_ventral_path` (3732) | highest-volume turns use the least structured path |
| G5 | Emission is uniform over authored strings | `response_gen.py:2227, 2275, 2429, 2490, 3064, 3088, 3102` | zero conditioning on input, graph, or history |
| G6 | "Composition" is a Cartesian product of authored strings | `response_gen.py:3098` comment `5 x 4 x 5 = ~100` | scales the table, not the mechanism |
| G7 | Surface variants are sampled, not generated | `surface_realizer.py:776` `random.choices(choices, weights=...)` | `_variant_weights` (756) only reweights a fixed authored menu |
| G8 | No cross-sentence state | `surface_realizer.py:48-50` fields never read | no anaphora, no ellipsis, no discourse continuity |
| G9 | Epistemic frame selection is a string hash | `surface_realizer.py:263` | hedge is uncorrelated with actual confidence |
| G10 | Basal ganglia gates concepts, not tokens | `basal_ganglia.py:206` `select_concept` | no competitive selection at emission time |
| G11 | Forward model is a degeneracy filter, not a predictor | `response_gen.py:8158` `_forward_model_check` | no efference-copy error signal to learn from |
| G12 | Feedback updates a fixed menu, not a construction space | `engine.py:10702` `learn_from_feedback(self.syntactic_assembly._last_frame, ...)`; `syntactic_cell_assembly.py:452` | learning cannot create a new construction, only reweight existing ones |
| G13 | Duplicate pipeline implementation | `engine_generation.py:186` vs `interface.py:1602`; the decoder is *also* constructed twice in one class at `response_gen.py:270` and `:419` | divergent behaviour across call sites |
| G14 | No energy/latency accounting on the emit path | none; `MAX_DECODER_VOCAB_SIZE` (`engine.py:508`) and `_DECODER_DIM` (`response_gen.py:169`) are the only shape declarations | cannot evaluate the 20 W constraint, cannot detect regressions |

**G5 + G7 together are the whole disease**: the decision is uniform sampling, and the only learnable surface is the weight of a hand-authored menu.

**G1 + G2 are the whole cure, if re-powered**: the machinery to do it properly is present, tested in isolation, and switched off by one boolean whose unset condition is already documented.

---

## 5. Mechanism-by-mechanism mapping

Verdict legend: **ADOPT** — implement (often: re-wire). **DEFER** — correct but not on the critical path now. **REJECT** — wrong for this system, with reason.

### Group A — architecture

#### M1. Production/comprehension asymmetry
- **What it is.** Separate systems for mapping meaning→signal and signal→meaning, integrated by forward models (Hickok & Poeppel 2007; Hickok 2012; Matchin & Hickok 2020; Pickering & Garrod 2013).
- **RAVANA today.** There is a routing split — `_ventral_path` (`response_gen.py:3732`) vs `_dorsal_path` (`:3798`) — but both are *retrieval* routes, and both call the same emitter. There is no separate production model.
- **Missing.** A production-side model with its own objective (predict the next lemma), not the comprehension-side scorers (`_coherence_ok` 7770, `_forward_model_check` 8158) repurposed as generators.
- **Verdict: ADOPT.** This is the single most consequential correction. **Evidence [S].**
- **Sketch.** Keep `_coherence_ok` and `_forward_model_check` as *rejectors only* — they never rank, never choose. Introduce `LemmaPlanner` as the production model: input = `(message_frame, active_graph_nodes)`, output = an ordered lemma sequence with roles. Train it online by `train_on_sentence` (`neural_decoder.py:489`) on turns that survived `_forward_model_check`. The comprehension-side graph remains a *conditioning source*, never a reward model — which also avoids the degenerate "score N candidates" search that §2.5 rules out.

#### M2. Lemma → lexeme, two stages
- **What it is.** Grammatical encoding selects a lemma (meaning+syntax); morpho-phonological encoding retrieves its form (Levelt et al. 1999).
- **RAVANA today.** `VerbLexicon.select_verb` (`verb_lexicon.py:500`) returns a **verb phrase string** — lemma and surface form fused. `SurfaceRealizer` then does morphology on top, so inflection is applied twice or not at all depending on path.
- **Missing.** The separation. Because form is fused into the lexicon, agreement and tense cannot be computed from the frame.
- **Verdict: ADOPT.** **Evidence [S].**
- **Sketch.** Add a `Lemma` record — `(concept_id, pos, features{sem, num, tense, aspect}, lexeme_key)` — and make `select_verb` return the lemma, with surface form produced by `SurfaceRealizer` from features. Migration is additive: keep the string-returning method as a thin wrapper so `syntactic_cell_assembly.bind_to_sentence` (200) keeps working during the transition. Learned online: `verb_lexicon.reinforce` (704) already updates relation→verb weights; extend the reinforced key from phrase-string to `(lemma, feature_bundle)` so agreement errors are attributable.

#### M3. Incremental grammatical encoding with unification
- **What it is.** Build the sentence as a feature-structure tree by unification, one fragment at a time (Kempen & Hoenkamp 1987; Vosse & Kempen 2000).
- **RAVANA today.** `SyntacticCellAssembly.bind_to_sentence` (`syntactic_cell_assembly.py:200`) binds subject/relation/object into a **fixed frame**; `rel_to_verb` is a static dict (`:293-298`). No unification, no feature agreement, no partial structures.
- **Missing.** A feature structure with slots that can be *partially* filled, and a unify operation that fails loudly (which is the learning signal).
- **Verdict: ADOPT.** **Evidence [S].**
- **Sketch.** Represent the in-progress utterance as `{role: node_id}` plus `{role: FeatureBundle}`. At each step, take the top-1 frame from `SyntacticCellAssembly` and attempt to unify the next graph node into the next open slot. On unification failure, emit the prediction-error signal (§M9) and either repair (try the next frame) or shrink the planning window (§M4). This is where the *frame* — not a string — becomes the source of function words: `determiner`, `aux`, `prep` are produced by slot-filling rules over the feature bundle. Learned online: `learn_from_feedback` (452) already receives `user_understood`; extend it to receive *which unification failed*, so frame weights become feature-conditioned rather than frame-global.

#### M4. Strategic (variable) incrementality
- **What it is.** Speakers plan one phrase ahead by default and extend the window only when the material is predictable (Ferreira & Swets 2002).
- **RAVANA today.** No planning window exists; the unit is the whole reply.
- **Missing.** A knob, and a signal to drive it.
- **Verdict: ADOPT.** **Evidence [S].**
- **Sketch.** `planning_window ∈ {1 lemma, 1 phrase, 1 clause}`, driven by `CerebellarNgram.predict_next` entropy (283): low entropy → widen (the continuation is cheap, so plan further); high entropy → narrow to one lemma. Because it reads an existing learned distribution, it is online-adaptive with no new parameters and no new training. Cap at one clause — this is the working-memory bound of M7.

#### M5. Convergence-zone semantic hub
- **What it is.** Semantic combination occurs where modality-specific representations converge; the result is structured, not a single vector (Hagoort 2005, 2013; Binder & Desai 2011).
- **RAVANA today.** The typed concept graph already *is* a convergence structure — nodes with typed edges. `_graph_fallback_response` (`chain_walker.py:2011`) walks it.
- **Missing.** Nothing structural. What is missing is that the walk's output is converted to a *sentence template* rather than to a *message* for the planner.
- **Verdict: ADOPT (as the conditioning substrate).** **Evidence [M].**
- **Sketch.** Define `Message = {predicate_edge, agent_node, patient_node, modifiers[], focus}` produced by the graph walk, and make `Message` the *only* input to `LemmaPlanner` (M1). Concretely: `_graph_fallback_response` (2011) and `_ventral_path`'s retrieval (`3732`) should both return `Message`, not text. This is a refactor of return type, not new science, and it is the change that makes G5 removable.

#### M6. Dual-stream routing
- **What it is.** Dorsal (sound↔articulation) and ventral (sound↔meaning) streams for comprehension (Hickok & Poeppel 2007).
- **RAVANA today.** `_ventral_path` (3732) / `_dorsal_path` (3798) exist, but they route *retrieval vs reasoning*, not *form vs meaning*.
- **Missing.** Nothing that matters for emission. The names are aspirational.
- **Verdict: DEFER.** Renaming them would be cosmetic; restructuring around a different axis is not on the critical path and risks destabilizing routing that works. Revisit only if M3's frame selection needs to be split by stream.

#### M7. Bounded working memory
- **What it is.** ~4 chunks of active content (Cowan 2001), which is what makes incremental planning tractable.
- **RAVANA today.** No explicit bound. Graph walks can return arbitrarily many nodes; `BasalGangliaGate.select_concept` (206) has `min_candidates_for_gating` but no upper bound on the active set.
- **Missing.** A hard `k`.
- **Verdict: ADOPT.** **Evidence [S].**
- **Sketch.** `ACTIVE_SET_MAX = 4` content nodes, enforced in the gate. This is both a biological fidelity fix and the main energy lever: attention cost in `PredictiveCodingGenerator._attention_logits` (97) is linear in `n_concepts`, so bounding it bounds the per-token MACs. Expose it as a constant and log the saturation rate — if the set is saturated >80 % of turns, the graph retrieval is over-recalling (§8 metric M5).

### Group B — control and inference

#### M8. Hierarchical predictive processing
- **What it is.** Top-down predictions, bottom-up errors, per level (Rao & Ballard 1999; Friston 2010; Heilbron et al. 2022).
- **RAVANA today.** None. There is no per-level prediction anywhere on the emit path. `_forward_model_check` (8158) is a single binary check at the very end.
- **Missing.** Three prediction levels and three error channels.
- **Verdict: ADOPT (as engineering principle, not as a claim about production neurobiology).** **Evidence [S] comprehension / [W] production.**
- **Sketch.** Three levels, each with a cheap learned predictor and a residual:
  - L1 *lemma*: `CerebellarNgram.predict_next` (283) predicts the next lemma; residual = surprisal of the chosen lemma.
  - L2 *role*: the frame's expected next slot; residual = unification failure (M3).
  - L3 *morpheme*: `SurfaceRealizer` agreement features; residual = feature mismatch.
  Feed the **summed residual** into (a) the learning rate of the local updates at each level, and (b) the gate: high residual raises `BasalGangliaGate`'s NoGo strength via `set_prediction_error` (`basal_ganglia.py:88`), which already exists and is currently never called from the emit path. No new model class required; this is wiring three existing signals into one existing setter.

#### M9. Surprise-gated (error-driven) learning
- **What it is.** Plasticity is gated by prediction error; predictable input produces almost no update (Friston 2010; Chang, Dell & Bock 2006).
- **RAVANA today.** Feedback learning is gated by `quality_score >= 0.55` (`engine.py:10700`), i.e. by an *outcome* judgement, not by a *prediction error*. `syntactic_assembly.learn_from_feedback` (452) and `verb_lexicon.reinforce` (704) both fire on the same coarse signal.
- **Missing.** The residual as the learning rate.
- **Verdict: ADOPT.** **Evidence [M].**
- **Sketch.** Multiply every existing update by `g(residual)`, e.g. `lr = lr0 * clip(residual, 0, 1)`. Concretely: `verb_lexicon.reinforce` (704) and `cerebellar_ngram.learn_chain` (87) gain an optional `surprisal` argument, defaulting to 1.0 for backward compatibility. Effect: a turn that unfolded exactly as predicted barely moves any weight (cheap, and prevents drift); a turn that surprised the model moves them. This is a two-line change per call site and makes every existing learner online-correct.

#### M10. Energy-based settling (attractor dynamics)
- **What it is.** Inference as relaxation to a low-energy state, not enumeration (Hopfield 1982; McClelland & Rumelhart 1981).
- **RAVANA today.** Fully implemented and disabled: `PredictiveCodingGenerator.generate` (`predictive_coding_generator.py:114`) settles toward a retrieved `AttractorMemory` trajectory; `_pick` (261) does top-p sampling from the settled logits; gated by `use_linggen=False` (`engine.py:1230`).
- **Missing.** Only the bit, and a conditioning fix (see M1/M5).
- **Verdict: ADOPT.** **Evidence [S].**
- **Sketch.** Do not rewrite. (1) Feed `generate` a `Message`-derived `conditioning_embs` instead of the current bag-of-words, via M5. (2) Set the `trajectory` argument from `AttractorMemory` over the *active set* (≤4 nodes, M7) rather than all retrieved nodes. (3) Flip `use_linggen` **only** behind the held-out CE gate that the code comment already specifies, on a per-route basis (social first — see §7). If `generate` returns `[]` it already fails closed to `realize_dim`, so the blast radius of a bad flip is bounded.

#### M11. Basal-ganglia Go/NoGo competitive selection
- **What it is.** Focused selection by amplifying one candidate and inhibiting competitors; early exit by inhibition (Mink 1996; Crosson et al. 2003, 2007).
- **RAVANA today.** `BasalGangliaGate.select_concept` (`basal_ganglia.py:206`) implements it correctly, with modulators (`set_dopamine_tone` 118, `set_prediction_error` 88, `set_arousal` 74, …). It selects **concepts**.
- **Missing.** It never selects a **token** or a **frame**. Emission has no competitive stage at all.
- **Verdict: ADOPT.** **Evidence [M].**
- **Sketch.** Extend the gate to a second decision point without changing its mathematics: `select_token(candidates)` where `candidates = [(token, logp_from_settle, confidence, role)]`. The Go/NoGo threshold (`compute_effective_go_threshold` 139) then fires on the settled distribution from M10, so the *decision* is gated while the *distribution* is neural. Crucially, this gives early exit: once one token clears threshold and inhibits the rest, we stop — which is the §2.5 economy in code. Wire `set_prediction_error` from M8 and `set_dopamine_tone` from the existing modulator dict (`set_all_from_modulators` 124) so production is affect-sensitive the way the rest of the system already is.

#### M12. Cerebellar forward model + efference copy
- **What it is.** The cerebellum learns to predict the sensory consequence of a motor command; the comparison with the actual consequence is the error signal (and the reason you cannot tickle yourself).
- **RAVANA today.** Two disjoint pieces: `CerebellarNgram` (`cerebellar_ngram.py:37`) learns transitions but is used as a *fluency prior* inside the bypassed pipeline; `_forward_model_check` (`response_gen.py:8158`) is a **post-hoc degeneracy filter**, explicitly documented as such ("pre-articulation guard"), not a predictor, and its output is not fed back into any learner.
- **Missing.** The loop. A prediction, a comparison, and an update.
- **Verdict: ADOPT.** **Evidence [M].**
- **Sketch.** Before emitting token *n*, ask `CerebellarNgram.predict_next` (283) for a distribution. After emitting, compute `residual = −log p_pred(actual)`. Two consumers: (a) `learn_chain` (87) already updates on success — pass `success = residual < τ` so the n-gram self-calibrates; (b) `_forward_model_check` gains a `predicted` argument and can reject *when the utterance diverges from its own predicted shape* — a much stronger and more interesting criterion than "is it word salad", and one that is genuinely an efference-copy analogue.

#### M13. Sparsity as a metabolic constraint
- **What it is.** ~1–15 % of neurons active at once (Attwell & Laughlin 2001; Olshausen & Field 1996, 2004); sparse codes are cheap and reduce interference.
- **RAVANA today.** The decoder is dense: `forward` (`neural_decoder.py:188`) produces a full `vocab`-width softmax (≤15 000, `engine.py:508`), and `_pick` (`predictive_coding_generator.py:261`) samples over all of it.
- **Missing.** A sparse candidate set.
- **Verdict: ADOPT (as a constraint, not a layer).** **Evidence [S].**
- **Sketch.** Do not add a sparsity penalty — that would need training. Instead *restrict the candidate set structurally*: at each step, candidates = {top-`k` from the n-gram prior} ∪ {tokens belonging to the active frame's open slots} ∪ {function words required by unification}, with `k ≈ 12`. Everything else is masked to `−inf` before `_pick`. This is biologically motivated, costs nothing to learn, cuts the sampling entropy (and therefore the per-token cost) by an order of magnitude, and is exactly the "narrow the distribution so the level below samples instead of searching" move of §2.5.

#### M14. Neuromodulatory gain
- **What it is.** Dopamine, arousal, and novelty change the *gain* on selection, not the content.
- **RAVANA today.** `BasalGangliaGate` has all the setters (`basal_ganglia.py:74-124`) and `set_all_from_modulators` (124), driven by the engine's modulator dict.
- **Missing.** Nothing on the gate. It is missing on the *emitter*: the settle temperature and the token gate threshold are constants.
- **Verdict: ADOPT.** **Evidence [M].**
- **Sketch.** Route the existing modulator dict to two parameters: `temperature` in `_pick` (`predictive_coding_generator.py:273`, already a settable attribute used at `:278`) and the effective Go threshold (`compute_effective_go_threshold` 139). High arousal → higher temperature (more varied), low → tighter. No new state; the modulators already exist and already drive concept selection. This makes emission feel alive without any authored variation.

### Group C — lexicon and learning

#### M15. Lexicon as a weighted network
- **What it is.** Words and constructions are nodes in a weighted graph; activation spreads; frequency/recency modulate edges (Dell 1986; Goldberg 2006; Tomasello 2003).
- **RAVANA today.** `VerbLexicon` (`verb_lexicon.py:82`) is a class-level structure with `select_verb` (500) and a `_init_hebbian_priors` (758) — this *is* a weighted lexicon, and it has the right shape.
- **Missing.** It is consulted only inside the bypassed pipeline, and it stores verb *phrases*, not lemmas (M2).
- **Verdict: ADOPT.** **Evidence [S].**
- **Sketch.** Promote `VerbLexicon` from garnish to the only source of predicative content: `LemmaPlanner` queries it, never a pool. Convert entries to lemma records incrementally (lazy: on first read of a phrase entry, split into lemma+features using `SurfaceRealizer`'s existing morphology). Learning already exists (`reinforce` 704); just add the surprisal gate from M9.

#### M16. Usage-based constructions (syntactic cell assemblies)
- **What it is.** Form–meaning pairings larger than a word are stored, weighted, and learned from use (Goldberg 2006; Tomasello 2003).
- **RAVANA today.** `SyntacticCellAssembly` (`syntactic_cell_assembly.py:45`) is a genuine construction inventory with `bind_to_sentence` (200) and `learn_from_feedback` (452). But `rel_to_verb` (`:293-298`) is a hardcoded dict, and feedback only reweights existing frames.
- **Missing.** The ability to **create** a construction. G12.
- **Verdict: ADOPT.** **Evidence [M].**
- **Sketch.** On a turn where (a) the output passed `_forward_model_check` *and* (b) no existing frame unified cleanly, allocate a new frame whose slots are the roles that were actually filled, initialized with a low weight. That is construction learning from a single successful use — the usage-based claim, and the mechanism that lets the system escape its authored menu. Bound the inventory (e.g. 512 frames, evict lowest-weight) so it stays in the sparse, cheap regime of M13. This directly closes G12 and is the only ADOPT that can produce a sentence no human authored.

#### M17. Interactive alignment / structural priming
- **What it is.** Interlocutors converge on each other's words and structures; alignment is a cheap route to mutual comprehensibility (Pickering & Garrod 2013).
- **RAVANA today.** Nothing. Each turn conditions on the graph and (no) history.
- **Missing.** Reuse of the user's structure.
- **Verdict: ADOPT.** **Evidence [M].**
- **Sketch.** After parsing the user's turn, bump the weight of the frame that matched their structure before frame selection, and seed `DiscourseState.previous_*` (`surface_realizer.py:48-50`, currently dead) from their subject/object/verb. Two payoffs: it makes G8 disappear, and it is the cheapest possible source of *conditioning on the input* — which is precisely what G5 says is absent. Decay the boost by ~0.5/turn so it is alignment, not perseveration.

#### M18. Sleep replay consolidation
- **What it is.** Offline replay of the day's trajectories strengthens what matters (Buzsáki; standard consolidation literature).
- **RAVANA today.** `NeuralDecoder.sleep_cycle` (`neural_decoder.py:1152`) exists.
- **Missing.** Nothing needed for emission.
- **Verdict: DEFER.** It is a maintenance operation. It improves quality over days, but it cannot make the emit path generative — that is decided per-turn. Revisit after M1/M10 are live, when there is a decoder worth consolidating.

### Group D — rejected

#### M19. Predictive coding as a complete theory of production
- **What it is.** The strong claim that production *is* hierarchical predictive processing.
- **Verdict: REJECT.** The empirical base (Heilbron et al. 2022; Kuperberg & Jaeger 2016) is comprehension. Production has no equally clean multi-level error demonstration. Adopting the strong claim would license architectural decisions (e.g. discarding a valid lemma because its prediction error is high) on evidence that does not exist. We adopt the *mechanism* (M8) as an engineering choice and keep the claim modest.

#### M20. A large gradient-trained language model on GPU
- **What it is.** The obvious solution: train or call a transformer.
- **Verdict: REJECT.** Violates three constraints simultaneously: (a) the ~20 W / portability budget — the current design is MAC-countable and CPU-resident, a transformer at useful scale is not; (b) no LLM in the generation path, explicit in the brief; (c) online-learnability — a frozen pretrained model cannot adapt per-turn, which is the whole point. Note this is *not* a claim that small GRUs are better than transformers in general. It is a claim about this system's constraints.

#### M21. Oscillatory / beta-band syntax binding
- **What it is.** The proposal that cortical rhythms implement syntactic binding (Buzsáki & Draguhn 2004 and the beta-syntax literature).
- **Verdict: REJECT.** There is no observable consequence we can implement or falsify in a text system. Adopting it would be decoration, which is exactly the sin G9 already commits.

#### M22. Sub-lexical morphological unification (MUC)
- **What it is.** Hagoort's Memory–Unification–Control: unification operates over morphemes, not just words (Hagoort 2013, 2019).
- **Verdict: DEFER.** Correct and relevant, but inflection is not the bottleneck — G5 (uniform sampling) is. Implementing MUC before M3/M16 would be optimizing the wrong layer. Revisit once frames are unified and lemma/lexeme (§M2) is split; at that point MUC is a natural extension of the feature bundles.

#### M23. Articulatory / prosodic planning
- **What it is.** Phonetic encoding, prosody generation, articulation.
- **Verdict: REJECT.** RAVANA emits text. There is no articulator. Prosody lives in punctuation, which `SurfaceRealizer` already handles.

---

## 6. Verdict summary

| # | Mechanism | Verdict | Evidence | Primary gap closed |
|---|---|---|---|---|
| M1 | Production/comprehension asymmetry | ADOPT | S | G1 |
| M2 | Lemma → lexeme | ADOPT | S | — |
| M3 | Incremental unification | ADOPT | S | G12 |
| M4 | Strategic incrementality | ADOPT | S | — |
| M5 | Convergence-zone hub → `Message` | ADOPT | M | G5 |
| M6 | Dual-stream routing | DEFER | S | — |
| M7 | Bounded working memory | ADOPT | S | G14 |
| M8 | Hierarchical predictive processing | ADOPT | S/W | G11 |
| M9 | Surprise-gated learning | ADOPT | M | G12 |
| M10 | Attractor settling | ADOPT | S | G2 |
| M11 | BG Go/NoGo token selection | ADOPT | M | G10 |
| M12 | Cerebellar forward model | ADOPT | M | G11 |
| M13 | Sparse candidate set | ADOPT | S | G14 |
| M14 | Neuromodulatory gain | ADOPT | M | — |
| M15 | Lexicon as network | ADOPT | S | G5 |
| M16 | Usage-based constructions | ADOPT | M | G12 |
| M17 | Interactive alignment | ADOPT | M | G8 |
| M18 | Sleep replay | DEFER | M | — |
| M19 | PC as complete production theory | REJECT | — | — |
| M20 | GPU language model | REJECT | — | — |
| M21 | Oscillatory binding | REJECT | — | — |
| M22 | Sub-lexical MUC | DEFER | M | — |
| M23 | Articulatory planning | REJECT | — | — |

**ADOPT 16 · REJECT 4 · DEFER 3.**

---

## 7. Target architecture

```
                 typed concept graph
                          |
                   graph walk / retrieval
                          |
                    Message  (M5, M1)
        {predicate, agent, patient, mods, focus}
                          |
              +-----------+-----------+
              |                       |
      BasalGangliaGate          AttractorMemory
      select_concept (206)      (trajectory over active set)
              |                       |
              +----------> LemmaPlanner (M1, M2)
                                |
                     incremental unify (M3)  <-- planning window (M4)
                                |
                    active set, k <= 4 (M7)
                                |
                PredictiveCodingGenerator.generate (114)
                     conditioning_embs from Message
                                |
                   settled logits  --mask-->  sparse candidates (M13)
                                |
                   + CerebellarNgram log-prior (M12)
                                |
                   BasalGangliaGate.select_token (M11)
                                |
                        token emitted
                                |
                SurfaceRealizer  (morphology only; M2)
                                |
                _forward_model_check (8158)  -> residual
                _coherence_ok        (7770)  -> reject
                _final_emit_guard    (841)
                                |
                               emit
                                |
                    ---- online update, gated by residual (M9) ----
                    syntactic_assembly.learn_from_feedback (452)
                    verb_lexicon.reinforce                  (704)
                    cerebellar_ngram.learn_chain            (87)
                    neural_decoder.train_on_sentence        (489)
```

Three properties of this diagram are the whole pivot:

1. **The arrow into the emitter is a `Message`, not a string.** Every ADOPT downstream depends on this; it is also the answer to "why did the decoder produce word salad" — it was given a topic, not a message.
2. **Every decision is a settle or a gate, never a search.** No stage enumerates candidates to score them. This is the 20 W property, and it is the property that template sampling accidentally has and must be preserved.
3. **The learning arrow closes the loop inside the same turn.** No corpus, no epochs. That is what makes the system learnable during normal operation, as required.

---

## 8. Plan, metrics, gate

### Phases

**P0 — Instrument, change nothing.** Add tracing behind `RAVANA_EMIT_TRACE=1`:
- which branch produced each turn (social / ventral / dorsal / fallback);
- the template-overlap flag for the emitted text (is it literally in an authored pool?);
- per-token decoder cross-entropy, even on turns the decoder did not produce (run it as a *shadow*, don't use it);
- MAC count per token from the declared shapes (`_DECODER_DIM` 169, `MAX_DECODER_VOCAB_SIZE` 508) — this is G14.
This phase is pure measurement and must be merged before any behaviour change, or the pivot cannot be evaluated.

**P1 — One narrow domain, end to end.** Target the social composers, because they are the highest-volume turns, the smallest vocabularies, and the ones that currently bypass everything (G4): `_compose_greeting` (3090), `_compose_wellbeing` (3130), `_compose_farewell` (3225). Implement M5 (`Message` for interoceptive state + graph) → M1/M2 → M3 → M10 with the existing `PredictiveCodingGenerator` → M11 → M13. Keep the existing pools as a fallback behind the CE comparison. Do not remove any string yet.

**P2 — Flip the gate where it is earned.** For each route separately, flip `use_linggen`/the decoder path only if §8's gate passes on held-out turns. Per-route, never global — the comment at `engine.py:1230` describes one bit, but the correct unit is a route.

**P3 — Close the discourse loop.** M17: populate `DiscourseState.previous_*` (48-50) and use it in conditioning. This is what makes multi-turn output stop reading as independent samples.

**P4 — Construction learning.** M16: allow new frames to be created from successful unifications. This is the step that can produce a sentence no human wrote, and it must come *after* P1–P3 have proven the emitter is coherent, or it will learn degenerate frames.

**P5 — Retire the tables.** Delete pool entries only where P2's gate has passed for ≥ N turns with no regression. Collapse the duplicate at `interface.py:1602` before retiring anything, so there is one pipeline.

### Metrics

| ID | Metric | Definition | Direction |
|---|---|---|---|
| M1 | `template_overlap@1` | fraction of emitted sentences matching an authored pool entry verbatim | → 0 |
| M2 | `distinct-2` / `distinct-3` | type-token ratio of bigrams/trigrams over 100 consecutive turns | ↑ |
| M3 | `surprisal_gap` | mean `CE_decoder − CE_template` on held-out turns | ≤ 0 to promote |
| M4 | `guard_pass` | pass rate of `_coherence_ok` (7770) and `_forward_model_check` (8158) | no regression |
| M5 | `grounding` | fraction of content words traceable to a typed edge in the retrieved subgraph | ↑ |
| M6 | `active_set_saturation` | fraction of turns with `k` at the M7 cap | < 0.8 |
| M7 | `p50/p95 latency` | per turn, on the reference CPU | no regression |
| M8 | `macs_per_token` | analytic count from decoder shapes | ≤ 1.5× today |
| M9 | `capability_regression` | existing capability test suite | 100 % |

### The promotion gate (mirrors `engine.py:1230`)

A route may switch from pool emission to generated emission only when **all** hold on a held-out set of ≥ 200 real turns:

1. `CE_decoder ≤ CE_template` (the gate already written in the code comment);
2. `template_overlap@1` drops below 0.2 for that route;
3. `guard_pass` does not regress by more than 2 points;
4. `p95 latency` and `macs_per_token` within the bounds above;
5. `capability_regression` is clean.

Otherwise the route keeps its pool. This makes the pivot **falsifiable per route** and makes a partial success a real success rather than a failed flip of one global bit.

### Risks

| Risk | Signal | Mitigation |
|---|---|---|
| Decoder collapses to high-frequency tokens | `distinct-2` falls, `surprisal_gap` negative but output repetitive | sparse candidate set (M13) + temperature from M14; top-p in `_pick` (261) already limits the tail |
| Empty conditioning → `generate` returns `[]` | fail-closed rate high | `generate` already fails closed (`predictive_coding_generator.py:144`); treat high fail-closed rate as a P1 blocker, not a silent fallback |
| Graph retrieval over-recalls | `active_set_saturation` > 0.8 | tighten the walk; the gate's `min_candidates_for_gating` is not a cap |
| Guards reject learned output | `guard_pass` drops | guards are correct; fix conditioning, never weaken `_forward_model_check` |
| Duplicate pipeline fixes only half the call sites | behaviour differs by entry point | collapse `interface.py:1602` into `engine_generation.py:186` first |
| Learned frames degenerate | new frames with 1 slot, or identical to existing | minimum-2-slot rule + dedup on creation; evict lowest weight at the 512 cap |

---

## 9. What this pivot explicitly does not do

- No LLM in the generation path.
- No new hardcoded reply strings. Every ADOPT either reuses an existing component or produces text from a learned distribution over graph-grounded content.
- No retraining from scratch. All updates are per-turn, local, and gated by surprise (M9).
- No removal of the honesty guards. `_forward_model_check` and `_coherence_ok` survive and get *stronger* (M12 gives the forward model something to predict).
- No claim that this reproduces the brain. It reproduces a small number of mechanisms the brain uses to stay cheap, and it is honest about which ones rest on production evidence and which are borrowed from comprehension.

---

## 10. References

Attwell, D., & Laughlin, S. B. (2001). An energy budget for signaling in the grey matter of the brain. *Journal of Cerebral Blood Flow & Metabolism*, 21(10), 1133–1145.

Binder, J. R., & Desai, R. H. (2011). The neurobiology of semantic memory. *Trends in Cognitive Sciences*, 15(11), 527–536.

Bock, K., & Levelt, W. J. M. (1994). Language production: Grammatical encoding. In M. A. Gernsbacher (Ed.), *Handbook of Psycholinguistics* (pp. 945–984). Academic Press.

Buzsáki, G., & Draguhn, A. (2004). Neuronal oscillations in cortical networks. *Science*, 304(5679), 1926–1929.

Chang, F., Dell, G. S., & Bock, K. (2006). Becoming syntactic. *Psychological Review*, 113(2), 234–272.

Clark, A. (2013). Whatever next? Predictive brains, situated agents, and the future of cognitive science. *Behavioral and Brain Sciences*, 36(3), 181–204.

Cowan, N. (2001). The magical number 4 in short-term memory: A reconsideration of mental storage capacity. *Behavioral and Brain Sciences*, 24(1), 87–114.

Crosson, B., Benefield, H., Cato, M. A., Sadek, J. R., Moore, A. B., Wierenga, C. E., … Briggs, R. W. (2003). Left and right basal ganglia and frontal activity during language generation. *Journal of the International Neuropsychological Society*, 9(7), 1061–1077.

Crosson, B., Benjamin, M., & Levy, I. (2007). Role of the basal ganglia in language production: The "GODIVA" model. In *Basal Ganglia and Thalamus* (pp. 1–26).

Dell, G. S. (1986). A spreading-activation theory of retrieval in sentence production. *Psychological Review*, 93(3), 283–321.

Fedorenko, E., & Blank, I. A. (2020). Broca's area is not a natural kind. *Trends in Cognitive Sciences*, 24(4), 270–284.

Fedorenko, E., Ivanova, A. A., & Regev, T. I. (2024). The language network as a natural kind within the broader landscape of the human brain. *Nature Reviews Neuroscience*, 25, 289–312.

Ferreira, F., & Swets, B. (2002). How incremental is language production? Evidence from the production of utterances requiring the computation of arithmetic sums. *Journal of Memory and Language*, 46(3), 623–640.

Friston, K. (2010). The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127–138.

Goldberg, A. E. (2006). *Constructions at Work: The Nature of Generalization in Language*. Oxford University Press.

Hagoort, P. (2005). On Broca, brain, and binding: A new framework. *Trends in Cognitive Sciences*, 9(9), 416–423.

Hagoort, P. (2013). MUC (Memory, Unification, Control) and beyond. *Frontiers in Psychology*, 4, 416.

Hagoort, P. (2019). The neurobiology of language beyond single words. *Annual Review of Neuroscience*, 42, 425–446.

Heilbron, M., Armeni, K., Schoffelen, J.-M., Hagoort, P., & de Lange, F. P. (2022). A hierarchy of linguistic predictions during natural language comprehension. *PNAS*, 119(32), e2201968119.

Hickok, G. (2012). Computational neuroanatomy of speech production. *Nature Reviews Neuroscience*, 13(2), 135–145.

Hickok, G., & Poeppel, D. (2007). The cortical organization of speech processing. *Nature Reviews Neuroscience*, 8(5), 393–402.

Hopfield, J. J. (1982). Neural networks and physical systems with emergent collective computational abilities. *PNAS*, 79(8), 2554–2558.

Indefrey, P., & Levelt, W. J. M. (2004). The spatial and temporal signatures of word production components. *Cognition*, 92(1–2), 101–144.

Kempen, G., & Hoenkamp, E. (1987). An incremental procedural grammar for sentence formulation. *Cognitive Science*, 11(2), 201–258.

Kuperberg, G. R., & Jaeger, T. F. (2016). What do we mean by prediction in language comprehension? *Language, Cognition and Neuroscience*, 31(1), 32–59.

Levelt, W. J. M. (1989). *Speaking: From Intention to Articulation*. MIT Press.

Levelt, W. J. M., Roelofs, A., & Meyer, A. S. (1999). A theory of lexical access in speech production. *Behavioral and Brain Sciences*, 22(1), 1–38.

Matchin, W., & Hickok, G. (2020). The cortical organization of syntax. *Cerebral Cortex*, 30(3), 1481–1498.

McClelland, J. L., & Rumelhart, D. E. (1981). An interactive activation model of context effects in letter perception. *Psychological Review*, 88(5), 375–407.

Mink, J. W. (1996). The basal ganglia: Focused selection and inhibition of competing motor programs. *Progress in Neurobiology*, 50(4), 381–425.

Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive field properties by learning a sparse code for natural images. *Nature*, 381, 607–609.

Olshausen, B. A., & Field, D. J. (2004). Sparse coding of sensory inputs. *Current Opinion in Neurobiology*, 14(4), 481–487.

Pickering, M. J., & Gambi, C. (2018). Predicting while comprehending language: A theory and review. *Trends in Cognitive Sciences*, 22(11), 1002–1014.

Pickering, M. J., & Garrod, S. (2013). An integrated theory of language production and comprehension. *Behavioral and Brain Sciences*, 36(4), 329–347.

Raichle, M. E., & Gusnard, D. A. (2002). Appraising the brain's energy budget. *PNAS*, 99(16), 10237–10239.

Rao, R. P. N., & Ballard, D. H. (1999). Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field effects. *Nature Neuroscience*, 2(1), 79–87.

Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*, 274(5294), 1926–1928.

Tomasello, M. (2003). *Constructing a Language: A Usage-Based Theory of Language Acquisition*. Harvard University Press.

Vosse, T., & Kempen, G. (2000). Syntactic structure assembly in human parsing: A computational model based on competitive inhibition and a lexicalist grammar. *Cognition*, 75(2), 105–143.

---

## Appendix — how to re-verify every code claim

From the repository root (`ravana/`):

```powershell
# G1: _generate_with_decoder has no call sites (only the def line matches)
Select-String -Path "ravana\src\ravana\**\*.py" -Pattern "_generate_with_decoder\b"

# G2: the disabled gate
Select-String -Path "ravana\src\ravana\chat\engine.py" -Pattern "use_linggen"

# G5: the uniform-RNG emission sites
Select-String -Path "ravana\src\ravana\chat\response_gen.py" -Pattern "random\.choice"

# G6: the Cartesian-product claim
Select-String -Path "ravana\src\ravana\chat\response_gen.py" -Pattern "5 x 4 x 5"

# G8: the discourse fields, and their absence of readers
Select-String -Path "ravana\src\ravana\language\surface_realizer.py" -Pattern "previous_(subject|object|verb)"

# G9: the hash-based frame picker
Select-String -Path "ravana\src\ravana\language\surface_realizer.py" -Pattern "sum\(ord\(c\)"

# G13: the duplicate pipeline
Select-String -Path "ravana\src\ravana\chat\*.py" -Pattern "def _generate_with_decoder_and_syntax"
```

All line numbers in this document were read from the working tree at the time of writing and are 1-based.
