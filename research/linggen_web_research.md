LINGGEN WEB RESEARCH REPORT
Local-only, tiny-model (vocab ~3200, embed 75-D, hidden 128-D, CPU/NumPy GRU) techniques
for coherent concept-conditioned free-form generation, and the "word salad" failure mode.
Compiled 2026-07-15. Web research only. Plain text; bullet lists OK.

================================================================================
TL;DR  (recommendations ranked by likelihood of helping a TINY local model)
================================================================================

1. (HIGHEST) Switch the architecture from "decoder competes with the KB" to
   "decoder enriches a retrieved KB definition with bounded analogy/elaboration."
   i.e. Option C, not Option B. Retrieval guarantees the 0.5 coherence floor; the
   GRU only adds local variation. CPU-feasible, no extra training of open-ended
   generation. This is also the brain-faithful angular-gyrus model (see item 7).

2. (HIGH) Make the concept pointer a REPEATED, STRONG conditioning signal, not a
   one-shot injection. Inject av (projected) at EVERY token step via
   concatenation or FiLM/feature-wise modulation, plus an auxiliary
   concept-reconstruction loss so the latent is "load-bearing." Fixes the
   "thin conditioning is ignored" failure (see items 3, 4, 9). CPU-feasible.

3. (HIGH) Train with SCHEDULED SAMPLING (or free-running curriculum) to close the
   teacher-forcing -> free-run exposure-bias gap that produces word salad at
   inference. Pure NumPy/CPU, tiny cost. Often the single biggest coherence win
   for a small recurrent LM (see item 2).

4. (MEDIUM-HIGH) Use RETRIEVAL-AUGMENTED SCAFFOLDING: at generation time, pull the
   best (concept, description) pair(s) from the existing KB and either (a) seed the
   GRU with retrieved n-grams, (b) constrain the decoding vocabulary to retrieved
   words, or (c) do retrieve-and-lightly-edit. Forces lexical coherence. CPU-only.

5. (MEDIUM) Replace the plain ridge W_sm (av -> word-embedding) with a stronger
   binding: a small MLP "angular gyrus" (av -> 75-D embedding) and/or inject av
   directly as the decoder's initial/recurrent context instead of only via the
   projected embedding. Ridge alone is a weak, easily-bypassed signal (items 4, 9).

6. (LOWER for open-ended text, but brain-faithful) Add explicit role-filler /
   Tensor-Product / VSA binding so the concept pointer actually "steers" word
   choice rather than being a static bias. Useful as a representational upgrade
   but heavier; see item 6.

================================================================================
THE FAILURE MODE: teacher-forcing -> gibberish from a thin conditioning signal
================================================================================

DIAGNOSIS (standard).
- Exposure bias: model is trained on ground-truth prefixes (teacher forcing) but
  at inference must condition on its OWN previous tokens, so one early mistake
  snowballs into word salad. Classic, well-documented.
  Source: Exposure Bias overview, emergentmind.com, https://www.emergentmind.com/topics/exposure-bias
  Source: "Teacher Forcing" handbook, Brenndoerfer 2025,
          https://mbrenndoerfer.com/writing/teacher-forcing-seq2seq-training-exposure-bias-scheduled-sampling
- Thin/ignored conditioning: with a single projected concept vector injected only
  at the start, the GRU can learn to largely ignore it and fall back to the
  marginal training distribution (generic Gutenberg phrasing). A 2026 diagnostic
  ("Cosine Misleads", Zhang & Fang, https://arxiv.org/html/2606.05753v1 ) shows
  that auxiliary/conditioning objectives often reshape the LM through shared
  PARAMETERS rather than through the latent itself; if the latent is not made
  "load-bearing" (e.g. via an auxiliary reconstruction loss + per-step injection),
  the concept vector is bypassed at inference. Directly explains RAVANA's salad.

KNOWN FIXES (ranked for tiny models).
- Scheduled sampling (Bengio et al. 2015, "Scheduled Sampling for Sequence
  Prediction"; https://arxiv.org/abs/1506.03099 ). Gradually replace some
  ground-truth tokens with the model's own predictions during training. Reduces
  exposure bias. CPU-feasible for a 3200-vocab GRU (just sample from the output
  distribution during training). Variants: inverse-sigmoid schedule, dynamic
  schedule (DySI, Lin et al. 2023, https://arxiv.org/abs/2301.13753 ).
- Stronger conditioning / make latent load-bearing (see TL;DR 2,5).
- Free-running / professor forcing (curriculum that increases self-feedback).
- Retrieval scaffolding (see item 4 and TL;DR 1,4).
- Note: scheduled sampling has caveats (inconsistent objective, can ignore the
  prefix; Korakakis & Vlachos 2022, https://aclanthology.org/2022.findings-emnlp.536/ ).
  Pair it with stronger conditioning so the model cannot "ignore the prefix."

================================================================================
TECHNIQUE CATALOG (name / mechanism / why it helps / CPU-feasibility / pointer)
================================================================================

[1] SCHEDULED SAMPLING (exposure-bias cure)
  Mechanism: During training, with probability epsilon(t) feed the model's own
  previous prediction instead of the teacher token; anneal epsilon from 0 to high.
  Why it helps word salad: Aligns train/test distributions so a single early
  generation error does not cascade into incoherent text at free-run time.
  CPU-feasibility: Trivial. Sample argmax/top-k from softmax each step. No extra
  params. Ideal for 3200-vocab GRU.
  Pointer: Bengio et al. 2015, "Scheduled Sampling for Sequence Prediction",
           https://arxiv.org/abs/1506.03099
           Follow-up: Mihaylova & Martins 2019 (Transformers), https://aclanthology.org/P19-2049/
           Dynamic variant DySI: https://arxiv.org/abs/2301.13753

[2] STRONGER / REPEATED CONDITIONING (FiLM, concatenation, adaptive modulation)
  Mechanism: Instead of injecting the concept only at t=0, concatenate the
  projected av to the embedding at EVERY step, or apply feature-wise linear
  modulation (FiLM: gamma/beta from av) to GRU hidden states, or use a small
  cross-attention over the concept vector. Add an auxiliary loss that reconstructs
  av from the final hidden state so the concept is "load-bearing."
  Why it helps word salad: Prevents the decoder from ignoring the thin signal and
  reverting to the generic marginal distribution; keeps the concept steering every
  word choice.
  CPU-feasibility: Fully feasible. FiLM/cross-attention over a 75-D vector is
  cheap. Auxiliary MSE/CE on av adds one small linear head.
  Pointer: "Conditioning on Embeddings in Generative AI" (input/attention/
           parametric conditioning taxonomy), https://medium.com/@ding.zhongqiang/conditioning-on-embeddings-in-llms-4db3d3b06c4a
           FiLM: Perez et al. 2018, "FiLM: Visual Reasoning with a General
           Conditioning Layer", https://arxiv.org/abs/1709.07871
           Practical note on projection + auxiliary classification loss jumping
           adherence 34%->91%: https://www.dhanu.dev/blog/gan-research-methodology

[3] CONTROL CODES / DISCRETE CONDITIONING (tiny-LM friendly)
  Mechanism: Prepend or interleave a discrete "control token" (e.g. concept id,
  category, attribute bag) to the input sequence; the LM learns p(text | code).
  Why it helps word salad: A discrete, high-signal token is far harder to ignore
  than a soft projected vector; gives the GRU an explicit, stable anchor.
  CPU-feasibility: Trivial - just extra vocabulary entries / a learned code
  embedding fed at t=0 and/or each step. No transformer needed.
  Pointer: CTRL (Keskar et al. 2019, "A Conditional Transformer Language Model for
           Controllable Generation"), https://arxiv.org/abs/1909.05858  - the
           *idea* (control codes = domain/entity/relation tokens) ports down to a
           GRU; you do not need 1.6B params. Repo: https://github.com/salesforce/ctrl
           Survey of controllable generation: https://arxiv.org/abs/2408.12599

[4] CONCEPT -> WORD-EMBEDDING via RIDGE: IS IT TOO WEAK?
  Mechanism (RAVANA current): av (65-D Binder attributes) --ridge W_sm--> 75-D
  embedding, used as the decoder seed/init.
  Diagnosis: Ridge projection is a LINEAR, low-capacity map. The Binder 2016
  literature shows word embeddings CAN be mapped to/from 65-D brain-based semantic
  features with ridge regression (Decoding Word Embeddings with Brain-Based
  Semantic Features, https://direct.mit.edu/coli/article/47/3/663/102823/Decoding-Word-Embeddings-with-Brain-Based-Semantic ),
  but (a) the map is lossy and (b) a linear seed is a weak, easily-bypassed signal
  (cf. "Cosine Misleads" latent-bypass finding). It is a plausible initializer but
  not, by itself, a strong enough steering signal for coherent free-run text.
  Fix ideas: (a) learn a small MLP instead of ridge (more capacity); (b) inject the
  projected vector at every step, not just as init; (c) add an auxiliary loss that
  the decoder must predict av from its hidden states.
  CPU-feasibility: MLP + per-step injection fully CPU-feasible.
  Pointer: Binder et al. 2016 semantic features; decoding paper above.
           Concept-embedding generation for LLMs (CoLLEGe) shows example-buffer +
           negative sampling helps concept learning: https://arxiv.org/html/2403.15362v2

[5] CONDITIONAL LSTM/GRU LM: ENCODE CONCEPT VECTOR -> LATENT -> DECODE
  Mechanism: An encoder (small MLP or GRU) maps the concept attribute vector to a
  latent that initializes/conditions the decoder hidden state; decoder reconstructs
  the description. This is exactly the "vec-to-text" pattern.
  Why it helps word salad: Gives a principled, dense conditioning pathway and
  exposes the concept as a proper input distribution rather than a one-shot seed.
  CPU-feasibility: Highly feasible at this scale (a few hundred params encoder).
  Pointer: "Conditional LSTM Language Models" tutorial (Vec_To_Text_Dataset,
           molecular fingerprint -> SMILES; directly analogous concept-vector ->
           text), https://darkmatterai.github.io/mrl/tutorials.generative_models.conditional_lstm_lm.html
           Small GRU text generation repo: https://github.com/MuhammetSonmez/GRU-TEXT-GENERATION

[6] ROLE-FILLER BINDING: TENSOR PRODUCTS / VSA / ANGULAR-GYRUS MLP
  Mechanism: Bind concept (filler) to a role vector via outer product / circular
  convolution (HRR) / Hadamard, so the concept pointer is a structured,
  unbinding-capable representation rather than a flat bias. An "angular gyrus" MLP
  can be the local binding/unbinding operator between semantic (av) and lexical
  (word-embedding) spaces.
  Why it helps word salad: A bound representation is harder to ignore and supports
  compositional, concept-specific word retrieval; gives the decoder a steering
  signal with algebraic structure.
  CPU-feasibility: Fixed-dimension VSA ops (Hadamard/circular convolution) are
  cheap in NumPy; TPR outer product is 75x65 - tiny. Feasible, but adds design
  complexity; best as a representational upgrade to [2]/[4].
  Pointer: Smolensky 1990 Tensor Product Representations; McCoy et al. 2019
           "RNNs Implicitly Implement Tensor-Product Representations" (TPDN demo),
           https://rtmccoy.com/techviz/tprn/tpr_demo.html  and paper https://arxiv.org/abs/1902.09749
           Plate 1995 Holographic Reduced Representations; Eliasmith et al. 2012
           Semantic Pointer Architecture (Spaun), https://compneuro.uwaterloo.ca/research/spa/semantic-pointer-architecture.html
           VSA survey (Kleyko et al. 2022), https://dl.acm.org/doi/10.1145/3538531
           Gayler 2004 VSA, https://simondlevy.academic.wlu.edu/files/publications/agi_2008_levy_gayler.pdf

[7] BRAIN-FAITHFUL "RETRIEVAL + GENERATIVE ELABORATION" (angular gyrus model)
  Mechanism: The angular gyrus acts as a bidirectional interface / convergence zone
  that integrates retrieved semantic knowledge with generated elaboration, rather
  than generating from scratch. Concept knowledge = retrieved (ATL/hub) content,
  enriched by generative simulation/analogy. This is the basis for Option C.
  Why it helps word salad: Matches human cognition - we rarely produce coherent
  novel descriptions from a thin pointer alone; we retrieve a known structure and
  elaborately vary it. Retrieval supplies the coherence floor; generation adds
  value without bearing the full burden.
  CPU-feasibility: Fully local - the "retrieval" is a lookup over the existing KB
  (already in the system); the GRU only performs bounded analogy/paraphrase.
  Pointer: Binder et al. 2009/2011 neurobiology of semantic memory,
           https://pmc.ncbi.nlm.nih.gov/articles/PMC6601832/  (semantic network)
           Distinct roles of ATL vs angular gyrus (hub vs integration/buffering),
           https://academic.oup.com/cercor/article/32/20/4549/6517439
           Angular gyrus as interface between reading network and semantic system,
           https://link.springer.com/article/10.1007/s00429-023-02624-z
           "A Unifying Account of Angular Gyrus Contributions to Episodic and
           Semantic Cognition", Trends in Neurosciences 2021 (dynamic buffering of
           multisensory representations).

[8] RETRIEVAL-AUGMENTED GENERATION ("RAG") / N-GRAM SCAFFOLDING
  Mechanism: At generation, retrieve relevant (concept, description) passages from
  the KB; then either (a) seed the GRU with retrieved phrase fragments, (b) restrict
  the candidate vocabulary/beam to retrieved terms (constrained decoding), or
  (c) retrieve-and-edit (generate a light paraphrase of the retrieved text).
  Why it helps word salad: Guarantees lexical and factual grounding; the open-ended
  LM only fills low-risk gaps, so it cannot collapse into salad.
  CPU-feasibility: Ideal - retrieval is a cosine/keyword lookup over the local KB;
  no LLM, no internet. Common in small-LM + RAG deployments (see ACM SLM+RAG
  education paper, https://dl.acm.org/doi/10.1145/3641554.3701844 ).
  Pointer: Lewis et al. 2020 "Retrieval-Augmented Generation for Knowledge-
           Intensive NLP Tasks", https://arxiv.org/abs/2005.11401
           RAG survey: https://arxiv.org/abs/2504.14891  and https://arxiv.org/html/2506.00054v1
           Small-LM + RAG deployment guide: https://dl.acm.org/doi/10.1145/3641554.3701844

[9] AUXILIARY CONCEPT-RECONSTRUCTION LOSS (make the latent load-bearing)
  Mechanism: Add a loss term requiring the decoder's hidden state (or output
  distribution) to be able to reconstruct av (or the concept id). This forces the
  conditioning to actually flow through the network.
  Why it helps word salad: Directly counters the "Cosine Misleads" bypass failure;
  without it a weak projected vector is silently dropped by the GRU.
  CPU-feasibility: One small linear layer + MSE/CE. Fully feasible.
  Pointer: "Cosine Misleads: Auxiliary Losses Reshape VLMs, Not Their Latents"
           (Zhang & Fang 2026), https://arxiv.org/html/2606.05753v1  - key
           diagnostic: supervise the latent AND verify it is load-bearing.
           GAN conditioning + auxiliary classification loss (34%->91% adherence):
           https://www.dhanu.dev/blog/gan-research-methodology

[10] ATTRIBUTE-DRIVEN / PER-STEP ATTRIBUTE FILTERING (from captioning)
  Mechanism: In image captioning, feeding ALL attributes at every step hurts;
  filtering the relevant attributes per timestep improves coherence. Analog: feed
  only the currently-relevant subset of av at each generation step.
  Why it helps word salad: Avoids overloading the GRU with irrelevant dimensions
  that act as noise; sharper, concept-relevant signal per word.
  CPU-feasibility: Feasible (gating/attention over 65 attributes).
  Pointer: Attribute-Driven Filtering (ADF) captioning, Engineering Applications of
           AI 2024, https://www.scienceddirect.com/science/article/pii/S0952197624012922
           Attribute-Guided Fusion captioning, https://link.springer.com/article/10.1007/s11042-024-19410-6

[11] CONCEPT-CONDITIONED CAPTIONING AT SMALL SCALE (direct analog)
  Mechanism: CNN/attribute encoder -> GRU decoder conditioned on the encoded
  concept/attributes; standard recipe achieving coherent captions on Flickr/COCO
  with a GRU (fewer params than LSTM, better on small data per Chung et al.).
  Why it helps word salad: Shows that a small GRU CAN produce coherent conditioned
  text when (a) conditioning is dense and per-step (attention) and (b) training
  corpus is (image-attributes, caption) pairs. Same recipe applies to
  (av-attributes, description) pairs.
  CPU-feasibility: Directly feasible; this is the exact scale.
  Pointer: GRU-based attention captioning (Khan et al. 2023),
           https://arxiv.org/abs/2310.07252  ; attention-GRU caption framework
           https://arxiv.org/pdf/2008.01663  ; "Describing Image with Attention
           based GRU" https://ieeexplore.ieee.org/document/9418171
           Chung et al. 2014 "Empirical Evaluation of GRU vs LSTM" (GRU better on
           small data): https://arxiv.org/abs/1412.3555

[12] (NOT for tiny models, listed for completeness) CONCEPT ACTIVATION VECTORS
  Mechanism: Steer an LLM by adding a concept vector to activations (GCAV). Powerful
  but requires a large pretrained model + gradient access; NOT CPU/NumPy-local.
  Pointer: Zhang et al. 2025 "Controlling LLMs Through Concept Activation Vectors",
           https://arxiv.org/html/2501.05764v1

================================================================================
DATASET / OBJECTIVE CHOICES FOR "CONCEPT -> COHERENT SENTENCE"
================================================================================

- (concept, description) PAIRS are the right corpus. Harvest (av-attributes,
  sentence) pairs, not just free Gutenberg sentences. The captioning literature
  (item 11) shows pair supervision is what makes a small GRU coherent. Your current
  "harvested Gutenberg descriptions" is on the right track IF each description is
  paired with its concept's av and the concept is conditioned.
- KB-AS-TEACHER / DISTILLATION: Use the curated KB definitions as high-quality
  targets the GRU is trained to reproduce (given av). This injects the 0.5-floor
  coherence into the training signal. Pair with scheduled sampling (item 1).
- CONTRASTIVE / RANKING objective: train the decoder so the correct concept's
  description scores higher than a mismatched one (similar to image-text contrastive
  but concept-text). Cheap, stabilizes the av->text map.
- PARAPHRASE objective: train the GRU to paraphrase a retrieved good definition
  (input: retrieved definition + av; output: varied wording). This is the Option-C
  training regime and is far easier than open-ended generation.
- AVOID pure language-modeling on harvested text WITHOUT concept conditioning -
  that is what produces a model that ignores av and emits generic salad.
- Curriculum: start training teacher-forced on (av, retrieved-KB-definition) pairs,
  then anneal to scheduled sampling, then optionally fine-tune on free paraphrases.

================================================================================
RANKING: OPTION B (decoder competes with KB) vs OPTION C (decoder enriches KB)
================================================================================

VERDICT: Option C is substantially more likely to yield coherent output from a tiny
local model. Recommendation: pursue Option C; keep Option B disabled (fail-closed)
until/unless items 1-3 above are implemented and measured.

WHY OPTION C wins for a 3200-vocab CPU GRU:
1. Coherence is guaranteed by retrieval. The curated KB already supplies good
   definitions (the 0.5 floor). The GRU only performs BOUNDED elaboration /
   analogy / paraphrase on top of retrieved text, so even if the GRU is imperfect,
   the output stays anchored to a coherent source. Option B asks the GRU to supply
   ALL coherence from a thin pointer - exactly the regime that produced the 0.23
   salad.
2. It matches the brain-faithful angular-gyrus model (item 7): retrieved semantic
   knowledge enriched by generative simulation, not generation-from-scratch.
3. Training is easier and more stable: a paraphrase/elaborate objective (item 12)
   is far simpler than open-ended conditional generation; it needs less data and is
   robust to the exposure-bias problem because the "hard part" (facts, structure)
   comes from retrieval.
4. CPU/NumPy feasibility: Option C needs no large model and no internet; retrieval
   is a local lookup, generation is a small GRU doing light edits. Option B would
   require a much larger/better-trained decoder to beat a curated KB - unrealistic
   at this scale.
5. Option B's only upside is novelty/variation for concepts the KB lacks - but for
   those, Option C can still retrieve the NEAREST concept's definition and
   analogize, preserving coherence. So Option C dominates even on coverage.

When Option B could be revisited: only after (a) scheduled sampling, (b) repeated
strong conditioning + auxiliary concept loss, and (c) (concept,description) pair
training are in place AND free-form coherence clears ~0.5 on held-out concepts.
Until then, enabling B regresses coherence - the current fail-closed choice is
correct.

================================================================================
KEY SOURCES (URLs)
================================================================================

- Exposure bias / scheduled sampling:
  https://www.emergentmind.com/topics/exposure-bias
  https://arxiv.org/abs/1506.03099  (Bengio 2015 Scheduled Sampling)
  https://arxiv.org/abs/2301.13753  (DySI dynamic scheduled sampling)
  https://mbrenndoerfer.com/writing/teacher-forcing-seq2seq-training-exposure-bias-scheduled-sampling
- Conditioning / control codes / FiLM:
  https://medium.com/@ding.zhongqiang/conditioning-on-embeddings-in-llms-4db3d3b06c4a
  https://arxiv.org/abs/1709.07871  (FiLM)
  https://arxiv.org/abs/1909.05858  (CTRL)
  https://arxiv.org/abs/2408.12599  (CTG survey)
- Latent-bypass diagnostic (why weak conditioning fails):
  https://arxiv.org/html/2606.05753v1  (Cosine Misleads)
- Conditional LSTM/GRU vec-to-text:
  https://darkmatterai.github.io/mrl/tutorials.generative_models.conditional_lstm_lm.html
  https://github.com/MuhammetSonmez/GRU-TEXT-GENERATION
- Ridge / concept -> embedding / Binder features:
  https://direct.mit.edu/coli/article/47/3/663/102823/Decoding-Word-Embeddings-with-Brain-Based-Semantic
  https://arxiv.org/html/2403.15362v2  (CoLLEGe concept embedding)
- Role-filler / TPR / VSA / semantic pointers:
  https://rtmccoy.com/techviz/tprn/tpr_demo.html  (+ https://arxiv.org/abs/1902.09749 )
  https://compneuro.uwaterloo.ca/research/spa/semantic-pointer-architecture.html
  https://dl.acm.org/doi/10.1145/3538531  (VSA survey)
  https://simondlevy.academic.wlu.edu/files/publications/agi_2008_levy_gayler.pdf
- Angular gyrus / retrieval + elaboration (brain model):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC6601832/  (semantic network)
  https://academic.oup.com/cercor/article/32/20/4549/6517439  (ATL vs AG roles)
  https://link.springer.com/article/10.1007/s00429-023-02624-z  (AG interface)
- Retrieval-augmented generation / scaffolding:
  https://arxiv.org/abs/2005.11401  (RAG)
  https://arxiv.org/abs/2504.14891  & https://arxiv.org/html/2506.00054v1  (surveys)
  https://dl.acm.org/doi/10.1145/3641554.3701844  (Small-LM + RAG deployment)
- Concept-conditioned captioning at small scale (direct analog):
  https://arxiv.org/abs/2310.07252  (GRU attention captioning)
  https://arxiv.org/pdf/2008.01663  (attention-GRU caption framework)
  https://arxiv.org/abs/1412.3555  (GRU vs LSTM, small-data)
  https://www.sciencedirect.com/science/article/pii/S0952197624012922  (attribute filtering)
- Concept Activation Vectors (LLM-only, not for tiny models):
  https://arxiv.org/html/2501.05764v1

================================================================================
BOTTOM LINE FOR RAVANA
================================================================================

The word salad is the textbook exposure-bias + thin-ignored-conditioning failure of
a teacher-forced small GRU. The cheapest, highest-impact local fixes are: (1) adopt
Option C (retrieve KB definition, let the GRU do bounded analogy/elaboration),
(2) scheduled sampling, (3) make av a repeated, load-bearing conditioner with an
auxiliary reconstruction loss, and (4) replace the lone ridge seed with a small MLP
"angular gyrus" plus per-step injection. Keep Option B disabled until free-form
coherence clears the 0.5 floor under those changes.
