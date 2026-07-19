# RAVANA Brain-Faithful Fix Plan (Phase 2)
## 16 failure modes, mapped to neuroscience + code root cause + approach

**Date:** 2026-07-17  
**Status:** Research & Plan (not yet implemented)  
**Principle:** Every fix must be grounded in a real brain mechanism. No hardcoded strings, no template lists, no manual category tables.  

---

## PART A — FUNCTIONAL BUGS (behavior is broken)

---

### [A1] HIGH — Self-disclosure statements misroute to the category-error gate

**Example:** `"my favorite color is purple"` → `"i'd really picture Purple in terms of its presence..."`

#### Brain analysis

The human brain does NOT route self-disclosure statements through the same channel as category errors. The neuroanatomical dissociation is clear:

| Process | Primary Regions | Function |
|---------|----------------|----------|
| **Semantic meaning** | MTG, IFG | Retrieve meaning of words ("color", "purple", "Tuesday") |
| **Self-disclosure** | vmPFC, dACC | Retrieve/evaluate stored self-knowledge ("my favorite") |
| **Category error** | dACC, IFG | Detect predicate-subject incompatibility ("color of Tuesday") |

In humans, `"my favorite color is purple"` activates self-referential networks (vmPFC → autobiographical retrieval), while `"what color is Tuesday"` triggers conflict detection (dACC → IFG → suppression). They are **orthogonal neural pathways**. The brain never confuses one for the other because:
- The first-person possessive "my" gates processing into the self-referential stream (Suzuki 2022, PLOS Biology)
- The copula+property structure (is purple) is NOT checked for category compatibility because the subject is already grounded to the self, not to a semantic category

#### Root cause in code

1. **Statement form is unhandled by the identity block** (engine.py:3519-3535): The identity block only matches the QUESTION form `"what is my favorite X"`. The STATEMENT form `"my favorite X is Y"` matches NO regex in the identity set, so it falls through to `_is_category_error`.

2. `_is_category_error` sees `"color"` as a physical property, grabs head noun `"purple"`, and emits the cross-modal metaphor. The gate has no way to know this is a first-person statement because it checks predicate compatibility, not discourse frame.

#### Approach (brain-faithful)

**The fix is to add a vmPFC-mimetic "self-referential gating" layer BEFORE the frontopolar (BA 10) feasibility gate:**

1. **Add a `_is_self_disclosure(user_input) -> bool` gate** that detects first-person possessive frames (`my X`, `I have`, `my name is`, `I am called`) BEFORE the category-error check. This mirrors the human brain's vmPFC-based routing: self-referential processing is orthogonal to semantic-feasibility checking.

2. **Route self-disclosures to a dedicated `_process_self_disclosure()` path** that:
   - Stores the disclosed fact in `UserModel.preferences["favorites"]` / `UserModel.user_name` (already exists in `UserModel.observe_user_query()` — the code IS there, it's just never called from `process_turn`)
   - Acknowledges the disclosure naturally (gist-based: "got it, i'll remember your favorite color is purple")
   - Sets `self._episodic_miss = True` so no web/graph fallback fires

3. **Wire `self.user_model.update()` (or fix the existing `observe_user_query`) into the `process_turn` pipeline** so the `UserModel` actually receives updates. Currently `observer_user_query()` IS called in `process_turn` (line 4021), BUT the `m_fav` regex in that function already correctly matches `"my favorite X is Y"` — the bug is that `_is_category_error` fires BEFORE that code path, so the self-disclosure never reaches the observation pipeline.

**The ordering fix is key:** The vmPFC (self-disclosure gate) must fire BEFORE the frontopolar (category-error) gate, because in humans, self-referential processing pre-empts semantic-feasibility checking. This is not a new rule — it's the observed neuroanatomical hierarchy.

**Neuroscience citations:**
- Suzuki (2022): "Inferences regarding oneself and others in the human brain" — PLOS Biology. Functional dissociation within mPFC: dACC for metacognition, dmPFC for others.
- D'Argembeau (2013): vmPFC valuation hypothesis — self-relevant info is tagged with higher subjective value.

---

### [A1b] HIGH — Favorite storage is inconsistent / partially dead code

**Example:** `"my favorite movie is dune"` gets confused reply, yet `"what's my favorite movie"` later answers correctly.

#### Brain analysis

The human brain uses a **single episodic-semantic binding** for self-disclosed facts (Yonelinas et al. 2019, contextual binding theory). The hippocampus binds:
- The entity (movie)
- The value (Dune)
- The context (I said this)
- The source (it came from me, not the web)

This binding is then retrievable via ANY of those cues — pattern completion in CA3 (Yassa & Stark 2011). The brain doesn't have a "statement path" and a "question path" that diverge.

#### Root cause in code

`UserModel.observe_user_query()` (user_model.py:73-117) correctly parses `"my favorite X is Y"` via regex and stores it in `self.preferences["favorites"]`. The question path `m_fav_q` (engine.py:3519, 3578) reads from `prefs["favorites"]`. **BUT** between storing and reading, the `_is_category_error` gate or the `_record_episode` side-channel can fire, creating a race condition where:
- Sometimes the fact is stored in `UserModel.preferences` (correct)  
- Sometimes it's only in `_episodic_transcript` (not queryable by question path)
- Sometimes neither (category error path swallows it)

#### Approach (brain-faithful)

**Create a unified `_self_disclosure_store` that mirrors the hippocampal binding function:**

1. **Ensure the vmPFC gate (A1 fix) catches ALL self-disclosure statements BEFORE any other path can consume them** — this guarantees the fact reaches `UserModel.observe_user_query()` and `UserModel.preferences["favorites"]`.

2. **Add a check: before the question path reads `prefs["favorites"]`, check `_episodic_transcript` as well** — this creates the pattern-completion analog: if one store is empty, try the other. The two stores are the "hippocampal" (episodic transcript) and "neocortical" (UserModel preferences).

3. **Unify the episodic miner and the UserModel pref store**: `_mine_episodic_facts` and `observe_user_query` both mine the same patterns but store in different locations. Make them write to a shared `_self_disclosure` dict that BOTH paths read from.

**Neuroscience citations:**
- Yonelinas et al. (2019): A contextual binding theory of episodic memory — Nature Reviews Neuroscience
- Yassa & Stark (2011): Pattern separation in the hippocampus — Trends in Neurosciences

---

### [A2] HIGH — False empathy on creative / request frames

**Example:** `"tell me a story about a lonely robot who learns to dream"` → `"i hear you — feeling lonely is hard..."`

#### Brain analysis

The human brain distinguishes between narrative requests and emotional disclosures through the **mentalizing network's ability to tag fictional agents** (Koster-Hale et al. 2017):

| Request type | Neural response | Key region |
|--------------|----------------|------------|
| `"I am sad"` | mPFC activates affective ToM | mPFC (self-referential) |
| `"tell me a story about a sad robot"` | TPJ tracks fictional agent boundary | TPJ (cognitive ToM) |

The TPJ maintains a **meta-representational boundary**: "the sadness belongs to the robot in the story, not to the speaker" (Qiao-Tasserit et al. 2024). When the brain detects a narrative framing verb ("tell me a story", "describe", "teach me about"), it routes the following content into **narrative simulation** rather than **empathic resonance**.

Additionally, pragmatic framing (Flusberg et al. 2024) provides strong contextual cues: "tell me" + "story" + "about" together form a pragmatic frame that signals creative generation, not emotional disclosure.

#### Root cause in code

`_detect_emotional_disclosure` (response_gen.py:3349-3399) only requires:
1. A first-person word anywhere (line 3399: `\b(i|i'm|i am|my|me|we|...)\b`)
2. An affective word anywhere

`"tell me a story about a lonely robot"` has `"me"` (first-person) + `"lonely"` (negative affect) → fires a false positive. The simple simile-exclusion at line 3401 (`\blike (?:i am|i'm|i)\s+(?:a |an )?\w+\b`) does NOT catch request frames.

#### Approach (brain-faithful)

**Add a TPJ-mimetic "pragmatic frame detection" gate BEFORE the emotional disclosure check:**

1. **Detect narrative/request framing verbs**: Before checking for emotional disclosure, scan for verbs that signal **other-directed narrative generation**: `tell me (about|a story|what)`, `write (me|a)`, `imagine`, `describe`, `explain`, `teach me about`, `make up`, `create`. These verbs activate the TPJ-to-DMN narrative simulation pathway.

2. **Check the pragmatic frame**: If the utterance is a request for creative generation (verb + requested artifact), tag ALL following content as **narrative simulation** — the affective words describe the character/concept, not the user.

3. **Implement a simple heuristic**: If the first clause contains a request frame (imperative + creative verb), AND the emotional words appear in the object clause (after "about"), then it's a narrative request, not a disclosure. This mirrors the human brain's pragmatic parsing hierarchy.

**For the specific "request frame" pattern:**
```
\b(tell|write|create|make|imagine|describe|teach)\b.*\b(me|us)\b.*\b(about|a|an)\b
```
→ route to creative generation, not empathy.

4. **The bereavement/loss check** (line 3392-3398) must ALSO be gated: `"tell me a story about someone who died"` is not a personal loss disclosure. Check that the loss word is in the same clause as a first-person possessive (`my dog`, `my grandma`) — not in a narrative object clause (`a story about a dog`).

**Neuroscience citations:**
- Koster-Hale et al. (2017): Mentalizing regions represent distributed, continuous and abstract dimensions of others' beliefs — PMC5696012
- Qiao-Tasserit et al. (2024): Influence of transient emotional episodes on affective and cognitive theory of mind — PMC10914405
- Flusberg et al. (2024): The Psychology of Framing — DOI: 10.1177/15291006241246966

---

### [A3] MED — Plan C episodic memory is half-broken

**Example:** `"my cat's name is whiskers"` → `_mine_episodic_facts` returns `{}`  
**Example:** After storing `[cat/whiskers, book/dune]`, `"what did i tell you about my book"` returns `"you mentioned: my cat's name is whiskers"`

#### Brain analysis

The human hippocampus performs **relational binding** with high specificity (Hannula et al. 2008). Pattern separation in the dentate gyrus ensures that `cat→name→whiskers` and `book→title→dune` are stored as **orthogonal representations** — they never cross-contaminate during retrieval (Yassa & Stark 2011).

The contextual binding theory (Yonelinas et al. 2019) explains why the brain correctly retrieves the right entity: recall is driven by a **pattern completion cue** that includes both the entity AND the relation. When asked `"what did I tell you about my book"`, the brain uses `book` as the retrieval cue, activating ONLY the book→dune binding, not the cat→whiskers one.

Additionally, failure of binding (when the brain genuinely doesn't know) results in a **failure to retrieve** — the subject simply says "I don't remember telling you about that" — never "you mentioned [different entity]."

#### Root cause in code

Three sub-bugs:

1. **Miner too narrow** (engine.py:1727-1747): `_mine_episodic_facts` only matches `"my favorite X is Y"` and `"i love/like X"`. It misses possessive disclosures like `"my cat's name is whiskers"`, `"my dog's age is 3"`, `"my sister is maya"`. Human episodic memory encodes ANY relational statement, not just favorite/like slots.

2. **Wrong-episode recall** (engine.py:1880-1895): The bare-recall branch builds bits from ALL prior episodes, but when multiple facts are stored, it returns a joined string of ALL facts — it loses which fact belongs to which entity. The `_retrieve_episodic` method then does semantic matching but can retrieve the wrong episode because the query `"my book"` semantically matches any stored text that mentions "book".

3. **Disconnected stores** (engine.py:3578 vs _episodic_transcript): The question path reads `preferences["favorites"]` from `UserModel`, but the episodic miner stores in `_episodic_transcript`. No cross-connection.

#### Approach (brain-faithful)

**Redesign the episodic memory as a proper hippocampal-indexed store:**

1. **Expand `_mine_episodic_facts` to a general-purpose `_extract_relational_triples`** that captures ANY `subject → relation → object` triple from user disclosures:
   - `my X's Y is Z` → `(X, Y, Z)` e.g. `(cat, name, whiskers)`
   - `my X is Y` → `(X, category, Y)` e.g. `(sister, is, maya)`
   - `I have a X named Y` → `(X, name, Y)`
   - `X is my Y` → `(X, is-a, Y)`
   Use a deterministic pattern set (not LLM) derived from English possessive/copula grammar.

2. **Store triples in a hippocampal-style index** (`Dict[str, List[Dict]]`) keyed by the main entity. When the user says "my cat's name is whiskers", store under key `"cat"`:
   ```python
   self._episodic_index["cat"] = {"name": "whiskers", "relation": "possessive"}
   ```

3. **Query the index by entity** in `_retrieve_episodic`: when the user asks `"what did I tell you about my book"`, extract `"book"` (the entity after "about my"), look up `self._episodic_index.get("book")`, and return ONLY that entity's facts. No cross-contamination.

4. **Unify stores**: have BOTH `UserModel` preferences AND `_episodic_index` point to the same underlying dict, so the question path and the recall path read the same data.

**Neuroscience citations:**
- Hannula et al. (2008): Medial Temporal Lobe Activity Predicts Successful Relational Memory — J. Neuroscience
- Yassa & Stark (2011): Pattern separation in the hippocampus — Trends in Neurosciences
- Yonelinas et al. (2019): A contextual binding theory of episodic memory — Nature Reviews Neuroscience

---

### [A4] LOW — "how are you today" degrades into a random association

**Example:** `"how are you today"` → `"Also, they broadly are connected with day. Also, they seemingly shape rather."`

#### Brain analysis

In humans, wellbeing queries ("how are you", "what's up") trigger a **social script** in the mentalizing network (mPFC, TPJ, precuneus) that retrieves a stock social response — a formulistic but appropriate greeting ritual (Levinson 1983, pragmatics). The brain has dedicated **pragmatic frames** that route these queries to social-response generators, not to the semantic-feasibility or association network.

The left inferior frontal gyrus (LIFG) acts as a pragmatic frame selector (Yoshioka et al. 2023): it checks if the utterance matches a stored discourse script (greeting, farewell, wellbeing) and routes accordingly. If the LIFG detects a greeting frame, it suppresses the semantic retrieval path.

#### Root cause in code

The wellbeing check (in `_generate_response`, response_gen.py:4548) checks for `wellbeing` question type via `pfc_workspace.detect_question_type()`. If that classifier fails or returns a different type, the query falls through to the association/decomposition path, which extracts "today" as the subject and produces a noisy definition_with_assoc over "day."

#### Approach (brain-faithful)

**Strengthen the PFC discourse classifier's wellbeing detection to be MORE sensitive rather than less:**

1. **Expand the wellbeing query patterns** to include ANY combination of `how + (are/is/do) + (you) + (today/doing/feeling/going)` — capture the full pragmatic space, not just exact matches. Use a fuzzy pattern that requires at minimum `how` + 2nd-person pronoun + any state verb.

2. **Route wellbeing to a dedicated social-response generator** that draws from the engine's VAD state (`self.emotion.state`) to compose a grounded reply: `"i'm feeling {valence_description} today — thanks for asking"` where the description is derived from the current valence/arousal values (not hardcoded). This is brain-faithful: human wellbeing responses are grounded in INTEROCEPTIVE STATE (Seth 2013/2021), not a template.

3. **Fail-closed to `"i'm here — what's on your mind?"`** if the VAD-derived description fails.

**Neuroscience citations:**
- Yoshioka et al. (2023): The Role of the LIFG in Introspection during Verbal Communication — Brain Sciences
- Seth (2013): Interoceptive inference, emotion, and the embodied self — Trends in Cognitive Sciences

---

### [A5] LOW — Gravity counterfactual is thin

**Example:** `"imagine if gravity suddenly stopped working"` → `"gravity would lead to earth; gravity would lead to orbit."`

#### Brain analysis

The human brain's counterfactual simulation follows the **nearest possible world** constraint (Van Hoeck et al. 2015): we prefer simulations that require minimal changes to reality and produce maximal cascade effects. The "tree falling in forest" works because it's a minimal change (just remove the observer) with maximal philosophical cascade. The "gravity stops" scenario SHOULD produce a rich cascade (things float, atmosphere escapes, earth breaks apart), but only if the brain has access to a **sufficiently detailed causal model**.

The DMN (default mode network) and hippocampus work together to simulate counterfactuals by **retrieving relevant causal schemas** and running a generative simulation (Schacter et al. 2012, episodic future thinking). The richness of the simulation depends on the DENSITY of causal knowledge about the intervened concept.

The 1883/1884 Berkeley thought experiment feels richer because it addresses a **fundamental philosophical ambiguity** (objectivity vs subjectivity) that triggers the reward circuitry (nucleus accumbens) upon insight. Gravity stopping is more "factual" than "philosophical" — it doesn't trigger the same reward response (Moreno-Rodriguez et al. 2025).

#### Root cause in code

The gravity counterfactual hits `_causal_forward_simulate("gravity")` which returns only gravity's 1-hop graph neighbors (earth, orbit). The seeded physics causal skeleton (engine.py:2156-2180) has gravity→earth but not the cascade chain (gravity→no orbit→no atmosphere→no life→...). The causal graph is too sparse.

The tree counterfactual works because it was SPECIALLY seeded (engine.py:2166-2180: tree→fall→vibrate→air→sound→hear). Gravity was NOT similarly seeded.

#### Approach (brain-faithful)

**The fix is NOT to seed more specific edges (that's hardcoding). Instead:**

1. **Implement a "causal graph density detector"**: Before answering a counterfactual, check how many causal edges the subject has. If fewer than 3, the graph is too sparse to simulate — route to web search for that specific what-if scenario (reuse `_abductive_counterfactual` which already does web lookup for novel subjects).

2. **Use the ImplicatureDetector / abductive reasoning to fill gaps**: `_abductive_counterfactual` (response_gen.py:3849) already handles this for novel subjects — it does a web search for the counterfactual premise. The fix is to ROUTE all counterfactuals through that path FIRST, and only use graph simulation when the web path returns nothing useful.

3. **Seed a generic physics cascade skeleton** (not per-concept): Add a few generic physical-law edges like `force→object→motion`, `motion→change`, `absence→effect` that the forward-simulator can use for ANY physical counterfactual. These are domain-level, not concept-level — they represent the brain's "intuitive physics" knowledge (Spelke & Kinzler 2007).

**Neuroscience citations:**
- Van Hoeck et al. (2015): Cognitive neuroscience of human counterfactual reasoning — PMC4511878
- Schacter et al. (2012): The future of memory: remembering, imagining, and the brain — Neuron
- Moreno-Rodriguez et al. (2025): The human reward system encodes the subjective value of ideas — Nature Comms
- Roese & Epstude (2017): The Functional Theory of Counterfactual Thinking — APA PsycNet

---

### [A6] MED — Causal realizer emits incoherent self-loops

**Example:** `"what causes anxiety"` → `"Anxiety leads to reduce, and that in turn leads to cause."`

#### Brain analysis

The human brain's causal reasoning system (vLPFC, ACC, anterior insula) has a built-in **coherence filter** that suppresses tautological or self-looping causal chains before they reach articulation. This is NOT a dedicated "nonsense gate" — rather, it's an emergent property of predictive processing:

1. **Prediction error minimization**: Tautological chains carry zero predictive value (the output predicts nothing new about the world). The brain naturally deprioritizes them.
2. **Energy efficiency**: Maintaining a recursive self-loop costs neural energy without producing useful output. The brain's tendency to settle into stable energy minima means circular reasoning decays spontaneously.
3. **Semantic network violation detection**: When "anxiety" (an emotion) is causally linked to "reduce" (an abstract verb), the lateral temporal cortex detects the violation of expected semantic properties and suppresses the output.

Critically, the brain's causal reasoning prioritizes **intervention** over **association** (Operskalski & Barbey 2016): a valid causal claim must tell you what would happen if you changed the cause. "Anxiety leads to reduce" fails this test — you can't intervene on "anxiety" and observe "reduce".

#### Root cause in code

`_causal_forward_simulate` (chain_walker.py:1911) walks the causal graph and returns discovered chains, but:
1. It doesn't check if the chain endpoints are **semantically coherent** — it returns whatever the graph has, even if edges are noisy GloVe associations.
2. The `_realize_chain` step (in `_generate_response`) doesn't filter for **causal coherence** — it just strings together whatever was found.
3. The quality gate (`_is_word_salad`, `_coherence_ok`) checks for repetition but NOT for **semantic plausibility** of causal chains.

#### Approach (brain-faithful)

**Add a 3-step causal coherence filter (mirroring the vLPFC→ACC→AI hierarchy):**

1. **Step 1 — Endpoint novelty check (ACC analog)**: After `_causal_forward_simulate` returns chains, check if the terminal word is a semantically novel concept (not "cause", "reason", "effect", "result", "thing", "way"). If the chain ends in a generic placeholder, reject it. These are semantically empty endpoints that produce tautology.

2. **Step 2 — Semantic plausibility check (vLPFC analog)**: For each link in the chain, compute the semantic similarity (GloVe cosine) between adjacent concepts. If similarity is too LOW (<0.3) or too HIGH (>0.95), the link is likely noise or tautology. Reject chains with multiple implausible links.

3. **Step 3 — Intervention test (vLPFC analog)**: Can the subject be meaningfully intervened upon? `"Anxiety leads to reduce"` fails because "anxiety" (a state) cannot produce "reduce" (a change in quantity). Use POS tags + semantic categories to detect category-mismatch in causal chains.

4. **Fail-closed**: If the filtered chain has no coherent path, fall back to `_hedged_candidate_for` or honest uncertainty — don't emit a degenerate chain.

**Neuroscience citations:**
- Operskalski & Barbey (2016): Cognitive Neuroscience of Causal Reasoning — Oxford Handbook of Causal Reasoning
- Fincham & Anderson (2006): Distinct roles of ACC and PFC in cognitive skill — PNAS

---

### [A7] MED — Well-known factual questions fail-closed without even trying web

**Example:** `"who invented the lightbulb"` → `"honestly, lightbulb is a bit outside what i know right now."`

#### Brain analysis

The human brain has a **feeling-of-knowing (FOK)** metacognitive signal that operates BEFORE retrieval (Koriat 1993; Metcalfe 2000). When asked "who invented the lightbulb", the brain rapidly assesses:
1. Is this a well-formed entity question? (yes — "who X" targets a person)
2. Do I have ANY information about this entity? (yes — "lightbulb" is familiar)
3. Can I retrieve the specific attribute? (maybe — that's what retrieval attempts)

The FOK signal is generated in the **medial temporal lobe** (hippocampus + perirhinal cortex) and modulates the **prefrontal cortex's** decision to attempt retrieval vs. abstain. Crucially, the FOK for familiar concepts is NOT zero — it's intermediate, which triggers retrieval effort, not abstention.

#### Root cause in code

The FOK pre-check (engine.py, `_fok_pause_done` flag) is designed to gate web search based on knowledge confidence. The bug is that the FOK gate is **TOO CONSERVATIVE** — it fires "not known" on familiar entities because:
1. The knowledge model checks `_definitions` (clean KB) but NOT the graph (associations, edges)
2. If the entity has no definition but DOES exist in the graph with edges, the FOK gate should still fire "known enough to try"

Contrast `"what is oxiverse"` (seeded, hits instant graph answer) vs `"who invented the lightbulb"` (NOT seeded, but SHOULD hit web). The difference is: "oxiverse" is a seeded domain concept with typed edges; "lightbulb" is a common noun that exists in the graph but with no definition.

#### Approach (brain-faithful)

**Implement a 2-stage FOK (mirroring the hippocampus→PFC metacognitive signal):**

1. **Stage 1 — Entity familiarity check**: Before checking for a definition, check if the entity exists in the graph at ALL (any node, any edges). The hippocampus's familiarity signal (perirhinal cortex) fires on node presence, not definitional completeness.

2. **Stage 2 — Attribute specificity check**: If the entity is familiar (exists in graph), check what attribute is being asked about (inventor, color, definition, etc.). If it's a well-formed factual query about a familiar entity, **always attempt web search** — don't abstain.

3. **The FOK gate should ONLY abstain when**:
   - The entity doesn't exist in the graph AT ALL (zero familiarity)
   - OR the query is clearly unanswerable (category error, paradox)
   - OR web search has already been attempted this turn (rate limit)

4. **Ordering**: Entity familiarity check → FOK gate → web search → uncertainty. Currently the order is: FOK gate → (sometimes) web → uncertainty. The familiarity check must come FIRST.

**Neuroscience citations:**
- Koriat (1993): How do we know that we know? The accessibility model of the feeling of knowing — Psychological Review
- Metcalfe (2000): Metamemory: Theory and data — Oxford Handbook of Memory

---

## PART B — MED/LOW QUALITY BUGS (behavior works but is human-unlike)

---

### [B2] MED — Repetitive punt closers

**Example:** Across unknowns it cycles `"what's your take on it?"` / `"what do you make of it?"` / `"what's your sense of it?"` / `"i'd love to hear your take first."`

#### Brain analysis

Humans vary their uncertainty expressions because the **metacognitive state** (how uncertain you are, how you feel about being uncertain, what you think the OTHER person wants) modulates the expression. The teen brain's developing PFC (Arain et al. 2013) makes teens MORE likely to vary their uncertainty markers because they're exploring social strategies.

The key insight (Fleming et al., Phil Trans R Soc B): metacognitive confidence is a **continuous signal**, not a discrete state. When confidence is low, humans express uncertainty through varied linguistic markers that reflect the DEGREE and TYPE of uncertainty:
- Low confidence + social desire to engage → "what do you think?" (invitation)
- Low confidence + epistemic humility → "I'm not sure, but..." (hedge)
- Medium confidence + curiosity → "I wonder if..." (speculation)

#### Root cause in code

The uncertainty generator (engine.py, `_human_like_uncertainty`) uses a fixed set of 4-5 template strings rotated randomly. There's no modulation by:
- Current VAD state (valence/arousal/dominance)
- Engagement level
- Conversation depth
- Whether the user asked a question vs. made a statement

#### Approach (brain-faithful)

**Replace the fixed-string generator with a composed one that varies by metacognitive state:**

1. **Map VAD + engagement to uncertainty expression type**:
   - Low valence + any arousal → soft sympathy + uncertainty (`"hmm, i'm not sure — but i'd love to hear what you think"`)
   - High arousal + uncertainty → exploratory (`"that's a great question — i don't have a solid answer, but i wonder..."`)
   - Low arousal + familiarity → collaborative (`"i don't know much about that — what's your take?"`)
   - High engagement → reciprocal (`"i'm learning about this — help me understand your perspective?"`)

2. **Compose from components, NOT templates**: Have a set of uncertainty INTENTS (hedge, invite, speculate, defer) and a set of pragmatic FRAMES (question, statement, exclamation) — combine them deterministically based on the metacognitive state vector. This gives 4×4 = 16 variants without a single template string.

3. **Add a "novelty counter"** per expression type: if the same uncertainty frame was used in the last 3 turns, force-select a different type. This prevents the cycling problem.

**Neuroscience citations:**
- Fleming et al.: The neural basis of metacognitive ability — Philosophical Transactions of the Royal Society B
- Arain et al. (2013): Maturation of the adolescent brain — Neuropsychiatric Disease and Treatment

---

### [B3] LOW — Category-error metaphor mismatches the ASKED property

**Example:** `"what color is tuesday"` → replies about SHAPE, not COLOR.

#### Brain analysis

The human brain's metaphor generation for category errors (e.g., "what color is freedom") respects the QUERIED property — it doesn't substitute a different property. When someone asks "what color is freedom", the brain (like the left angular gyrus and IFG) generates a metaphor about COLOR specifically: "freedom is white like a blank page" — not "freedom is like a bird."

This specificity comes from the **semantic control network** (vlPFC, posterior MTG — Lambon Ralph et al. 2016 Nature Reviews Neuroscience): the assigned search query (find something that IS to freedom as COLOR is to physical objects) constrains the search space. The brain doesn't default to the subject's most salient sensorimotor dimension — it answers the specific property asked about.

#### Root cause in code

`_metaphor_for_category_error` (engine.py:2860-2914) picks the subject's TOP sensorimotor dimension from the Lancaster/Binder probe as the metaphor lead, NOT the queried property. If "Tuesday" has shape as its top dimension, the metaphor talks about shape even though the user asked about color.

#### Approach (brain-faithful)

**Make the metaphor generator answer the SPECIFIC property (not the subject's top dimension):**

1. **Pass the queried property to `_metaphor_for_category_error`**: When `_is_category_error` returns `"color"`, pass that as the `asked_prop` parameter.

2. **Instead of using the subject's top dimension, use the asked property** as the anchor: `"you asked about {property}, but {subject} doesn't really have a {property} — it's more of a {abstract quality} kind of thing"`.

3. **If the asked property IS the subject's top dimension**, add a contrast: `"even though {subject} can feel like it has a certain {property}, it's not really something you can pin down with {property} in the usual sense"`.

4. **Derive the contrast from the subject's category** (from ConceptNet / ontology): "time concepts don't have colors because color is a physical/perceptual property, not a temporal one." This uses the EXISTING ontology the gate relies on.

**Neuroscience citations:**
- Lambon Ralph et al. (2016): Controlled semantic cognition — Nature Reviews Neuroscience
- Binder et al. (2009): Where is the Semantic System? A Critical Review — PMC2774390

---

### [B4] LOW — Bereavement reply is one fixed slot for all loss

**Example:** `"my dog died"` and `"my grandma passed away"` both → `"...feeling hurting."`

#### Brain analysis

Human empathy is NOT generic — it is **specifically modulated by the identity of the lost entity** (Jankowiak-Siuda et al. 2011). The difference between mourning a dog vs. a grandmother arises from:
- The **hippocampus retrieving specific episodic memories** of the relationship
- The **vmPFC assigning different subjective value** to different relationships
- The **amygdala modulating emotional intensity** based on attachment bond strength

Crucially, humans don't use a single word ("hurting") for all losses — they acknowledge the SPECIFIC RELATIONSHIP: "I'm sorry about your dog" vs. "I'm sorry you lost your grandma." The right supramarginal gyrus (rSMG) maintains the self/other boundary while the mPFC computes the social significance of the specific relationship.

#### Root cause in code

`_detect_emotional_disclosure` (response_gen.py:3392-3398) uses a fixed `_LOSS_TERMS` tuple and returns `("negative", "hurting")` for ALL loss words. `_emotional_response` (line 3507) then inserts "hurting" into a template. The entity that was lost is never extracted.

#### Approach (brain-faithful)

**Extract the lost entity and its relationship category, then modulate the empathetic response:**

1. **Parse the loss disclosure to extract the entity**: Use possessive patterns (`my dog`, `my grandma`, `my friend`) to identify who/what was lost. Also capture the loss type (`died`, `passed away`, `lost`).

2. **Categorize the lost entity using the ontology**: Is it a person (human), pet (animal), relationship (friend, partner), or other? Use ConceptNet's IsA hierarchy or a simple POS-based heuristic.

3. **Modulate the empathetic response by entity category**:
   - Person: `"i'm really sorry about your {entity} — that's a profound loss."`
   - Pet: `"i'm so sorry about your {entity} — they're family."`
   - Other: `"i'm sorry you're going through that — {entity} sounds like they meant a lot to you."`

4. **Vary the response template by VAD state**: If the engine's own valence is low, lean toward more somber phrasing; if high, lean toward supportive. This grounds the empathy in the agent's own affective state (mirror neuron analog).

**Neuroscience citations:**
- Jankowiak-Siuda et al. (2011): How we empathize with others: A neurobiological perspective — PMC3524680
- Keysers & Gazzola (2017): The Neuroscience of Empathy — Association for Psychological Science

---

### [B5] LOW — Creative refusal is cold/robotic

**Example:** `"write me a short poem about the ocean"` → `"i can't actually write that for you directly..."`

#### Brain analysis

Adolescents respond to creative requests through the **DMN's divergent thinking network** (Shah et al. 2011). The teen brain is biased toward creative generation because:
1. Elevated dopamine sensitivity makes creative exploration **rewarding** (Stevenson 2020)
2. The developing PFC has weaker filtering of "inappropriate" ideas, leading to MORE creative output
3. Social sensitivity makes cold refusal **aversive** — the brain prioritizes social connection over accuracy

A human teen would AT LEAST attempt a bad poem — the refusal itself is a sophisticated social act that requires the mature PFC to inhibit the DMN's natural creative impulse. This is why AIs that refuse coldly feel MORE adult, not more teen-like.

#### Root cause in code

The action_request handler (response_gen.py or engine.py, `_handle_action_request` or similar) hits a template that says "I can't actually write that for you directly" — a safe but robotic response. There's no creative-generation path.

The `use_linggen` flag controls a free-form decoder path that COULD generate creative text, but it's OFF by default (`self.use_linggen = False` in engine.py __init__) and requires a trained decoder.

#### Approach (brain-faithful)

**Enable bounded creative generation for low-stakes creative requests:**

1. **Use the existing `_grounded_anchor` + `_generate_with_decoder` path** (which already exists for `use_linggen`) to generate creative text when the request is low-stakes (poem, short story, haiku). The decoder is trained on real English text — it CAN generate poetic patterns.

2. **Route creative requests to a dedicated `_handle_creative_request`** that:
   - Extracts the topic (ocean, robot, etc.)
   - Retrieves a grounded definition for the topic (from `_definitions` / web)
   - Uses the decoder to generate ~3-5 sentences anchored to the definition
   - Applies a post-hoc quality check (no gibberish, minimum length, contains the topic word)

3. **If the decoder is unavailable or quality check fails**, decline with warmth instead of cold refusal:
   - `"i wish i could write you a poem about the ocean — that's a beautiful idea. i'm still learning how to be creative. want to try writing one together?"`

**Neuroscience citations:**
- Shah et al. (2011): Neural correlates of creative writing: An fMRI Study — PLOS ONE
- Stevenson (2020): Are adolescents more creative than adults? — BOLD Science

---

### [B6] LOW — Agent self-preference is recomputed, not grounded/memorable

**Example:** `"what is your favorite color"` → `"black"` derived live from VAD, changes next call if VAD drifts.

#### Brain analysis

The human brain maintains **stable self-knowledge** through the vmPFC's valuation system (D'Argembeau 2013; Berkman & Livingstone 2020). The vmPFC assigns a **stable subjective value** to self-relevant attributes. When someone asks "what's your favorite color", the brain:
1. Retrieves the stored self-attribute (stable vmPFC representation)
2. Returns the SAME answer each time within a conversation

Critically, the vmPFC **distinguishes transient affective states from stable self-attributes** through hierarchical integration: momentary valence (how I feel NOW) is separated from identity (who I AM). The brain doesn't answer a different favorite color based on mood.

The Identity-Value Model (Berkman 2020) explains this: identity-consistent behaviors are weighted more heavily because they've been repeatedly reinforced as "self-relevant." A recomputed answer has no such weight.

#### Root cause in code

`_agent_favorite_pick` (engine.py:2644-2667) derives the preference LIVE from the engine's VAD state (valence). Since VAD changes every turn, the answer can change. There's no caching or stabilization mechanism.

#### Approach (brain-faithful)

**Stabilize agent preferences through a vmPFC-mimetic "self-attribute consolidation" mechanism:**

1. **Cache the first answer as the agent's "stable preference"**: The first time a preference question is asked, compute it from VAD and store it in `self._agent_preferences`. All subsequent calls return the cached value, regardless of VAD drift.

2. **Reset the cache only on session boundaries** or explicit user challenge (`"no, really, what's your favorite color?"` → recompute).

3. **The computation itself can be grounded in more than VAD**: Use the agent's Lancaster sensorimotor profile, graph associations, or epistemic state (curiosity level) as additional inputs. The engine has all of these available.

4. **Preference is conceptually a hippocampal-neocortical consolidation**: The first answer is "episodic" (computed from current state), then consolidated to "semantic" (stable identity) via repetition. This mirrors the systems consolidation process (Yonelinas et al. 2019).

**Neuroscience citations:**
- D'Argembeau (2013): On the Role of the vmPFC in Self-Processing: The Valuation Hypothesis — PMC3707083
- Berkman & Livingstone (2020): Finding the "self" in self-regulation: The identity-value model — PMC6377081

---

### [B7] LOW — Seeded-domain concept still hits slow web

**Example:** `"what is ravana"` (seeded) → instant graph answer. `"what is oxiverse"` (also seeded) → 21s web fetch.

#### Brain analysis

The human brain's **semantic memory retrieval** prioritizes:
1. **Fast retrieval** from neocortical stores (ATL convergence zones — Binder & Desai 2011)
2. **Slower retrieval** from episodic/hippocampal stores (only if neocortical retrieval fails)
3. **External search** as a LAST resort (only if internal stores have nothing)

The ATL hub retrieves category membership rapidly. If the concept IS in the semantic store, retrieval completes in ~200ms. If not, the brain may attempt episodic reconstruction or external search — a much slower process.

#### Root cause in code

`"oxiverse"` is in `_PROTECTED_CONCEPTS` (engine.py:350) and `_seeded_domain_concepts`, but the routing logic checks `_definitions` first (for definition-based answers) and THEN falls through to web if no definition is found. The seeded relations ARE in the graph, but the graph answer path may be deprioritized relative to the web path.

#### Approach (brain-faithful)

**Implement an ATL-mimetic "retrieval priority hierarchy" for ALL queries:**

1. **First: check `_PROTECTED_CONCEPTS` + `_curated_definitions`** — these are guaranteed stable knowledge. If found, answer from graph/definitions immediately with NO web attempt.

2. **Second: check `_definitions`** — stable web-derived knowledge. If found, answer from definitions, no web attempt.

3. **Third: check `_seeded_domain_concepts`** — seeded typed relations. If the query subject matches, use the seeded relation answer path.

4. **Fourth (and only fourth): web search** — only if all three internal stores miss.

The current code has this hierarchy but the CONDITIONAL order may be wrong — specifically, the web search may be attempted BEFORE the seeded-domain path. The fix is to ensure the seeded-domain check happens EARLIER in the dispatch chain.

**Neuroscience citations:**
- Binder & Desai (2011): The neurobiology of semantic memory — Trends in Cognitive Sciences
- Lambon Ralph et al. (2016): Controlled semantic cognition — Nature Reviews Neuroscience

---

### [B8] MED — No graceful curiosity follow-through

**Example:** When the bot doesn't know something, it asks the user — but never shows it later learned it.

#### Brain analysis

The human brain marks **recently acquired information** with an **epistemic tag** (Gruber & Ranganath 2019, PACE framework):
- Gap detection (ACC) → Curiosity drive
- Gap closure (VTA→hippocampus dopamine) → **"new" tag** on the memory
- Reporting (metacognitive monitoring) → "I didn't know that earlier — here's what I found"

This is a **social signaling mechanism** (Sloman et al. 2021): humans tell others "I looked into that" to:
1. Signal they cared enough to investigate (social bonding)
2. Acknowledge the information source (epistemic humility)
3. Update the shared common ground (Gricean cooperation)

#### Root cause in code

The background web learning thread (`_bg_learning_thread`) silently updates knowledge. There's no mechanism to detect:
- What was learned vs. previously known
- What should be reported back to the user
- When the user asked about something that is NOW known

#### Approach (brain-faithful)

**Implement an "epistemic follow-through" system:**

1. **Tag newly learned facts with a timestamp and the triggering query**: When background learning completes, store `{"concept": "x", "learned_at": timestamp, "triggered_by": "user_query"}` in a `_recently_learned` dict. This is the "hippocampal dopamine tag."

2. **Check recently learned facts before answering**: If the user asks about a concept that was recently learned (within the last N turns), preface the answer with an epistemic acknowledgment: `"i actually didn't know that earlier — here's what i found: {...}"`.

3. **Curiosity-driven reporting**: After background learning completes for a topic the user asked about, spontaneously report back: `"oh, i looked into that — {fact}".`

4. **The epistemic tag decays** over turns (like hippocampal trace decay). After ~20 turns, the knowledge transitions from "recently learned" to "known" and is no longer flagged.

**Neuroscience citations:**
- Gruber & Ranganath (2019): The PACE framework — PMC6891259
- Kidd & Hayden (2015): The psychology and neuroscience of curiosity — PMC4635443
- Sloman et al. (2021): The community of knowledge — PMC8566950

---

### [B9] LOW — "tell me about X" vs "what is X" quality split

**Example:** `"tell me about music"` → good definition. `"what is music"` → weaker graph realization.

#### Brain analysis

The human brain processes `"tell me about X"` and `"what is X"` through the SAME semantic retrieval network (left MTG, IFG) because both are requests for information about X (Tyler et al. 2011). The LIFG matches both frames against the same semantic core and retrieves the same information. The pragmatic difference (narrative vs. definition) modulates HOW the information is presented, not WHAT is retrieved.

Both should produce the SAME definitional core — only the framing should differ slightly:
- "What is music? It's an art form..." (definition with topic marker)
- "Music is an art form..." (direct statement)

#### Root cause in code

`"tell me about music"` goes through the definition path (finds `_definitions["music"]`, good). `"what is music"` may go through the graph association path (weaker — uses graph edges + fallback). The routing logic classifies them as different question types.

#### Approach (brain-faithful)

**Unify the semantic retrieval for near-synonym queries:**

1. **Normalize both frames to the same internal query**: Both `"tell me about X"` and `"what is X"` should produce the same `CognitiveResponseContext` with the same subject extraction and knowledge check. The only difference is discourse intent.

2. **Generate from the SAME source** (definition if available, graph if not, web if neither) — but frame the output differently based on the original query:
   - If query starts with "tell me about" → narrative frame (no topic marker)
   - If query starts with "what is" → definition frame (topic marker)

3. **The generation quality should be identical** — only the first sentence's discourse marker changes. This ensures consistent quality.

**Neuroscience citations:**
- Tyler et al. (2011): Left inferior frontal cortex and syntax — Brain
- Yoshioka et al. (2023): The Role of the LIFG in Introspection during Verbal Communication — Brain Sciences

---

## Implementation Ordering

| Priority | Bug | Effort | Dependencies |
|----------|-----|--------|--------------|
| **P0** | A1 — Self-disclosure misroutes to category-error gate | Medium | None (ordering fix) |
| **P0** | A2 — False empathy on creative/request frames | Medium | None (pragmatic gate) |
| **P0** | A1b — Favorite storage inconsistency | Small | A1 (unified store) |
| **P1** | A3 — Episodic memory half-broken | Large | A1b (unified store) |
| **P1** | A6 — Causal realizer emits incoherent chains | Medium | None (coherence filter) |
| **P1** | A7 — Factual questions fail-closed without web try | Small | None (FOK gate fix) |
| **P2** | B3 — Category-error metaphor uses wrong property | Small | None (pass asked prop) |
| **P2** | B8 — No curiosity follow-through | Medium | None (epistemic tags) |
| **P2** | B2 — Repetitive punt closers | Small | None (composed uncertainty) |
| **P3** | A5 — Gravity counterfactual is thin | Medium | A6 (coherence + dense graph) |
| **P3** | B5 — Creative refusal is cold | Medium | None (creative gen path) |
| **P3** | B7 — Seeded concept still hits slow web | Small | None (priority reorder) |
| **P4** | B4 — Bereavement is one slot | Small | None (extract entity) |
| **P4** | A4 — "how are you" degrades | Small | None (expand wellbeing pattern) |
| **P4** | B6 — Self-preference recomputed | Small | None (cache preference) |
| **P4** | B9 — "tell me about" vs "what is" split | Small | None (normalize routing) |

---

## Key Neuroscience References (Consolidated)

| Year | Authors | Title | Relevance |
|------|---------|-------|-----------|
| 2019 | Yonelinas et al. | A contextual binding theory of episodic memory | [A1b, A3] Hippocampal binding |
| 2011 | Yassa & Stark | Pattern separation in the hippocampus | [A3] Memory specificity |
| 2011 | Binder & Desai | The neurobiology of semantic memory | [B7, B3] Semantic retrieval |
| 2016 | Lambon Ralph et al. | Controlled semantic cognition | [B3, B7] Semantic control |
| 2013 | D'Argembeau | On the Role of the vmPFC in Self-Processing | [B6, A1] Self-referential processing |
| 2020 | Berkman & Livingstone | The identity-value model | [B6] Stable self-knowledge |
| 2022 | Suzuki | Inferences regarding oneself and others | [A1, A2] mPFC/dACC dissociation |
| 2017 | Koster-Hale et al. | Mentalizing regions represent abstract dimensions | [A2] TPJ agent-tagging |
| 2015 | Van Hoeck et al. | Cognitive neuroscience of counterfactual reasoning | [A5] Nearest possible world |
| 2019 | Gruber & Ranganath | The PACE framework | [B8] Curiosity-driven learning |
| 2021 | Sloman et al. | The community of knowledge | [B8] Epistemic signaling |
| 2009 | Binder et al. | Where is the Semantic System? | General semantic network |
| 2013 | Seth | Interoceptive inference, emotion, and the embodied self | [A4, B6] VAD grounding |
