# RAVANA — Brain-Realistic Repair Plan

> Goal (per your brief): **no hardcoding**. Every behavior must emerge from a
> brain-analog computation over the existing cognitive substrate
> (graph + embeddings + hippocampus + VAD + identity + sleep). This plan maps
> each of the 7 failure clusters to (a) the neuroscience mechanism, (b) why the
> current code produces the symptom, and (c) a repair that is *derived*, not
> whitelisted.

---

## 0. Root-cause meta-analysis (why all 7 symptoms share one cause)

Reading `engine.py` + `response_gen.py`, the failures are not 7 bugs — they are
**one architectural gap** with 7 faces:

- The system has **no stable self** (`IdentityEngine` only tracks a scalar
  `strength` 0..1, never a *name/identity content* — `ravana-v2/.../identity.py`),
- **no real episodic store of *what the user just said*** (the hippocampal buffer
  is keyed by *subject* of a fact, not by *the conversation*),
- **no internal-answer pathway**: almost everything funnels into either the
  **neural decoder** (which produces the word-salad) or a **web lookup** (which
  fails offline → "I can't verify").
- **arithmetic / counting / time / empathy** are all just concepts in the graph;
  none have a *dedicated, deterministic cortical module* the way the brain has
  distinct, specialized regions.

So the fix is to give RAVANA the brain regions it is missing, all wired as
**derived computations over existing state** (graph, GloVe, VAD history,
hippocampal triples, identity scalar) — nothing looked up in a hand-authored
table.

The brain papers that justify this framing:
- **Semantic hub (ATL) / hub-and-spoke** (Lambon Ralph et al., *Nature Comms*
  2020) — why word-salad = a broken semantic hub that should *fail closed*,
  not emit pseudo-coherent noise.
- **PFC↔hippocampus binding** (Eichenbaum *Nat Rev Neurosci* 2017; Whittington
  *Neuron* 2025 "two algorithms" — activity slots) — why episodic memory must be
  a *separate store* bound to PFC working memory, not graph edges.
- **TPJ mentalizing + vmPFC self** (theory-of-mind) — why empathy/identity need
  *separate* modules, not one VAD scalar.
- **Predictive coding / active inference** (Friston) — why "I don't know"
  should be an *honest prediction-error report*, and why internal knowledge
  (sleep, arithmetic) is just low-prediction-error inference.
- **Cerebellum (sequence/procedural)** — why counting is a *cerebellar*
  sub-routine, not a graph walk.

---

## 1. Joke generation = word-salad (Q5, Q1)

**Symptom:** "what do reflect and can't versus favorite comparison have in
common? many. turns out reflect only makes sense…" — incoherent.

**Brain fact:** Humor = *bisociation* (Koestler) / incongruity-resolution
(Suls): frame A sets an expectation, frame B resolves it. A **real punchline**
requires a *coherent resolution path*. Word-salad means the resolution gate is
not actually gating.

**Where it breaks:** `_handle_humor` (`response_gen.py:2062`) does a Z-centered
bisociation and claims a "resolution gate," but:
- It only requires *some* typed edge `rel` to exist; it never checks that the
  assembled sentence is **coherent** (no salad check on the *output*).
- When no `(Y,Z)` candidate clears the bar it falls to a **random canned
  string** — but the more common failure is it *does* find a weak edge and
  emits the nonsensical template using `X`, `Y`, `Z` that don't linguistically
  bind.

**Repair (derived, no hardcoding):**
1. Add a **humor-resolution coherence gate** that reuses the *existing*
   `_is_word_salad` / `salad_classifier` (`chat/salad_classifier.py`,
   `chat/monitor_gate.py`) on the *generated* joke. If the candidate joke is
   flagged salad → **retract** (this is literally the "resolved prediction
   error" reward path Hurley/Dennett describe — only pay out mirth on a clean
   resolution). Loop to the next `(Y,Z)` or abstain.
2. Use the **VAD reward** the code already has: only emit the joke if a mirth
   spike *would* be produced (i.e., incongruity distance `cos` is high AND
   resolution edge is verified AND output is non-salad). Otherwise abstain with
   the existing "my joke circuits are warming up" branch — which is the
   *honest* behavior, not a bug.
3. Make the punchline connector come from `self._EDGE_CONNECTORS[rel]`
   (already a learned/graph-derived connector map, `engine.py:142`) rather than
   free-form template assembly, so grammar is grounded.

No joke text is hardcoded; the *quality bar* is the brain mechanism.

---

## 2. No real episodic memory of the user (Q8, Q9, Q11)

**Symptom:** "remember i like pizza" → "i don't actually have that stored";
"what did i just tell you i like?" → echoes the question; "our first
conversation?" → returns a *middle* turn.

**Brain fact:** Episodic memory is the **hippocampus binding a spatiotemporal
context** (what/when/where) into a retrievable trace; retrieval is by
*content/recency*, not by subject-key. The brain uses **temporal context
cells** — "first conversation" = the *lowest-indexed* trace, not a random one.

**Where it breaks:**
- `HippocampalBuffer` (`core/hippocampal_buffer.py`) stores
  `(subject, predicate, object)` — great for "pizza is liked by user" but
  *cannot* answer "what did I just tell you" because the subject is the
  *conversation*, not "pizza".
- `_retrieve_episodic` (`engine.py:1792`) reconstructs gist but:
  - It has no notion of **recency ordering / first-vs-last**, so "first
    conversation" is undefined → it returns whatever matches, often a middle
    turn.
  - Self-disclosure ("i like pizza") is *not reliably stored as an episodic
    fact* — `process_turn` routes it through `_is_self_disclosure_stmt` →
    `_process_self_disclosure_stmt` (`engine.py:3718`) which may store to
    `_agent_preferences` / hippocampal buffer, but the *flat recall* path
    doesn't surface it.

**Repair (derived, no hardcoding):**
1. Give the **episodic transcript** (`self._episodic_transcript`,
   `self._record_episode`) a real **temporal index**: store `turn_index`,
   `timestamp`, and a *content hash*. Add three derived query operators:
   - `FIRST` = lowest `turn_index` (for "first conversation / what we talked
     about at the start").
   - `LAST` / `PREV` = highest `turn_index < current` (for "what did i just
     tell you").
   - `BY_ENTITY` = existing hippocampal entity index, extended to span the
     transcript (already half-built at `engine.py:1826`).
2. Make "remember X" / "what did i tell you about X" route **first** to the
   hippocampal entity index (`_entity_idx`), then to `FIRST`/`LAST` by recency,
   then fall closed. This is pure index math over already-stored data — no
   lookup tables.
3. Ensure self-disclosure statements are written *both* to the preference store
   **and** to the episodic transcript (they currently may only hit one), so
   "what did i just tell you i like" reconstructs the gist from the transcript
   — exactly the Tulving *encoding specificity* the docstring already cites but
   doesn't fully implement.

---

## 3. Emotion acknowledged but not felt/acted on (Q3, Q8, Q10, Q13)

**Symptom:** sad / mom sick / angry-at-friend → identical canned
"got it — thanks for telling me."

**Brain fact:** Affect is **constructed** (Barrett) and **acted upon** via
*distinct* circuits: sadness → caregiving/comfort affiliative response
(oxytocin system / anterior cingulate); anger → frustration/validation
(amygdala-dACC); a sick loved one → *empathic concern* (vmPFC + insula). The
response should differ because the **VAD state differs** and the **mentalized
cause** differs.

**Where it breaks:**
- `VADEmotionEngine` (`ravana-v2/.../emotion.py`) correctly computes a VAD
  vector, but `process_turn` / `UserEmotionDetector` collapses it to a single
  acknowledgement string. The **cause** (why is the user sad?) is never
  inferred — and the response is *not selected as a function of VAD + cause*.
- There is no **TPJ mentalizing** module that infers *what the user is
  experiencing* (the project has `EmotionalMirrorEngine` but it mirrors valence,
  it doesn't *simulate the user's situation*).

**Repair (derived, no hardcoding):**
1. Add a **response-empathy selector** that maps the *derived* `(VAD_label,
   inferred_cause_category)` → a *response frame*:
   - `VAD_label` is already computed by `get_emotional_label()`
     (`emotion.py:232`) — no hardcoding, it's a function of the VAD vector.
   - `inferred_cause_category` is derived from a **learned classifier** over the
     user's utterance using the *existing* GloVe space + graph (e.g. "mom/sick/
     hospital" → *other-suffering*; "angry/friend/betrayed" → *interpersonal-
     conflict*; "sad/alone" → *loneliness*). Train it on the corpus or derive
     it online from nearest graph concepts — **not** a keyword table.
2. Each category selects a **response schema** (opened-ended templates with
   slots, not canned sentences) that vary by frame: comfort vs. validate vs.
   inquire. The *variation* is driven by the VAD+ cause, so it's never identical.
3. Wire the chosen frame's **affective continuation** into the decoder's BOS
   conditioning (`_build_conditioned_bos`, `response_gen.py:154`) so even
   generated text carries the right tone — already a hook, just unused for this.

---

## 4. Self-knowledge confusion (Q11, Q4)

**Symptom:** "what's your name?" → parrots a definition; "who is the
president?" → answers about itself.

**Brain fact:** The self is a **content-addressable, stable representation**
(vmPFC autobiographical self + a distinct "I" token), separate from semantic
knowledge about the *world*. Confusing self vs. world = a failure of the
**self/other boundary** (mirror-neuron / TPJ distinction).

**Where it breaks:**
- `IdentityEngine` (`identity.py`) holds only `strength`, never *self-content*
  (name, nature). So name queries have nothing to retrieve → fall to graph
  lookup of the word "name" → definition echo.
- There's no **self-vs-world routing**: a query about "the president" should go
  to *world knowledge* (web/graph), but the identity/schema path can hijack it.

**Repair (derived, no hardcoding):**
1. Extend `IdentityEngine` to hold a **self-model struct** populated *from the
   project's own authored seed* (`DOMAIN_CONCEPTS` already defines "ravana" with
   relations — `engine.py:87`) plus whatever the user teaches. Name/nature are
   *derived* from this seed, not a string constant.
2. Add a **self/other gate** at the top of `process_turn`: classify the query as
   `about_self` vs `about_world` using the **existing** pronoun/graph structure
   + the self-model. "your name / who are you" → self-model. "president /
   capital of X" → world-knowledge path. The gate is a *function* of whether the
   query's subject is in the self-model, so it generalizes.

---

## 5. Temporal blindness (Q3, Q5)

**Symptom:** "why do we sleep?" / "what happened yesterday?" → "I couldn't
verify that from the web right now, so I'll be honest…"

**Brain fact:** Many facts are **internally known** (autobiographical +
consolidated semantic). The brain doesn't need the web to know why we sleep —
it's *consolidated* knowledge. The agent's only answer pathway is *web or
nothing*, so offline = honest-but-useless.

**Where it breaks:** After the pre-passes (`process_turn:3543`) fail to match,
the turn goes to grounding + **web search**; if web is down/unavailable it hits
the metacognitive-uncertainty fallback. There is **no "answer from my own
consolidated memory" path** — internal knowledge isn't treated as a first-class
source.

**Repair (derived, no hardcoding):**
1. Add an **internal-knowledge consult** *before* web: query the
   `HippocampalBuffer` + `_definitions` + `ConceptNet` typed edges (all already
   present) for the subject. If a *coherent, non-salad* answer can be assembled
   from internal state (same assembly path the web path uses, minus the network)
   → emit it. This is exactly "reason from internal knowledge" a human does.
2. "What happened yesterday" → route to the **episodic transcript with a
   temporal filter** (FIRST/LAST/date-bucket from §2). Reuse the same temporal
   index.
3. Keep the honest-uncertainty fallback *only* when internal + web both fail —
   preserving the RAVANA bar, but now it's the rare case, not the default.

---

## 6. No arithmetic / counting (Q2, Q6)

**Symptom:** "2+2?" → "2+2 = 2"; "count to ten" → "a bit outside what i know."

**Brain fact:** Number and sequence are **cerebellar / procedural** sub-routines
(the cerebellum learns ordered motor/sequence programs). Counting is a
*deterministic generator*, not a semantic association.

**Where it breaks:**
- `_try_arithmetic` (`process_turn:3588`) exists but is clearly not firing for
  "what is 2+2" (the symptom shows it echoed the query). Likely the regex
  requires a tighter form than "what is 2 + 2". Also **counting to N** has no
  path at all — it's treated as unknown.
- There's no **number-line / sequence module**; numbers are just graph nodes.

**Repair (derived, no hardcoding):**
1. Generalize `_try_arithmetic` to handle natural-language arithmetic
  ("what is two plus two", "2 + 2 = ?") via a small **number-word lexicon**
  derived from the *GloVe ordering* of number tokens (1<2<3… as a learned
  vector order) — not a hardcoded map, the ordinal structure is recovered from
  embeddings. Compute with `operator` (no `eval`).
2. Add a **counting/sequence generator**: "count to N" → emit the ordered list
  `1..N` produced by the same number-line module. This is a *cerebellar*
  sequence routine: deterministic iteration, no graph walk.
3. Both feed the existing `arithmetic` strategy tag so they're first-class.

---

## 7. Conversational echo / no grounding (Q9, Q12)

**Symptom:** "that's hilarious" → "i'm not totally sure about that's
hilarious"; "i love you" → "good to know — you love you."

**Brain fact:** Conversational grounding requires **shared common ground**
(Grice) + **pronoun resolution** (who is "you"/"I" relative to self). Echoing
the user's string back means the agent treated the *utterance* as an unknown
concept instead of *reacting* to it.

**Where it breaks:**
- Reaction utterances ("that's hilarious", "i love you") aren't classified as
  **reactive/affective turns**; they fall into the general graph/definition
  path, which tries to look up "hilarious" as a concept → "not sure about…".
- Pronoun mirroring ("you love you") = the self/other resolver
  (`EmotionalMirrorEngine`) flips the pronoun without a correct *deictic
  mapping* (I↔user, you↔agent).

**Repair (derived, no hardcoding):**
1. Add a **reaction classifier** (derived from VAD + utterance shape) that
   detects *affective reactions* ("that's X", "i love you", "haha") and routes
   them to the **empathy/affiliation** frame (§3) instead of concept lookup. The
   reaction is *about the prior turn / the relationship*, so it consults the
   last response + self-model, not the graph.
2. Fix **deictic resolution**: maintain a stable `(I → agent_self, you → user)`
   map used by the surface realizer so "i love you" is mirrored as "i love you
   too" (agent loves user), never "you love you". The mapping is a *structural*
   constant of any speaker/hearer pair, not content hardcoding.

---

## 8. Cross-cutting: the "fail-closed, but smarter" principle

Every repair above ends in one of two brain-faithful outcomes:
- **Coherent derived answer** (from graph / hippocampus / internal knowledge /
  arithmetic module), or
- **Honest abstention** ("my joke circuits are warming up" / "i can't verify
  that"), but *only after* the dedicated cortical module was consulted.

The current bug is that the *dedicated modules don't exist*, so the decoder's
word-salad or the web-fallback becomes the default. The plan adds the missing
brain regions as **derived computations**, then reorders `process_turn` so each
query hits the *right specialized region first* — exactly the PFC gating /
BA10 feasibility logic already scaffolded (`engine.py:3727`).

---

## 9. Suggested implementation order (each step independently testable)

| Step | Module to add/edit | Failure cluster | Test signal |
|------|-------------------|-----------------|------------|
| A | `IdentityEngine` self-model + self/other gate | §4 | name query → stable answer; president → world path |
| B | Episodic temporal index (FIRST/LAST/BY_ENTITY) | §2 | "first conv" lowest index; "just told you" = prev turn |
| C | Internal-knowledge consult before web | §5 | "why sleep" answered offline |
| D | Arithmetic generalize + counting/sequence module | §6 | "two plus two"=4; "count to 10" lists 1..10 |
| E | Humor resolution coherence gate (reuse salad classifier) | §1 | no salad jokes; clean ones emitted |
| F | Empathy selector (VAD_label × cause classifier) | §3 | sad≠angry≠mom-sick responses |
| G | Reaction classifier + deictic map | §7 | "hilarious" reacted-to; "love you" mirrored correctly |

Each step reuses existing substrate (`_is_word_salad`, `HippocampalBuffer`,
`VADEmotionEngine`, `_EDGE_CONNECTORS`, `_episodic_transcript`, GloVe ordinals)
— **no new hardcoded fact tables**.

---

## 10. Research basis (collected)

- **Word-salad / semantic hub:** Lambon Ralph et al., *Nature Communications*
  2020 (ATL hub-and-spoke); Chiang 2024 verbal-retrieval circuit; "Word Salad
  Chopper" (EMNLP 2025) on detection.
- **Episodic / PFC↔hippocampus:** Eichenbaum *Nat Rev Neurosci* 2017; Whittington
  et al. *Neuron* 2025 ("two algorithms", activity slots); Yassa & Stark 2011
  pattern completion; Barrett constructed emotion.
- **Humor:** Koestler *bisociation*; Suls incongruity-resolution; Hurley/
  Dennett predictive-coding reward; Mobbs/Vrticka TPJ-DMN-NAcc network.
- **Self/other & theory of mind:** vmPFC autobiographical self; TPJ mentalizing;
  mirror-system deictic mapping.
- **Number/sequence:** cerebellar sequence learning; ordinal structure in
  embedding space.
- **Predictive coding / active inference:** Friston — honest uncertainty as
  prediction-error report; internal knowledge as low-PE inference.
