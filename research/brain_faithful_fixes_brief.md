# RAVANA — Brain-Faithful Fixes: Web-Research Brief

Goal: replace the *mechanical/hardcoded* phrasings and *over-aggressive gates* in
RAVANA's chat engine with mechanisms that mirror how a human brain actually
handles the same situation. Each prompt below targets ONE deficiency, names the
exact code anchor, and asks the researcher to return a concrete, implementable
specification (algorithm + how it maps to RAVANA's existing primitives: the
ConceptGraph, GloVe 64-D vectors, the VAD emotion engine, the self-monitor
guards).

How to use: paste one prompt at a time into a web-research tool / model. Bring
back the answer (mechanism + computational formulation + mapping). I will then
implement it at the cited anchor. Do NOT ask the researcher to write RAVANA
code — only to specify the *mechanism* in algorithm form.

---

## PROMPT 1 — Counterfactual / "what if" simulation (feels like a placeholder)

CODE ANCHOR: `ravana/src/ravana/chat/response_gen.py`
  - `_simulate_counterfactual`  (line ~3100)
  - `_removal_causal_lines`     (line ~3252)  — emits the SAME two fallback
    lines at lines 3264, 3275, 3279, 3280, 3314, 3326:
        "everything that depends on X would shift"
        "other systems would scramble to fill the gap"
  - lead-in boilerplate at 3212 and 3356:
        "if X were different, here's what I'd expect to follow:"

PROBLEM: Every hypothetical opens with the identical lead-in and, when the
causal graph is sparse, falls back to the same two canned sentences. A human
simulating "what if the sun disappeared" does a *modal* forward simulation
with graded confidence, not a template.

RESEARCH ASK:
  Define the brain mechanism for *offline / counterfactual simulation* and give
  an implementable computational formulation. Cover:
  1. The hippocampus–prefrontal–default-mode-network forward simulator
     (Schacter & Addis; Gerstenberg et al. 2021 "Causal Judgments from
     Counterfactual Simulation"; Hassabis & Maguire scene construction).
  2. How humans *tag simulated content with modal certainty* (epistemic
     possibility vs. likelihood) instead of asserting it flatly.
  3. How a sparse knowledge base still yields a *plausible, specific* simulation
     (e.g. via analogy to the nearest known causal chain) rather than a generic
     placeholder.
  Return: (a) the mechanism in 2-3 sentences, (b) a pseudo-algorithm RAVANA can
  implement using a typed edge graph + vector similarity, (c) how to vary the
  lead-in naturally and how to express graded confidence ("probably", "one
  consequence would be", "I'm not sure but") without a fixed template.

---

## PROMPT 2 — "I don't know" that sounds like a person (not a slot fill)

CODE ANCHOR: `ravana/src/ravana/chat/response_gen.py`
  - `_reflective_uncertainty` openers at lines 3938-3963, especially the
    evidence clause `... and {top[0]} appears to be {_evidence}` which produced
    the garbled line: "power seems to be topics referred to by the same term".
  - fallback uncertainty at 3743-3781 (`_human_like_uncertainty`).

PROBLEM: When RAVANA lacks a definition it surfaces a *weak graph association*
("makes me think of power and trust") dressed up as insight, or emits a fixed
"a bit outside what i know" line. A human, when they don't know, signals the
*type* of not-knowing (never heard of it / vaguely recall / disagree with the
framing) and often offers a related thread WITHOUT asserting it as fact.

RESEARCH ASK:
  Define the cognitive science of *metacognitive ignorance signaling* and give
  an implementable formulation. Cover:
  1. Feeling-of-knowing / tip-of-the-tongue (Koriat 1993; Nelson & Narens
     metacognitive monitoring) — how the brain distinguishes "I have no
     representation" from "I have a partial/related representation".
  2. How humans hedge a *related-but-uncertain* association without presenting
     it as verified knowledge (epistemic markers: "reminds me of", "might be
     related to", "not sure if that's right").
  3. Why surfacing a *low-confidence association as if it were a definition* is
     a specific error class (overclaiming from sparse activation), and what a
     brain-faithful system should do instead (either withhold, or explicitly
     mark the association as a guess).
  Return: (a) mechanism, (b) pseudo-algorithm that, given a subject with NO
  stored definition but SOME graph neighbors, decides between
  [withhold | offer-a-guess-with-hedge | offer-related-thread-with-hedge],
  (c) a small set of *natural* English hedge frames (not templates to copy, but
  the linguistic pattern) RAVANA can realize.

---

## PROMPT 3 — Web-answer confidence gate (stop throwing away correct answers)

CODE ANCHOR: `ravana/src/ravana/chat/engine.py`
  - `_best_answer_snippet`     (line ~5163)
  - `_snippet_quality`         (line ~5638)
  - `_SNIPPET_PLAUSIBILITY_FLOOR = 0.38`  (line ~5742)
  - `_snippet_plausibility`    (line ~5744)
  - `_web_snippet_search` quality floor `if quality < 1.5: continue` (line ~5855)
  - `_forward_model_check`     (response_gen.py ~4906) — now exempts web, but the
    *pre-emission* salad gate still applies to everything else.

PROBLEM: A retrieved encyclopedic snippet scoring just below an arbitrary floor
(q=1.00 < 1.5) gets discarded, and the plausibility monitor (0.38) can withhold
a correct answer. The floors are hand-tuned constants, not derived from how the
brain decides "do I actually know this / is this source trustworthy".

RESEARCH ASK:
  Define how the brain does *reality monitoring* and *source credibility* for
  retrieved (external) memories, and give an implementable formulation RAVANA
  can use WITHOUT an LLM. Cover:
  1. Reality monitoring (Johnson & Raye 1981; Mitchell & Johnson 2009) — how
     the brain tags a memory as self-generated vs. externally sourced and flags
     low-fidelity retrieval.
  2. Source credibility / trust as a *learned, per-source* signal (the
     Track-B "source-trust" work already exists as a stub in the engine — find
     `use_source_trust`). How should credibility modulate, not gate, the answer?
  3. Why a *fixed numeric floor* is the wrong shape: confidence should be
     *comparative* (is this the best available snippet? is it coherent with what
     I already believe?) not absolute.
  Return: (a) mechanism, (b) pseudo-algorithm for a *comparative* plausibility
  check (best-of-available + belief-coherence) that replaces the fixed floor,
  (c) how to express remaining uncertainty in the answer ("according to
  [source type]...") naturally.

---

## PROMPT 4 — Degeneracy monitor that doesn't kill good sentences

CODE ANCHOR: `ravana/src/ravana/chat/response_gen.py`
  - `_forward_model_check`      (line ~4906)  — pre-articulation self-monitor
  - `_strip_degenerate_clauses` (line ~4838)  — clause-level salad detector
  - `_is_word_salad` (engine)   — the shared degeneracy detector

PROBLEM: The inner-speech / forward-model monitor is the right *idea* (Levelt's
pre-articulation loop; Pickering & Garrod) but it currently uses a
content-word-ratio + subject-presence heuristic that flags SHORT, correct,
honest answers ("I couldn't verify that...") and valid web snippets as
"degenerate". A human's inner monitor checks *does this match my intent*, not
*does this have enough content words*.

RESEARCH ASK:
  Define the brain's *inner-speech / forward-model self-monitoring* and give an
  implementable formulation that does NOT collapse correct short answers. Cover:
  1. Levelt's Blueprint (1989) pre-articulation monitor + Pickering & Garrod
     (2013) alignment; Yao et al. 2025 inner rehearsal. What exactly does the
     monitor compare against (the communicative intent, not a word-count)?
  2. Why a *content-word-ratio* proxy is a poor stand-in for "did I say what I
     meant" — and what a cheaper, intent-anchored check looks like (e.g. does
     the utterance address the question type: factual? emotional? hypothetical?).
  3. How to exempt *intentionally short* speech acts (acknowledgments, honest
     unknowns, empathic one-liners) from the salad check by their speech-act
     type rather than by a hardcoded strategy allowlist.
  Return: (a) mechanism, (b) pseudo-algorithm that tags a candidate utterance by
  speech-act/intent and only withholds when intent≠content (not when short),
  (c) how to compute "intent_match" from the already-available ctx fields
  (raw_input, subject, strategy, emotion state).

---

## PROMPT 5 — Query grounding that reads like comprehension (not regex surgery)

CODE ANCHOR: `ravana/src/ravana/chat/engine.py`
  - `_ground_query`      (line ~8198)  — multi-strategy grounding
  - `_strip_eli5_tail`   (line ~8180)  — ELI5 tail stripper (just added)
  - `_SUBJECT_CONTEXT_WORDS` (line ~7167) — the hardcoded verb/role stop-list
  - compositional block (~8248-8311) — joins first 2-3 words, splits clauses

PROBLEM: Grounding is a pile of regex + a giant hardcoded stop-list of verbs
("invent", "discover", "cause"...) and role words ("capital", "inventor"...).
Every new query shape needs a new hand-added word. A human parses the *syntax*
(thematic roles: who did what to whom) and recovers the topic from structure,
not from a dictionary of banned words.

RESEARCH ASK:
  Define the brain's *sentence comprehension / thematic-role assignment* and give
  an implementable, *dictionariless* formulation RAVANA can approximate with what
  it has (GloVe vectors, a typed dependency parse if available, or a lightweight
  POS/role heuristic). Cover:
  1. Theta-role / thematic-role assignment (Fillmore Case Grammar; Bornkessel &
     Schlesewsky extended argument dependency model) — how the brain recovers
     the topic from syntactic structure (agent/patient/theme) rather than a stop
     list.
  2. How to implement "the real topic is the *patient/theme*, not the verb or
     the role noun" without enumerating verbs — e.g. dependency-parse head-finding
     or, failing that, a vector-similarity "which content word is the semantic
     head" heuristic.
  3. How clause connectors (but/and) should trigger *topic segregation* (two
     separate questions) rather than a fused subject — link to human
     constituent parsing.
  Return: (a) mechanism, (b) pseudo-algorithm that takes a parsed/lightly-tagged
  query and returns the theme/patient as the subject (replacing the hardcoded
  _SUBJECT_CONTEXT_WORDS list), (c) a fallback when no parser is available.

---

## DELIVERABLE FORMAT I NEED BACK

For each prompt, return a short structured answer:
  - MECHANISM: 2-3 sentences, cite 1-2 key papers (author year).
  - ALGORITHM: pseudo-code or bullet steps, using only primitives RAVANA has
    (ConceptGraph nodes/edges, GloVe 64-D vectors, VAD state, ctx fields,
    existing monitors). No LLM calls.
  - MAPPING: which existing function/anchor it replaces or augments.
  - HEDGE/SURFACE: the natural-language pattern for expressing uncertainty or
    confidence (so the reply stops sounding templated).

I will implement each into the cited anchor, add/extend the regression tests in
`tests/unit/test_chat_fixes.py`, and verify end-to-end.
