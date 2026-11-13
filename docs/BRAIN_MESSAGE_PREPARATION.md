# How the brain prepares a message before lexical selection

Research deliverable, 2026-11-13. Question: the RAVANA decoder's GRU hidden state encodes
*where in the sentence we are* (collapsing to `<eos>`) rather than *what word comes next*.
Before redesigning it, establish what the brain actually does at the stage that precedes
word selection.

## The stage ordering

Levelt's model of speech production separates **conceptual preparation** (pre-lexical) from
**lexical access** (lemma retrieval) and from phonological/phonetic encoding. The ordering is
robust across studies:

| Stage | Window after stimulus |
|---|---|
| Conceptual preparation (macro + microplanning) | ~0–200 ms |
| Lemma retrieval | ~200–275 ms |
| Phonological code | ~250–330 ms |
| Syllabification | ~330–455 ms |
| Phonetic encoding / articulation | ~455–600 ms |

(Indefrey & Levelt 2004, *Cognition* 92:101–144. Windows are approximate and task-dependent;
MEG/EEG work puts category/conceptual decoding at 150–250 ms.)

The point for RAVANA: **there is a representational stage before any word is chosen**, and it
is not a word sequence.

## Macroplanning vs microplanning

Levelt splits conceptual preparation in two:

- **Macroplanning — what to say.** Elaborates the communicative intention into subgoals,
  selects and orders information, linearizes, manages discourse focus and theory of mind.
- **Microplanning — how to say it.** Gives each speech act propositional shape and
  perspective, recodes non-verbal images into propositional format, and maps states of affairs
  onto **lexical concepts** (concepts that have words in the language).

The output of microplanning is Levelt's **message**.

## The message is not a word sequence

The message is a **non-linear, propositional/relational structure** — a who-did-what-to-whom
framework with **thematic role–concept bindings** (e.g. `AGENT ↔ DOG`). It is amodal in format,
but its terminal elements are lexical concepts (Levelt 1989 p.73; Bock & Ferreira 2014;
Konopka & Brown-Schmidt 2014).

This is the single most important fact for the decoder redesign. RAVANA's GRU is trained to
predict the next token from a linear left context — i.e. it is asked to do microplanning's job
*and* lexical selection in one step, with no message representation in between. The brain does
not do this: the relational structure exists first, and words are selected from it.

## The transmodal hub

The left anterior temporal lobe (ATL) is a **transmodal semantic hub** in a hub-and-spoke
architecture: modality-specific spokes (visual, auditory, motor) converge on an amodal hub.
Disrupting ATL with rTMS slows specific-level naming and eliminates semantic congruency
effects (Pobric et al. 2007, *J Neurosci* 27(31):8328–8333; 2009/2010 *Neuropsychologia*).
Lambon Ralph et al. 2017, *Nat Rev Neurosci* 18(1):42–55.

For RAVANA this is the convergence-zone architecture: a relational hub with modality-specific
spokes, not a single flat embedding table.

## The prefrontal goal buffer

Speaking intention activates the whole speech network **2–4 seconds before speech**:
frontopolar BA10, ACC, SMA/pre-SMA, caudate, and lateral PFC (Kell et al. 2011, *NeuroImage*;
Gehrig et al. 2012, *NeuroImage*). Ventromedial and bilateral ventral PFC encode the speaking
condition and social context in advance.

This is a **goal buffer that configures the network before message encoding** — the content
to be expressed is set up seconds ahead, and it constrains what the downstream lexical system
can produce.

## Lemma selection is competitive and incremental

WEAVER/WEAVER++ selects the target lemma under **competition**, using a **Luce-ratio hazard
rule**:

```
P(select target per time unit) = a(target) / Σ a(all lemmas)
```

with activation spreading and decaying: `a(m, t+Δt) = a(m,t)(1−d) + Σ r·a(n,t)`.
Competitor activation increases selection time (Nozari, Dell & Oppenheim 2013). Lemma
selection window ~100–150 ms (Roelofs 1992, *Cognition* 42:107–142).

Dell's interactive two-step model adds cascading activation and feedback between semantic
features → lemmas → phonemes (Dell & O'Seaghdha 1991/1992). The "Dark Side" model
(Oppenheim, Dell & Schwartz 2010) uses a booster that floods the lexical layer until one node
exceeds the others.

## The most directly implementable architecture

**Chang–Dell–Bock dual-path model (2006, *Psych Review* 113(2):234–272):**

- **Meaning System** holds event semantics; the message is a set of temporary **Role ↔ Concept
  bindings**.
- **Sequencing System** is a simple recurrent network (SRN) that predicts the next word from
  left context.
- Restricted communication between the two systems; **error-driven (backprop) learning**.
- Accounts for structural priming persistence and the comprehension–production priming
  symmetry.

This maps onto RAVANA almost exactly: the ConceptGraph is the Meaning System, the GRU is the
Sequencing System, and the missing piece is the **message layer with role–concept bindings**
between them.

## Convergence with CPG

The Committed-Proposition Gate is **Levelt's microplanning output**: it commits to a
graph-supported proposition before form encoding. That is the right mechanism, but it is the
*second* half of conceptual preparation. What RAVANA lacked was the **macroplanning /
intention-specification stage** that *builds* that proposition from a communicative goal.

That stage now exists: `ravana/src/ravana/learn/macroplanning.py`.

## Implementation core for a small neural system

1. **Goal/intention buffer** — a prefrontal-style state that configures the network before
   content encoding (seconds-scale in the brain; here, turn-scale).
2. **Non-linear message graph** with role–concept bindings (amodal hub + modality-specific
   spokes = convergence-zone architecture).
3. **Competitive Luce-ratio selection** over candidate lexical concepts with decay and
   spreading activation.
4. **Incremental, ease-gated planning scope** — hierarchical when event gist is easy to
   encode, linear (one referent at a time) when hard (Konopka & Meyer 2014).
5. **Prediction-error / implicit-learning loop** for online adaptation.

## Settled vs contested

**Settled:** conceptual preparation precedes lexical selection; the message is a non-linear
relational structure with role–concept bindings; lemma selection is competitive and incremental
with a Luce-ratio rule; left ATL is a transmodal semantic hub; planning is incremental and
flexible; prefrontal intention signals precede message encoding by seconds.

**Contested:** modal vs amodal representations (unresolved); lexical vs abstract syntactic
control of grammatical encoding; whether the message is cast in a propositional language of
thought or whether linguistic creatures think directly in lexical concepts; whether
microplanning simplifies conceptual structure; exact time-course boundaries; WEAVER++'s
discrete production rules vs Dell's continuous interactive activation.

## Citations

- Levelt, W. J. M. (1989). *Speaking: From Intention to Articulation*. MIT Press. DOI 10.7551/mitpress/6393.001.0001
- Levelt, Roelofs & Meyer (1999). A theory of lexical access in speech production. *BBS* 22(1):1–38. DOI 10.1017/S0140525X99001776
- Roelofs, A. (1992). A spreading-activation theory of lemma retrieval in speaking. *Cognition* 42:107–142. DOI 10.1016/0010-0277(92)90041-f
- Indefrey & Levelt (2004). The spatial and temporal signatures of word production components. *Cognition* 92:101–144. DOI 10.1016/j.cognition.2002.06.001
- Chang, Dell & Bock (2006). Becoming syntactic. *Psych Review* 113(2):234–272. DOI 10.1037/0033-295X.113.2.234
- Dell & O'Seaghdha (1991). Mediated and convergent lexical priming. *Psych Review* 98(4):604–614. DOI 10.1037/0033-295X.98.4.604
- Dell & O'Seaghdha (1992). Stages of lexical access in language production. *Cognition* 42:287–314. DOI 10.1016/0010-0277(92)90046-k
- Oppenheim, Dell & Schwartz (2010). The dark side of incremental learning. *Cognitive Psychology* 61(3):217–250
- Nozari, Dell & Oppenheim (2013). How does the brain select its candidates? *Brain and Language* 127(1):1–12
- Konopka & Meyer (2014). Planning ahead. *Cognitive Psychology* 73:1–40. DOI 10.1016/j.cogpsych.2014.04.001
- Kell et al. (2011). Speaking intention: neural correlates of the speech network. *NeuroImage*
- Gehrig et al. (2012). The neural costs of the auditory–speech interface. *NeuroImage*
- Pobric, Jefferies & Lambon Ralph (2007). Anterior temporal lobes mediate semantic representation. *J Neurosci* 27(31):8328–8333
- Lambon Ralph et al. (2017). The neural and computational bases of semantic cognition. *Nat Rev Neurosci* 18(1):42–55
- Bock & Ferreira (2014). Syntactically speaking. In *Oxford Handbook of Language Production*. DOI 10.1093/oxfordhb/9780199735471.013.008
- Roelofs (2014). A unilateral processing advantage in the WEAVER++ model? *Cortex* 59:33–48
- Momma (2021). *The Production of Linguistic Knowledge*. MIT Press
