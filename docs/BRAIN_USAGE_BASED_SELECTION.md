# Usage-based construction selection: the cue→frame association must be learned, not hard-coded

**Date:** 2026-11-19
**Scope / repo state:** `C:/Users/Likhith/Documents/projects/ravana`, tree clean at `7e55cda`, Windows/MSYS, read-only investigation. **No code changed, no other doc edited, no `eng.save()` called.**
**Question this doc answers:** RAVANA now selects a construction from the question cue (FUNCTION / PART_WHOLE / LOCATIVE / PROPERTY), but `extract_construction_cue` (`response_gen.py:205`) and the `if/elif` ladder in `_productive_use_response` (`response_gen.py:7590–7611`) hard-wire *which frame each cue means*. The brain-faithful next step is **usage-based**: the cue→frame association should be acquired from feedback and usage. What mechanism does the human brain use, what does that imply must change here, and how would we observe that it changed?

**Method note.** Every empirical claim carries an inline author–year citation plus a ledger-backed numbered citation `[n]`; the ledger maps each number to a retrieved URL in the Sources block. Peer-reviewed journal articles, monographs and handbook chapters are labelled and not conflated. Claims from model knowledge are `[unverified]`; engineering choices are labelled as such. No runtime score is reported: a cold engine boot is ~200 s and this task forbids code changes, so every code claim below was checked by reading the exact source lines and cross-checking identifiers with ripgrep.

**Companion docs (do not re-read):** `docs/BRAIN_CUE_SENSITIVE_CONSTRUCTIONS.md` (CVCC: cue-validity construction competition; fixed cue→relation table with a promise to "later count from accepted turns" — this doc delivers that "later"), `docs/BRAIN_CROSS_FRAME_PRODUCTIVITY.md` (SCC-RF: one frame → repertoire), `docs/BRAIN_PRODUCTIVE_VOCABULARY.md`, `docs/CUE_SELECTION_CODEMAP.md`, `docs/CROSS_FRAME_CODEMAP.md`, `docs/CAPABILITY_LEARNED_CONSTRUCTIONS_PERSIST_REUSE.md` (M16). This document is the **next** gap: learning the association itself.

---

## 0. Executive summary

Four findings, kept separate:

1. **The strength of a cue→construction pairing is a learned association, not a definition.** In collostructional analysis the unit of learning is the *collexeme*: a lexeme is a collexeme of a construction if it occurs in that construction's slot more often than expected by chance, and collostructional strength is computed from a 2×2 contingency of observed vs expected co-occurrence, with attraction and repulsion both measured (Stefanowitsch & Gries 2003, *International Journal of Corpus Linguistics*, 8(2), 209–243 [1]). The Competition Model makes the same claim about cue validity: a cue's strength is the product of its availability and its reliability, and those strengths are *adjusted by experience* (MacWhinney, Bates & Kliegl 1984, *Journal of Verbal Learning and Verbal Behavior*, 23(2), 127–150 [11]). A hard-coded wh-word→frame table has zero learned association: it is reliability fixed at 1 or 0 before any experience.

2. **Repetition entrenches a specific pairing (and can freeze it), while varied use across types builds a productive, general pairing.** Token frequency strengthens the memory representation of a specific instance and gives it a *conserving effect* — "High-frequency sequences become more entrenched in their morphosyntactic structure and resist restructuring on the basis of productive patterns that might otherwise occur" (Bybee 2006, *Language*, 82(4), 711–733 [2]); token repetition entrenches, while *type* repetition (many different items in the same slot) produces schematization, generalization and productivity (Bybee 2006 [2]; Gries, handbook chapter [9]). Children likewise begin with item-specific, lexically restricted schemas ("verb islands" / constructional islands) and only later abstract across items (Tomasello 2003, *Constructing a Language*, Harvard University Press — monograph [12]; Tomasello, *Constructions* [3]; a corpus study of item-based early grammar [10]). **This is exactly the failure mode to design against:** reinforcing one cue→frame pair many times will entrench it and can freeze selection; variety of use (across words and cues) is what keeps the association general rather than petrified.

3. **Learning from feedback is a prediction-error-driven, opponent process — reinforce the winner, suppress the competitor.** Midbrain dopamine prediction error has a *causal* role in cue–outcome learning: optogenetic activation timed to mimic a prediction error was sufficient to produce long-lasting increases in cue-elicited behaviour (Steinberg et al. 2013, *Nature Neuroscience*, 16, 966–973 [4]); the human feedback-related RewP scales with prediction error and drives learning (Weber & Bellebaum 2024, *Scientific Reports*, 14, 9674 [8]). The basal ganglia implement this with **separate Go and NoGo pathways differentially modulated by positive and negative reinforcement**, so an accepted action is facilitated and a rejected action is suppressed (Frank, Seeberger & O'Reilly 2004, *Science*, 306, 1940–1943 [5]). A learning rule that only ever *boosts* the chosen frame is half the mechanism: rejection must subtract from the chosen association and thereby let a competitor win.

4. **A system can avoid freezing while remaining deterministic by adding a novelty/optimistic term to an argmax, not by sampling.** Dopamine novelty responses are explained as *exploration bonuses* — fictitious value added to an option to encourage exploration — and "optimistic initial values" are the standard way to make an agent try untested options before settling (Kakade & Dayan 2002, *Neural Networks*, 15, 549–559 [6]); human striatal choice is best fit by a model that assigns higher initial value to novel options (Wittmann, Daw, Seymour & Dolan 2008, *Neuron*, 58(6), 967–973 [7]). A deterministic `argmax` over scores that include an optimism term preferentially selects the *least-attempted* admissible frame when evidence is tied, then disconfirms it through feedback — exploration without RNG, so byte-identical runs from identical state remain possible.

**One-sentence mechanism.** Replace the fixed `if cue == ...: frame = ...` ladder with a **learned association table** `assoc(cue, frame)` stored in the SCC's persisted state: initialise every admissible pair optimistically (novelty bonus), select by a deterministic `argmax` of `log(availability) + β·assoc(cue,frame) + γ·type_frequency(frame) − λ·recency`, then update `assoc` with a signed prediction-error step (`reinforce_construction(frame, success)` plus a negative counter-step on rejection) so the cue→frame mapping is acquired, can be reversed by feedback, generalizes across taught words, and never depends on a random draw.

Name: **Usage-Based Cue Association with Optimistic Exploration (UBCAE)** — a learned replacement for CVCC's fixed reliability table; it keeps CVCC's admissible-set, grounding and persistence machinery and changes only how `reliability(f, cue)` is obtained and how selection is scored.

---

## 1. Code-level diagnosis (read-only, at `7e55cda`)

| current behaviour | exact location | why it is not usage-based |
|---|---|---|
| cue is *detected* | `extract_construction_cue`, `response_gen.py:205–225` | Returns one of `PART_WHOLE / LOCATIVE / FUNCTION / PROPERTY / NONE` by regex. Detection is fine; the problem is what consumes it. |
| cue→frame mapping is fixed | `_productive_use_response`, `response_gen.py:7574`, `7590–7611` | `_cue = extract_construction_cue(...)`; then `if _cue == "FUNCTION": sent = f"{subject} {predicate}."`, `elif _cue == "PART_WHOLE" ...`, `elif LOCATIVE ...`, `elif PROPERTY ...`, else the modulo-2 repeat counter. The association is a literal `if/elif`, identical on turn 1 and turn 10 000. |
| the learned inventory is *not consulted* by this path | `SyntacticCellAssembly.learned_frames` at `syntactic_cell_assembly.py:107`; `allocate_construction` `:591`, `reinforce_construction` `:616`, `find_construction` `:581`, `get_state/set_state` `:735–768` | The M16 machinery that could carry a learned association exists and persists, but `_productive_use_response` never reads `learned_frames` and never calls `reinforce_construction`. |
| a separate, unused learning signal exists | `SurfaceRealizer.learn_from_feedback`, `surface_realizer.py:213–217` (`+0.05` success / `−0.025` failure on `_variant_weights`); `_recent_variants` `:182`, `:618–621`, fatigue `:801` | This is a *surface-variant* preference (merge order), not a *constructional relation* association. It is the right shape of learning signal but is attached to the wrong level: changing `_variant_weights` cannot turn a functional answer into a part-whole answer. |
| tie-break still a counter | `response_gen.py:7603–7611` | `_productive_use_counts[subject] % 2`, in-memory only, not in any `get_state`/`set_state` blob — so even the fallback does not learn or persist. |

**What is genuinely missing:** there is no stored value anywhere for "how strongly does cue *c* predict frame *f*," and no feedback path that writes one. `learned_frames[f].weight` (`SyntacticFrame.weight`, `syntactic_cell_assembly.py:40`) is per-*frame* availability, not per-*cue–frame* reliability; the same weight cannot express that a frame is good under `LOCATIVE` but poor under `FUNCTION`. That distinction is the collexeme relation [1] and is the load-bearing addition.

**Instrument caution.** The four emitted strings are treated as given; no probe was run (cold boot ~200 s, no code changes permitted). Before wiring UBCAE, reproduce the four phrasings with an instrumented run logging `(cue, selected_frame_id, assoc_before, assoc_after)` per turn, per the measurement discipline in `brain-faithful-switch-rule`.

---

## 2. Research question 1 — what does experience change about a cue→frame mapping?

### 2.1 Collexeme association: the mapping is a measured, signed association

Collostructional analysis starts from a construction and asks which lexemes are *attracted* or *repelled* by a slot, using a 2×2 table of (word in construction, word elsewhere, construction with other words, other constructions with other words) and an association measure that compares observed with expected frequency (Stefanowitsch & Gries 2003 [1]). Two properties matter here:

- **signed:** repulsion is as informative as attraction — a cue that actively disfavours a frame after repeated failure must be representable, not merely "not boosted";
- **contingency-based:** the association is not a raw count, it is a count *relative to how often the frame is used under other cues and the cue under other frames* — i.e. it is a conditional reliability, the same quantity cue validity calls reliability [11].

**Consequence for RAVANA.** `assoc(cue, frame)` must be a signed, relative measure updated from the *joint* history of cue and frame, not a per-frame scalar. `SyntacticFrame.weight` is the availability factor; the new table is the reliability factor; their product is cue validity [11], which is exactly CVCC's score split carried one step further into learned territory.

### 2.2 Entrenchment and type frequency: why repeated reinforcement freezes, and what prevents it

Bybee's exemplar account gives two opposing effects of repetition: high *token* frequency entrenches a specific instance and conserves it against restructuring, while high *type* frequency (many distinct items in a slot) strengthens a schematic, productive pattern [2][9]. Tomasello's item-based/verb-island findings are the acquisition-side counterpart: early schemas are tied to specific lexical items and only become abstract and general after experience with many items [12][3][10].

**Consequence for RAVANA.** If the only update is `reinforce_construction(frame, True)` after each successful cued turn, the most-used `(cue, frame)` pair will entrench and *freeze* — Bybee's conserving effect [2]. Two design responses follow, both literature-backed:

- keep a **type count** `types(frame)` = number of distinct `(word, cue)` pairs under which the frame was accepted, and give it a positive score term — this instantiates type-frequency-driven productivity/generalization [2][9];
- give every not-yet-disconfirmed pair an **optimistic initial association** so it can still be sampled — the novelty/optimism term [6][7] (§4).

Together these keep the mapping learnable rather than petrified: entrenchment supplies stability once evidence accumulates, type diversity and novelty supply plasticity while it does not.

---

## 3. Research question 2 — how does feedback re-weight competing frames?

### 3.1 Prediction error is the teaching signal

Phasic dopamine encodes a reward prediction error and causally drives cue–outcome learning: mimicking a prediction error with optogenetic stimulation produces long-lasting increases in cue-elicited behaviour (Steinberg et al. 2013 [4]); the human RewP is a prediction-error signal that scales with the unexpectedness of feedback and drives subsequent learning (Weber & Bellebaum 2024 [8]). The computational content is a signed update: `δ = outcome − expected`; accepted-better-than-expected → strengthen, rejected-worse-than-expected → weaken.

**Consequence for RAVANA.** `user_understood`/acceptance is the RAVANA outcome and `assoc(cue, frame)` is the expectation. A success should move the association up; a failure should move it *down*, not merely fail to move it. The SCC already exposes the required two-signed operation: `reinforce_construction(frame, success)` uses `+learning_rate` on success and `−0.5·learning_rate` on failure (`syntactic_cell_assembly.py:616–621`). The missing half is a matching negative step on the *cue-specific* association, so a rejected frame loses its hold on that cue.

### 3.2 Competition is opponent, not additive-only

Frank et al. (2004) show with a probabilistic selection task and a biologically based computational model that the basal ganglia contain separate Go and NoGo pathways, differentially modulated by positive and negative reinforcement: reinforcement facilitates the chosen response while negative feedback suppresses the incorrect response, and dopamine medication shifts the balance between the two [5]. This is the mechanism that resolves *competition* between responses rather than merely recording which one won.

**Consequence for RAVANA.** Reinforcing frame *f* under cue *c* must have two observable effects: `assoc(c, f)` rises **and** `assoc(c, f′)` falls for competing frames *f′* under the same cue (lateral suppression). A learning rule that only increments the winner can never reverse a wrong early association; a Go/NoGo-style pair of updates can. This maps cleanly onto the existing `learn_from_feedback(variant, success)` asymmetry (`surface_realizer.py:215`: `+0.05 / −0.025`) — but applied to constructional associations, not surface variants.

### 3.3 Where the cue→frame association lives

Cue validity is a form–function mapping whose strength is learned from reliability and availability [11]; construction-level reliability is at least as good a cue as the verb itself (Goldberg, Casenhiser & Sethuraman 2005, cited in the companion doc), so a construction-level reliability table is the right level of representation. In RAVANA that table must be persisted through `SyntacticCellAssembly.get_state/set_state` (`syntactic_cell_assembly.py:735–768`) — the same blob that already persists `learned_frames`. Keeping the load-bearing association there (rather than in `SurfaceRealizer._variant_weights`, whose persistence is not established) is what makes the learning survive save/load.

---

## 4. Research question 3 — how can selection be adaptive yet deterministic?

Sampling (`random.choice`, `random.choices`, softmax with temperature) would acquire associations but break the reproducibility criterion the companion doc already established (byte-identical logs from identical state). The brain-inspired route to exploration without noise is an **optimism/novelty bonus added to a deterministic choice rule**:

- Kakade & Dayan (2002) model dopaminergic novelty responses as exploration bonuses — fictitious value added to an option's reward/value that encourages exploration — and note that optimistic initial values are a principled way to make an agent try each option before settling [6];
- Wittmann et al. (2008) show human choice is best captured by a model assigning higher initial value to novel options, with the effect tracked in striatum [7];
- Bybee's type-frequency result gives the linguistic reason the same mechanism is needed: without varied use, a high-token-frequency pairing is conserved and stops generalizing [2].

**The deterministic rule.** For admissible frames *f* under cue *c* and word *w*:

```
score(c, f, w) = log(availability(f, w))            # learned_frames[f].weight + deployments
               + β · assoc(c, f)                    # signed, learned collexeme-style association
               + γ · types(f)                       # type frequency -> productivity / generalization
               + η · bonus(f)                       # optimism ∝ time-or-trials since f last used
               − λ · recency(f, w)                  # demoted tie-break, NOT the driver
f* = argmax_f score ; ties broken by stable relation_type string   # NO RNG
```

`bonus(f)` decreases as *f* is used under this cue (its optimism is spent), so on a tied/novel cue the least-recently-attempted admissible frame wins first, is then confirmed or disconfirmed by feedback, and the association table converges. Because every term is a stored number and `argmax` has a stable tie-break, the same persisted state plus the same feedback history yields the same frame every time. `λ` is chosen below the smallest positive base-score gap so the cue term dominates fatigue, exactly as in CVCC — fatigue remains a tie-break, never the selector.

**What this is not:** it is not `epsilon`-greedy (no sampling), not a learned stochastic policy, and not the claim that the brain literally computes this score. The brain evidence supports the *components*: association learning [1], entrenchment [2], and type generalization [9]. Signed prediction-error updating [4] and opponent Go/NoGo competition [5] are likewise supported. Novelty-driven exploration is supported too [6][7]. The algebraic form and the gains `β, γ, η, λ` are engineering choices to sweep.

---

## 5. Observable criteria: learned usage-based selection vs a fixed cue mapping

Five machine-checkable discriminators. Each must be logged from the emitted string **and** from `(cue, selected_frame_id, assoc, types)`. Report counts with denominators (trial × cue × subject); build the gold set from a source the mechanism cannot see, not in the same pass that chose it. A fixed cue mapping is a step function from trial 1; the discriminators below are all experience-dependent.

| # | Criterion | Operational test | Fixed cue mapping fails because | Evidence |
|---|---|---|---|---|
| **U1** | **Experience-dependent association curve.** | From identical blank persisted state, issue cue *c* N times, always accepting the frame, and log `assoc(c,f)` and P(f\|c) each trial. Learned: monotonic rise with saturation; a previously-untrained competitor's score falls (opponent update). | A literal `if/elif` returns the same frame and has no `assoc` value to rise — flat at step 1/0 from trial 1. | [1][11][2] |
| **U2** | **Feedback reversal (NoGo).** | Entrench frame *f* under cue *c* with ~5 successes, then deliver ~3 rejections, then re-issue *c*. Learned: selection switches to the next-ranked admissible frame within 1–2 cued turns and `assoc(c,f)` decreases on rejection. | Fixed mapping is invariant to feedback by construction; it never stores a negative association. | [5][4][8] |
| **U3** | **Transfer across words grows with type frequency.** | Train cue *c*→*f* on one taught word A, test on novel word B; then repeat after *f* has been accepted under words A, C, D, E. Learned: transfer to B strengthens as `types(f)` grows. | Fixed mapping transfers identically regardless of experience — its transfer has no dependence on type count. | [12][2][9] |
| **U4** | **No freezing under uniform evidence, yet reproducible.** | (a) two runs from identical persisted state and feedback history → byte-identical frame logs; (b) under uniform/no differential feedback, 20 repetitions of cue *c* cycle among admissible frames rather than locking to the first; (c) after entrenching *f*, flip the reward contingency and confirm a switch within k trials. | A fixed mapping passes (a) trivially but fails (b) — it locks to one frame forever — and (c) because it cannot be reversed. A `random.choice` fix passes (b) but fails (a). Only a deterministic rule with an optimism/novelty term plus signed updating passes all three. | [6][7][2] |
| **U5** | **Suppression of untrained competitors.** | Measure P(f′\|c) for a competitor before and after reinforcing *f* under *c*. Learned: P(f′\|c) drops without f′ ever being directly trained. | Fixed mapping has P(f′\|c)=0 always, so there is no change to observe; a boost-only rule leaves competitors' latent scores untouched. | [5][1] |

**Verdict on the current implementation.** U1 fails (no association stored), U2 fails (no feedback path), U3 fails (no type count), U4 passes only reproducibility and fails no-freezing, U5 fails (no competitors). The system has **zero learned cue sensitivity**.

---

## 6. Recommended mechanism — UBCAE, implemented in RAVANA's existing inventory

**Constraints:** no new neural component, no retraining, no RNG, no weakening of `_sm_response_grounded` / `_is_word_salad` / the echo gate, no change to the intent/performative normalizers. Every operation below is an existing method; the only genuinely new storage is one association dictionary, persisted through the existing SCC state blob.

### S0 — cue detection stays; the decision ladder goes

Keep `extract_construction_cue` (`response_gen.py:205`) as a **feature extractor** returning the cue label. Delete its use as a selector: the `if/elif` ladder at `response_gen.py:7590–7611` becomes a single call to a selector that receives `(ctx.raw_input, subject, definition_bundle)`. The cue is information-state input to a competition, not a branch.

### S1 — admissible frame set (unchanged from CVCC)

From the definition bundle plus the subject's graph neighbours, build grounded candidates tagged `functional`, `part_of`, `located_in`, `has_property`, and the appositive naming frame; veto `is_a`/`isa` and any candidate whose object is the definition string. If the cued relation has no graph-supported target, decline to compose rather than substitute another frame — cue sensitivity must not become confabulation. Reuse `find_construction` / `bind_to_sentence` (`syntactic_cell_assembly.py:581`, `:207`) and `VerbLexicon.select_verb(relation_type)`.

### S2 — the learned association table (new state, existing persistence)

Add to `SyntacticCellAssembly` a table `cue_frame_assoc: Dict[(cue, relation_type), (a, b, n, last_success)]`, persisted as a new key in `get_state/set_state` (`syntactic_cell_assembly.py:735–768`) alongside `learned_frames`. Define, per cue *c* and frame *f*:

```
assoc(c, f) = log( (a[c,f] + prior) / (expected[a | b, n] + prior) )   # signed collexeme-style association [1]
```

- `a[c,f]` = accepted uses of *f* under cue *c*; `b` = uses of *f* under other cues; `n` = uses of cue *c* under other frames — the 2×2 contingency of collostructional analysis [1].
- `prior > 0` implements optimistic initialization: unexplored pairs start attractive and can be tried [6][7]; the optimism decays as `a` and `n` accumulate.
- availability `A(f,w)` remains `learned_frames[f].weight` (`:40`) plus a deployments count, both already persisted.

### S3 — score and select (the load-bearing step)

Use the score in §4. `types(f)` = distinct `(word, cue)` pairs accepted under *f* (type frequency → productivity [2][9]). `bonus(f) = η / (1 + deployments_since_last_use[f])`, the deterministic novelty term [6][7]. `recency(f,w)` reuses `SurfaceRealizer._recent_variants` (`surface_realizer.py:182`, `:618`) demoted by `λ` below the minimum base gap. Select `f* = argmax` with a stable `relation_type` tie-break; **do not call `random.choice`**. Use `MERGE_VARIANTS` only *after* the relation is fixed, to mark cued focus (`cleft`/`pseudo_cleft`) or given subject (`as_for_topic`).

### S4 — feedback update (signed, opponent, prediction-error-driven)

Define acceptance from an observable signal — no repair turn / no explicit correction / not rejected by the echo or grounding gate. On **accept**:

- `allocate_construction(frame, weight)` or `reinforce_construction(frame, True)` (`syntactic_cell_assembly.py:591`, `:616`) — +`learning_rate` on the frame's availability (entrenchment [2]);
- `SurfaceRealizer.learn_from_feedback(variant, True)` (`surface_realizer.py:213`) — +0.05 on the surface preference;
- increment `a[c,f]` and `deployments`; increment `types(f)` if this `(word, cue)` is new (type frequency [2][9]).

On **reject**:

- `reinforce_construction(frame, False)` — −0.5·`learning_rate` on availability;
- `SurfaceRealizer.learn_from_feedback(variant, False)` — −0.025;
- decrement `assoc(c,f)` (negative prediction error [4][8]) and apply a smaller positive increment to the competing frames under *c* (NoGo on the chosen, relative release of alternatives [5]); then **relax the choice and try the next-ranked admissible frame** — never relax the grounding criterion.

### S5 — what NOT to do

- Do not leave the `if cue == FUNCTION → frame` ladder as the selector; it is the defect.
- Do not use `random.choice`/`random.choices` or an unseeded softmax; it breaks U4(a).
- Do not treat `MERGE_VARIANTS` / `_variant_weights` as the construction inventory — they are surface realization, not relation choice.
- Do not make the association a per-frame scalar; it must be per `(cue, frame)` to express that a frame is good under one cue and poor under another [1].
- Do not implement boost-only learning: rejection must subtract, or U2/U5 can never pass [5].
- Do not invent a target when the cued relation is ungrounded; fail closed.
- Do not persist the load-bearing association in a non-persisted attribute; put it in the SCC state blob (§3.3).

### Falsifiers (measurable predictions)

1. Freeze `assoc(c,f)` at its prior: U1 and U2 collapse. If selection still adapts, learning is leaking in through `learned_frames.weight` rather than the cue association.
2. Remove the novelty/optimism term (`η=0`): U4(b) fails — uniform evidence locks selection to the first frame. Sweep `η` to find the crossover; it is an empirical parameter, not a brain constant.
3. Set the rejection step to zero (boost-only): U2 and U5 fail — a wrong early cue→frame association is never reversed. This is the Frank et al. (2004) NoGo ablation [5].
4. Set `γ` (type-frequency term) to zero and train one word only: U3 fails. Re-enable with many words and confirm transfer grows — Bybee's type-frequency effect [2][9].
5. Set `λ` above the minimum base-score gap: cued criteria U1/U2 still pass but U4(b) reverts to cycling by recency — the signature of the counter-driven system.
6. Clear the persisted `cue_frame_assoc` key on save/load: post-restart selection must regress to the prior-prior distribution. If it does not, the table is not actually being persisted through `get_state/set_state`.

---

## 7. Honest flags

- **No dedicated "cue-association" brain region.** Collostructional analysis is a corpus method [1] and usage-based theory is a linguistic framework [12][3]; the neural results concern domain-general selection and learning, not a dedicated construction module. Dopamine RPE [4] and BG Go/NoGo [5] support the teaching signal, and striatal novelty [7] supports exploration. UBCAE is an extrapolation from those general mechanisms to a symbolic architecture, not a claim that the brain contains this table.
- **Prediction-error evidence comes from reward/action tasks, not syntactic construction choice.** Steinberg et al. (2013 [4]) and Weber & Bellebaum (2024 [8]) establish the teaching signal in cue–reward and feedback paradigms; using it as a signed update to a frame weight is an analogy labelled as such.
- **Optimism/novelty is theory plus striatal correlates, and the deterministic instantiation is ours.** Kakade & Dayan (2002 [6]) and Wittmann et al. (2008 [7]) support an exploration bonus; a noiseless `argmax + bonus` (rather than stochastic softmax) is chosen to preserve reproducibility, not read off the brain.
- **There is currently no explicit feedback signal for productive-use turns.** `user_understood` must be defined from an observable proxy (repair, re-ask, explicit correction, echo-gate rejection). Without such a signal the association cannot be learned at all — this is a real prerequisite, not a detail. `[unverified]` that any existing RAVANA field already carries it for this path.
- **Type frequency is not universally dominant.** Some L2 studies find token frequency outweighs type frequency in production, so `γ` must be swept, not assumed [9].
- **No direct human experiment matches "learn a cue→construction association from a one-turn-taught word."** U1–U5 are designed diagnostics for this system, not measured human norms.

---

## References (full citations)

**Peer-reviewed journal articles**

- **Stefanowitsch, A., & Gries, S. T.** (2003). Collostructions: Investigating the interaction of words and constructions. *International Journal of Corpus Linguistics*, 8(2), 209–243. doi:10.1075/ijcl.8.2.03ste
- **MacWhinney, B., Bates, E., & Kliegl, R.** (1984). Cue validity and sentence interpretation in English, German, and Italian. *Journal of Verbal Learning and Verbal Behavior*, 23(2), 127–150. doi:10.1016/S0022-5371(84)90093-8
- **Bybee, J. L.** (2006). From usage to grammar: The mind's response to repetition. *Language*, 82(4), 711–733. doi:10.1353/lan.2006.0186
- **Steinberg, E., Keiflin, R., Boivin, J. R., Witten, I. B., Deisseroth, K., & Janak, P. H.** (2013). A causal link between prediction errors, dopamine neurons and learning. *Nature Neuroscience*, 16, 966–973. doi:10.1038/nn.3413
- **Frank, M. J., Seeberger, L. C., & O'Reilly, R. C.** (2004). By carrot or by stick: Cognitive reinforcement learning in Parkinsonism. *Science*, 306(5703), 1940–1943. doi:10.1126/science.1102941
- **Kakade, S., & Dayan, P.** (2002). Dopamine: Generalization and bonuses. *Neural Networks*, 15(4–6), 549–559. doi:10.1016/S0893-6080(02)00048-5
- **Wittmann, B. C., Daw, N. D., Seymour, B., & Dolan, R. J.** (2008). Striatal activity underlies novelty-based choice in humans. *Neuron*, 58(6), 967–973. doi:10.1016/j.neuron.2008.04.027
- **Weber, C., & Bellebaum, C.** (2024). Prediction-error-dependent processing of immediate and delayed positive feedback. *Scientific Reports*, 14, 9674. doi:10.1038/s41598-024-60328-8
- **Tomasello, M.** *Constructions* (journal article, journal *Constructions*, Heinrich-Heine-Universität Düsseldorf) — item-based schemas / constructional islands.

**Monographs and handbook chapters (theory, not journal articles)**

- **Tomasello, M.** (2003). *Constructing a Language: A Usage-Based Theory of Language Acquisition*. Harvard University Press. (monograph)
- **Gries, S. Th.** New advances in statistical analysis for usage-based linguistics. In *Handbook of Usage-Based Linguistics* — token frequency ↔ entrenchment, type frequency ↔ productivity. (handbook chapter)
- **A corpus study on the item-based nature of early grammar acquisition** — *Colorado Research in Linguistics* (item-based units / constructional islands). (journal article)

*Goldberg, Casenhiser & Sethuraman (2005, Journal of Child Language 32(2), 407–426) is cited here through the companion doc's full reference and is not re-registered in this ledger.*

---

## Sources

[1] https://stgries.info/research/2003_AS-STG_Collostructions_IJCL.pdf
[2] http://ling.unm.edu/assets/documents/bybee2006fromusage.pdf
[3] https://constructions.journals.hhu.de/article/download/452/469/1808
[4] https://www.nature.com/articles/nn.3413
[5] https://www.science.org/doi/10.1126/science.1102941
[6] https://www.gatsby.ucl.ac.uk/~dayan/papers/kakday02.pdf
[7] https://pmc.ncbi.nlm.nih.gov/articles/PMC2535823
[8] https://www.nature.com/articles/s41598-024-60328-8
[9] https://www.stgries.info/research/2023_STG_NewAdvStatAnals_HdbkUsgBased.pdf
[10] https://journals.colorado.edu/index.php/cril/article/download/241/221/501
[11] https://psyling.talkbank.org/years/1984/cueval.pdf
[12] https://www.hup.harvard.edu/books/9780674044395
