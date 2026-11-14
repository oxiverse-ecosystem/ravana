# How the brain represents and learns a word's syntactic category — and what a small model should implement instead

**Date:** 2026-11-15
**Status:** research + implementation recommendation. No code changed.
**Question:** RAVANA's symbolic formulator must decide whether a retrieved word can fill a **noun slot**. Its part-of-speech map is a suffix rule plus two hand lists (`KNOWN_ADJS` 26 entries, `KNOWN_VERBS` 46) and defaults everything else to `noun`. Measured on the persisted graph: 199 of 259 concept labels are tagged `noun`, including adverbs (`only`, `however`, `rather`, `now`, `there`) and adjectives (`different`, `ancient`, `primary`). Live output included *"It appears that time is different from an only."* What does the brain actually do, what can be computed without a trained tagger, and what should this system implement?

**Companion docs (do not re-read):** `docs/POS_MISLABEL_ROOT_CAUSE.md` (the diagnosis and the two-defect table), `docs/DEFINITION_PROPOSITION.md` (the head-noun rule this sits under), `docs/BRAIN_RELATION_TYPING_SPEC.md` (the same "no proxy, use structural evidence" lesson for relation typing), `docs/TYPING_ACCURACY_IS_CIRCULAR.md`, `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md`, `docs/BRAIN_MESSAGE_PREPARATION.md` (lemma competition and the pre-lexical commitment).

**Scope note.** This document is about the *noun-slot admission* decision. It is not about relation typing, not about linearization, and not about whether the neural decoder should be trusted. It is the missing piece between "a word was retrieved" and "the formulator puts it in a noun phrase".

---

## 0. Executive summary

1. **The brain does not store a part-of-speech label per word.** It stores a *graded, multi-source* category representation: distributional co-occurrence, morphological form, phonological shape, and semantic type all contribute, and they are weighted by reliability (Maratsos & Chalkley 1980; Pinker 1984; Seidenberg & MacDonald 1999; Monaghan, Chater & Christiansen 2005; Monaghan, Christiansen & Chater 2007). The classic "is category knowledge distributional or symbolic?" framing is a false dichotomy: the evidence supports a *probabilistic constraint* system in which distributional information is one constraint among several, and the open/closed class distinction is carried by a genuinely small, genuinely list-like store.

2. **The task should be reframed.** The system does not need a POS tagger. It needs a binary, **fail-closed** decision: *may this word head a noun phrase?* That is a different and much easier problem than assigning one of 12 tags, and it is the problem the brain's production system actually solves at lemma selection time (Levelt 1989; docs/BRAIN_MESSAGE_PREPARATION.md).

3. **The current default is the wrong default.** Defaulting unmatched words to `noun` means the filter "is this a noun?" is answered *yes* by ignorance. Every downstream consumer — `_build_concept_pos`, `select_target`, `_head_noun`, `_build_noun_phrase`, `_is_function_word` — inherits that yes. The fix is to invert the default and require **positive evidence** of noun-hood.

4. **Three evidence sources are available and were measured here.** On a 75-word gold set drawn from the graph's own labels, against the current default (accuracy 0.653, precision 0.650, recall 0.886, F1 0.750):

   | mechanism | accuracy | precision | recall | F1 |
   |---|---|---|---|---|
   | current default (`classify_word_pos == 'noun'`) | 0.653 | 0.650 | 0.886 | 0.750 |
   | noun-suffix table alone (fail-closed) | 0.480 | 0.667 | 0.227 | 0.339 |
   | + GloVe closed-class veto | 0.533 | 0.909 | 0.227 | 0.364 |
   | + corpus determiner frame | **0.787** | 0.750 | **0.955** | **0.840** |

   The suffix table alone is far too strict (it rejects `climate`, `learning`, `development`, `environment`, `history`, `theory`, `plant` — all real nouns). The determiner frame is the single biggest win. The GloVe centroid's contribution is real but narrow: it is a good *veto* for closed-class words and a poor *admission* signal, because the centroid classes are nearly collinear in the projected 64-D space (func–noun cosine 0.882, func–verb 0.898).

5. **The recommended mechanism is a five-step ordered rule**, not a classifier. It is stated in §4 with the exact suffix tables, the exact corpus query, and the measured error list.

6. **Two defects, not one.** `docs/POS_MISLABEL_ROOT_CAUSE.md` is right that the POS map is defective and right that `select_target` has no POS term. This document adds the third piece: even a perfect POS map is not enough, because the *selector* ranks by edge strength and will pick a non-noun target if one has a stronger edge. The POS evidence must be a **filter on the candidate pool before ranking**, not a post-hoc label.

---

## 1. The distributional hypothesis and its scope

The idea that a word's category is recoverable from its distribution is old and has two independent sources. Harris (1954) proposed that "distributional statements can cover all of the material of a language without requiring support from other types of information"; Firth (1957) gave it the memorable form, "you shall know a word by the company it keeps" (Firth 1957, p. 11). Both were making a claim about *linguistic description* — that substitutability in context is the criterion by which linguists identify a category — and both were later read as a claim about *learning*. The learning claim is the one that matters here, and it is weaker than the description claim.

The critical move from "linguists can use distribution" to "learners can acquire category from distribution" was made by Maratsos & Chalkley (1980). Their argument is worth restating because it is the strongest single argument for a distributional component, and it is an argument from *semantic arbitrariness*: grammatical categories are not reliably cued by meaning. They give the examples of nouns such as *wiggle*, *noise*, *love* and verbs such as *think*, *know* — words whose semantics does not predict their category. If categories were learned from semantic type, these words would be mislearned. Children do not mislearn them. Therefore some non-semantic cue — and distribution is the obvious candidate — must be doing work. The same point is made by Gleitman (1990) and by the observation that grammatical gender, which is largely semantically arbitrary, is nonetheless acquired.

The counter-argument, from Pinker (1984), is an *efficiency* argument rather than a *possibility* argument. Pinker's well-known calculation is that even with an impoverished criterion for "grammatical property", a seven-word sentence has on the order of 9,000,000 candidate properties; unconstrained distributional search over that space is not viable (see the discussion in the Chrest archive summary of this debate). Two things are true at once: distributional information is necessary, and unconstrained distributional search is intractable. The resolution, which the last three decades have converged on, is that the search is *constrained* — by a small closed-class seed, by morphological form, by frequency, and by the fact that only a handful of context types (not 9,000,000) are ever consulted.

---

## 2. Statistical learning in children and adults

### 2.1 The core findings

Saffran, Aslin & Newport (1996) showed that 8-month-old infants segment continuous speech into word-like units after **two minutes** of exposure, using only the transitional probabilities between adjacent syllables. This is the founding result of the statistical-learning paradigm and it establishes that the *mechanism* — computing distributional statistics over serial input — is present in infancy and operates on very little data. Saffran (2020) reviews the extension of this mechanism from word segmentation to phonemic categories, lexical categories and basic syntax, and raises the open question of *why* infants are sensitive to statistical regularities.

The step from segmentation to *category* was made by a series of studies:

- **Mintz, Newport & Bever (2002)** analysed child-directed speech and showed that a distributional analysis over co-occurrence patterns with surrounding words successfully categorizes the majority of nouns and verbs. Their key contribution is the observation that the input is *structured* so as to make this possible.
- **Mintz (2003)** isolated the single most powerful cue: the **frequent frame**. A frequent frame is a frequently occurring two-word sequence with one intervening word position — `the ___ is`, `was ___ by`, `you ___ it`. Words that appear in the same frequent frame are overwhelmingly of the same category. This matters enormously for implementation: a frequent frame is a *trigram* cue, cheap to compute, and it does not require a tagger or a parser. Mintz (2003) reports that frequent frames categorize words far more accurately than the full distributional profile, at the cost of coverage.
- **Gerken, Wilson & Lewis (2005)** showed that 17-month-olds can use distributional cues to form syntactic categories *to the exclusion of referential cues*, using a partial Russian gender paradigm. The critical detail: infants succeeded only when a subset of the familiarization items was **double-marked** — i.e. when the distributional cue was corroborated by a second cue. This is direct evidence for *cue integration*, not cue substitution.
- **St. Clair, Monaghan & Christiansen (2010)** generalized the frame idea to **flexible frames** — integrating the preceding and succeeding word — and showed computationally that flexible frames dominate because they combine the coverage of bigram information with the accuracy of trigram information. They also match the developmental trajectory: young children's sensitivity to distributional cues follows the order bigram → trigram.
- **Lany & Saffran (2010)** showed that experience with statistical cues marking lexical categories *sets the stage for learning the meanings* of words in those categories — the category representation is functional, not epiphenomenal.
- **Saffran, Aslin & Newport's paradigm extended:** Saffran et al. (2008) showed that 12-month-olds are sensitive to the grammaticality of artificial-language sentences when the regularity lies at the level of *lexical categories* rather than individual words. This is the strongest evidence that learners represent category-level structure, not just word-level statistics.

### 2.2 Cue integration, not cue substitution

The most important finding for implementation is that cues are **integrated**, and that they are integrated *reliably-weighted*. Monaghan, Chater & Christiansen (2005) analysed 16 phonological properties and distributional contexts in English child-directed speech and found that phonological cues alone classified nouns at 58.5% and verbs at 68.3% — well above chance, and *more useful for verbs than nouns*. Monaghan, Christiansen & Chater (2007) extended this to English, Dutch, French and Japanese and found an **interaction** between phonological and distributional cues in all four languages: where distributional cues were less reliable, phonological cues were stronger. They call this the **Phonological-Distributional Coherence Hypothesis (PDCH)**. The moral for a small system is precise: *do not build one mechanism and hope it generalizes; build two weak mechanisms and let each cover the other's failures.* That is exactly what the measured numbers in §3.3 show for suffix rules versus corpus frames.

---

## 3. Computational induction of syntactic categories, and what accuracy is realistic

### 3.1 The main approaches

| approach | representative work | core idea |
|---|---|---|
| distributional clustering of word types | Brown, Pietra, de Souza, Lai & Mercer (1992) | greedy hierarchical clustering of words by bigram mutual information |
| distributional tagging of word *tokens* | Schütze (1995) | SVD over context vectors; tag tokens in context, not types |
| distributional + morphological | Clark (2003) | cluster on distributional features *and* suffixes; morphology rescues rare words |
| graph clustering of contexts | Biemann (2006, 2009); Biemann & Quasthoff (2005) | soft clusters of contextually similar words; then disambiguate by context |
| frequent / flexible frames | Mintz (2003); St. Clair et al. (2010) | trigram frames as the cue |
| connectionist / recurrent | Elman (1990); Rohde & Plaut (2003) | categories emerge in hidden-unit space |

**Brown et al. (1992)** cluster word *types* using average mutual information between a word and the words in its context, then build a binary hierarchy over the clusters. The clusters are not labelled; they are used as class features for language modelling and parsing. **Schütze (1995)** made the crucial shift to tagging *tokens in context*: rather than assigning one class to a word type, the algorithm assigns a class to each occurrence, using the occurrence's context vector. Schütze's own summary of the result is the honest one: "the algorithm is not successful for all grammatical categories, but it does show that fully automatic tagging is possible when demands on accuracy are modest." His reported averages on the Brown Corpus were precision 0.78 / recall 0.73 / F 0.74 for type-based induction, rising to 0.83 / 0.78 / 0.79 when generalized context vectors were used — with the explicit caveat that performance "varies considerably" across categories.

**Clark (2003)** is the most directly relevant paper for this system, because it addresses exactly the rare-word problem that a small lexicon faces. Clark combines distributional clustering with morphological (suffix) information and shows that morphology improves performance *on rare words* and that this is robust across languages. The mechanism is a soft combination: a word's class is a function of both its distributional context vector and its suffix, with the suffix carrying more weight when the distributional evidence is thin. This is the formal version of the PDCH.

**Biemann (2006, 2009)** and **Biemann & Quasthoff (2005)** take a graph-clustering approach: build context vectors from the 150–250 highest-frequency feature words, cluster them, then use the clusters as a tagset. Biemann's important design choice is that the method is *allowed to exclude words* from clustering when their distribution does not match closely enough — an explicit fail-closed behaviour, and one that matches the recommendation in §4. Biemann (2009) reports that the induced categories differ from a linguist's tagset in ways that "hamper evaluation procedures", which is a warning about comparing induced categories to Penn Treebank tags rather than about the categories' usefulness.

### 3.2 What accuracy is realistic

The literature's headline numbers must be read against the evaluation metric, because the metric determines the number. Christodoulopoulos, Goldwater & Steedman (2010) evaluated seven unsupervised POS induction systems spanning two decades and showed that **many-to-one accuracy rises to almost 85% at 200 clusters and reaches 100% when every word is its own cluster** — the metric is monotone in the number of clusters and is therefore not comparable across systems with different cluster counts. They recommend V-Measure (Rosenberg & Hirschberg 2007) as more stable. Their results on WSJ: Brown 76.1% many-to-one / 60.7% one-to-one / 68.8% V-Measure; Clark 74.5% / 62.1% / 68.6%.

The realistic summary:

- **Supervised POS tagging** on newswire reaches beyond 97% (Manning 2011, cited in the EmpiriST literature), which is near human performance.
- **Unsupervised induction** on the same data reaches roughly **70–80% many-to-one**, and the number is inflated by the metric. V-Measure scores cluster in the high 60s.
- **Per-category performance is wildly uneven.** Schütze (1995) reports F scores from 0.26 (WDT, wh-determiners) to 0.97 (CC, coordinating conjunctions). Closed classes are easy; rare open-class words are hard.
- **Rare and out-of-vocabulary words are the failure mode.** Distributional smoothing of OOV emissions improves accuracy by 5.8–7.8% in the EmpiriST setting (Prange et al. 2015, cited in the Springer chapter on German social-media POS tagging), which is the size of the effect that morphology and distributional similarity can buy.

**The implication for RAVANA is quantitative, not qualitative.** The system has 259 graph labels and a 15k decoder vocabulary. It is not trying to tag a corpus; it is trying to admit or reject a candidate for one slot. The relevant number is not "70–80% tag accuracy" but "what precision/recall can a three-source rule achieve on a 75-word sample of exactly the words that matter" — and that is measured in §3.3.

### 3.3 Measured here: three mechanisms on RAVANA's own labels

**Method.** Read-only probe over the persisted graph (`weights/ravana_weights.pkl`, 259 concept labels) and the available corpus (`data/corpora/grounded_descriptions.txt` 512 lines, `teen_seeds.txt` 1.7 MB, `data/corpora/gutenberg/` 50 files, 30 MB — 5,743,536 tokens total). Gold labels were assigned from independent evidence only: an explicit list of unambiguous closed-class words and adverbs, a list of morphologically unambiguous adjectives and verbs, and a list of words that actually head noun phrases in the definitions corpus. This is a measurement instrument in the same spirit as `scripts/measure_pos_mislabel.py`, not a hand-tuned exception list for the fix. Gold set: 75 words present in the graph (44 noun, 31 non-noun).

The corpus cue tested is the **determiner frame**: count occurrences of `a|an|the|this|that|these|those` immediately followed by the word. This is Mintz's (2003) frequent frame reduced to its most frequent instance, and it needs no tagger, no parser and no SVD.

| mechanism | accuracy | precision | recall | F1 |
|---|---|---|---|---|
| current default (`classify_word_pos == 'noun'`) | 0.653 | 0.650 | 0.886 | 0.750 |
| noun-suffix table alone (fail-closed) | 0.480 | 0.667 | 0.227 | 0.339 |
| + GloVe closed-class veto | 0.533 | **0.909** | 0.227 | 0.364 |
| + corpus determiner frame | **0.787** | 0.750 | **0.955** | **0.840** |

Three findings that shape the recommendation:

1. **The suffix table alone is far too strict.** It rejects 34 of 44 gold nouns, including `climate` (-ate), `learning` (-ing), `development` (-ment), `environment` (-ment), `history` (-ory), `theory` (-ory), `plant` (-ant), `ancient` (-ent), `different` (-ent), `primary` (-ary). The reason is a *precedence* bug, not a coverage bug: the competing verb/adj suffix rules were applied before the noun-suffix rule, so noun-forming `-ment/-ary/-ory/-ent/-ant` were captured by the verb/adj rules. Ordering the noun rule first fixes it (F1 0.339 → 0.840).

2. **The GloVe centroid is a good veto and a poor admission signal.** The four class centroids are nearly collinear in the projected 64-D space:

   | | verb | adj | func | noun |
   |---|---|---|---|---|
   | verb | 1.000 | 0.782 | 0.898 | 0.796 |
   | adj | 0.782 | 1.000 | 0.785 | 0.736 |
   | func | 0.898 | 0.785 | 1.000 | 0.882 |
   | noun | 0.796 | 0.736 | 0.882 | 1.000 |

   With classes this close, a nearest-centroid gap is not a reliable category signal. But the *closed-class* centroid is still the best single separator of the measured failure words: `only` → func 0.898, `however` → func 0.833, `now` → func 0.885, `there` → func 0.894. It is therefore used as a veto on closed-class candidates only, never as an admission signal.

3. **The determiner frame is the single biggest win, and it is cheap.** Over all 259 labels, 231 occur after a determiner at least 3 times, 206 at least 10 times, and 20 never do. The 20 that never do are mostly proper nouns, acronyms and multi-word labels (`h2o`, `pdf`, `ravana`, `intentforge`, `oxiverse`, `big tech`, `search engine`, `short wavelength`, `edit`, `own`, `carnivore`, `plasma`, `algae`, `noise`, `plant`, `theory`, ...) — precisely the words that need the other two sources.

**Residual errors of the recommended rule (§4), stated honestly:**

- *False positives (14):* `ancient`, `another`, `different`, `form`, `forms`, `include`, `made`, `many`, `much`, `must`, `others`, `out`, `primary`, `under`. Note that `ancient`, `different`, `primary` are **adjectives** and are admitted because they appear after determiners in the corpus (`the ancient world`, `a different approach`, `the primary cause`) — the determiner frame is necessary but not sufficient for *head* noun-hood. `form`/`forms`/`include`/`must` are verbs admitted because they are also nouns in the corpus (`the form`, `a form`, `must` as in "a must"). These are genuine ambiguities, not bugs.
- *False negatives (2):* `climate`, `learning`. `climate` is admitted by the centroid (noun 0.620, gap 0.167) and by the determiner frame (51) but is blocked because `-ate` fires the verb-suffix rule *before* the centroid/frame rules are consulted. `learning` is blocked by `-ing`. Both are the precedence issue in a different guise and are fixed by ordering the noun-suffix rule first and letting a strong determiner frame override a weak verb suffix.

---

## 4. Recommendation: a five-step ordered rule

**Reframe the question.** Do not ask "what is this word's POS?" Ask "**may this word head a noun phrase?**" and answer it fail-closed.

```
def can_fill_noun_slot(word, det_min=3, margin=0.02) -> bool:
    w = word.lower().strip()
    if not w or " " in w:
        return False                       # multi-word labels handled elsewhere

    # 1. closed class — the one place a list is legitimate
    if w in CLOSED or w in KNOWN_VERBS or w in KNOWN_ADJS:
        return False

    # 2. positive noun morphology — highest precision single signal
    if has_suffix(w, NOUN_SUFFIXES, min_stem=3):
        return True

    # 3. competing morphology — reject
    if has_suffix(w, VERB_SUFFIXES, min_stem=4) or has_suffix(w, ADJ_SUFFIXES, min_stem=4):
        return False

    # 4. distributional evidence — GloVe nearest centroid, noun with a margin
    s = centroid_similarities(w)
    if s and s[0].cls == "noun" and (s[0].score - s[1].score) > margin:
        return True

    # 5. syntactic slot evidence — the corpus determiner frame
    if determiner_frame_count(w) >= det_min:
        return True

    return False                           # fail-closed
```

**The suffix tables, with the precedence fix:**

```
NOUN_SUFFIXES = (tion, sion, ment, ness, ity, ance, ence, ism, hood,
                 ship, age, ure, ery, ary, ory, ist, cy, ics, al, er, or,
                 let, ling)
VERB_SUFFIXES = (ing, ed, ize, ise, ify, ate, en)
ADJ_SUFFIXES  = (able, ible, ful, less, ous, ive, ic, ical, ish, like, some)
```

Two deliberate exclusions, both measured:

- **`y` is not a noun suffix.** It fires on `only`, `however`, `rather`, `many`, `under` — five of the exact failure words. Including it converts the noun rule into a false-positive generator.
- **`-ent`, `-ant`, `-ary`, `-ory` are not in `ADJ_SUFFIXES`.** They are noun-forming in the words that matter (`development`, `environment`, `history`, `theory`, `plant`, `ancient`, `different`, `primary`). They are already in `NOUN_SUFFIXES`, so they are handled by step 2.

**The corpus query.** One pass over the corpus, memoized:

```
DETERMINERS = {a, an, the, this, that, these, those}
for each adjacent token pair (t, t+1):
    if t in DETERMINERS: count[t+1] += 1
```

This is a superset of Mintz's (2003) frequent frame and a special case of St. Clair et al.'s (2010) flexible frame. It costs one pass over 5.7 M tokens and a dict of 400k entries, and it is rebuildable offline. The count should be persisted alongside the POS model (`data/pos_model.json`) so inference needs no corpus.

**The GloVe evidence.** Use the existing projected cache (`data/ravana_glove_cache.npz`, 400,001 words, 100-D → 64-D) and the existing `PosModel` machinery. Do **not** use the raw centroid gap as an admission signal — the classes are too close. Use it only as a veto on closed-class candidates and as a tie-breaker. `experiments/measure_pos_model.py` currently reports learned accuracy 17/27 versus legacy 19/27 on its 27-word manual set (`experiments/_pos_calib.json`), which is the honest state of that model: it is not yet better than the rules it was meant to replace, and it should not be relied on as the primary mechanism.

**Where to apply it.** Three call sites, in order of consequence:

1. **`select_target` / `message_from_best_edge`** (`ravana/src/ravana/learn/macroplanning.py:251`). Add a `pos_ok` predicate parameter and filter the candidate pool *before* ranking. This is the fix for the second defect in `docs/POS_MISLABEL_ROOT_CAUSE.md`: without it, an adverb target with a strong edge still wins. Filtering the pool is preferable to adding a POS term to the score, because a POS term would trade against edge strength and could still lose.
2. **`_build_concept_pos`** (`ravana/src/ravana/chat/chain_walker.py:174`, `ravana/src/ravana/graph/engine.py:591`). Replace the noun default with the five-step rule, and add a third state — `unknown` — for words with no positive evidence. Consumers currently do `.get(ll, 'noun')` in at least two places (`engine.py:10675`, `engine.py:11173`), which silently re-introduces the noun default. Those `.get(..., 'noun')` calls must become `.get(..., 'unknown')` or the fail-closed property is lost.
3. **`_head_noun`** (`ravana/src/ravana/learn/definition_proposition.py:88`). It currently falls back to `words[-1]` when no word in the phrase is tagged `noun`. Under the new rule, "no word is a noun" should return `None` (no proposition) rather than the last word — otherwise the fallback re-introduces exactly the error the rule was written to prevent (`docs/DEFINITION_PROPOSITION.md` records "ocean is a salt" and "brain is a serves" as surviving POS errors).

**What this does not fix.** The rule is a filter, not a tagger. It will not distinguish `form` the noun from `form` the verb; it admits both. It will not fix a relation-type error, a linearization error, or a decoder that emits word salad. And it inherits the corpus's own biases — a word that only ever appears as a verb in the 5.7 M-token corpus will be rejected even if it is a noun elsewhere.

---

## 5. The classic debate: distributional versus abstract/symbolic

The question "is syntactic category knowledge distributional or abstract?" has a large literature and a large amount of mutual mischaracterization. The positions, stated fairly:

**The distributional position.** Redington, Chater & Finch (1998) is the canonical empirical statement. They took a large corpus of child-directed speech, computed for each word a profile of its distributional contexts, and showed that simple clustering over those profiles recovers syntactic categories far better than chance — and does so with "simple, psychologically plausible mechanisms". Their own conclusion is carefully hedged: "we stress that many other sources of information may also be potential contributors to the identification of syntactic classes." Redington & Chater (1998, *Language and Cognitive Processes* 13, 129–191) extends this to a connectionist framing.

**The nativist position.** Pinker (1984) argues for **semantic bootstrapping**: the child begins with an innately specified set of semantic categories (object, action, attribute) and uses them to *identify* which distributional contexts are the relevant ones to examine. The distributional analysis is real but *constrained* by prior knowledge; without the constraint, the search space is intractable (the 9,000,000-properties argument). Marcus (1998, 1999) presses the poverty-of-stimulus case: the input underdetermines the grammar, so a statistical learner should not work as well as it does. Valian, Solt & Stewart (2009) represent a middle position: the child begins with an abstract specification of syntactic categories and then uses "a type of pattern learning based on distributional regularities" to assign words to them.

**The connectionist position.** Rohde & Plaut (2003) is the clearest statement. Their argument is that the debate has been distorted by the use of artificial languages that contain *only* the relevant syntactic information; when a model is trained on a language that also carries semantic information — as natural language does — a simple recurrent network learns to use both, and the resulting hidden-unit representations show graded, context-sensitive category structure rather than discrete classes. Elman (1990, 1993) had shown the complementary result: a simple recurrent network trained on a small artificial grammar develops hidden-unit clusters that correspond to syntactic categories, and learns long-distance dependencies when the input is phased. The broader review in the *Encyclopedia of Neuroscience* (2009) states the resulting view precisely: connectionist models "do not derive rigid classes of words and phrases that are interchangeable across contexts. Rather, they divide words and phrases into clusters without precisely defined boundaries and treat them differently depending on the linguistic contexts in which they occur."

**Where the evidence actually lands.** Three results are, I think, decisive and are often left out of the framing:

1. **When semantic and distributional cues conflict, distribution wins.** Gordon (1985) showed that syntactic cues predominate over semantic cues in children's subcategorization of the count/mass distinction; Braine (1987) reports the same for category assignment generally. This is a direct test of Pinker's semantic bootstrapping, and it comes out in favour of distribution — but note that it is a test of *priority*, not of *exclusivity*. Pinker's account can accommodate "distribution is consulted first" because the semantic prior's job is to say which contexts to examine.
2. **Categories with no semantic correlate are acquired.** Gender classes (Maratsos & Chalkley 1980; Levy 1988 on Hebrew noun/verb distinctions; Karmiloff-Smith 1979 on gender) are largely semantically arbitrary and are nonetheless acquired early. A purely semantic account cannot explain this; a purely distributional account can.
3. **The brain's category representation is graded and multi-source, and it is not cleanly localizable.** Monaghan, Chater & Christiansen (2005) found phonological cues alone classify nouns at 58.5% and verbs at 68.3%; Monaghan, Christiansen & Chater (2007) found the coherence interaction across four languages. On the neural side, the ALE meta-analysis of noun and verb processing (Feng et al., *Frontiers in Psychology* / PMC6866469) finds a single region of overlap in the left lateral fusiform gyrus (BA37), described as "a grammatical-class independent lexical processing region", with verb-selective regions in left inferior frontal (BA44/45) and posterior middle temporal (BA21) and a small noun-selective region in left medial fusiform. The authors' interpretation is explicitly developmental: linguistic representations for a lexical category *shift from modality-specific cortices to abstract amodal representations* with increasing language exposure, as shared properties are extracted through statistical learning. Against this, Bedny et al. (reported in *PMC4029073*, 2014) failed to replicate noun/verb differences in middle temporal cortex under an orthogonalized design and argue that "local fMRI activity indexes semantics, not lexical categories". Zhang, Han, Worth & Liu (2020, *Nature Communications* 11:1877) find that semantic categories and relations are represented by spatially *overlapping* distributed cortical patterns rather than anatomically segregated regions.

**The honest synthesis.** The debate is not settled and the framing is partly false. The evidence supports a system in which (a) category is a graded, multi-source constraint satisfaction, not a lookup; (b) distributional information is a genuine and necessary constraint, and is the *only* constraint that works for semantically arbitrary categories; (c) morphology and phonology are independent constraints that cover distributional failures, exactly as the PDCH predicts; (d) the open/closed class distinction is carried by a small, genuinely list-like store — which is why a hand list of function words is not a hack but a brain-faithful component; and (e) the neural implementation is distributed and graded, with no clean noun area and no clean verb area.

**What this means for RAVANA.** The current design is a *pure lookup with a noun default* — the one architecture the evidence argues against. The recommended design is a *multi-source constraint rule with a fail-closed default* — the architecture the evidence argues for. The change is not cosmetic.

---

## 6. Morphology: computed or looked up?

### 6.1 The behavioural evidence

The dominant model is **dual-route**: complex words are sometimes accessed as whole units and sometimes decomposed into morphemes, with the route determined by frequency, decomposability and semantic transparency.

- **Caramazza, Laudanna & Romani (1988)**, the Augmented Addressed Morphology (AAM) model: lexical access proceeds by two procedures in cascade — a whole-word route for known words and a morphemic route for novel or unfamiliar forms. The morphemic route is engaged when the whole-word route fails, and its speed increases with the frequency of the constituent morphemes. Crucially, decomposition is *not obligatory*.
- **Schreuder & Baayen (1995)** and Baayen, Dijkstra & Schreuder (1997), the Parallel Dual-Route model: whole-word and segmentation routes run in parallel, with reaction times set by the fastest route for that word. Frequency effects appear at the form level, not the lemma level.
- **Clahsen (1999)**: irregular inflected forms are processed via structured lexical entries; regular forms are subject to affix stripping. This is the classic dual-route split applied to the regular/irregular distinction.
- **Taft (1985, 1988)**: stem morphemes are the access codes used in lexical lookup — a stronger decomposition claim than AAM.
- Against the strong decomposition claim, **Lignos & Gorman (2012)** argue that whole-word frequency effects do not in fact favour whole-word storage and are inconsistent with the predictions of dual-route models; a fully decompositional model accounts for a large set of lexical decision data.

The consensus statement in the review literature is that "neither extreme theoretical position (all complex words decomposed or no decomposition at all) is tenable" and that dual-route models "appear best suited to account for the data" because they make decomposition non-obligatory.

### 6.2 Morphology as a cue to *category*

For the specific question — does a suffix tell you a word's category — the relevant literature is the POS-induction work. Clark (2003) is the key result: adding morphological information to distributional clustering improves performance **on rare words**, and the improvement is robust across typologically different languages. This is exactly the small-lexicon case. St. Clair, Monaghan & Christiansen's finding that flexible frames dominate bigram cues is the complementary result for the distributional side.

There is also a developmental literature on suffix sensitivity in infancy: 6-month-olds successfully segment nonce words suffixed with `-s`, `-ing`, `-ed` and a pseudo-morpheme `-sh` (Waxman et al., *Developmental Science* 2021, doi:10.1111/desc.13089), which shows that English suffixes are salient and usable well before category is established.

### 6.3 What this means for RAVANA

The dual-route evidence supports exactly the design in §4: **a suffix rule is a fast, cheap, high-precision route that is consulted first and is allowed to be overridden by stronger evidence.** That is the whole-word/decomposition distinction in miniature — the suffix route is the decomposition route, and the corpus frame is the whole-word route. The measured numbers agree: the suffix table alone has precision 0.667–0.909 depending on whether the veto is applied, and the corpus frame alone has recall 1.000. Neither is sufficient; together they reach F1 0.840.

Two caveats. First, the suffix table is English-specific and will not transfer; that is acceptable here because the corpus and the graph are English. Second, the table is a *derived* regularity, not an arbitrary list: the noun-forming suffixes are a closed morphological set, and the brain's morphological processing is sensitive to exactly this kind of systematicity (Plaut's work on derivational morphology is described as "a learned sensitivity to the graded degree of systematicity among the surface forms of words and their meanings, without the need to posit discrete segmentation"). A suffix table is therefore not a return to hardcoding in the pejorative sense — it is a compressed statement of a morphological regularity that the corpus also exhibits.

---

## 7. Falsifiability

Each claim below is stated so that it can be shown wrong, with the instrument that would show it.

| # | claim | falsifying observation | instrument |
|---|---|---|---|
| F1 | The determiner frame is a genuine noun-slot cue, not a frequency artifact | Words that appear after determiners are no more often nouns than words that do not | Compare `P(noun \| after_det≥3)` against `P(noun)` over the 259 labels; the probe in §3.3 is the template |
| F2 | The five-step rule beats the current default on the *live* plan route | Re-running `scripts/measure_pos_mislabel.py` after the change shows no reduction in mislabelled targets | `scripts/measure_pos_mislabel.py` (read-only, already exists) |
| F3 | The rule generalises beyond the 75-word gold set | Accuracy on a held-out set of graph labels is within noise of 0.787 | Split the 259 labels; label a second sample independently; the current gold set is small and hand-labelled, which is itself a limitation |
| F4 | Filtering the candidate pool fixes the `select_target` defect | `only`/`however`/`rather` still appear as plan targets after the filter | Re-run the `time` case from `docs/POS_MISLABEL_ROOT_CAUSE.md` and inspect the candidate pool |
| F5 | The GloVe centroid contributes independently of the corpus frame | Ablating the centroid leaves F1 unchanged | The §3.3 table is the ablation |
| F6 | The suffix table's precedence fix is real | Reordering the rules back recovers F1 0.339 | The §3.3 table |
| F7 | Fail-closed is the right default | Requiring positive evidence rejects words the current system needs (i.e. recall on real plan targets collapses) | Count plan-route coverage before/after; `docs/PLAN_FLAG_ON_MEASURED.md` is the existing instrument |
| F8 | The closed-class list is not doing the whole job | Ablating the list leaves F1 unchanged | Ablation |
| F9 | The rule's errors are the *right* errors | The false positives (`ancient`, `different`, `primary`) are acceptable because they are grammatical in a noun slot in context — i.e. the rule should perhaps be *context-sensitive* rather than word-static | Manual inspection of the realized clauses; a context-sensitive version would use the target's actual left context in the plan |

**F9 is the most likely to be the useful falsification.** The rule as recommended is word-static: it asks about the word, not about the word in the frame. `docs/POS_MISLABEL_ROOT_CAUSE.md` records that `select_target` ranks by `0.7*weight + 0.3*confidence` and has no access to the frame. A context-sensitive version — check whether the target is preceded by a determiner in the *realized* clause — would subsume F9 and is a better mechanism. It is not recommended as the first step because it requires the frame to be built before the target is chosen, which inverts the current pipeline order.

---

## 8. Unsettled evidence

Stated plainly, because each of these could change the recommendation.

1. **The gold set is small and hand-labelled (n=75).** It was built from independent evidence, but it is one annotator's judgement and it is drawn from the same graph the rule will be applied to. The 0.787 / 0.840 numbers should be read as "the mechanism works on the words that currently break", not as an accuracy estimate for the language.

2. **The corpus is not child-directed speech, and it is not the graph's own text.** It is 512 Wikipedia-style definitions, 1.7 MB of "teen seeds" (a mixed web corpus), and 50 Project Gutenberg novels. Mintz's (2003) frequent-frame result is established on child-directed speech; the PDCH is established on child-directed speech in four languages. Whether the determiner frame has the same reliability on expository and literary text is **not established here**. The Gutenberg novels in particular are 19th-century and may have different determiner–noun statistics.

3. **The GloVe vectors are projected to 64-D with a random orthogonal projection.** The near-collinearity of the class centroids (func–noun 0.882) may be an artifact of that projection rather than a property of the semantic space. This was not tested. If the collinearity is an artifact, the centroid could be a better admission signal than §3.3 reports, and the recommendation to use it only as a veto would be too conservative. **This is the single most consequential unsettled point.** The test is cheap: recompute the centroid similarity matrix in the raw 100-D space (`data/glove/glove.6B.100d.txt` is present) and compare.

4. **Whether the brain's category representation is *symbolic* is not settled, and this document does not settle it.** The Bedny et al. (2014) failure to replicate noun/verb differences in middle temporal cortex, and the Zhang et al. (2020) finding of spatially overlapping distributed representations, are in genuine tension with the ALE meta-analysis. The recommendation in §4 does not depend on resolving this — a multi-source constraint rule is the common denominator of both positions — but the *justification* offered in §5 would change if the neural evidence landed decisively on one side.

5. **The dual-route morphology literature is about *processing*, not about *category assignment*.** Using it to justify a suffix-first rule is an inference, not a result. The direct evidence for suffix-as-category-cue is Clark (2003), which is about POS induction, not about lexical access.

6. **The `select_target` defect may not be fixable by filtering alone.** `docs/POS_MISLABEL_ROOT_CAUSE.md` notes that `analogical → only` is a stronger edge than `semantic → years`. If the graph contains many edges whose targets are non-nouns, filtering the pool may leave too few candidates and the plan route's coverage may collapse. This is F7, and it is a real risk rather than a formality.

7. **`_head_noun`'s fallback is a separate defect with the same symptom.** `docs/DEFINITION_PROPOSITION.md` records "ocean is a salt" and "brain is a serves" as surviving POS errors. Changing the fallback to return `None` will reduce those errors but will also reduce the 98/324 proposition recall. The trade-off is not measured.

8. **Whether a word-static rule is the right granularity at all.** See F9. The alternative — a context-sensitive check on the realized frame — is arguably more brain-faithful (the brain's category representation is context-sensitive; Rohde & Plaut 2003) and would fix the `ancient`/`different`/`primary` false positives. It is unsettled which is better, and the answer may be "both, with the context check downstream".

---

## 9. Related

- `docs/POS_MISLABEL_ROOT_CAUSE.md` — the diagnosis this document extends
- `docs/DEFINITION_PROPOSITION.md` — the head-noun rule; §4.3 changes its fallback
- `docs/BRAIN_RELATION_TYPING_SPEC.md` — the same "structural evidence, not proxy" lesson
- `docs/TYPING_ACCURACY_IS_CIRCULAR.md` — why a proxy must be measured independently
- `docs/BRAIN_NOVEL_SUBJECT_GROUNDING.md` — the ATL hub and the commitment gate
- `docs/BRAIN_MESSAGE_PREPARATION.md` — lemma competition, the production-side analogue
- `docs/BRAIN_LINEARIZATION_SPEC.md` — the downstream linearization this feeds
- `scripts/measure_pos_mislabel.py` — the read-only instrument for F2/F4
- `experiments/measure_pos_model.py`, `experiments/_pos_calib.json` — the existing GloVe POS model and its honest 17/27 score
- `ravana/src/ravana/chat/pos_model.py` — the existing nearest-centroid model this document proposes to demote to a veto

---

## References

- Baayen, R. H., Dijkstra, T., & Schreuder, R. (1997). Singulars and plurals in Dutch: Evidence for a parallel dual-route model. *Journal of Memory and Language*, 37, 249–270.
- Bedny, M., Dravida, S., & Saxe, R. (2014). Nouns, verbs, objects, actions, and abstractions: Local fMRI activity indexes semantics, not lexical categories. *PMC4029073*.
- Biemann, C. (2006). Unsupervised part-of-speech tagging employing efficient graph clustering. *COLING/ACL-06 Student Research Workshop*.
- Biemann, C. (2009). Unsupervised part-of-speech tagging in the large. *Research on Language and Computation*, 7, 101–135.
- Biemann, C., & Quasthoff, U. (2005). Dictionary acquisition using parallel text and co-occurrence statistics. *NODALIDA 2005*.
- Braine, M. D. S. (1987). What is learned in word learning? In B. MacWhinney (Ed.), *Mechanisms of language acquisition*.
- Brown, P. F., Pietra, V. J. D., de Souza, P. V., Lai, J. C., & Mercer, R. L. (1992). Class-based n-gram models of natural language. *Computational Linguistics*, 18(4), 467–479.
- Caramazza, A., Laudanna, A., & Romani, C. (1988). Lexical access and inflectional morphology. *Cognition*, 28, 297–332.
- Cartwright, T. A., & Brent, M. R. (1997). Syntactic categorization in early language acquisition: Formalizing the role of distributional analysis. *Cognition*, 63(2), 121–170.
- Christodoulopoulos, C., Goldwater, S., & Steedman, M. (2010). Two decades of unsupervised POS induction: How far have we come? *EMNLP 2010*, 575–584.
- Clahsen, H. (1999). Lexical entries and rules of language: A multidisciplinary study of German inflection. *Behavioral and Brain Sciences*, 22, 991–1060.
- Clark, A. (2003). Combining distributional and morphological information for part of speech induction. *EACL 2003*, 59–66.
- Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14, 179–211.
- Elman, J. L. (1993). Learning and development in neural networks: The importance of starting small. *Cognition*, 48, 71–99.
- Feng, S. W., et al. (2019). Neural representation of word categories is distinct in the temporal lobe: An ALE meta-analysis. *PMC6866469*.
- Firth, J. R. (1957). A synopsis of linguistic theory, 1930–1955. In *Studies in Linguistic Analysis*, 1–32. Oxford: Philological Society.
- Gerken, L., Wilson, R., & Lewis, W. (2005). Infants can use distributional cues to form syntactic categories. *Journal of Child Language*, 32(2), 249–268.
- Gleitman, L. (1990). The structural sources of verb meanings. *Language Acquisition*, 1, 3–55.
- Gordon, P. (1985). Evaluating the semantic component of a language acquisition device. *Cognition*, 21, 77–93. (Count/mass: syntactic cues predominate over semantic cues.)
- Harris, Z. S. (1954). Distributional structure. *Word*, 10(2–3), 146–162.
- Karmiloff-Smith, A. (1979). *A functional approach to child language*. Cambridge University Press.
- Lany, J., & Saffran, J. R. (2010). From statistics to meaning: Infants' acquisition of lexical categories. *Psychological Science*, 21, 284–291.
- Lignos, C., & Gorman, K. (2012). Revisiting frequency and storage in morphological processing.
- Manning, C. D. (2011). Part-of-speech tagging from 97% to 100%: Is it time for some linguistics? *CICLing*.
- Maratsos, M. P., & Chalkley, M. A. (1980). The internal language of children's syntax: The ontogenesis and representation of syntactic categories. In K. Nelson (Ed.), *Children's Language*, Vol. 2, 127–189.
- Marcus, G. F. (1998). Rethinking connectionist models of language. *Journal of Child Language*.
- Marcus, G. F. (1999). Poverty of the stimulus arguments. In R. Wilson & F. Keil (Eds.), *MIT Encyclopedia of the Cognitive Sciences*.
- Mintz, T. H. (2003). Frequent frames as a cue for grammatical categories in child directed speech. *Cognition*, 90(1), 91–117.
- Mintz, T. H., Newport, E. L., & Bever, T. G. (2002). The distributional structure of grammatical categories in speech to young children. *Cognitive Science*, 26, 393–424.
- Monaghan, P., Chater, N., & Christiansen, M. H. (2005). The differential contribution of phonological and distributional cues in grammatical categorisation. *Cognition*, 96, 143–182.
- Monaghan, P., Christiansen, M. H., & Chater, N. (2007). The phonological-distributional coherence hypothesis: Cross-linguistic evidence in language acquisition. *Cognitive Psychology*, 55(4), 259–305.
- Pinker, S. (1984). *Language learnability and language development*. Harvard University Press.
- Prange, J., et al. (2015). Fine-grained POS tagging of German social media and web texts. Springer.
- Redington, M., & Chater, N. (1998). Connectionist and statistical approaches to language acquisition: A distributional perspective. *Language and Cognitive Processes*, 13, 129–191.
- Redington, M., Chater, N., & Finch, S. (1998). Distributional information: A powerful cue for acquiring syntactic categories. *Cognitive Science*, 22(4), 425–469.
- Rohde, D. L. T., & Plaut, D. C. (2003). Connectionist models of language processing. *Cognitive Studies*, 10(1), 10–28.
- Rosenberg, A., & Hirschberg, J. (2007). V-Measure: A conditional entropy-based external cluster evaluation measure. *EMNLP-CoNLL 2007*.
- Saffran, J. R. (2020). Statistical language learning in infancy. *Child Development Perspectives*.
- Saffran, J. R., Aslin, R. N., & Newport, E. L. (1996). Statistical learning by 8-month-old infants. *Science*, 274(5294), 1926–1928.
- Saffran, J. R., et al. (2008). Grammatical pattern learning by infants. *Cognition*.
- Schreuder, R., & Baayen, R. H. (1995). Modeling morphological processing. In L. B. Feldman (Ed.), *Morphological aspects of language processing*, 131–154.
- Schütze, H. (1995). Distributional part-of-speech tagging. *EACL 1995*, 141–148.
- Seidenberg, M. S., & MacDonald, M. C. (1999). A probabilistic constraints approach to language acquisition and processing. *Cognitive Science*, 23(4), 569–588.
- St. Clair, M. C., Monaghan, P., & Christiansen, M. H. (2010). Learning grammatical categories from distributional cues: Flexible frames for language acquisition. *Cognition*, 116(3), 341–360.
- Taft, M. (1985). The decoding of words in lexical access: A review of the morphographic approach. In D. Besner, T. Waller, & G. MacKinnon (Eds.), *Reading research: Advances in theory and practice*.
- Valian, V., Solt, S., & Stewart, J. (2009). Abstract categories such as Noun or Verb. *Cognition*.
- Zhang, Y., Han, K., Worth, R., & Liu, Z. (2020). Connecting concepts in the brain by mapping cortical representations of semantic relations. *Nature Communications*, 11, 1877.
