# Calibrated gap instead of associative filler in `_reflective_response`

## Defect (measured, n=203; `scratch/front_probes_n30.jsonl`)

15/203 replies were the same fixed template:

> `the closest i get to earthquakes is rains and maybe environments -- and from what i've seen, water that falls from the sky as drops of liquid.`

The reply echoes the subject, chains two unrelated GloVe neighbours, and then
asserts a definition of the **neighbour** ("rains") as if it answered the
question about the **subject** ("earthquakes"). Grounded-factual scored 30/100
and speaking 26/100 largely on this one failure mode.

## Root cause

1. **Selection.** `_generate_response` reaches `_reflective_response` as the
   terminal fallback whenever there is no stored definition, no web fact and no
   trusted causal chain, but shallow associations exist. The presence of any
   GloVe neighbours was treated as "partial knowledge worth sharing".
2. **Evidence misattribution.** The method fetched a KB definition of `top[0]`
   (an *association*) and spliced it after a hedge head as `from what i've
   seen, <definition>`. The only gate that checked subject relevance,
   `_sentence_grounded`, was a **dead reference**: it is a nested local function
   elsewhere in `response_gen.py`, never an attribute of the mixin, so
   `getattr(self, "_sentence_grounded", None) is None` short-circuited the check
   to True on every call. The surviving gate (`_coherence_ok`) tests tautology
   and inter-clause coherence, not *aboutness*, so a definition of "rain" passed
   as an answer about "earthquakes".

## Brain basis for the fix

The brain does not emit a fixed associative string when semantic retrieval
fails. It distinguishes accessible partial information from verified knowledge,
binds retrieved content to the retrieval cue, and recognises an unfillable gap
as the start of curiosity:

- **Koriat, A. (1993).** How do we know that we know? The accessibility model of
  the feeling of knowing. *Psychological Review, 100*(4), 609–639.
  DOI 10.1037/0033-295X.100.4.609. FOK is computed during retrieval from the
  *accessibility* of partial information, regardless of its correctness; people
  cannot directly tell a correct clue from an incorrect one. Promoting an
  accessible neighbour to a claim about the subject is exactly this
  accessibility/correctness confusion.
- **Badre, D., & Wagner, A. D. (2002).** Semantic retrieval, mnemonic control,
  and prefrontal cortex. *Behavioural and Cognitive Neuroscience Reviews, 1*(3),
  206–218. DOI 10.1177/1534582302001003002. Automatic bottom-up activation
  follows associative strength; when the cue–target link is weak it fails and
  controlled, cue-bound retrieval (left anterior inferior PFC) is required.
  Associative spillover is not a verified recollection.
- **Buckner, R. L., Andrews-Hanna, J. R., & Schacter, D. L. (2008).** The brain's
  default network: anatomy, function, and relevance to disease. *Annals of the
  New York Academy of Sciences, 1124*, 1–38. DOI 10.1196/annals.1440.011.
  Unfiltered semantic association is a default-mode product, not knowledge.
- **Loewenstein, G. (1994).** The psychology of curiosity: A review and
  reinterpretation. *Psychological Bulletin, 116*(1), 75–98.
  DOI 10.1037/0033-2909.116.1.75. An information gap opens when a question is
  activated but its answer is not known with certainty; the calibrated,
  socially natural response is to acknowledge the gap and stay curious.

## Implemented decision

In `ResponseGenMixin._reflective_response` (and the new
`_subject_grounded_evidence` helper):

- A KB sentence about a neighbour is usable **only** if it actually mentions the
  subject (token binding, with a light singular/plural variant). This implements
  controlled retrieval / cue binding; it is structural, not a per-topic list.
- If such partial material exists, it is stated plainly
  (`what i do know is this: ...`, strategy `reflective_grounded`).
- If none exists, the method delegates to `_human_like_uncertainty`, the
  existing FOK + turn-back route (strategy `metacognitive_uncertainty`), which
  also folds in any partial web candidates. No associative filler is emitted.
- Legitimate analogy/creativity remains in the graph-edge-gated bisociation
  path (`_generate_creative` / `_handle_humor`), which is unchanged.

Regression test: `tests/unit/test_reflective_calibrated_gap.py`.
