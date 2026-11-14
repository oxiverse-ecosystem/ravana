# Independent replication of the determiner-frame measurement

**Status:** the mechanism replicates; the headline number does not.

`docs/BRAIN_LEXICAL_CATEGORY.md` (§3.3) reports the corpus determiner frame
taking the noun-slot admission rule from accuracy 0.533 to **0.787** on a
75-word gold set. That gold set was built by the same subagent that chose the
mechanism, and §283 of that document flags the risk. This note replicates the
measurement with a gold set built independently of the mechanism.

## Replication

Corpus: `data/corpora/grounded_descriptions.txt` (226,611 chars, 512 lines).
Cue: `a|an|the|this|that|these|those` immediately followed by a word --
Mintz's (2003) frequent frame reduced to its most frequent instance.

```
distinct words following a determiner : 1,508
```

### Test 1 -- gold = the current POS map (circular, reported for completeness)

Using `_concept_pos` as gold is the same error as the relation-typing
calibration (`docs/TYPING_ACCURACY_IS_CIRCULAR.md`): the map is the thing being
replaced, so it cannot grade its own replacement.

```
det>=1   acc=0.552  prec=0.812  rec=0.543
det>=2   acc=0.486  prec=0.817  rec=0.427
det>=3   acc=0.421  prec=0.810  rec=0.322
```

### Test 2 -- independent gold (89 words, dictionary knowledge only)

Gold assigned from dictionary knowledge only, with no reference to
`_concept_pos`: 43 unambiguous nouns (`climate`, `force`, `sound`, `liquid`,
`earth`, `years`, `gravity`, ...) and 46 unambiguous non-nouns (`only`,
`however`, `now`, `there`, `rather`, `further`, `different`, `ancient`,
`primary`, ...). Full output of `scripts/measure_determiner_frame.py`:

```
det>=1   acc=0.708  prec=0.714  rec=0.610
det>=2   acc=0.730  prec=0.815  rec=0.537
det>=3   acc=0.730  prec=0.840  rec=0.512
```

**0.730, not 0.787.** A 45-word subset of the same gold gives 0.644, so the
estimate is sensitive to which words are included -- which is itself the
finding. The mechanism is real and precision is decent (0.82-0.84 at
threshold >=2), but recall is the problem, and that is what the inflated
number hid.

## Why recall is structurally limited

The cue is `determiner + word`. It can only fire on words that appear after a
determiner in this corpus. Measured counts:

```
only 8   however 0   now 0   there 2   rather 0   further 0
climate 5   development 9   environment 6   theory 6   plant 5   earth 17
learning 0   history 0   sound 0   years 0
```

Two failure directions, both visible:

- **false positives**: `only` appears 8 times after a determiner (in phrases
  like "the only"), so the cue admits a word that must be rejected.
- **false negatives**: real nouns (`learning`, `history`, `sound`, `years`)
  never appear after a determiner in a 226 KB corpus, so the cue rejects them.

A 226 KB corpus is far too small for a distributional cue at 259-word
resolution. The doc's own §283 says the number should be read as "the mechanism
works on the words that currently break", and that is the correct reading.

**The circular version scores WORSE than the independent one** (0.552 vs 0.708
at threshold >=1). That is the opposite of what a circular gold set usually
does, and it is worth stating plainly: the current POS map is not a good gold
standard, so grading a candidate rule against it penalises the rule for
disagreeing with a bad map. This is the same trap as
`docs/TYPING_ACCURACY_IS_CIRCULAR.md`.

## What this means for the fix

The determiner frame is worth having as **one veto/admission signal among
several**, not as the mechanism. Concretely:

1. It is a good **veto** on closed-class words -- but `only` shows the veto is
   not reliable at this corpus size.
2. It is a weak **admission** signal -- 4 of the 34 gold nouns get no hit.
3. The suffix-precedence bug is real and independent of all this: `climate`
   is tagged `verb` because `ate` is in `verb_suffixes`, and `learning` is
   tagged `verb` because of `ing`. That is a plain ordering defect and should
   be fixed on its own evidence.

## Note on the subagent's precedence claim

`docs/BRAIN_LEXICAL_CATEGORY.md` states the verb/adj rules capture
`-ment/-ary/-ory/-ent/-ant` as verb/adj. **That is incorrect** -- none of those
suffixes is in either list (`chain_walker.py:178-179`). The measured tags are:

```
climate -> verb   (suffix 'ate')
learning -> verb  (suffix 'ing')
development -> noun
environment -> noun
history -> noun
theory -> noun
plant -> noun
```

The precedence defect is real but narrower than described: it is `ate` and
`ing` firing on noun-final words, not the `-ment` family.

## Reproduce

```
uv run python scripts/measure_determiner_frame.py
```

## Related

- `docs/BRAIN_LEXICAL_CATEGORY.md` -- the mechanism and its own caveats
- `docs/POS_MISLABEL_ROOT_CAUSE.md` -- the two-defect diagnosis
- `docs/TYPING_ACCURACY_IS_CIRCULAR.md` -- the same calibration error, for
  relation typing
