# Why web learning never admits a node

**Status: ROOT CAUSE FOUND, NOT FIXED.** This is the substrate blocker behind
`why does it rain` returning a definition.

## The chain of gates

`WebToGraph._ensure_node` (`web/web_to_graph.py:126`) admits a scraped concept
only after three gates:

1. **junk gate** — `junk_score(label) >= _theta` (θ = 0.5) → reject
2. **provisional buffer** — the label is held, not written
3. **promotion** — after `_promote_min_sources` (= 2) **distinct sources**, the
   node is created

Measured: after three `learn_text` calls on real rain text, `ocean` was promoted
(nid 29) but `clouds` was not, and `_provisional_nodes` was **empty** — so
`clouds` was rejected at gate 1.

## The defect: the learned classifier is degenerate

```
junk_score('clouds', glove_mag=1.0) = 0.501
junk_score('ocean',  glove_mag=1.0) = 0.501
junk_score('rain',   glove_mag=1.0) = 0.501
```

**Identical to three decimal places for every ordinary noun.** With θ = 0.5,
every real concept scores *just* over the threshold and is rejected.

Inspecting the classifier (`junk_scorer._get_classifier()`):

```
weights: [-5.2508  0.  -5.2508  -2.6254  0.  0.0058]
features('clouds'): [0, 0, 0, 0, 0, 1]
features('zzqxjunk'): [0, 0, 0, 0, 0, 0]
```

`predict` is a logistic `1/(1+exp(-w·x))`. With only the last weight non-zero
(0.0058), `z ≈ 0.006` and the output is **0.501 regardless of input**. The
learned model carries **no discriminative signal** — it has collapsed to a
constant.

Note `_structural_floor` correctly returns **0.000** for these words, and the
cold-start soft combination (`glove_low*0.35 + salad*0.4`) also returns 0.
The 0.501 comes entirely from the **learned** branch, which is why the bug only
appears in a process where the classifier has been fitted.

## Why this matters

This single defect explains the whole substrate problem:

- web learning reports success but writes **no causal nodes** for ordinary
  topics (`clouds`, `evaporate`, `condense` never became nodes)
- the graph's causal substrate stays at 23 hand-authored `curated` edges
- `why does it rain` can only ever return a definition or an honest hedge

It also explains the earlier `rain` finding: `rain -> heat` is
`edge_kind='unknown'` — a learned edge that was never provenance-stamped,
because the nodes around it were admitted by a different path.

## What a fix must do

The classifier must not be consulted when it carries no signal. Options, in
order of preference:

1. **Guard on discriminative range** — if the fitted weights are near-zero
   (or the predicted spread over a probe set is ~0), fall back to the
   cold-start structural formula. This is the minimal, safest change.
2. **Refit from the structural floor** — reinitialise `w` so the cold-start
   weights are reproduced, then require labels before trusting the learned
   branch.
3. Investigate **why the refit collapsed** — `refit()` runs SGD over weak
   labels; if the labels are near-constant the weights collapse. That is the
   deeper fix.

Option 1 is a two-line guard and unblocks web learning immediately. Option 3
is the real fix and needs its own measurement.

## Reproduction

```
uv run python -c "
import sys; sys.path.insert(0,'ravana/src')
from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.junk_scorer import junk_score
eng = CognitiveChatEngine(data_dir='weights', user_suffix='')
print([junk_score(w, glove_mag=1.0) for w in ('clouds','ocean','rain')])"
```

Expected (buggy): `[0.501, 0.501, 0.501]`
Expected (fixed): values below 0.5 for ordinary nouns.
