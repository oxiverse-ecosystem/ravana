# RSI relation-target evidence channel — implementation & result (NO-GO)

Branch `invest/rsi-reltarget-evidence` (worktree `ravana_rsi_reltarget`, based on `main`).
Companion pre-registration: `docs/investigations/RSI_RELTARGET_PREREG.md` (written before data).

## Outcome

Built a NEW, independent, licensed **relation-target** evidence channel for RSI,
froze a real source snapshot, and ran the full pre-registered control suite.
The channel is correct and safe (licence/provenance, negation veto, credibility,
answer-key independence, default-off, no production change) but **NO-GO for
deployment**: it cannot carry the RSI endpoint because real relation assertions
almost never name a *displayed candidate*. Pre-registered **C4** (candidate-target
coverage ≥ 0.50) failed at **0.0042** (1/240 rows). No gate, threshold, item,
prompt, scorer or production file was changed; the module stays default-OFF and
unwired.

## What was built

| artifact | purpose |
|---|---|
| `scratch/rsi_reltarget_fetch.py` | one-shot WDQS fetch of real assertions for the 141-word RSI vocabulary; reads only the public `OBJECTS` word list, never the answer key |
| `scratch/rsi_reltarget_wikidata_raw.json` | frozen CC0 snapshot: 371 statement-level bindings, each with rank, statement URI, item/value QIDs and the exact request URL |
| `scratch/rsi_relation_target_evidence.py` | pure, default-OFF channel: `(subject, relation, target)` records with provenance, rank-derived credibility, explicit negation veto, wrong-tag SHAM, AST independence proof |
| `scratch/rsi_reltarget_controls.py` | boot-free coverage audit + permutation / negation / candidate-order / flag-off controls |
| `scratch/rsi_reltarget_results.json` | all per-row decisions and criterion outcomes |

Source / licence: **Wikidata**, **CC0-1.0** (https://creativecommons.org/publicdomain/zero/1.0/),
endpoint https://query.wikidata.org/sparql. This is a different property set
from M2's P31/P279 type facts: P828/P1542 (cause), P361/P527 (part-whole),
P155/P156 (order), P461 (opposition). SUPPORTS and RESEMBLES have no clean WD
property and are reported uncovered, not fabricated.

## Measured results (240 query rows = 120 items × exact/far)

| metric | value |
|---|---|
| snapshot assertions (all CC0, provenanced) | 371 |
| relation-family mix | PART_OF 304, PRECEDES 28, OPPOSITE 20, CAUSE 19 |
| query vocabulary subjects | 141 |
| subjects with ≥1 assertion (C3, need ≥0.80) | 107 / 141 = **0.7589 — FAIL** |
| rows with any admitted candidate-target evidence (C4, need ≥0.50) | 1 / 240 = **0.0042 — FAIL** |
| rows with a pick | 1 / 240 |
| accuracy against authored key (grader-only) | 0.0042 |
| licence + provenance on every admitted record (C1) | PASS |
| real relation, never `is_a` (C2) | PASS |
| AST answer-key independence (C5a) | PASS |
| permutation inside displayed option set: keys changed / pick invariant / permuted-key accuracy (C5b) | 240 / yes / 0.0000 — PASS |
| negation veto on picking rows / wrong-tag SHAM admitted (C6) | 1 / 0 — PASS |
| candidate-order invariance (C7) | 1 / 1 — PASS |
| flag-OFF returns `None` (C8) | PASS |
| **verdict** | **NO-GO** |

The only hit (`rsi_031`, subject `root`, relation `PART_OF`, options
herb/sapling/bark/seed, key `bark`) comes from a Wikidata `P527` "has part"
statement read as `PART_OF` — the channel currently collapses P361 and P527
without direction handling, so even the single hit is a direction artifact.
This weakens rather than strengthens the case for deployment.

## Why the channel cannot move RSI

`build_items()` (`scripts/measure_learn5_rsi.py`) constructs the world model
entirely from hash-selected object pairs:

```
demo_objs = _hash_pick(seed + "|D", OBJECTS[domain], 16)
pairs     = [(demo_objs[i], demo_objs[i+1]) for i in range(0, 16, 2)]
reserved  = _hash_pick(seed + "|R", OBJECTS[domain], 8, exclude=...)
s_exact, t_exact = reserved[0], reserved[1]      # the "gold target"
wm[(relation, s)] = {t} for every such pair
```

The gold target is `reserved[1]` — an arbitrary same-domain word, not a
real-world `(subject, relation, target)` fact. A real CC0 channel therefore
has ~1/240 chance of naming it among four displayed candidates, exactly as
measured. This is an **information-deficit in the source space**, not a reader
defect: the M3/M123 reader machinery already proved it emits a pick when a
genuine candidate target is supplied (`rsi_f7_rootcause_probe.py` oracle arm).

Following `knowledge-channel-audits.md`: NO-GO is a result. Do **not** author
channel facts to fill the gap (that would make the channel the answer key), do
not read `world_model`/`gold`, and do not tune a threshold to force C4.

## Brain mechanism (peer-reviewed; verified against publisher/index records)

The design is brain-grounded, not ad-hoc. Each mapping is an analogy to a
symbolic read path, not a claim that the brain stores Wikidata statements.

- **Gentner (1983)**, Structure-Mapping: A Theoretical Framework for Analogy,
  *Cognitive Science* 7(2):155–170, doi:10.1207/s15516709cog0702_3 — analogy
  transfers **relations**, not object attributes → the channel must supply
  `(subject, relation, target)`, which is why the M2 type-label channel failed.
- **Eichenbaum et al. (2009)**, Relational Memory and the Hippocampus:
  Representations and Methods, *Frontiers in Neuroscience*,
  doi:10.3389/neuro.01.023.2009 — the hippocampus binds arbitrary relations
  among elements → relation-target binding is a distinct memory operation from
  item/type encoding.
- **Tulving & Thomson (1973)**, Encoding specificity, *Psychological Review*
  80(5):352–373, doi:10.1037/h0020071 — a cue encoded as a type cannot retrieve
  a relation-target trace (the exact label-space mismatch measured here).
- **Tomasello (2000)**, The item-based nature of children's early syntactic
  development, *Trends in Cognitive Sciences* 4(6):209–214 — role knowledge
  begins as concrete, item-based schemas (verb islands) before abstraction →
  relation-target evidence is item-specific, not a global type hierarchy.
- **Carey (2010)**, Beyond Fast Mapping, *Language Learning and Development*
  6(3):184–205, doi:10.1080/15475441.2010.484379 — fast mapping establishes a
  fragile initial mapping later refined by evidence → an admitted assertion is
  graded by source credibility (WD rank), not accepted as authoritative.
- **Kucker, McMurray & Samuelson (2015)**, Slowing Down Fast Mapping,
  *Child Development Perspectives* 9(2):74–78, doi:10.1111/cdep.12110 — fast
  mapping is a gradual, evidence-sensitive process, not one-shot certainty.
- **Johnson, Hashtroudi & Lindsay (1993)**, Source monitoring, *Psychological
  Bulletin* 114(1):3–28, doi:10.1037/0033-2909.114.1.3 — source and item memory
  dissociate → trust is read from provenance, never from confidence alone.

## Implementation status

- New module is **pure and default-OFF** (`RAVANA_RSI_RELTARGET=1` opt-in only),
  imports production `schema_induction` constants/trust predicate **unchanged**,
  and is not imported by any production file. No engine boot, no store write,
  no network at import.
- Frozen CC0 snapshot is checked in as data; no live WDQS call at run time.
- **NOT wired** into RSI, M1/M2/M3, or any gate/scorer. Kept as a negative
  result plus a reusable, independently audited channel for any benchmark
  whose key really is a world fact.
- No gate/threshold/item/scoring change; no production code changed.

## Reproduce (boot-free)

```bash
cd C:/Users/Likhith/Documents/projects/ravana_rsi_reltarget
PYTHONHASHSEED=0 PYTHONPATH="scripts;ravana/src;scratch;." \
C:/Users/Likhith/Documents/projects/ravana/.venv/Scripts/python.exe \
  scratch/rsi_reltarget_controls.py
```
