# M1 — unified durable-provenance adapter (identity-keyed act substrate)

**Status:** implemented, default-OFF behind `RAVANA_DURABLE_PROVENANCE=1`.
No gate, threshold, item, expected decision, prompt or scoring rule was changed.
`evaluate_source` and every existing flag (`RAVANA_SOURCE_EVAL`,
`RAVANA_PRIOR_ACT_TRACE`) are byte-identical when the new flag is unset.

## 1. The gap this closes

Two partial default-off substrates already exist on `main`:

| partial | what it keeps | where it lives | failure |
|---|---|---|---|
| claim-ID prior-act trace (`RAVANA_PRIOR_ACT_TRACE`) | one `Decision.ENDORSE` per claim key | in-memory dict on the engine | **lost on cold reload**; no provenance or negation |
| g05 provenance anchor (`wfa_claim_token` + `_write_asserted_wfa_fact`) | origin / kind / status on collision-safe claim-id graph nodes | the WFA harness teaching path | harness-only; no prior act; not a single record |

The measured symptom is `R_INVALIDATION_NO_PRIOR`: a discredited turn cannot
withdraw because the engine's only memory of its own earlier endorsement is a
single global slot (`_last_source_evaluation`) that any later turn overwrites,
and that disappears entirely on reload. The two partials fix halves of this,
in different places, with different lifetimes. M1 binds the four aspects into
**one identity-keyed record** that is written at the act and restored after a
cold reload.

## 2. Brain mechanisms and citations (all independently verified this session)

The relevant capacity is *persistent source/act memory*: knowing not just the
item but that **I** endorsed it, from **which** origin, and that the origin was
later discredited.

1. **Hippocampal indexing** — Teyler & DiScenna (1986), *Behavioral &
   Neural Biology* 100:147–152, "The hippocampal memory indexing theory";
   updated in Teyler & Rudy (2007), *Hippocampus* 17:1171–1177, and indexed in
   PubMed 17696170. The hippocampus stores a **short, stable index** of
   distributed neocortical activity; a partial cue re-activates the indexed
   pattern. Verified: "the hippocampus stores an index to neocortical patterns
   that can be used to retrieve the memory"; "a partial cue ... can activate
   the neocortical patterns and thus retrieve the memory of the episode."
   **Computable implication:** one short sha1 claim anchor must bind identity,
   provenance, act and negation so a later cue re-accesses all four together.
   **Evidence grade:** [S] as a theory of hippocampal function; [M] as a direct
   analogue for a symbolic key (engineering instantiation, not a neural claim).

2. **Prefrontal source monitoring** — Johnson, Hashtroudi & Lindsay (1993),
   *Psychological Bulletin* 114(1):3–28, doi:10.1037/0033-2909.114.1.3
   (PubMed 8346328, APA 1993-40167-001): the source-monitoring framework names
   *internal source monitoring* (what I thought/said vs. what I perceived) and
   holds that source is **evaluated at retrieval** from remembered
   characteristics, dissociable from item familiarity, with frontal systems
   supporting the systematic form. Ranganath, Johnson & D'Esposito (2000),
   *J. Neurosci.* 20(22):RC108: left anterior PFC supports systematic,
   source-specific monitoring. **Computable implication:** the record stores
   origin/kind/status and the emitted decision, never a reliability or
   familiarity score; it is read at staging time, not baked into the fact.
   **Evidence grade:** [S] behavioural/neuropsychological; [M] module analogy.

3. **Reconsolidation** — Nader, Schafe & Le Doux (2000), *Nature*
   406:722–726, doi:10.1038/35021052 (PubMed 10963596): a consolidated memory,
   once reactivated, returns to a labile state in which new information is
   written into the **old** trace. Verified: "consolidated fear memories, when
   reactivated during retrieval, return to a labile state ... requires de novo
   protein synthesis for reconsolidation." **Computable implication:** the act
   is written *at the act* and later updated **in place**; a negation cue is
   merged into the same record rather than inferred fresh each turn. Monotonic
   `ever_endorsed` and `negated` bits implement "write into the old trace."
   **Evidence grade:** [S] for labile reconsolidation; the symbolic in-place
   record is an extrapolation, not a biological claim.

4. **Complementary learning systems** — McClelland, McNaughton & O'Reilly
   (1995), *Psychological Review* 102(3):419–457: a fast, pattern-separated
   hippocampal system for episodes plus a slow, distributed neocortical system
   that gradually consolidates them. **Computable implication:** the in-memory
   ledger is the fast index; an atomically written JSON sidecar is the slow
   durable mirror, exactly as the engine's pkl/SQLite pair is for the graph.
   **Evidence grade:** [S] as a computational/architectural theory; the
   sidecar analogy is [M].

**Support vs extrapolation.** (1)–(3) directly motivate a stable key, a
retrieval-time source evaluation and an in-place durable act record. The
symbolic `dict[sha1] -> dataclass` and the JSON sidecar are engineering
instantiations. No claim is made that the engine has a hippocampus or that a
single-token `W` is a human withdrawal.

## 3. What was built

New module `ravana/src/ravana/chat/durable_provenance.py`:

- `claim_key(claim)` — byte-identical to the existing `claim_act_key` and the
  WFA `wfa_claim_token`, so provenance, prior act and ledger share one identity.
- `ProvenanceRecord` — frozen, JSON-round-trippable: `claim_key`, `claim`,
  `subject`, `subject_key`, `origin_id`, `source_type`, `edge_kind`,
  `trust_class`, `prior_decision`, `ever_endorsed`, `negated`, `version`.
  `fingerprint()` hashes only these bound fields (no item answer/label).
- `DurableProvenanceLedger` — identity→record map, deterministic serialisation,
  `apply_to_context()` which re-stages a durable prior act + negation into the
  ClaimContext **before** the unchanged gate.
- `DurableProvenanceStore` — atomic (`mkstemp`+`fsync`+`os.replace`),
  LF-normalised sidecar in the engine's data dir; corrupt/missing file fails
  open to an empty ledger.
- `record_decision()` — provenance is **first-write-wins** (stays stable across
  turns), `ever_endorsed` and `negated` are monotonic, `prior_decision` keeps
  the latest emitted decision for audit.

Wiring: 49 added lines in `engine.py`, all inside one flag check, in
`_apply_source_monitoring_gate`: re-stage before the decision, persist after
it. Provenance is read only from the engine's own staged `SourceStatus`; no
item field is ever consulted. The existing `RAVANA_PRIOR_ACT_TRACE` path is
untouched and independent.

## 4. Boot-free A/B (executed)

`scratch/m1_durable_provenance_ab.py` (12 claims, fresh per-claim data dir per
arm; real `process_turn` + real emission gate on a duck-typed engine; **no
engine boot**). Each arm: credible turn → ENDORSE; then a fresh engine object
with empty in-memory state and the same data dir → invalidated turn.

| arm | flag | withdraw after cold reload | substitutions | sidecar |
|---|---|---|---|---|
| A | unset | 0/12 (0.00) | 0 | absent |
| A0 | `"0"` | 0/12 (0.00) | 0 | absent |
| B | `"1"` | **12/12 (1.00)** | 12 | present |

Controls, all true: `flag_off_equals_explicit_zero`, `no_false_withdrawal`,
`key_isolation`, `item_answer_key_invariant`, `evaluator_unchanged`,
`anchor_matches_existing`. Manifest hash
`9f7467ad80261b470818b092e1cbdb0279658538a4dfe36a5b94bd78f8200082` (no commit
metadata in the hashed payload). Per-row audit in
`scratch/m1_durable_provenance_ab.json`.

**Cold-reload evidence.** The "cold reload" arm constructs a brand-new engine
object with a fresh ledger (`"_durable_provenance_store" not in __dict__`),
reads the durable record from the sidecar on disk, and produces WITHDRAW with
`R_SOURCE_INVALIDATED`; identity, provenance and `ever_endorsed` fields are
byte-identical before/after, and `negated` becomes durable. Repeat-reload tests
show a later WITHDRAW does not erase the earlier endorsement. This is a
boot-free cold reload of the substrate; a full weight-store boot was
**deliberately not run** so the 1.2 GB shared store was never opened by a live
engine during this task.

**Tests.** `tests/unit/test_durable_provenance.py` (13 tests) together with
`test_source_prior_act_trace.py`, `test_source_gate_early_return.py`,
`test_source_evaluation_wiring.py`, `test_source_evaluation.py` in one pytest
process: **79 passed**, with `PYTHONPATH` pointed at this worktree (import
verified as the worktree module via `__file__`).

## 5. Falsifiers (pre-registered; measurable)

1. **Cold reload loses the act.** If a fresh engine on the same data dir returns
   `N / R_INVALIDATION_NO_PRIOR` after a prior ENDORSE with the flag on, the
   sidecar round-trip is broken. Predicted by the mechanism: `W`.
2. **Flag leakage.** If flag unset or `"0"` creates a sidecar, substitutes a
   prior act, or changes any emitted token, the "default-off/inert" claim is
   false. Predicted: no file, zero substitutions, byte-identical tokens.
3. **Item-answer leakage.** If changing an item's `expected_answer`/`gold`
   changes `claim_key` or `fingerprint()`, the adapter reads the answer key.
   Predicted: unchanged.
4. **Key collision.** If an invalidated claim B with no prior act emits `W`
   after claim A was endorsed, identity scoping is broken. Predicted: `N`.
5. **Provenance clobber.** If a later invalidated turn overwrites the stored
   `origin_id`/`edge_kind`, first-write-wins is broken. Predicted: original
   origin/kind retained, `negated=True` added.
6. **Endorsement erasure.** If a second cold reload after a WITHDRAW emits `N`,
   the monotonic `ever_endorsed` bit was lost. Predicted: `W` again.
7. **Gate change.** If `evaluate_source` output differs with the flag on vs off
   for the same ClaimContext, a threshold/rule was changed. Predicted:
   byte-identical `as_dict()`.
8. **Real-store falsifier (not run here).** Boot two engines on a copied weight
   store; if the second cannot WITHDRAW after the first endorsed and saved, the
   `_save_path`-derived sidecar path is wrong in the real constructor. This is
   the remaining verification a full engine boot would provide.

## 6. Caveats

- The internal-source path still uses a deterministic rule table; a saturated
  12/12 endpoint is G6 mechanism evidence, **not** autonomous human-like
  source recollection.
- n=12 is a mechanism probe, not a capability verdict; no full WFA harness run
  was performed and no gate/sample-size criterion is claimed to be met.
- The sidecar is a fourth store; it is mitigated by atomic writes, LF
  normalisation, fail-open reads, and flag-gating (never written when off).
- No real weight-store boot was run (store protected); falsifier 8 is open.
