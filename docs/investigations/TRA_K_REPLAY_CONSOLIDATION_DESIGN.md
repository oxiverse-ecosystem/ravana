# TRA-K consolidation/replay mechanism design (scratch only, no engine boot)

Branch: `invest/trak-replay-consolidation` · base `main` @ `51eee3e`.
Static probe: `scratch/trak_replay_consolidation_probe.py`.
Results: `scratch/trak_replay_results.json`.

## 0. What the reported `retention=0.000 / growth=+0.000` actually is

The focused live run (`docs/investigations/TRA_K_FOCUSED_LIVE.md`, commit
`0829813`, 40 families) is a valid benchmark FAIL with all instrument controls
OK. Before designing a mechanism, three static checks locate the number:

1. **Persistence is not the failure.** A read-only census of all 40 persisted
   cold-reload stores (`scratch/tra_k_harness/live_data/tra_k_*/ravana_weights.db`,
   SQLite `mode=ro`) finds **468 trusted typed-slot edges** stamped
   `slot_writer='tra_typed_slot'`, `edge_kind='curated'`, `source='conversation'`,
   `source_type='user'`, `relation_type='is_a'`. The typed facts do survive cold
   reload. Some are decayed (107/468 below the 0.20 congruence floor) and some
   were written as unprovenanced `edge_kind='unknown' / relation_type='episodic'`
   traces (e.g. `guild→association` in `tra_k_0020`) — a write-time
   acquisition drop, not a save/reload drop.
2. **The recorded delayed probes are not scorable by a typed-slot reader.**
   Rows from that run carry domain-word options (`glow, mist, hail, gale`),
   while stored typed slots carry source-native Wikidata labels (`photometeor`,
   `snow storm`). That run predates the typed-option-space fix on `main`.
   The probe marks this scope `INSTRUMENT INVALID for typed read` — the 0/16
   across all arms there is a vocabulary mismatch, not a replay result.
3. **Emission is a separate gap.** Recorded P4/P6 engine responses are the
   honesty deflection register ("i don't really have a solid grasp on that so
   far", "outside what i know"), and the typed-slot read primitive is
   default-OFF and unwired (`hook None` in the one-item trace). No storage
   mechanism can make an unwired reader speak.

Consequence: the mechanism below targets the **write→store→read seam** and is
explicitly falsified if the live held-out endpoint stays flat after the read
primitive is wired.

## 1. Brain mechanism and evidence ledger

Grade key: **[S]** strong (multiple studies / causal or meta-analytic),
**[M]** moderate (review, converging indirect evidence), **[W]** weak.
"Support" states what the source actually shows; "Extrapolation" states what
this design assumes beyond it.

| Mechanism | Source | Evidence type / grade | Support | Extrapolation in this design |
|---|---|---|---|---|
| Compressed replay of recent experience during rest/sleep | Wilson & McNaughton (1994), *Science* 265:676–679, doi:10.1126/science.8036517 | rodent ensemble neurophysiology **[S]** | Same-cell sequences recur during SWS | Replay is a symbolic offline pass over typed edges, not a ripple |
| Forward/reverse replay during sharp waves | Diba & Buzsáki (2007), *Nat Neurosci* 10:1241–1242, doi:10.1038/nn1961 | rodent neurophysiology **[S]** | Bidirectional re-enactment at sharp waves | We use one forward pass; no reverse-order claim |
| Replay is causal for consolidation | Girardeau et al. (2009), *Nat Neurosci* 12:1222–1223, doi:10.1038/nn.2384 | rodent causal intervention **[S]** | Ripple suppression impairs spatial memory | Wrong-tag/sham arms are computational analogues, not lesions |
| Complementary learning systems | McClelland, McNaughton & O'Reilly (1995), *Psychol Rev* 102:419–457, doi:10.1037/0033-295X.102.3.419; O'Reilly et al. (2014), *Cogn Sci* 38:1229–1248, doi:10.1111/j.1551-6709.2011.01214.x | computational + review **[M]** | Fast hippocampal store reinstates into slow neocortex | `episodic_buffer` → `ConceptGraph` is the closest RAVANA analogue; graph ≠ distributed cortex |
| Schemas permit one-trial rapid consolidation | Tse et al. (2007), *Science* 316:76–82, doi:10.1126/science.1135935 | rodent behaviour + lesions **[S]** | Prior schema makes one-trial traces rapidly hippocampal-independent | We treat the existing trusted typed-slot family as schema; gold keys are never used to define it |
| Replay prioritizes weakly learned items | Schapiro et al. (2018), *Nat Commun* 9:3920, doi:10.1038/s41467-018-06213-1 | human fMRI (indirect) **[M]** | Weaker items replayed more; predicts gain with sleep | Edge `confidence` is a strength proxy; `eta=0.25` is not data-fitted |
| Spacing improves retention | Cepeda et al. (2006), *Psychol Bull* 132:354–380, doi:10.1037/0033-2909.132.3.354 | human meta-analysis **[S]** | Distributed > massed; optimal lag grows with retention interval | Three save-boundary cycles are not a behavioural retention interval |
| Testing/retrieval strengthens memory | Roediger & Karpicke (2006), *Psychol Sci* 17:249–255, doi:10.1111/j.1467-9280.2006.01693.x | human behavioural **[S]** | Retrieval practice improves long-term retention | R2 below is *replay*, not a test; the success-gated variant is kept as a separate arm only |
| Sleep supports systems consolidation | Diekelmann & Born (2010), *Nat Rev Neurosci* 11:114–126, doi:10.1038/nrn2762 | review/human behaviour **[M]** | SWS supports systems, REM synaptic consolidation | `save()`/`cold reload` is an event boundary, not a sleep stage |
| Reactivation makes traces labile | Nader, Schafe & LeDoux (2000), *Nature* 406:722–726, doi:10.1038/35021052 | rodent causal **[S]** | Retrieval returns memory to a labile state | Justifies restoring *earned* provenance only, and the mutation controls |
| Nonspatial replay in humans | Schuck & Niv (2019), *Science* 364:eaaw5181, doi:10.1126/science.aaw5181 | human fMRI **[M]** | Sequential replay of nonspatial task states | Does not establish replay of `is_a` facts specifically |

**Honest boundary.** No cited source establishes replay of typed semantic
`is_a` facts in a symbolic graph. The evidence is strong for the *shape* of the
mechanism (fast trace → offline reactivation → slower durable integration,
weak items preferentially), and weak-to-absent for its *content* in this
substrate. Everything below the ledger's "extrapolation" column is a design
choice to be tested, not a brain fact.

## 2. Candidate mechanism in RAVANA's own primitives

1. **Write-time replay journal (new, scratch).** For each admitted channel fact
   actually asserted as a teaching turn, record: `(subject, is_a, object_type_label)`,
   the earned conversation provenance (`source=conversation`,
   `source_type=user`, `extractor=tra-typed-slot-assertion`), the channel's own
   CC0 source URL / QIDs / sense as audit fields, the resulting edge key, and
   the confidence at write time. Inputs are the teaching surfaces and the
   channel fact only: never `type_sets`, `world_model`, gold or domain words.
2. **Offline commit at the save/cold-reload boundary.** Walk the journal. If an
   edge is an intact trusted typed slot → skip. If the exact pair exists but was
   stored as an unprovenanced/episodic trace → restore **only** the earned
   fields the trust predicate reads (`edge_kind`, `source`, `source_type`,
   `relation_type`, `type_label`, `slot_label`, `slot_writer`). If no exact
   stored trace exists → count an `absent_no_stored_trace` and do **not**
   fabricate a node or edge. No threshold and no trust-set change.
3. **Spaced reactivation (R2).** Three bounded cycles, `c ← min(1, c + 0.25(1−c))`,
   applied only to trusted, stored, source-credible taught traces. Deliberately
   admitted `evidence_role='misleading'` facts and subjects whose fact was
   taught negated are excluded. Failed retrieval writes nothing in the
   retrieval-practice variant.
4. **Held-out read.** Route the already-existing, default-OFF
   `slot_keyed_forced_choice` over `typed_slot_views_from_graph`; floor 0.20 and
   margin 0.10 unchanged; no eligible trusted slot ⇒ abstain. This is wiring an
   existing primitive, not a new gate.
5. **Sanitizer-safe labels.** Type labels of ≥5 tokens are minted under a short
   id (`pdr_<sha1>[:16]`) with the full label authoritative in the index, then
   resolved by exact node identity/provenance — only for journaled, taught
   facts.

## 3. Exact A/B plan (falsifiable, gate-preserving)

- **State.** One engine per item, isolated `data_dir`. Freeze one shared
  post-teach/post-maintenance snapshot (deep-copied via `copy.deepcopy`, because
  `__getstate__` returns live dicts) and one shared write-time journal capture.
  Arms differ **only** in the commit-time overlay; no arm's loops see
  arm-dependent graph state.
- **Arms:** `OFF` baseline; `R1` provenance restamp; `R2` = R1 + 3 reactivation
  cycles; `R1_CONF` confidence-only sham on the same traces; `R1_WRONG` identical
  loop with `edge_kind='unknown', source='system'`; `ORACLE` = gold supplied to
  the scorer only, labelled **NON-DEPLOYABLE**.
- **Compute parity:** assert equal `edges scanned`, `restamps attempted`,
  `cycles`, `prunes`, `intake calls`; pre-registered wall-clock ratio ≤ 1.12.
  Warm up once before timing.
- **Endpoints:** primary cold-reload `delayed / immediate` at k=8, `growth =
  immediate − baseline`, near/far/foil/negative, plus continuous mean stored
  confidence. Gates C1–C15, subject-independence, channel-independence, G6 and
  M-A/M-B unchanged.
- **Pass:** `R2 − OFF ≥ +0.10` on primary delayed near/far, foil and negative
  reject drop ≤ 0.05, `R1_CONF` and `R1_WRONG` flat, ORACLE > 0.90, controls OK,
  sealed pool replicates, all gains survive cold reload, counters equal.
- **Sample:** focused 20/pool is a hypothesis probe; full 200/pool only after a
  clean focused result.
- **Leakage audit:** `channel_overlap()` answer-key leak = 0; mutate item gold
  and assert the replay-state fingerprint is unchanged (implemented in the
  static probe); no oracle field in any deployable input.

## 4. Pre-registered falsifiers

1. `R2 − OFF < +0.05` with restored traces > 0 and controls OK ⇒ replay does
   not close the storage gap. Stop tuning replay.
2. `R1_CONF` or `R1_WRONG` moves the endpoint ⇒ the trust gate is not binding;
   the "gain" is a stamp artifact. Reject the mechanism.
3. ORACLE ≤ 0.50 ⇒ information deficit in the taught channel; fix teaching,
   not consolidation.
4. R2 raises near/far but drops foil/negative reject > 0.05 ⇒ boosting broadens
   acceptance. Reject. (An early uncorrected variant did exactly this: negative
   reject 0.65 → 0.25; restricting replay to source-credible, non-negated
   traces restored 0.65.)
5. Live delayed endpoint stays 0 while the static graph read is ≥ 0.40 after
   the existing read primitive is wired ⇒ articulation/emission gap; the next
   work item is the answer-emission path, not replay.
6. Cold census shows trusted typed slots for > 90% of probe subjects while live
   retention stays 0 ⇒ locate answer construction, not storage.
7. Wall clock or maintenance counters differ > 12% ⇒ compute confound; revert.
8. Mutating item gold changes the replay fingerprint ⇒ answer-key leakage; reject.
9. `channel_teaching_surfaces` truncated a multi-word label to its first word
   (the harness's `_TAUGHT_SLOT_RE` does) ⇒ the probe measures its own parser;
   discard the run. The committed probe parses the full object phrase.

## 5. Static probe result (real output, hypothesis only)

`uv run python scratch/trak_replay_consolidation_probe.py` — no engine boot, no
write, no network. Vocabulary-consistent scope, current frozen item bank,
40 recorded cold-reload stores, n=20 per pool:

| arm | primary near | primary far | foil reject | negative reject |
|---|---|---|---|---|
| OFF | 0.30 | 0.25 | 0.95 | 0.65 |
| R1 (restamp) | 0.30 | 0.30 | 0.95 | 0.65 |
| R1_CONF (sham) | 0.30 | 0.30 | 0.95 | 0.65 |
| R1_WRONG (wrong tag) | 0.30 | 0.25 | 0.95 | 0.65 |
| R2 (restamp + reactivation) | **0.45** | **0.45** | 0.95 | 0.65 |
| ORACLE (non-deployable) | 0.95 | 1.00 | 0.95 | 0.65 |

Instrument controls OK; answer-key mutation leaves replay state unchanged;
answer-key leak 0. 468 typed-slot edges censused; 5 damaged traces restorable
under the near-scope rule, 122 taught pairs have no exact stored trace.
Sealed pool: OFF 0.30/0.35 → R2 0.55/0.65 with foil 0.95 and negative 0.75
unchanged. The recorded-delayed scope is marked **INSTRUMENT INVALID** for a
typed reader (option vocabulary mismatch, section 0).

**Reading:** a hypothesis that R2 (not R1) moves the graph-derived read with
foils/rejects intact; it does **not** establish that live held-out retention
will move, because emission is not exercised here. No gate or threshold was
changed, and no production file was modified.

## 6. References (all verified to resolve)

- Wilson & McNaughton 1994, *Science* 265:676–679, doi:10.1126/science.8036517
- Diba & Buzsáki 2007, *Nat Neurosci* 10:1241–1242, doi:10.1038/nn1961
- Girardeau et al. 2009, *Nat Neurosci* 12:1222–1223, doi:10.1038/nn.2384
- McClelland, McNaughton & O'Reilly 1995, *Psychol Rev* 102:419–457, doi:10.1037/0033-295X.102.3.419
- O'Reilly et al. 2014, *Cogn Sci* 38:1229–1248, doi:10.1111/j.1551-6709.2011.01214.x
- Tse et al. 2007, *Science* 316:76–82, doi:10.1126/science.1135935
- Schapiro et al. 2018, *Nat Commun* 9:3920, doi:10.1038/s41467-018-06213-1
- Cepeda et al. 2006, *Psychol Bull* 132:354–380, doi:10.1037/0033-2909.132.3.354
- Roediger & Karpicke 2006, *Psychol Sci* 17:249–255, doi:10.1111/j.1467-9280.2006.01693.x
- Diekelmann & Born 2010, *Nat Rev Neurosci* 11:114–126, doi:10.1038/nrn2762
- Nader, Schafe & LeDoux 2000, *Nature* 406:722–726, doi:10.1038/35021052
- Schuck & Niv 2019, *Science* 364:eaaw5181, doi:10.1126/science.aaw5181
