# Encoding-gate A/B on frozen state — negative result (2026-11-21)

Instrument: `scripts/measure_encoding_gate_ab.py`. Isolated copies of the
repaired production weights, captured web snapshots
(`tests/fixtures/web_snapshots/web_fixtures.json`, 1,087 queries; 55 with ≥2
causal tokens; 24 probe queries used after deterministic hash selection),
`RAVANA_OFFLINE=1`, fixed hash seed. Four arms, each ONE engine process,
sequential; no live fetch. Written endpoints were measured directly (not
guessed from query words) and each arm was saved.

## Result

| Arm | Reads admitted | Facts written | Written endpoints | Assertable chains |
|---|---|---|---|---|
| OFF (flag=0) | 24/24 | 146 | 265 | **0** |
| ON, no signal (floor) | **0** | 0 | 0 | 0 |
| ON + external re-ask | **3** (budget cap) | 21 | 42 | **0** |
| ON + wrong tag (negative control) | 3 | 21 | 42 | **0** |

**Verdict: NO_GAIN.** All four falsifier checks that the gate itself must pass
are green (floor admits zero, OFF admits all, ON is bounded at 3, wrong-tag
control does not gain). The positive endpoint does not move.

## Why — measured, not inferred

Census of the saved arms' edges with `fixture.local` provenance:

- ON arm: 6 edges, all `edge_kind='curated'`, but 5 are `is_a` and 1 `causal`.
- The labels are phrase fragments, not canonical concepts:
  `learn → osmosis factors affect`, `ocean currents → responsible transporting
  warm water`, `typical circumstances wind → landscapes slowly`,
  `energy wave → driven wind propagated`.
- A trusted 2+ hop chain needs canonical concept nodes that link to each
  other; these phrase nodes are singletons. The extractor produces *facts* but
  not connectable knowledge.

This is falsifier #1/#3 from `WEB_ENCODING_GATE.md` §2.5: the corpus/extraction
ceiling, not a gate failure. It replicates the earlier finding that only 14.6%
of 1,087 queries contain ≥2 extractable causal triples — and even those
triple candidates here are phrase-level junk. Tuning the read schedule further
cannot help; the gate is doing exactly what it was designed to do.

## Claim discipline

- This does **not** show the encoding gate is useless: it shows selective
  admission works and the bottleneck is downstream extraction quality.
- This does **not** license changing the gate thresholds or turning the flag on
  by default. Per pre-registration, the arm is recorded as a negative and the
  default stays OFF.
- Next axis (untried): an extractor/anchor stage that maps phrase-level OpenIE
  output onto canonical graph concepts before writing, with the same A/B.
