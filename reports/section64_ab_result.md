# Section 6.4 — A/B Result (measured verdict)

## Runs (all: --skip-train --no-curiosity --semantic-grade, warmed snapshot)

| benchmark          | phase0(cold) | A: warm, flag OFF | B: warm, flag ON | B-A    |
|--------------------|-------------|-------------------|------------------|--------|
| lamp_test          | 1.000       | 1.000             | 1.000            | 0      |
| self_evaluation    | 0.817       | 0.817             | 0.817            | 0      |
| consult            | 0.571       | 0.571             | 0.571            | 0      |
| reasoning (LogiQA) | 0.374       | 0.374             | 0.374            | 0      |
| temporal           | 0.774       | 0.774             | 0.774            | 0      |
| locomo             | 0.720       | 0.720             | 0.720            | 0      |
| long_mem_eval      | 0.680       | 0.680             | 0.680            | 0      |
| adversarial        | 0.380       | 0.360             | 0.400            | +0.040 |
| memory_consistency | 0.732       | 0.817             | 0.895            | +0.078 |
| OVERALL            | 0.672       | 0.679             | 0.692            | +0.013 |

## Attribution probe (scripts/tmp_ab_attribution.py)

Counter wrapped around _triplet_mc_answer, rerun of memory_consistency +
adversarial with flag ON: consulted=2, ANSWERED=0. The candidate never
produced an answer in any benchmark case.

Therefore the apparent B-A gains are RUN VARIANCE, not the flag:
memory_consistency across four runs = 0.732 / 0.817 / 0.895 / 0.791
regardless of flag state (candidate answered 0 in all). The seven
deterministic benchmarks are bit-identical across A and B, consistent
with a candidate that never fires.

## Zero-regression check

Protected manifest: A closure=192/syll=0, B closure=194/syll=0
(phase0: 195/0; +-3 calls is buffer-state variance across runs, not
displacement — the candidate runs only where closure already abstained).
No benchmark scored lower in B than in A. PASS.

## Decision (per plan 3.3, delta==0 branch)

use_triplet_candidate stays DEFAULT OFF. Wiring is correct and fail-closed
(verified by 8 unit tests incl. live syllogism answer with warmed profile),
but on the current benchmarks the gated inference finds no case where
(a) every evidence handler abstained, (b) clean premises exist, and
(c) a Wilson-gated chain reaches exactly one option.

Root cause matches the opencode plan's own prediction (5.3): warmup
opened only the 'is' gate; LogiQA's operative predicates are
argumentative ('supports', 'weakens', 'assumes') and its premises are
prose blobs the parsers cannot mine. Growth path: web/conversation
exposure accumulates evidence in persistent profiles; re-measure at a
later tick with the same protocol. The ClauseSegmenter idea (plan 4.2)
is the next investment if exposure alone doesn't move coverage.
