# WFA after the provenance fix: prior-endorsement binding is the next experiment

**Branch:** `invest/wfa-prior-endorsement` (new worktree from `main` @ `6a6fef6`).
**Constraints honoured:** SCRATCH + static only. No `CognitiveChatEngine` boot, no
weight-store open, no edit under `ravana/src`, no change to `scripts/`, tests,
gates, thresholds, item bank, prompts, expected decisions or grading rules.
The committed artifact is this doc only; the probe lives in gitignored `scratch/`
and is reproduced verbatim in Appendix A. Every function under test is imported
UNMODIFIED from the repo.

## 1. Exact state after the g05 provenance fix

The claim-ID-anchor fix (`_write_asserted_wfa_fact`, #497) is present in the
harness that produced `scratch/learn5_wfa_summary.json`
(SHA-256 `0fdc0cd297c5e6b4aedbd3a473a528b07ee2f75bdf57a6f565cc4a25ceff80c3`
of the read-only copy used by the probe). Full-run result:

- `pass = false`, all 26 original WFA rules and all instrument controls pass;
  **all 8 process-dissociation gates and both `unparseable_ceiling_*` gates fail**
  (`pd_unparseable_rate = 1.00`, `forced_choice_explicit_rate = 0.00`,
  `R = 0.0`, `F = 0.0` on both pools). Those are the free-text/forced-choice arms
  and were already diagnosed as the *absent decision routine*
  (`WFA_ECHO_IMMUNE_RESULT.md`); the wired routine is the internal-source arm.
- Internal-source arm: primary all three conditions 1.00; **sealed discredited
  = 0.80** (`endorse_rate 0`, `abstain_rate 0.2`, `withdraw_rate 0.8`), sealed
  credible and unknown 1.00. Anti-tautology true on both pools.
- The four sealed failures are **gg039, gg032, g64, g62**. Every one has
  `rule = R_INVALIDATION_NO_PRIOR`, `prior_decision = None`, and
  `invalidated = 2` or `3`. Provenance was found and the invalidation cue
  **tagged 2–3 stored edges** in each case.

That signature rules out the g05 class of failure. g05 lost its edge during the
prior turn (`invalidated = 0`); these four still hold a discredited, tagged
origin. The stored source is doing its job. What is missing is the engine's
record that *it previously endorsed this claim* — without that, rule 4 of
`evaluate_source` (`source_evaluation.py:249-254`) correctly fails closed to
ABSTAIN instead of WITHDRAW.

## 2. Brain mechanism and competing hypotheses

The relevant distinction is not "is this claim credible?" but **"did I previously
commit to it, and has the source of that commitment been discredited?"** That is
an act of *internal source monitoring* (did I do/say this?) rather than an item
judgment. Source monitoring is a set of explicit, fallible judgments over
qualities of experience, dissociable from item familiarity and supported by
frontal regions (Johnson, Hashtroudi & Lindsay 1993, *Psychol Bull* 114(1):3–28,
doi:10.1037/0033-2909.114.1.3 — verified PubMed 8346328 / APA 1993-40167-001).
Recollecting one's own earlier act is autonoetic ("remember"), whereas merely
knowing a fact is noetic ("know"); a bare answer does not carry the prior act
(Tulving 1985, *Can Psychol* 26(1):1–12, doi:10.1037/h0080017 — verified APA
1985-27196-001). Controlled recollection and automatic familiarity can be
separated by inclusion/exclusion (Jacoby 1991, *J Mem Lang* 30(5):513–541,
doi:10.1016/0749-596X(91)90025-F — verified). Reevaluating a just-learned
mapping after the speaker is shown unreliable has been demonstrated in
30-month-olds (Dautriche, Goupil, Smith & Rabagliati 2021, *Open Mind* 5:1–19,
doi:10.1162/opmi_a_00038, PMID 34485794 — verified publisher PDF and PubMed),
and trust is calibrated on the communicator, not the content's fluency
(Sperber et al. 2010, *Mind & Language* 25(4):359–393,
doi:10.1111/j.1468-0017.2010.01394.x). Retrieval returns a consolidated trace
to a labile state in which new information is written into the old trace
(Nader, Schafe & Le Doux 2000, *Nature* 406:722–726, doi:10.1038/35021052).

Three candidate explanations, with a discriminating prediction each:

| hypothesis | claim | discriminator |
|---|---|---|
| **H_PEB (prior-endorsement binding)** — *the refined hypothesis* | The engine's own prior ENDORSE act is not available at the withdrawal turn. Provenance and invalidation are intact; the *recollection of the earlier commitment* is missing. Fail-closed ABSTAIN is the correct behaviour absent that recollection. | Supplying only `prior_decision=ENDORSE` (from the engine's own first-turn audit, never from item metadata) flips ABSTAIN→WITHDRAW with stored provenance byte-identical. |
| **H_SLEEPER (cue dissociation)** | The discounting cue has dissociated from the claim content over intervening turns, so the claim regains force even though the prior endorsement was recorded (sleeper effect: Pratkanis, Greenwald, Leippe & Baumgardner 1988, *JPSP* 54(2):203–218; Kumkale & Albarracín 2004 meta-analysis, *Psychol Bull* 130(1):143–172, doi:10.1037/0033-2909.130.1.143). | With `prior_decision=ENDORSE` held fixed, withdrawal *decays* as a function of intervening turns/delay, and a re-binding cue at retrieval restores it. |
| **H_GATE (trust threshold too strict)** | The evaluator rejects discredited provenance at a bad threshold. | Ruled out by the mutation control: changing only stored `edge_kind` flips ENDORSE→ABSTAIN, and familiarity=1.0 with no provenance stays ABSTAIN. No threshold is touched. |

H_PEB and H_SLEEPER are not mutually exclusive; the experiment below separates
them by crossing prior-act availability with retention interval. The brain
analogy supports *withdrawal requires a recollected prior act*, but the specific
engine fields (`prior_decision`, `source_status`) are an instantiation, not a
neural claim. The verbal W token is a proxy for the toddler's behavioural
reevaluation, and the invalidation cue is experimenter-delivered — both recorded
as extrapolations.

## 3. Boot-free prototype — RESULT (run, not predicted)

Probe: `scratch/wfa_prior_endorsement_probe.py`. For each of the four recorded
failures it builds a fresh in-memory `ConceptGraph` (no boot, no store), writes
the authoritative claim via the REAL production `w._write_asserted_wfa_fact`,
applies the REAL `w.apply_source_invalidation`, stages with the REAL
`w.context_from_graph_provenance`, and decides with the REAL
`evaluate_source`. Four arms vary only `prior_decision` and invalidation.

| item | A: prior=None, invalidated | B: prior=ENDORSE, invalidated | staged provenance identical? |
|---|---|---|---|
| gg039 | ABSTAIN / N / `R_INVALIDATION_NO_PRIOR` | WITHDRAW / W / `R_SOURCE_INVALIDATED` | yes |
| gg032 | ABSTAIN / N / `R_INVALIDATION_NO_PRIOR` | WITHDRAW / W / `R_SOURCE_INVALIDATED` | yes |
| g64 | ABSTAIN / N / `R_INVALIDATION_NO_PRIOR` | WITHDRAW / W / `R_SOURCE_INVALIDATED` | yes |
| g62 | ABSTAIN / N / `R_INVALIDATION_NO_PRIOR` | WITHDRAW / W / `R_SOURCE_INVALIDATED` | yes |

Arm A reproduces the full-run observation exactly; arm B produces the expected
endpoint. The staged fingerprint, trust class, matching-edge count and
invalidation count are byte-identical between A and B — **the only changed
input is the prior-endorsement record**. Controls: C1 item-metadata mutation
leaves the staged fingerprint unchanged (f19a3423…); C2 corrupting stored
`edge_kind` flips ENDORSE→ABSTAIN; C3 familiarity=1.0 with no provenance stays
ABSTAIN/`R_UNKNOWN_PROVENANCE`; C4 three repeats are byte-identical.

**Verdict: `PRIOR_ENDORSEMENT_IS_THE_DISCRIMINATING_VARIABLE`.** All seven
assertions true (4 × reproduce × flip × only-prior, plus C1–C4). This is a
unit-level, boot-free result: it establishes *where the decision variable is*,
not a new headline score and not where the prior ENDORSE is lost in the engine.

## 4. The experiment to run next (live; NOT run here)

Pre-register before any live run. Keep every `TH` key, gate and item unchanged;
the only manipulated variable is engine isolation and, in E3, the provenance of
the `prior_decision` input (engine audit vs harness bookkeeping).

**E2 — shared vs fresh engine, full n≥20/pool.** Run the full harness twice with
the same seed and frozen fixtures: (P) default shared `is_engine`; (F)
`WFA_IS_FRESH_ENGINE=1` (flag already present at `measure_learn5_wfa.py:2522`).
No code or threshold change. Record sealed/primary discredited rates, the four
item IDs, their `prior_decision`, `invalidated` and `staged_fingerprint`.
Expected: P reproduces 0.80 with four `R_INVALIDATION_NO_PRIOR`; F yields 1.00
if shared-engine state is what drops the first-turn ENDORSE (the earlier focused
fresh-engine confirmation with the anchor fix was 15/15). If F still yields
0.80 on the same four, H_PEB's *record* is lost even in isolation.

**E3 — prior-act re-staging, shared engine.** At the first (prior-commitment)
turn, capture `engine._last_source_evaluation.decision` immediately after the
turn and pass that object as `prior_decision` into the second turn's
`context_from_graph_provenance` call; do not recompute it from item fields.
Expected: sealed discredited → 1.00 with no change to any stored edge, if the
loss is in cross-turn bookkeeping. If it stays 0.80 while E2's F arm passes,
the loss is inside engine state, not the harness.

**E4 — sleeper/delay arm.** Keeping `prior_decision=ENDORSE` from the engine
audit, insert 0 / 1 / 3 / 6 intervening `process_turn` calls before the
invalidation decision, matched for compute. Expected under H_PEB: withdrawal
flat at 1.00 across delays (the prior act is still present). Expected under
H_SLEEPER: monotonic decay; a retrieval cue naming the original source restores
it. Only a decay pattern justifies adding a re-binding mechanism.

**E5 — no-staged-context falsifier.** Ask the same source question with the
gate's staged context withheld (no `set_source_context_for_turn`) and count
explicit Y/N/W decisions. Expected ~0: that would confirm the internal-source
arm's near-1.00 is **G6-saturated mechanism implementation**, not autonomous
source recollection (`SOURCE_EVALUATION_DESIGN.md` §6), and must be reported as
such. Do not tune thresholds to make this arm move.

**Proposed next mechanism only if E2/E3 locate an absent prior-act trace:** a
default-off, scratch-tested episodic record keyed by the same claim-ID anchor
that fixed g05, written by the engine at the endorsement turn and read at
retrieval. It must be shown to change the endpoint under E2/E3 with E5's
controls intact before any production wiring; otherwise the gap is bookkeeping
and the fix belongs in the harness.

## 5. Pre-registered discriminating outcomes

| observed | conclusion | next action |
|---|---|---|
| E2 F = 1.00 and E3 = 1.00 | Remaining 0.80 is shared-engine bookkeeping, **not** a source-monitoring capability gap | Harness isolation or audit re-staging; no threshold, no new mechanism |
| E2 F = 0.80, E3 = 1.00 | Prior ENDORSE is lost inside the shared engine between turns | Engine-state fix, default-off; re-run E5 |
| E2 F = 0.80, E3 = 0.80 | No durable prior-act record exists in the engine | Implement claim-ID-keyed prior-act trace in scratch, A/B with E5 intact |
| E4 decays with delay | H_SLEEPER contributes: cue-content dissociation, not only a missing record | Add re-binding cue at retrieval; do not loosen trust gate |
| E5 ≈ 0 explicit decisions | G6 saturation; autonomous source attribution still absent | Keep the caveat with every internal-source number; do not cite as capability |

## 6. Falsifiers (measurable)

1. If supplying `prior_decision=ENDORSE` does **not** flip any of the four to
   WITHDRAW while `invalidated>=2`, H_PEB is false — re-open provenance/trust.
2. If the staged fingerprint changes when `prior_decision` changes (it does not
   in the boot-free probe), the variable is leaking from item metadata.
3. If damaging stored provenance leaves ENDORSE unchanged, the endpoint is not
   reading the engine's store (C2 must flip).
4. If familiarity=1.0 with no provenance endorses, the no-shortcut property is
   broken (C3 must abstain).
5. If E2 P differs from the recorded summary on the same four items with the
   same seed and fixtures, re-measure baseline on current main before diagnosing.
6. If E5 yields many explicit decisions with no staged context, the "no decision
   routine" diagnosis is stale and the mechanism claim changes.
7. If E4 decays but E3 restores, delay rather than absence is the cause.

## 7. Limits

- Boot-free only. No engine, store, mirror, cold reload or 120-condition run was
  exercised; no new headline score is claimed.
- The evaluator is a deterministic rule table. Even a live 1.00 on the
  internal-source arm is a **G6 FLAG** (mechanism implementation), not evidence
  of human-like autonomous source recollection. Keep `0 < R < 0.999`
  informativeness reporting; do not tune to a mid-range number.
- The sleeper-effect literature is attitude-change evidence; applying it to
  RAVANA's assertion store is an analogy, not a homologue. Cue-timing conditions
  (Pratkanis et al. 1988) are contested in their boundary conditions; E4 is the
  test, not an assumption.
- g05 itself is fixed and not among the four failures; this experiment is about
  a different, downstream variable.

## Appendix A — probe script (verbatim, gitignored `scratch/`)

```python
#!/usr/bin/env python3
"""WFA prior-endorsement binding -- boot-free discriminating probe.

NO CognitiveChatEngine boot, NO weight store, NO production edits, NO gate or
threshold change. Every function under test is imported UNMODIFIED from the
repo harness (measure_learn5_wfa) and the pure evaluator (source_evaluation).

Question after the g05 claim-ID-anchor provenance fix: the full WFA run still
scores sealed discredited = 0.80. Four sealed items fail with
`rule=R_INVALIDATION_NO_PRIOR`, `invalidated>=2` and `prior_decision=None`.
Provenance tags therefore FIRED; what is missing is the record of the engine's
OWN prior endorsement. This probe asks whether that single field is the whole
difference between the observed ABSTAIN and the expected WITHDRAW, with every
other stored input held identical.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
for _p in (os.path.join(PROJ, "scripts"),
           os.path.join(PROJ, "ravana", "src"), PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import measure_learn5_wfa as w  # noqa: E402
from ravana.chat.source_evaluation import (  # noqa: E402
    ClaimContext, Decision, evaluate_source)
from ravana_ml.graph import ConceptGraph  # noqa: E402

SUMMARY = os.path.join(HERE, "learn5_wfa_summary.json")
OUT = os.path.join(HERE, "wfa_prior_endorsement_results.json")


class GraphEngine:
    """Minimal in-memory graph holder: real ConceptGraph, no boot, no store."""

    def __init__(self):
        self.graph = ConceptGraph(dim=8)


def recorded_failures():
    """Read-only: the four sealed discredited rows the full run got wrong."""
    with open(SUMMARY, encoding="utf-8") as fh:
        summary = json.load(fh)
    bad = []
    for row in summary.get("internal_source", {}).get("audit", []):
        flags = row.get("is_discredited_flags") or {}
        if row.get("pool") == "sealed" and flags.get("correct") is False:
            bad.append({
                "id": row["id"],
                "recorded_decision": flags.get("decision"),
                "recorded_token": flags.get("token"),
                "recorded_rule": flags.get("rule"),
                "recorded_invalidated": flags.get("invalidated"),
                "recorded_fingerprint": flags.get("fingerprint"),
            })
    return bad


def run_case(claim, source_id, prior, do_invalidate=True, corrupt_stored=False):
    """One arm: real teach -> optional real invalidation -> real stage -> decide."""
    eng = GraphEngine()
    written = w._write_asserted_wfa_fact(eng, claim, source_id)
    if corrupt_stored:
        # Mutation control: damage ONLY stored provenance (edge_kind), before
        # the invalidation tag, so the trust class must move.
        tok = w.wfa_claim_token(claim)
        for (s, t), edge in w._iter_graph_edges(eng):
            labels = (w._graph_node_label(eng.graph, s) + " " +
                      w._graph_node_label(eng.graph, t))
            if tok in labels:
                edge.source_metadata["edge_kind"] = "co_occurrence"
                break
    n_inv = w.apply_source_invalidation(eng, source_id) if do_invalidate else 0
    ctx, audit = w.context_from_graph_provenance(
        eng, claim, source_id=source_id, prior_decision=prior)
    dec = evaluate_source(claim, ctx)
    return {
        "written": written,
        "invalidated": n_inv,
        "prior_decision": prior.value if prior is not None else None,
        "staged_trust": audit["trust_class"],
        "n_matching_edges": audit["n_matching_edges"],
        "staged_invalidated": ctx.invalidated,
        "familiarity": ctx.familiarity,
        "decision": dec.decision.value,
        "token": dec.token,
        "rule": dec.rule,
        "fingerprint": audit["fingerprint"],
    }


def main():
    items = {it["id"]: it for it in w.build_items()}
    failures = recorded_failures()
    result = {
        "probe": "WFA prior-endorsement binding (boot-free)",
        "no_engine_boot": True,
        "source_of_failures": "scratch/learn5_wfa_summary.json (read-only copy)",
        "n_recorded_failures": len(failures),
        "cases": {},
        "controls": {},
        "verdict": None,
    }

    for fail in failures:
        item = items[fail["id"]]
        claim, sid = item["claim"], w.is_source_id(item["claim"], "discredited")
        arms = {
            "A_as_recorded_prior_none": run_case(claim, sid, None),
            "B_prior_endorse_restored": run_case(claim, sid, Decision.ENDORSE),
            "C_prior_endorse_no_invalidation": run_case(
                claim, sid, Decision.ENDORSE, do_invalidate=False),
            "D_prior_none_no_invalidation": run_case(
                claim, sid, None, do_invalidate=False),
        }
        result["cases"][fail["id"]] = {
            "claim": claim,
            "source_id": sid,
            "recorded": fail,
            "arms": arms,
            "reproduces_ABSTAIN": (
                arms["A_as_recorded_prior_none"]["token"] == "N" and
                arms["A_as_recorded_prior_none"]["rule"] ==
                "R_INVALIDATION_NO_PRIOR"),
            "flips_to_WITHDRAW": (
                arms["B_prior_endorse_restored"]["token"] == "W" and
                arms["B_prior_endorse_restored"]["rule"] ==
                "R_SOURCE_INVALIDATED"),
            "only_prior_changed": (
                arms["A_as_recorded_prior_none"]["fingerprint"] ==
                arms["B_prior_endorse_restored"]["fingerprint"] and
                arms["A_as_recorded_prior_none"]["staged_trust"] ==
                arms["B_prior_endorse_restored"]["staged_trust"] and
                arms["A_as_recorded_prior_none"]["invalidated"] ==
                arms["B_prior_endorse_restored"]["invalidated"]),
        }

    first = failures[0]["id"]
    claim = items[first]["claim"]
    sid = w.is_source_id(claim, "discredited")

    # C1 anti-tautology: item metadata must not reach the staged context.
    eng = GraphEngine()
    w._write_asserted_wfa_fact(eng, claim, sid)
    w.apply_source_invalidation(eng, sid)
    _, base = w.context_from_graph_provenance(eng, claim, sid)
    mut = dict(items[first])
    mut.update({"expected_answer": "X", "source_status": "credible",
                "source_id": "forged", "reliability": 1.0})
    _, changed = w.stage_context_for_item(eng, mut, sid)
    result["controls"]["C1_item_metadata_tautology"] = {
        "base": base["fingerprint"], "mutated": changed["fingerprint"],
        "unchanged": base["fingerprint"] == changed["fingerprint"]}

    # C2 stored-provenance mutation: damaging ONLY stored edge_kind must move
    # the endpoint on the non-invalidated arm (no discredited tag to dominate).
    base_arm = run_case(claim, sid, Decision.ENDORSE, do_invalidate=False)
    mut_arm = run_case(claim, sid, Decision.ENDORSE, do_invalidate=False,
                       corrupt_stored=True)
    result["controls"]["C2_stored_provenance_mutation"] = {
        "stored_intact": base_arm["decision"],
        "stored_damaged": mut_arm["decision"],
        "flipped": base_arm["decision"] != mut_arm["decision"]}

    # C3 no-shortcut: high familiarity with unknown provenance stays ABSTAIN.
    eng = GraphEngine()
    _, audit = w.context_from_graph_provenance(eng, claim, sid)
    no_source_ctx = ClaimContext(
        source=None, prior_decision=None, invalidated=False, familiarity=1.0)
    no_source = evaluate_source(claim, no_source_ctx)
    result["controls"]["C3_familiarity_no_shortcut"] = {
        "n_matching_edges": audit["n_matching_edges"],
        "decision": no_source.decision.value, "rule": no_source.rule,
        "abstains": no_source.decision is Decision.ABSTAIN}

    # C4 determinism: same arm repeated is byte-identical.
    reps = [run_case(claim, sid, Decision.ENDORSE) for _ in range(3)]
    result["controls"]["C4_determinism"] = {
        "tokens": [r["token"] for r in reps],
        "identical": len({json.dumps(r, sort_keys=True) for r in reps}) == 1}

    all_cases = list(result["cases"].values())
    result["verdict"] = (
        "PRIOR_ENDORSEMENT_IS_THE_DISCRIMINATING_VARIABLE"
        if (len(failures) == 4 and
            all(c["reproduces_ABSTAIN"] and c["flips_to_WITHDRAW"] and
                c["only_prior_changed"] for c in all_cases) and
            all(c["unchanged"] for c in
                [result["controls"]["C1_item_metadata_tautology"]]) and
            result["controls"]["C2_stored_provenance_mutation"]["flipped"] and
            result["controls"]["C3_familiarity_no_shortcut"]["abstains"] and
            result["controls"]["C4_determinism"]["identical"])
        else "NOT_ESTABLISHED")

    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, default=str)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print("VERDICT:", result["verdict"])
    print("SHA256(summary input):",
          hashlib.sha256(open(SUMMARY, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
```

## Appendix B — reproduced run output (key lines)

```
reproduces_ABSTAIN / flips_to_WITHDRAW / only_prior_changed: true for gg039, gg032, g64, g62
controls: C1 unchanged=true; C2 flipped=true (ENDORSE->ABSTAIN);
          C3 abstains=true (R_UNKNOWN_PROVENANCE); C4 identical=true
verdict: PRIOR_ENDORSEMENT_IS_THE_DISCRIMINATING_VARIABLE
SHA256(summary input): 0fdc0cd297c5e6b4aedbd3a473a528b07ee2f75bdf57a6f565cc4a25ceff80c3
```
