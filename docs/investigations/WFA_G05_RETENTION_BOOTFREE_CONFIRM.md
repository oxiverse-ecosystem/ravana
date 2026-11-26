# WFA g05 provenance retention — independent boot-free confirmation

**Branch:** `invest/wfa-g05-retention-scratch` (new worktree from `main` @ `0829813`).
**Constraint compliance:** no edit under `ravana/src`, no change to `scripts/`, tests,
gates, thresholds, item bank, prompts or grading rules; **no `CognitiveChatEngine`
boot** and no weight store opened (a TRA-K focused run owns the store). Every function
under test is imported UNMODIFIED from the repo. The only committed artifact is this
doc; the probe lives in the gitignored `scratch/` by repo convention and is reproduced
verbatim in Appendix A.

## Exact finding

g05's taught provenance is **overwritten in place**, not pruned, on the first turn.
The authoritative claim in `build_items()` is:

> Mirrors reflect glow through specular reflection, where the inclination of incidence equals the inclination of reflection.

The engine's own extractors, printed by the probe on the exact first-turn prompt,
return:

- `OpenIEExtractor.extract(claim)` -> `('mirrors', 'causes', 'glow', 0.6)`
- `ChainWalkerMixin._asserted_causal_propositions(prompt)` -> `[('mirrors', 'glow')]`

Teaching therefore writes the WFA origin (`wfa-internal://discredited/f889f80bc0d6`,
`edge_kind=curated`, `source_type=seed`, **confidence 0.40725**, `relation_type=causal`)
on the natural-language `mirrors -> glow` pair — the same confidence 0.40725 observed
in the original failure. On the first `process_turn`,
`_consolidate_asserted_causal_facts` (`chat/chain_walker.py:878-930`) re-extracts the
**same pair** from the verbatim `Claim:` line inside the source-question prompt and
routes it through `KnowledgeIntake._write_edge` (`learn/intake.py:705-746`). Because the
edge already exists, `learn/intake.py:729`:

```python
meta.update(v.provenance.as_metadata())
```

replaces `source` with `"conversation"` and bumps `revisions`. The scoped census
falls 1 -> 0, `apply_source_invalidation` tags 0 edges, the staged context falls to
`trust_class=unknown`, and the engine correctly ABSTAINs. This is a **write-side
provenance collision**, not a prune, not a confidence floor, and not the decision
rule. The `0.40725` value is a symptom of the first-sighting write, not the cause.

The shipped fix (`scripts/measure_learn5_wfa.py:_write_asserted_wfa_fact`) writes the
authoritative record on short claim-hash nodes (`wfa_<sha1(claim)>[:12]` and
`...|target`); for g05 these are `wfa_d6281ca036b3` / `wfa_bafcaf57a0c2`. Those labels
contain no spaces (boot-sanitizer safe) and are disjoint from every word the assertion
extractor can mint, so consolidation cannot address that pair.

## A/B result (boot-free, one process, identical claim and origin)

| arm | rows after teach | `consolidation_written` | rows after first turn | outcome |
|---|---|---|---|---|
| A — natural-language pair (pre-fix write) | 1 (`source`=WFA origin) | 1 | **0** (`source`=conversation) | `provenance_lost = true` |
| B — synthetic claim-id anchor (shipped fix) | 1 (`source`=WFA origin) | 1 (harmless, natural pair) | **1** (`source`=WFA origin) | `provenance_retained = true` |

Staged endpoint on arm B: before invalidation `trust_class=credible`, fingerprint
`3bed25baa74f2b561c7e59ebef2a48bd247f334fe0c3c511e79d283d666ebc4f`;
`apply_source_invalidation` tags **1** edge; after invalidation `trust_class=discredited`,
fingerprint `b24931267bae835b1a9234d6756e7d7ac1e003c453168584b8c67b714d2fceac`.
The latter is byte-equal to the known-good teach-only fingerprint recorded for g05 in
`docs/investigations/WFA_ISOLATED_CONFIRM.md`, obtained here by an independent route
with no engine boot.

## Controls (all pass)

| control | result |
|---|---|
| C1 power — force the REAL `_write_edge` onto the anchor's own pair with conversation provenance | `source` WFA origin -> `conversation`, anchor destroyed. Arm B's survival is not a no-op. |
| C2 anti-tautology — mutate item `expected_answer` and `source_status` | staged fingerprint unchanged. No item metadata reaches the context. |
| C3 mutation — flip the stored edge's `edge_kind` to `co_occurrence` | staged trust flips `credible` -> `unknown`. The graded endpoint reads real stored provenance. |
| C4 collision — synthetic labels vs extractor words | disjoint (`{wfa_d6281ca036b3, wfa_bafcaf57a0c2}` vs `{mirrors, glow}`). |

All seven assertions (A reproduces, B credible-before, B discredited-after, C1-C4)
are true: **VERDICT `CONFIRMED_FIX`**.

## Probe validity — retracted runs, stated plainly

1. The first execution reported `NOT_CONFIRMED` because the probe compared
   `trust_class` against uppercase strings; the engine emits lowercase enum values.
   That was an instrument defect, retracted, not a system finding. Re-run against
   `TrustClass.CREDIBLE.value` / `TrustClass.DISCREDITED.value` gives all-true.
2. An earlier exploratory extractor probe used a paraphrase ("reflect light") instead
   of the bank's claim and returned `('mirrors','light')`. Retracted: the authoritative
   claim says "glow"; the final probe reads `build_items()` and prints
   `('mirrors','glow')`, converting the earlier doc's *inference* into direct evidence.

## Brain grounding

Item memory and source memory dissociate: a fact can persist while its origin is
silently reassigned, which is why a confidence-only or prune-only diagnosis misses
this failure (Johnson, Hashtroudi & Lindsay 1993, *Psychological Bulletin*
114(1):3-28, doi:10.1037/0033-2909.114.1.3 — verified). Retrieval returns a
consolidated trace to a labile state in which new information is written into the old
trace (Nader, Schafe & Le Doux 2000, *Nature* 406:722-726, doi:10.1038/35021052 —
verified); `meta.update()` is that labile write. The fix does not disable
reconsolidation or weaken the gate — it places the authoritative source record on a
pair the consolidation route cannot reach, the same principle as the XdG protection
already used by `WebToGraph` for context-established synapses.

## Scope limits and falsifiers

- This is a **unit-level, boot-free** confirmation of the mechanism and the fix. It
  does not re-run the 120-condition `>=20/pool` suite and does not establish a new
  headline score. No engine, store, mirror or cold reload was exercised.
- Falsifier 1: if a full live run still loses g05 provenance, dump the anchor's
  `source_metadata` before/after `process_turn` and check for a **store re-seed or
  SQLite-mirror round-trip** that overwrites the in-memory write; this probe cannot
  see that path.
- Falsifier 2: if `mirrors`/`glow` resolve to different node ids under the live label
  index than at teach time, the collision would not occur; the A arm reproduced the
  exact 0.40725 curated edge using the real extractor and real node minting, so this
  is unlikely but only a live trace can close it.
- Falsifier 3: a confidence floor or trust-gate relaxation is **not** an acceptable
  repair; C3 shows the gate correctly follows stored provenance.

## Reproduce

From the worktree root: `uv run python scratch/wfa_g05_retention_scratch.py`
(writes `scratch/wfa_g05_retention_results.json` and stdout log). `scratch/` is
gitignored by founder convention, so the script is embedded verbatim below.

## Appendix A — probe script

```python
"""WFA g05 provenance retention -- boot-free scratch verification.

NO CognitiveChatEngine boot, NO weight store, NO production edits, NO gate or
threshold changes. Every function under test is imported UNMODIFIED from the
repo (chain_walker, intake, graph, and the WFA harness).

Question: does g05's taught provenance die at the first process_turn, and is
the shipped claim-id anchor fix sufficient?  A falsifiable A/B plus controls.
"""
import json
import os
import sys

sys.path.insert(0, "scripts")
import measure_learn5_wfa as w  # noqa: E402
from ravana.chat.chain_walker import ChainWalkerMixin  # noqa: E402
from ravana.chat.source_evaluation import Decision, TrustClass  # noqa: E402
from ravana.learn.intake import (Fact, IntakeVerdict, KnowledgeIntake,  # noqa: E402
                                 Provenance)
from ravana.web.openie import OpenIEExtractor  # noqa: E402
from ravana_ml.graph import ConceptGraph  # noqa: E402

ITEM = next(it for it in w.build_items() if it["id"] == "g05")
CLAIM = ITEM["claim"]
SID = w.is_source_id(CLAIM, "discredited")
PROMPT = w.source_question_prompt(CLAIM)
RESULT = {"claim": CLAIM, "source_id": SID, "prompt": PROMPT, "arms": {},
          "controls": {}, "verdict": None}


class FakeEngine(ChainWalkerMixin):
    """Minimal in-memory graph engine: real ConceptGraph, no boot, no store.

    Inherits ChainWalkerMixin so the REAL consolidation method can be called
    on it; only the small engine surface that method touches is emulated.
    """

    def __init__(self):
        self.graph = ConceptGraph(dim=8)
        self._all_labels = {}
        self._concept_keywords = {}
        self._concept_labels = set()
        self._provisional_nodes = {}
        self._junk_theta = 0.0
        self._promote_min_sources = 1
        self._trace_enabled = False
        self._intake = KnowledgeIntake(gate=None, ge=self)
        self._staged_ctx = None

    def _glove_vector(self, label):
        return None

    def _ensure_assertion_node(self, label):
        # REAL production node minting, plus the engine label index the gate
        # reads (a real engine maintains this alongside _concept_keywords).
        nid = ChainWalkerMixin._ensure_assertion_node(self, label)
        if nid is not None:
            self._all_labels[(label or "").strip().lower()] = nid
        return nid

    def _get_knowledge_intake(self):
        return self._intake

    def set_source_context_for_turn(self, claim, ctx):
        self._staged_ctx = ctx


def rows(eng):
    return w.graph_provenance_for_claim(eng, CLAIM, SID)


def verdict(fact, prov, route="novel_write"):
    return IntakeVerdict(
        fact=fact, consistency=0.5, elaboration=0.5, entrenchment=0.0,
        salience=1.0, route=route, contradiction=None, retention_prior=0.4,
        review_interval=4, provenance=prov, reason="scratch-teach")


def teach_on_natural_pair(eng):
    """Emulate the REAL teach write: WebToGraph fact -> intake._write_edge."""
    props = object.__new__(ChainWalkerMixin)._asserted_causal_propositions(PROMPT)
    cause, effect = props[0]
    s, t = eng._ensure_assertion_node(cause), eng._ensure_assertion_node(effect)
    assert s is not None and t is not None and s != t
    prov = Provenance(source=SID, source_type="seed", url=SID,
                      extractor="openie", operations=("openie-triple",))
    v = verdict(Fact(subject=cause, relation="causes", obj=effect,
                     confidence=0.40725, source_sentence=CLAIM), prov)
    ok = eng._intake._write_edge(v, 0.40725)  # REAL production write
    return ok, (cause, effect), (s, t)


# ---- 0. what the real extractors see (prior doc left this as inference) ----
openie_facts = [(f.subject, f.relation, f.obj, f.confidence)
                for f in OpenIEExtractor().extract(CLAIM)]
props = object.__new__(ChainWalkerMixin)._asserted_causal_propositions(PROMPT)
RESULT["extraction"] = {
    "openie_facts_from_claim": openie_facts,
    "asserted_propositions_from_first_turn_prompt": props,
    "claim_token": w.wfa_claim_token(CLAIM),
    "synthetic_labels": [w.wfa_claim_token(CLAIM),
                         w.wfa_claim_token(CLAIM + "|target")],
}

# ---- ARM A: pre-fix reproduction, natural-language node pair ----
a = FakeEngine()
ok, pair, ids = teach_on_natural_pair(a)
a_before = rows(a)
a_written = ChainWalkerMixin._consolidate_asserted_causal_facts(a, PROMPT)
a_after = rows(a)
RESULT["arms"]["A_pre_fix_natural_pair"] = {
    "write_ok": ok, "extracted_pair": pair, "node_ids": ids,
    "rows_after_teach": a_before, "consolidation_written": a_written,
    "rows_after_first_turn": a_after,
    "provenance_lost": len(a_before) == 1 and len(a_after) == 0,
}

# ---- ARM B: shipped fix -- synthetic claim-id anchor, same turn replay ----
b = FakeEngine()
n_anchor = w._write_asserted_wfa_fact(b, CLAIM, SID)  # REAL production fix
b_before = rows(b)
b_written = ChainWalkerMixin._consolidate_asserted_causal_facts(b, PROMPT)
b_after = rows(b)
RESULT["arms"]["B_synthetic_anchor"] = {
    "anchor_written": n_anchor, "rows_after_teach": b_before,
    "consolidation_written": b_written, "rows_after_first_turn": b_after,
    "provenance_retained": len(b_before) == 1 and len(b_after) == 1,
}

# staged context + invalidation on the retained anchor (the graded endpoint)
ctx0, aud0 = w.context_from_graph_provenance(b, CLAIM, SID)
n_inv = w.apply_source_invalidation(b, SID)
ctx1, aud1 = w.context_from_graph_provenance(
    b, CLAIM, SID, prior_decision=Decision.ENDORSE)
RESULT["arms"]["B_synthetic_anchor"]["staged"] = {
    "before": aud0, "invalidated_edges": n_inv, "after": aud1,
    "credible_before": aud0["trust_class"] == TrustClass.CREDIBLE.value,
    "discredited_after": aud1["trust_class"] == TrustClass.DISCREDITED.value,
}

# ---- CONTROLS -------------------------------------------------------------
# C1: power check -- overwriting the SAME anchor pair with conversation
#     provenance must destroy it (proves ARM B's survival is not a no-op).
c1 = FakeEngine()
w._write_asserted_wfa_fact(c1, CLAIM, SID)
tok = w.wfa_claim_token(CLAIM)
anchor_key = None
for (s, t), e in w._iter_graph_edges(c1):
    labels = (w._graph_node_label(c1.graph, s) + " " +
              w._graph_node_label(c1.graph, t))
    if tok in labels:
        anchor_key = (s, t)
        break
assert anchor_key is not None
old = dict(c1.graph.get_edge(*anchor_key).source_metadata)
conv_prov = Provenance(source="conversation", source_type="user", url="",
                       extractor="conversational-assertion",
                       operations=("taught-in-conversation",))
# Stub ONLY the gate's label->id lookup so the REAL _write_edge hits the SAME
# pair; the overwrite logic under test is still production code.
c1._intake.gate = type("G", (), {
    "_node_id": staticmethod(
        lambda ge, lab: anchor_key[0] if "mirror" in lab.lower() else anchor_key[1]),
    "_edge_type_for": staticmethod(lambda f: "causal"),
    "_edge_confidence": staticmethod(lambda e: getattr(e, "confidence", 0.5) or 0.5),
})()
c1._intake._write_edge(verdict(Fact("mirrors", "causes", "light", 0.9), conv_prov),
                       0.9)
RESULT["controls"]["C1_power_same_pair_overwrite"] = {
    "before_source": old.get("source"),
    "after_source": c1.graph.get_edge(*anchor_key).source_metadata.get("source"),
    "destroyed": c1.graph.get_edge(*anchor_key).source_metadata.get("source") == "conversation",
}

# C2: anti-tautology -- item metadata must not move the staged context.
c2 = FakeEngine()
w._write_asserted_wfa_fact(c2, CLAIM, SID)
_, base_audit = w.context_from_graph_provenance(c2, CLAIM, SID)
item2 = dict(ITEM); item2["expected_answer"] = "X"; item2["source_status"] = "credible"
_, mut_audit = w.stage_context_for_item(c2, item2, SID)
RESULT["controls"]["C2_item_metadata_tautology"] = {
    "base_fingerprint": base_audit["fingerprint"],
    "mutated_fingerprint": mut_audit["fingerprint"],
    "unchanged": base_audit["fingerprint"] == mut_audit["fingerprint"],
}

# C3: mutation control -- editing stored provenance must flip staged trust.
c3 = FakeEngine()
w._write_asserted_wfa_fact(c3, CLAIM, SID)
_, base3 = w.context_from_graph_provenance(c3, CLAIM, SID)
for (s, t), e in w._iter_graph_edges(c3):
    if tok in (w._graph_node_label(c3.graph, s) + " " +
               w._graph_node_label(c3.graph, t)):
        e.source_metadata["edge_kind"] = "co_occurrence"
        break
_, mut3 = w.context_from_graph_provenance(c3, CLAIM, SID)
RESULT["controls"]["C3_stored_provenance_mutation"] = {
    "base_trust": base3["trust_class"], "mutated_trust": mut3["trust_class"],
    "flipped": base3["trust_class"] != mut3["trust_class"],
}

# C4: collision impossibility -- synthetic label cannot be minted by extractor.
syn = set(RESULT["extraction"]["synthetic_labels"])
minted = {x for p in props for x in p}
RESULT["controls"]["C4_no_label_collision"] = {
    "synthetic": sorted(syn), "extractor_words": sorted(minted),
    "disjoint": syn.isdisjoint(minted),
}

# ---- verdict ---------------------------------------------------------------
A = RESULT["arms"]["A_pre_fix_natural_pair"]
B = RESULT["arms"]["B_synthetic_anchor"]["staged"]
ctrl = RESULT["controls"]
passes = (A["provenance_lost"] and B["credible_before"] and
          B["discredited_after"] and
          all(ctrl[k][list(ctrl[k])[-1]] for k in ctrl))
RESULT["verdict"] = "CONFIRMED_FIX" if passes else "NOT_CONFIRMED"
RESULT["all_assertions"] = {
    "A_reproduces_failure": A["provenance_lost"],
    "B_credible_before_invalidation": B["credible_before"],
    "B_discredited_after_invalidation": B["discredited_after"],
    **{k: list(v.values())[-1] for k, v in ctrl.items()},
}

out = "scratch/wfa_g05_retention_results.json"
with open(out, "w", encoding="utf-8") as fh:
    json.dump(RESULT, fh, indent=2, ensure_ascii=False, default=str)
print(json.dumps(RESULT, indent=2, ensure_ascii=False, default=str))
print("VERDICT:", RESULT["verdict"])

```
