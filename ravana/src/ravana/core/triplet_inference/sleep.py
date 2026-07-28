"""Sleep-stage schema extraction over the triplet relational index.

Extends the existing consolidation machinery (core/consolidation.py
Consolidator) rather than duplicating it: the Consolidator promotes
episodic triples into semantic-graph edges; THIS extractor computes
batch relational statistics (offline replay) and folds them into the
RelationProfiles with the slow-integration bias of NREM (0.7 old /
0.3 batch — the CLS interleaving ratio, matching the plan and the
prior-session sleep-update convention), then records robust patterns
as RelationalSchema entries.

REM sabotage hook: mirrors core/sleep.py _rem_dream_sabotage — a small
random perturbation of transitivity evidence to test hypotheses
(Miconi & Kay 2025); post-sleep experience either reinforces or erodes
the perturbation because scores are count-backed.
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from .core import RelationalSchema
from .memory import TripletMemory


class SleepSchemaExtractor:
    def __init__(self, rng: random.Random = None):
        self.rng = rng or random.Random(0)

    def extract_schemas(self, memory: TripletMemory) -> int:
        """Batch (replay) pass over active triples. Returns the number
        of schemas recorded/updated."""
        by_pred: Dict[str, list] = defaultdict(list)
        for t in memory.active_triples():
            by_pred[t.predicate].append(t)

        n_schemas = 0
        for pred, triples in by_pred.items():
            prof = memory.profile(pred)

            # Batch transitivity over the replayed set.
            sp = {(t.subject, t.object) for t in triples}
            chains = 0
            closed = 0
            objs_of = defaultdict(set)
            for s, o in sp:
                objs_of[s].add(o)
            for s, o in sp:
                for c in objs_of.get(o, ()):
                    if c == s:
                        continue
                    chains += 1
                    if (s, c) in sp:
                        closed += 1
            if chains:
                batch_rate = closed / chains
                # NREM slow integration: blend batch evidence into the
                # counts at 30% of the batch chain volume.
                w = max(1, int(round(chains * 0.3)))
                prof.transitivity_pos += int(round(batch_rate * w))
                prof.transitivity_neg += w - int(round(batch_rate * w))

            # Record schema when the pattern is robust: the predicate's
            # Wilson lower bound clears the decision boundary AND the
            # exemplar count is above the cross-predicate mean
            # (distribution-relative, no fixed SCHEMA_MIN_EXEMPLARS).
            mean_count = (sum(len(v) for v in by_pred.values())
                          / max(1, len(by_pred)))
            if (prof.transitivity_lower() > 0.5
                    and len(triples) >= mean_count):
                sc = RelationalSchema(
                    pattern_type="transitive-chain", predicate=pred,
                    confidence=prof.transitivity_score,
                    n_exemplars=len(triples))
                memory.schemas[sc.key()] = sc
                n_schemas += 1
            if (prof.symmetry_lower() > 0.5
                    and len(triples) >= mean_count):
                sc = RelationalSchema(
                    pattern_type="symmetric-pair", predicate=pred,
                    confidence=prof.symmetry_score,
                    n_exemplars=len(triples))
                memory.schemas[sc.key()] = sc
                n_schemas += 1
        return n_schemas

    def rem_sabotage(self, memory: TripletMemory, rate: float = 0.1) -> int:
        """Randomly perturb a fraction of profiles' transitivity evidence
        by one optimistic count — hypothesis testing during REM. Count-
        backed scores mean unsupported perturbations wash out."""
        perturbed = 0
        for prof in memory.profiles.values():
            if self.rng.random() < rate and prof.transitivity_n > 0:
                prof.transitivity_pos += 1
                perturbed += 1
        return perturbed
