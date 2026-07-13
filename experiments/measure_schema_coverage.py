"""Measure schema-coverage substrate (research item: Schema Completion).

Parallel to the B/E/D/C/A calibration harnesses: MEASURE before changing.

Counts, for the engine's narrative-generation paths:
- TEMPLATE: subjects served by hardcoded EventSchema process strings
  (event_schema.py seed schemas).
- VSA_EVENT: subjects served by a VSA event/process schema in SchemaLibrary
  (currently 0 — SchemaLibrary only has syntactic SVO schemas).
- VSA_SYN: syntactic SVO schemas available (always >0).
- FREE_ASSOC: subjects that fall through to free-association only.

This establishes the baseline: N hardcoded-template subjects, 0 VSA-event
subjects. After Phase 2 (extend SchemaLibrary with event schemas + migrate
seeds + rewire response_gen), VSA_EVENT should cover the templated subjects
and the template path is deprecated.

No web, fast.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src")):
    sys.path.insert(0, p)

from ravana.core.event_schema import EventSchemaLibrary
from ravana.language.schemas import SchemaLibrary
from ravana.core.vsa import VSAManager

# Battery subjects (representative of what the agent is asked about).
BATTERY = [
    "trust", "knowledge", "love", "learning", "change", "friendship",
    "growth", "creativity", "communication", "respect", "hope", "truth",
    "life", "sleep", "dream", "memory", "time", "fear", "anger", "music",
    "art", "science", "language", "death", "freedom",
]

esl = EventSchemaLibrary()
esl.seed_default_schemas()
vsm = VSAManager()
sl = SchemaLibrary(vsm)

template_subjects = sorted(esl._schemas.keys())
vsa_syn_structures = [s.structure for s in sl.schemas]
# VSA event schemas would be tagged template_name starting with "EVENT-".
vsa_event_subjects = [s.template_name.replace("EVENT-", "")
                      for s in sl.schemas if s.template_name.startswith("EVENT-")]

coverage = {}
for subj in BATTERY:
    s = subj.lower()
    if s in template_subjects:
        path = "TEMPLATE"
    elif s in vsa_event_subjects:
        path = "VSA_EVENT"
    else:
        path = "FREE_ASSOC"
    coverage[s] = path

from collections import Counter
counts = Counter(coverage.values())

dashboard = {
    "item": "SCHEMA",
    "substrate": "utterance-generation path coverage",
    "template_subjects_total": len(template_subjects),
    "template_subjects": template_subjects,
    "vsa_syn_structures": vsa_syn_structures,
    "vsa_event_subjects_total": len(vsa_event_subjects),
    "vsa_event_subjects": vsa_event_subjects,
    "battery_path_counts": dict(counts),
    "verdict": (
        f"{len(template_subjects)} subjects served by HARDCODED EventSchema "
        f"templates; {len(vsa_event_subjects)} by VSA event schemas (SchemaLibrary "
        f"currently has only syntactic SVO schemas). Baseline established: "
        f"template path is the primary narrative generator and must be replaced "
        f"by VSA event schemas in Phase 2."
    ),
}

out = os.path.join(_PROJ, "experiments", "_schema_coverage.json")
with open(out, "w") as f:
    import json
    json.dump(dashboard, f, indent=2)

print(f"[schema] template subjects ({len(template_subjects)}): {template_subjects}")
print(f"[schema] vsa SYN structures: {vsa_syn_structures}")
print(f"[schema] vsa EVENT subjects ({len(vsa_event_subjects)}): {vsa_event_subjects}")
print(f"[schema] battery path counts: {dict(counts)}")
print(f"[schema] wrote {out}")
print("[schema] VERDICT:", dashboard["verdict"])
