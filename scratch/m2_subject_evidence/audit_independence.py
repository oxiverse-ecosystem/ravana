"""Informational independence + reproducibility audit (no gate changes)."""
import hashlib, json, os, sys
sys.path.insert(0, 'scratch/m2_subject_evidence')
sys.path.insert(0, 'scripts')
import subject_type_evidence as ste
import experiment as ex
import tra_k_items

snap = json.load(open('data/subject_type_evidence/snapshot.json', encoding='utf-8'))
subs = ex.public_subjects()
os.environ[ste.ENV_FLAG] = '1'

role_vocab = set()
gold_vocab = set()
option_vocab = set()
for it in tra_k_items.build_items():
    for vals in it['type_sets'].values():
        role_vocab.update(str(v).lower() for v in vals)
    for vals in it['world_model'].values():
        gold_vocab.update(str(v).lower() for v in vals)
    for c in ('near', 'far'):
        rec = it['unseen_test_words'][c]
        gold_vocab.add(str(rec['gold']).lower())
        option_vocab.update(str(v).lower() for v in rec['candidates'])

emitted = set()
for s in subs:
    ev, _ = ste.evidence_for_subject(s, snap)
    emitted.update(e.type_name.lower() for e in ev)

print('emitted distinct type names', len(emitted))
print('overlap with type_sets values', len(emitted & role_vocab))
print('overlap with world_model/gold values', len(emitted & gold_vocab))
print('overlap with displayed options', len(emitted & option_vocab))
print('sample hash', hashlib.sha256(('|'.join(subs)).encode()).hexdigest()[:16])
print('facts sha', snap['provenance']['facts_sha256'])
