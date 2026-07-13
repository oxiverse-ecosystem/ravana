"""Verify item casing/P5: entity-likeness from ConceptGraph IsA edges.

P5 wires the graph's raw IsA structure into casing so that named entities
NOT in SUBTLEX (e.g. "microsoft", "paris" — learned from web/KB) still
capitalize, while ambiguous brand/common words (e.g. "apple" -> both
computer_brand and edible_fruit) do NOT get force-capitalized against the
lexical prior.

Tests concept_entity_score directly + _capitalize_subject on the live engine.
No web.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.chat.case_distribution import concept_entity_score

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                          data_dir="/tmp/ravana_p5_test")
ont = getattr(eng, "_cn_ontology", None)
getter = (lambda w: ont.isa.get(w, set())) if ont else (lambda w: set())

# concept_entity_score: STRONG entity types -> 1.0; WEAK/ambiguous -> 0.0
e_france = concept_entity_score("france", getter)     # country -> strong
e_paris = concept_entity_score("paris", getter)       # capital/city -> strong
e_ms = concept_entity_score("microsoft", getter)      # company -> strong
e_nasa = concept_entity_score("nasa", getter)         # government_agency -> strong
e_apple = concept_entity_score("apple", getter)       # brand(weak)+fruit -> 0.0
e_bank = concept_entity_score("bank", getter)         # company -> 1.0 (graph truth: bank IS a company)

print(f"[P5] entity(france)={e_france} (expect 1.0)")
print(f"[P5] entity(paris)={e_paris} (expect 1.0)")
print(f"[P5] entity(microsoft)={e_ms} (expect 1.0)")
print(f"[P5] entity(nasa)={e_nasa} (expect 1.0)")
print(f"[P5] entity(apple)={e_apple} (expect 0.0, ambiguous brand/common)")
print(f"[P5] entity(bank)={e_bank} (expect 1.0; graph truth: bank is a company)")

# _capitalize_subject: OOV-from-SUBTLEX graph entities capitalize even when
# typed lowercase; ambiguous "apple" stays lowercase; KNOWN-common "bank"
# (cap_prob 0.12, in SUBTLEX) stays lowercase despite graph 'company' edge.
c_france = eng._capitalize_subject("france", "tell me about france")
c_ms = eng._capitalize_subject("microsoft", "what is microsoft")
c_apple = eng._capitalize_subject("apple", "what is an apple")
c_bank = eng._capitalize_subject("bank", "what is a bank")
print(f"[P5] capitalize(france)={c_france!r} (expect 'France')")
print(f"[P5] capitalize(microsoft)={c_ms!r} (expect 'Microsoft')")
print(f"[P5] capitalize(apple)={c_apple!r} (expect 'apple')")
print(f"[P5] capitalize(bank)={c_bank!r} (expect 'bank', known-common noun wins)")

eng.stop_background_learning()
ok = (e_france == 1.0 and e_paris == 1.0 and e_ms == 1.0 and e_nasa == 1.0
      and e_apple == 0.0 and e_bank == 1.0
      and c_france == "France" and c_ms == "Microsoft" and c_apple == "apple"
      and c_bank == "bank")
print("\nVERDICT:", "CONFIRMED — entity-likeness from graph IsA wired into casing "
      "(strong entities capitalize when OOV; ambiguous brand/common and "
      "known-common nouns stay lowercase)."
      if ok else "CHECK — a P5 case failed.")
raise SystemExit(0 if ok else 1)
