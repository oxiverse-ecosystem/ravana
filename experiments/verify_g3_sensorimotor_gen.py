"""Verify G3: sensorimotor-aware generation (Lancaster dual-coding).

Exercises the boundary files only:
- engine.py: _build_combined_encoder / _sensorimotor_vector / _sensorimotor_confidence
- verb_lexicon.py: set_sensorimotor_fn + embodiment-biased verb selection
- surface_realizer.py: set_sensorimotor_fn + sensorimotor-adjusted hedging

Checks (no web):
1. Engine exposes a sensorimotor read-out (CombinedAttributeEncoder built at
   init) and _sensorimotor_confidence returns 1.0 for well-grounded words and
   < 1.0 (or None->1.0) otherwise; concrete words score higher embodiment than
   abstract words.
2. VerbLexicon.select_verb is modulated by embodiment: for an embodied subject
   (e.g. "hand"/"tool") the selected verb root is more often an EMBODIED_ROOT
   than for an abstract subject (e.g. "trust"/"concept"), across many samples.
3. SurfaceRealizer._sensorimotor_adjusted_fe raises FE for OOD (low-confidence)
   words vs grounded words, so hedging increases for ungrounded concepts.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.language.verb_lexicon import VerbLexicon
from ravana.language.surface_realizer import SurfaceRealizer

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                          data_dir="/tmp/ravana_g3_verify")

ok = True

# ── 1. sensorimotor read-out ────────────────────────────────────────────────
enc = getattr(eng, "_combined_attr_encoder", None)
print(f"[G3] combined encoder built: {enc is not None}")
if enc is None:
    ok = False

# concrete vs abstract embodiment (uses VerbLexicon._embodiment_score)
emb_concrete = VerbLexicon._embodiment_score("hand")
emb_abstract = VerbLexicon._embodiment_score("trust")
print(f"[G3] embodiment hand={emb_concrete:.3f} trust={emb_abstract:.3f}")
if not (emb_concrete > emb_abstract):
    ok = False
    print("[G3] FAIL: concrete not more embodied than abstract")

conf_known = eng._sensorimotor_confidence("hand")
conf_oov = eng._sensorimotor_confidence("zzqxnope")
print(f"[G3] sensorimotor_confidence hand={conf_known:.3f} zzqxnope={conf_oov:.3f}")
if not (conf_known == 1.0 and conf_oov == 0.0):
    ok = False
    print("[G3] FAIL: OOV should be 0.0 confidence (hedge), grounded 1.0")

# ── 2. verb selection modulated by embodiment ────────────────────────────────
def embodied_root_rate(subject, n=400):
    counts = 0
    for _ in range(n):
        ph = VerbLexicon.select_verb("causal", subject=subject,
                                     object="thing", dopamine_tone=0.5)
        root = ph.split()[0].rstrip("s")
        if root in VerbLexicon.EMBODIED_ROOTS:
            counts += 1
    return counts / n

rate_concrete = embodied_root_rate("hand")
rate_abstract = embodied_root_rate("trust")
print(f"[G3] embodied-root rate hand={rate_concrete:.2f} trust={rate_abstract:.2f}")
if not (rate_concrete > rate_abstract):
    ok = False
    print("[G3] FAIL: embodied subject did not yield more embodied verbs")

# ── 3. surface realizer hedging adjusted by sensorimotor confidence ───────────
sr = SurfaceRealizer()
sr.set_sensorimotor_fn(eng._sensorimotor_confidence)
fe_grounded = sr._sensorimotor_adjusted_fe("hand", 0.2)
fe_oov = sr._sensorimotor_adjusted_fe("zzqxnope", 0.2)
print(f"[G3] adj FE hand={fe_grounded:.3f} zzqxnope={fe_oov:.3f}")
if not (fe_oov > fe_grounded):
    ok = False
    print("[G3] FAIL: OOD word did not raise free energy (hedging)")

eng.stop_background_learning()

print("\nVERDICT:",
      "CONFIRMED — G3 sensorimotor-aware generation wired: engine sensorimotor "
      "read-out + embodiment-biased verb selection + sensorimotor-driven hedging."
      if ok else "CHECK — a G3 case failed.")
raise SystemExit(0 if ok else 1)
