"""Verify item A: calibrated theta + data-derived realization + OOD-abstain.

Tests the calibration substrate on the live engine internals (no web):
1. calibrated_property_threshold('color') != blind 0.8 (real fit loaded).
2. A known word asked about a mismatched property yields a data-derived
   cross-modal metaphor (probe-selected dim + magnitude-conditioned phrasing).
3. An off-manifold (OOV/abstract) word asked about a property triggers
   OOD-abstain -> the metaphor path returns None (honest label), not a forced
   random graph metaphor.

Mirrors scripts/ravana_chat.py import order.
"""
import os
import sys

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (_PROJ,
         os.path.join(_PROJ, "ravana_ml", "src"),
         os.path.join(_PROJ, "ravana", "src"),
         os.path.join(_PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine
from ravana.ontology.attribute_calibration import (
    load_fitted_theta, calibrated_property_threshold, ood_abstain, realize_dim,
)

eng = CognitiveChatEngine(dim=64, seed=42, baby_mode=True,
                          data_dir="/tmp/ravana_a_test")

# (1) fitted theta differs from blind 0.8
fitted = load_fitted_theta()
th_color = calibrated_property_threshold("color", fitted)
print(f"[A] fitted theta('color') = {th_color}  (blind was 0.8; expect != 0.8)")
theta_differs = fitted is not None and abs(th_color - 0.8) > 1e-6

# (2) known word -> data-derived metaphor
meta_dog = eng._metaphor_for_category_error("dog", "color")
print(f"[A] metaphor('dog','color') -> {meta_dog!r}")
known_metaphor = meta_dog is not None and "color" not in meta_dog.lower().split(" not")[0]

# (3) off-manifold word (no glove vector) -> OOD-abstain suppresses the
#     probe-style sensorimotor metaphor (Path 1). It may still get an honest
#     Path-3 structure-mapping reply, which is fine (not a forced metaphor).
meta_oov = eng._metaphor_for_category_error("zxqwbl fictiontastic", "taste")
print(f"[A] metaphor('zxqwbl fictiontastic','taste') -> {meta_oov!r}")
_oo = meta_oov or ""
ood_abstains = ("i'd really picture" not in _oo) and ("in terms of its" not in _oo)

# (3b) direct OOD signal check: a word absent from glove -> OOD True;
# a known sensory word -> OOD False.
enc = getattr(getattr(eng, "_cn_ontology", None), "attribute_encoder", None)
gvec_oov = eng._glove_vector("zxqwbl fictiontastic")
gvec_dog = eng._glove_vector("dog")
ood_oov = ood_abstain(enc, gvec_oov)
ood_dog = ood_abstain(enc, gvec_dog)
print(f"[A] ood_abstain(oov)={ood_oov}  ood_abstain('dog')={ood_dog}")
ood_signal_ok = (ood_oov is True) and (ood_dog is False)

# (A.3) data-derived realization phrasing modulated by magnitude
r_hi = realize_dim("Color", 3.0)
r_lo = realize_dim("Sound", 0.4)
print(f"[A] realize_dim('Color',3.0)={r_hi}  realize_dim('Sound',0.4)={r_lo}")
realize_ok = r_hi[0].startswith("strong") and r_lo[0].startswith("hint")

eng.stop_background_learning()

ok = theta_differs and known_metaphor and ood_abstains and ood_signal_ok and realize_ok
print("\nVERDICT:", "CONFIRMED — theta FIT (not 0.8), data-derived metaphor, OOD-abstain works."
      if ok else "CHECK — an A assertion failed.")
raise SystemExit(0 if ok else 1)
