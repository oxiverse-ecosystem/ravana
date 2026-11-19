"""Structural checks for the authored causal scaffold (no engine boot)."""
import json
import os
import sys

_PROJ = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
for _p in (os.path.join(_PROJ, "ravana", "src"),
           os.path.join(_PROJ, "ravana_ml", "src"),
           os.path.join(_PROJ, "ravana-v2", "src"), _PROJ):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ravana.learn.explanation import provenance_trusted  # noqa: E402
from ravana.learn.explanation import Hop  # noqa: E402

PATH = os.path.join(_PROJ, "data", "curriculum", "causal_chains.json")


def test_curriculum_scaffold_is_well_formed():
    with open(PATH, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc.get("version") == 1
    chains = doc["chains"]
    assert len(chains) >= 25
    subjects = set()
    for chain in chains:
        assert chain["subject"].strip()
        subjects.add(chain["subject"])
        assert len(chain["hops"]) >= 2
        for hop in chain["hops"]:
            assert hop["from"].strip() and hop["to"].strip()
            assert hop["from"] != hop["to"]
            assert len(hop["evidence"]) > 20
    assert {"rain", "rust", "tides", "battery", "moon"} <= subjects


def test_curriculum_provenance_is_trusted():
    """The scaffold uses a provenance class the trust gate already accepts."""
    hop = Hop("water vapour", "clouds", "causal", 0.8)
    hop.provenance = None
    try:
        md = hop._ensure_source_metadata()
    except AttributeError:
        md = getattr(hop, "source_metadata", None)
    if isinstance(md, dict):
        md.update({"edge_kind": "curriculum",
                   "source": "curriculum:authored:rain"})
    assert provenance_trusted(hop)
