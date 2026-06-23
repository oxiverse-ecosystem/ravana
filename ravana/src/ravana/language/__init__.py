"""
RAVANA Language Module
Biologically-plausible language production modules for the RAVANA cognitive architecture.
Inspired by cortico-basal ganglia-thalamocortical (CBGTC) loops, the GODIVA model (Guenther 2016),
and Pulvermüller's neural syntax cell assemblies (2010).

Phase 6: Added VerbLexicon — semantically-driven verb selection (Levelt 1989).
Replaces the old VERB_PHRASES template system with competitive lemma selection
based on concept vector similarity.
"""
from .basal_ganglia import BasalGangliaGate
from .cerebellar_ngram import CerebellarNgram
from .prefrontal_workspace import PrefrontalWorkspace, DiscourseIntent, DiscoursePlan, DiscourseType
from .syntactic_cell_assembly import SyntacticCellAssembly, SyntacticFrame
from .surface_realizer import SurfaceRealizer, DiscourseState
from .verb_lexicon import VerbLexicon
