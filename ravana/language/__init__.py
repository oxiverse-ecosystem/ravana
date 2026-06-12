"""
RAVANA Language Module
Biologically-plausible language production modules for the RAVANA cognitive architecture.
Inspired by cortico-basal ganglia-thalamocortical (CBGTC) loops, the GODIVA model (Guenther 2016),
and Pulvermüller's neural syntax cell assemblies (2010).
"""
from ravana.language.basal_ganglia import BasalGangliaGate
from ravana.language.cerebellar_ngram import CerebellarNgram
from ravana.language.prefrontal_workspace import PrefrontalWorkspace, DiscourseIntent, DiscoursePlan, DiscourseType
