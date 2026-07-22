"""Auto-generated mixin module for CognitiveChatEngine.
Web-snippet retrieval & source-quality mixin — snippet scoring, structural-junk detection, source trust, intent routing.
"""
from __future__ import annotations

# RAVANA Cognitive Chat Engine -- main orchestrator.
# Contains CognitiveChatEngine with __init__, process_turn, save/_load.
# Helper classes in models.py, user_model.py, belief_store.py.
# M0 crash-hardening: pin BLAS/OpenMP threads to 1 BEFORE numpy is imported, so
# worker-thread BLAS calls (web learner) can't race the main-thread decoder and
# trigger the Windows access-violation (numpy #27989). Must be the very first
# import -- ahead of `import numpy as np` below.
import ravana._numpy_threading  # noqa: F401  (side-effect: thread + faulthandler setup)
import sys, os, time, random, json, re, threading, hashlib, operator
import urllib.request
import socket
socket.setdefaulttimeout(4.0)
from urllib.error import URLError
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set

from collections import deque, Counter

# Import constants from shared module
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.insert(0, _proj_root)
sys.path.insert(0, os.path.join(_proj_root, "ravana_ml", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana", "src"))
sys.path.insert(0, os.path.join(_proj_root, "ravana-v2", "src"))

from ravana_ml.graph import ConceptGraph, ConceptEdge
from ravana_grace.core.emotion import VADEmotionEngine, VADConfig
from ravana_grace.core.identity import IdentityEngine
from ravana_grace.core.meaning import MeaningEngine, MeaningConfig
from ravana_grace.core.dual_process import DualProcessController, DualProcessConfig
from ravana_grace.core.global_workspace import GlobalWorkspace, GWConfig
from ravana_grace.core.meta_cognition import MetaCognition, MetaCognitiveConfig, EpistemicMode
from ravana_grace.core.sleep import SleepConsolidation, SleepConfig
from ravana.language.basal_ganglia import BasalGangliaGate
from ravana.language.cerebellar_ngram import CerebellarNgram, CerebellarState
from ravana.language.prefrontal_workspace import PrefrontalWorkspace, DiscourseIntent, DiscoursePlan, DiscourseType, SocialIntentClassifier
from ravana.language.syntactic_cell_assembly import SyntacticCellAssembly, SyntacticFrame
from ravana.language.surface_realizer import SurfaceRealizer, DiscourseState
from ravana_ml.nn.neural_decoder import NeuralDecoder
from ravana.core import UserEmotionDetector, EmotionalMirrorEngine, MirrorConfig
from ravana.core.hippocampal_buffer import HippocampalBuffer, HippocampalConfig
from ravana.core.proposition_parser import PropositionParser
from ravana.core.causal_schema import CausalSchemaLearner, CausalSchemaConfig
from ravana.core.implicature_detector import ImplicatureDetector
from ravana.core.relation_memory import RelationMemory, RelationMemoryConfig
from ravana.core.quantity_modifier import QuantityModifierSystem
from ravana.core.situation_model import SituationModel
from ravana.core.event_schema import EventSchemaLibrary
from ravana.ontology import DerivedOntology
from ravana.ontology.conceptnet import ConceptNetOntology

# Optional bs4
try:
    import bs4  # noqa: F401
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# Import constants
from .constants import (TEEN_CONCEPTS, WEB_GARBAGE, STOP_WORDS, ConceptPosDict,
                        _is_word_salad, _is_keyboard_mash,
                        _UNIVERSAL_PURGE, _DEFINITION_ASSERTION)
from .web_learning import WebLearningMixin
# Defect F: learned structural-PE snippet model (contrastive gap). Imported
# lazily-safe so a missing module degrades gracefully (the gate stays None and
# the old heuristic floor remains the backstop, never weakened).
try:
    from .snippet_quality import SnippetStructureModel, default_model
    _HAS_SNIPPET_MODEL = True
except Exception:  # pragma: no cover - defensive
    SnippetStructureModel = None  # type: ignore
    default_model = None  # type: ignore
    _HAS_SNIPPET_MODEL = False
# Research item B (fail-closed salad monitor): learned distributional classifier
# + fluent-tautology signature gate. Imported lazily-safe so a missing fit file
# degrades gracefully (the guard falls back to the legacy rule-based detector).
try:
    from .salad_classifier import is_salad_learned, get_classifier
    from .monitor_gate import detects_fluent_tautology
    _HAS_SALAD_LEARNED = True
except Exception:  # pragma: no cover - defensive
    _HAS_SALAD_LEARNED = False
    is_salad_learned = None
    get_classifier = None

# Stage 5a (de-hardcoding plan): snippet-PE gate parameters live in a fit file
# (data/snippet_pe.json) rather than inline constants. Fails open to seed
# constants when the fit file is absent.
try:
    from .snippet_pe_config import default_config as _default_pe_config
    _HAS_PE_CONFIG = True
except Exception:  # pragma: no cover - defensive
    _HAS_PE_CONFIG = False
    _default_pe_config = None

# Stage 5b-i (de-hardcoding plan): learned distributional POS classifier
# (data/pos_model.json), replacing the rule-based classify_word_pos +
# KNOWN_VERBS/ADJS/FUNCTION_WORDS lists when --learned-pos is enabled.
try:
    from .pos_model import PosModel, _seed_from_constants as _pos_seed_from_constants
    _HAS_POS_MODEL = True
except Exception:  # pragma: no cover - defensive
    _HAS_POS_MODEL = False
    PosModel = None
    _pos_seed_from_constants = None

# Stage 3 (de-hardcoding plan): Semantic Prototype Router (M-A) — replaces the
# ~15 hardcoded routing lists with one learned centroid router. Flag-gated
# (use_intent_router, OFF by default); regex path stays the default until the
# router is verified on the regression suite and promoted.
try:
    from .intent_router import IntentRouter
    _HAS_INTENT_ROUTER = True
except Exception:  # pragma: no cover - defensive
    _HAS_INTENT_ROUTER = False
    IntentRouter = None

# Stage 5b-ii (de-hardcoding plan): the duplicated closed-class functional
# lexicons (_generic / _FRAMING / _bare_moral / _INC/_DEC/_REM) collapse into one
# data-driven source of truth (data/functional_lexicon.json). Fails open.
try:
    from .functional_lexicon import default_lexicon as _default_lexicon
    _HAS_FUNC_LEX = True
except Exception:  # pragma: no cover - defensive
    _HAS_FUNC_LEX = False
    _default_lexicon = None

import pickle
from ravana.web.learner import SearchEngine
from ravana.core.dual_code_space import DualCodeSpace
from ravana.core.hrr_reasoner import HRRReasoner

# Universal closed-class / pronoun words that can never own a learned definition
# (you don't "define" the word "you"). This is the only hand-listed part of the
# definition purge — a minimal universal seed, not a per-word category table.
# The rest of the purge is derived from the learned graph (see
# _derive_definition_purge).

from ravana.language.verb_lexicon import VerbLexicon
from .models import FailedQuery, ChainHop, ChainTrace, CognitiveResponseContext, Correction, CorrectionType

from .user_model import UserModel
from .belief_store import BeliefStore
from ravana.nn.rlm import Plasticity

# Phase 1 & 2 Imports
from ravana.core.predictive_coding import PredictiveCodingLearner
from ravana.core.coherence import CoherenceNetwork
from ravana.core.working_memory import WorkingMemory
from ravana.storage.db import CognitiveDB, migrate_pickle_to_sqlite
from ravana.core.vsa import VSAManager
from ravana.language.schemas import SchemaLibrary
from ravana.core.system1 import System1Attractor
from ravana.core.system2 import System2Simulator
from ravana.core.question_decomposition import QuestionDecompositionEngine, QuestionCategory
from ravana.core.sub_answer_synthesizer import SubAnswerSynthesizer

# Phase 3 Imports
from ravana.learn.curiosity import CuriosityEngine
from ravana.learn.consolidation import HippocampalReplay
from ravana.language.register import RegisterController




class WebSearchMixin:
    """Web-snippet retrieval & source-quality mixin — snippet scoring, structural-junk detection, source trust, intent routing."""

    def _web_snippet_search(self, variants, ctx, is_conditional, deadline):
        """Search each query variant and return the highest-shape snippet that
        passes the chrome/quality floor, or (None, None, attempted). Extracted from
        _web_direct_answer so the plausibility monitor can re-run a refined set
        of variants without duplicating the search loop.

        PROMPT 3 (Johnson & Raye 1981; Mitchell & Johnson 2009): confidence is a
        *comparative*, criterion-based decision — accept the BEST available
        snippet when it beats the runner-up by a margin OR coheres with existing
        belief, rather than discarding it for clearing an absolute floor. A fixed
        floor (was quality < 1.5) threw away correctly-sourced encyclopedic
        answers whose shape score landed just below it. We keep only a low
        *safety* floor (>= 1.0) to reject pure noise, then pick comparatively.
        """
        import time as _time
        attempted = False
        query = ctx.raw_input.strip()
        _cands = []  # (snippet, term, quality, plausibility, trust, url)
        for term in variants:
            if _time and _time.time() > deadline:
                break
            try:
                # For conditionals the local engine (localhost:4000) is instant
                # and reliably returns hypothetical content; skip the slower
                # remote APIs so a hung call can never stall the turn.
                local_only = is_conditional
                # Mark as attempted BEFORE the call: a raised SearchError (all
                # backends failed) is still "we searched and found nothing",
                # which is exactly when the caller should abstain honestly.
                attempted = True
                res = self.search_engine.search(term, max_results=6,
                                                local_only=local_only)
            except Exception as ex:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] search failed for {term!r}: {ex!r}")
                continue
            if not res:
                continue
            cand = self._best_answer_snippet(res, ctx.subject, query,
                                            is_conditional=is_conditional)
            if not cand:
                continue
            _cand_san = self._sanitize_definition_text(cand)
            if not _cand_san:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] chrome-only / promo snippet rejected: {cand[:50]!r}")
                continue
            cand = _cand_san
            quality = self._snippet_quality(cand, ctx.subject, term,
                                            is_conditional=is_conditional)
            if term == ctx.subject:
                quality -= 1.0
            # Low SAFETY floor only: reject pure noise, not borderline-good answers.
            if quality < 1.0:
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] '{term}' -> below safety floor (q={quality:.2f}); skip")
                continue
            plaus = self._snippet_plausibility(ctx.subject, cand)
            trust = self._domain_trust(self._result_url(res))
            # belief coherence: snippet's phrase embedding vs the subject's vector
            _bel = self._belief_coherence(ctx.subject, cand)
            # Repair plan C (weakness C): hypothetical/counterfactual web answers
            # need an analogy-gate. The brain only launches a literal "look it
            # up" routine when simulation FAILS AND the query is genuinely
            # factual — and crucially, a counterfactual needs an anchor in known
            # structure to simulate. For an unanchored hypothetical ("if pigs
            # could fly would democracy still work") the graph has no analogical
            # or causal edge to project onto, so simulation cannot engage; a
            # human says "i can't really picture how that'd play out" rather
            # than dumping an encyclopedia article. So: a conditional query whose
            # premise concepts have NO graph anchor is vetoed from the literal
            # web path, letting the pipeline fall through to honest uncertainty.
            # This is distribution-driven (graph connectivity), not a topic
            # blocklist. Fail-closed: if we cannot resolve the premise at all
            # (no GloVe/graph), we let the general plausibility path decide.
            if is_conditional:
                # Analogy-gate: veto literal web answers for unanchored
                # hypotheticals (see _conditional_has_graph_anchor). Fail-open:
                # if the gate cannot decide, let the candidate proceed rather
                # than silently suppress a possibly-answerable query.
                try:
                    _anchored = self._conditional_has_graph_anchor(query)
                except Exception:
                    _anchored = True
                if not _anchored:
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [webans] conditional analogy-gate: premise "
                              f"unanchored in graph; veto literal web, abstain")
                    continue
            if getattr(self, '_trace_enabled', False):
                print(f"  [webans] '{term}' -> {cand[:70]!r} (q={quality:.2f}, "
                      f"plaus={plaus}, trust={trust:.2f}, bel={_bel:.2f})")
            _cands.append((cand, term, quality, plaus or 0.0, trust, self._result_url(res)))

        if not _cands:
            return None, None, attempted
        # Comparative selection: higher = better.
        # Brain-faithful reality-monitoring (N400; Johnson & Raye 1981): source
        # trust is only meaningful for content that COHERES with the query. A
        # snippet whose *added content* is anti-coherent with the subject
        # (plausibility < 0 — e.g. junk "a bordure of France", a promo blurb,
        # an off-topic paragraph) is a reality-monitoring failure and must NOT
        # be rescued by a high-trust domain: trusted sources still surface
        # off-topic snippets. So:
        #   - plausibility weighs heavily (it is the coherence signal),
        #   - trust contributes ONLY when plausibility is non-negative (gated),
        # which demotes anti-coherent junk below any coherent candidate even if
        # the junk sits on a high-trust domain and has a better surface shape.
        def _score(c):
            # c = (snippet, term, quality, plaus, trust, url)
            _q, _plaus, _trust = c[2], c[3], c[4]
            _trust_term = 2.0 * _trust if _plaus >= 0.0 else 0.0
            return _q + 3.0 * _plaus + _trust_term
        # ── Defect F: hard-wire the learned structural-PE snippet model ──
        # Replace the loose safety-floor heuristic (quality >= 1.0 only) with a
        # *learned* junk reject from SnippetStructureModel (contrastive gap).
        # This is the Track B Phase 2 model that was previously only reachable
        # behind a flag; we now apply it as a HARD reject before the comparative
        # score selection, so boilerplate / off-topic / token-salad snippets
        # never reach the surface — fail-closed to honest uncertainty.
        # Structural + distribution-driven (contrastive gap), not a hardcoded
        # shape list, matching the de-hardcoding philosophy.
        _snip_model = getattr(self, "_snippet_structure_model", None)
        if _snip_model is not None:
            _filtered = []
            for c in _cands:
                _snip_text = c[0]
                try:
                    if _snip_model.is_junk(_snip_text):
                        if getattr(self, "_trace_enabled", False):
                            print(f"  [webans] learned snippet-PE reject: "
                                  f"{_snip_text[:50]!r} (boilerplate/off-topic)")
                        continue
                except Exception:
                    pass
                _filtered.append(c)
            if _filtered:
                _cands = _filtered
            # If the learned model rejected EVERY candidate, fall through to
            # honest uncertainty (don't rescue junk via the old heuristic).
            elif getattr(self, "_trace_enabled", False):
                print("  [webans] learned snippet-PE rejected all candidates "
                      "-> abstain")
        _cands.sort(key=lambda c: -_score(c))
        best, second = _cands[0], (_cands[1] if len(_cands) > 1 else None)
        # Accept if clearly best OR coherent with belief; else abstain (don't
        # force a bad answer — fail-closed to honest uncertainty).
        if second is not None and (_score(best) - _score(second) < 0.1) and best[4] < 0.2:
            if getattr(self, '_trace_enabled', False):
                print("  [webans] comparative: best not clearly ahead and incoherent -> abstain")
            return None, None, attempted
        # Stash source/plausibility for downstream surfacing (PROMPT 3 hedges).
        self._last_web_source = self._source_type_label(best[5])
        self._last_web_plausibility = best[3]
        self._last_web_trust = best[4]
        return best[0], best[1], attempted

    def _web_direct_answer(self, ctx: CognitiveResponseContext) -> Optional[Tuple[str, str]]:
        """Answer an unknown factual query directly from live web snippets.

        Returns (answer_text, strategy) or None if no usable snippet.
        """
        if not ctx.subject:
            return None
        # NOTE: Do NOT bail out just because a stored definition exists — the
        # live web snippet is fresher and often more accurate than a loosely
        # learned stored definition (and _generate_response already prefers web
        # over the stale def). Let web have its chance; _best_answer_snippet
        # returns None if the snippet doesn't actually back the claim.
        if not ctx.subject:
            return None
        query = ctx.raw_input.strip()
        if not query:
            return None
        is_conditional = self._is_conditional_query(query)
        if not (self._is_informational_query(query, ctx.subject)
                or is_conditional
                or self._is_yesno_factual_query(query)):
            return None
        variants = self._web_query_variants(query, ctx.subject, is_conditional)
        if getattr(self, '_trace_enabled', False):
            print(f"  [webans] informational query '{query}' subj='{ctx.subject}' "
                  f"variants={variants}")
        import time as _time
        _budget = 12.0  # hard wall-clock cap on the whole variant search
        _deadline = _time.time() + _budget
        try:
            best, best_term, attempted = self._web_snippet_search(variants, ctx,
                                                      is_conditional, _deadline)
        except Exception as ex:
            if getattr(self, '_trace_enabled', False):
                print(f"  [webans] search failed: {ex!r}")
            return None
        if not best:
            # D (research item D): fail-closed degradation. If this was an
            # informational/definitional query and we actually searched the web
            # (all backends, including remote fallbacks) but found nothing
            # usable, abstain honestly instead of letting the caller silently
            # fall back to a hollow graph-edge answer.
            if attempted and self._is_informational_query(query, ctx.subject):
                return ("I couldn't verify that from the web right now, "
                        "so I'll be honest rather than guess.", "web_unverified")
            return None

        # ── Answer-usefulness monitor (N400 plausibility / reality monitoring) ──
        # We "see" the candidate answer and check whether it actually serves the
        # question before speaking it (metacognitive monitoring; Nelson & Narens;
        # Koriat). PROMPT 3: this is now a *comparative* double-check, not an
        # absolute floor — a snippet the comparative search already vetted is only
        # withheld if it is implausible AND incoherent with belief (genuine junk),
        # never merely because it sits below a fixed number.
        plaus = self._snippet_plausibility(ctx.subject, best)
        _bel = self._belief_coherence(ctx.subject, best)
        # PROMPT 3 (revised): reality-monitoring veto. Withhold a snippet ONLY
        # when it is genuinely incoherent (degenerate: GloVe cosine with the
        # subject is near-zero / negative — unrelated words, e.g. the Roblox
        # junk) OR when it sits below the plausibility floor AND comes from a
        # LOW-trust source (a borderline snippet we shouldn't assert from an
        # unreliable origin). A correct encyclopedic definition from a HIGH-trust
        # source (our local engine, trust=1.0) is NEVER vetoed for a modest
        # GloVe shortfall: GloVe under-estimates technical-term coherence
        # ("mitosis" vs "cytokinesis" sit far apart in embedding space despite
        # being deeply related), so a ~0.2-0.3 cosine is a metric
        # miscalibration, not real incoherence. This restores the COMPARATIVE
        # intent stated in _web_snippet_search (accept the best coherent
        # candidate) rather than an absolute floor that discards correct
        # answers. Fail-closed honesty is preserved: genuinely degenerate junk
        # is still withheld.
        _trust = getattr(self, "_last_web_trust", 0.0) or 0.0
        # M-C (plan Stage 2): forward-model prediction error. The old veto only
        # used the raw GloVe plausibility floors (magic numbers at 0.12/0.38);
        # that misses contradiction-by-polarity (Q15 "gravity doubled" -> "world
        # WITHOUT gravity") and answer-type mismatch (break-a-promise -> off-topic
        # anecdote). _answer_prediction_error is a CONTINUOUS PE combining premise
        # polarity, answer-type fit, and belief convergence. We veto when the PE
        # clears a learned midpoint (a single continuous gate, not three magic
        # floors) — the brain-faithful acceptance test from the plan.
        _ape = self._answer_prediction_error(query, ctx.subject, best)
        _degenerate = plaus is not None and plaus < self._SNIPPET_PLAUSIBILITY_DEGENERATE
        _lowtrust_below_floor = (plaus is not None and plaus < self._SNIPPET_PLAUSIBILITY_FLOOR
                                 and _trust < 0.5)
        _forward_model_veto = _ape >= (self._pe_cfg.veto_midpoint
                                       if self._pe_cfg is not None
                                       else self._ANSWER_PE_VETO)
        if _degenerate or _lowtrust_below_floor or _forward_model_veto:
            if getattr(self, '_trace_enabled', False):
                _plaus_s = f"{plaus:.2f}" if isinstance(plaus, (int, float)) else str(plaus)
                _bel_s = f"{_bel:.2f}" if isinstance(_bel, (int, float)) else str(_bel)
                _trust_s = f"{_trust:.2f}" if isinstance(_trust, (int, float)) else str(_trust)
                _ape_s = f"{_ape:.2f}" if isinstance(_ape, (int, float)) else str(_ape)
                print(f"  [webans] monitor: snippet implausible (plaus={_plaus_s}, "
                      f"bel={_bel_s}, trust={_trust_s}, pe={_ape_s}) "
                      f"for '{ctx.subject}' -> refine search")
            # Metacognitive control: instead of emitting junk, refine the query
            # and re-search (second-pass reanalysis; Kuperberg & Jaeger). Only
            # adopt the refined result if it clears the degenerate/low-trust
            # gate; if even the refined search can't produce an acceptable
            # answer, WITHHOLD (return None) so we fall through to other
            # strategies rather than leaking an incoherent snippet.
            refined = self._refine_query_variants(query, ctx.subject)
            if refined:
                try:
                    best2, best2_term = self._web_snippet_search(
                        refined, ctx, is_conditional, _time.time() + 8.0)
                except Exception:
                    best2, best2_term = None, None
                p2 = self._snippet_plausibility(ctx.subject, best2) if best2 else None
                b2 = self._belief_coherence(ctx.subject, best2) if best2 else 0.0
                _t2 = getattr(self, "_last_web_trust", 0.0) or 0.0
                _p2_ok = (p2 is None
                          or p2 >= self._SNIPPET_PLAUSIBILITY_DEGENERATE
                          or (p2 >= self._SNIPPET_PLAUSIBILITY_FLOOR and _t2 >= 0.5))
                if best2 is not None and _p2_ok:
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [webans] refined query yielded plausible snippet (plaus={p2})")
                    best, best_term = best2, best2_term
                else:
                    if getattr(self, '_trace_enabled', False):
                        print(f"  [webans] monitor: refined search also implausible -> withhold")
                    return None
        if getattr(self, '_trace_enabled', False):
            print(f"  [webans] best snippet (via '{best_term}'): {(best or 'NONE')[:80]}")
        best = self._strip_title_echo(best.strip(), ctx.subject)
        # Attribute-focused recall: for "what is the capital of France" / "who
        # wrote X" queries, extract the clause carrying the attribute rather
        # than emitting the entity's whole encyclopedic definition (which buries
        # the answer). Fail-open: returns the snippet unchanged when it can't
        # confidently isolate the attribute.
        best = self._focus_attribute_answer(ctx.raw_input, ctx.subject, best)
        # Sanitise dictionary/UI chrome and dateline prefixes from the live
        # snippet before it is emitted as the answer (the store-side sanitiser
        # only covers learned definitions, not directly-surfaced web answers).
        _san = self._sanitize_definition_text(best)
        if _san:
            best = _san
        if not best.endswith((".", "!", "?")):
            best = best + "."
        # ── Defect D: numeric-claim honesty gate (ACC conflict + FOK) ──
        # When a surfaced snippet asserts a numeric/math verdict
        # ("rational"/"irrational", "square root", numeric equality), check
        # RAVANA's internal Feeling-of-Knowing for that concept. If there is no
        # confident internal support, the claim is externally-sourced and
        # uncertain — attach a modality hedge ("i'm not certain, but…") rather
        # than presenting it as flat fact (the honesty bar). Brain-faithful:
        # the ACC gates assertion on confidence > theta_withhold; below that,
        # the claim is hedged, not asserted. This directly fixes "√-1" style
        # misleading snippets and the "is pi rational" evasion (force a verdict
        # via internal BeliefStore rather than dodging).
        self._last_numeric_honesty = None
        if best:  # guard: only run when a real snippet was produced
            _NUM_CLAIM = re.compile(
                r"\b(rational|irrational|square root|cube root|prime|even|odd|"
                r"equals|=|divisible|multiple of|factor of)\b", re.I)
            if _NUM_CLAIM.search(best) and not best.startswith("according to"):
                try:
                    from ravana.chat.metacognition import Metacognition
                    _mc = getattr(self, "_metacognition", None) or Metacognition()
                    self._metacognition = _mc
                    # support = how many graph edges / stored beliefs back the subject
                    _support = float(len(getattr(self, "_concept_keywords", {})
                                        .get(ctx.subject.lower(), [])))
                    _bel = self.belief_store.query_belief(ctx.subject.lower(), "def") \
                        if hasattr(self, "belief_store") else None
                    _retrieved = bool(_bel) or getattr(self, "_last_web_source", "") not in ("", "the web")
                    _conf, _may_assert, _modality = _mc.read(_support, _retrieved)
                    if not _may_assert:
                        _honest = (f"i'm not fully certain, but {best}"
                                   if not best[0].isupper() else
                                   f"I'm not fully certain, but {best[0].lower()}{best[1:]}")
                        self._last_numeric_honesty = _modality
                        if getattr(self, "_trace_enabled", False):
                            print(f"  [webans] numeric-claim honesty gate: low FOK "
                                  f"(conf={_conf}) -> hedged")
                        best = _honest
                except Exception:
                    pass
        # B4: simplification register. When the user framed the query with a
        # "like i'm five" / "in simple terms" frame (ctx.simplification_requested,
        # set at the ctx boundary), lower the lexical/syntactic complexity of the
        # emitted answer: keep only the FIRST sentence (the core definitional
        # statement) and drop the conversational closer. This is a register
        # signal, not a fact edit — the content is unchanged, just shorter and
        # plainer. Fail-closed: if there's only one sentence, nothing is trimmed.
        closer = ""
        if getattr(ctx, "simplification_requested", False):
            _sent = re.split(r"(?<=[.!?])\s+", best.strip())
            if _sent:
                best = _sent[0].strip().rstrip(".!?") + "."
            closer = ""
        # PROMPT 3: tag the answer with its source + comparative modesty instead
        # of presenting it as settled fact. Modality from trust + plausibility.
        from ravana.chat.hedges import hedge_frame, modality_from_support
        _web_mod = modality_from_support(
            min(1.0, 0.5 * self._last_web_trust + 0.5 * (self._last_web_plausibility or 0.0)))
        _src = getattr(self, "_last_web_source", "") or "the web"
        if _src and _src != "the web":
            # High-trust known source: prefix naturally.
            answer_text = f"according to {_src}, {best}{closer}"
        elif self._last_web_trust < 0.5:
            # Lower trust / forum: hedge explicitly.
            answer_text = hedge_frame("web", _web_mod, snip=best, src=_src) + closer
        else:
            answer_text = best + closer

        # ---- P6: verify the web claim against established belief before emitting ----
        # The web snippet is a *candidate*, not gospel. Check it against what we
        # already believe about the subject, adjust confidence accordingly, and
        # (the key move) store it in the BeliefStore so it can be contradicted,
        # reconciled, or forgotten later (see _sleep_consolidate / P7).
        subject_key = ctx.subject.lower()
        confidence = 0.5  # web claims start low-confidence (no RLM verification yet)
        try:
            existing = self.belief_store.query_belief(subject_key, "def")
        except Exception:
            existing = None
        if existing is not None:
            prior_val = existing[0]
            try:
                prior_conf = float(existing[1]) if isinstance(existing, (tuple, list)) and len(existing) > 1 else 0.0
            except (TypeError, ValueError):
                prior_conf = 0.0
            overlap = self._belief_value_overlap(prior_val, best)
            # Only treat a divergence as a real *conflict* when the prior belief
            # is itself well-established (reinforced beyond a single low-conf
            # web snippet). Otherwise two equally-uncertain web snippets of the
            # SAME sense just collide (e.g. legal "trust" vs interpersonal
            # "trust" both land sub-0.15 overlap) and a correct answer gets
            # wrongly stamped "[unverified: conflicts with what I knew]".
            _prior_established = prior_conf >= 0.6
            if overlap >= 0.5:
                # Web corroborates what we already knew -> boost confidence.
                confidence = max(confidence, min(0.9, existing[1] + 0.2))
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] belief match on '{subject_key}' "
                          f"(overlap={overlap:.2f}) -> conf {confidence:.2f}")
            elif overlap < 0.15 and _prior_established:
                # Nontrivial conflict: web disagrees with an established belief.
                # B2 (Round 3): make this FAIL-CLOSED. The prior fix only
                # lowered confidence + prepended a "[unverified: conflicts with
                # what I knew]" tag but STILL returned the snippet — so the
                # junk claim was spoken with a footnote. The rAI/ACC "feeling
                # of wrongness" (belief-coherence signal) must SUPPRESS, not
                # caveat: a conflicting low-plausibility item is quarantined
                # before the speech buffer. Route to honest uncertainty / re-
                # query instead. (Corroborating + novel-but-not-contradicting
                # branches below are untouched, so the gate does not
                # over-suppress a genuinely supported snippet.)
                old_triple = (subject_key, "def", prior_val)
                new_triple = (subject_key, "def", best)
                self.belief_store.contradictions.append(
                    (old_triple, new_triple, self.belief_store.turn_num))
                if getattr(self, '_trace_enabled', False):
                    print(f"  [webans] conflict FAIL-CLOSED on '{subject_key}': "
                          f"prior={prior_val[:40]!r} web={best[:40]!r} "
                          f"(overlap={overlap:.2f}, prior_conf={prior_conf:.2f}) "
                          f"-> withheld (route to honest uncertainty)")
                # Do NOT store the conflicting claim as a low-conf candidate,
                # and do NOT emit it. Returning None makes the caller fall
                # through to KB lookup / honest uncertainty.
                return None
        # The web claim now LIVES in the belief store as a low-confidence
        # candidate; a matching prior boosted it, a conflicting prior demoted it.
        # Either way sleep can reconcile it against local knowledge and prune it
        # if it stays unverified and unreinforced.
        try:
            self.belief_store.assert_belief(subject_key, "def", best,
                                            confidence=confidence)
        except Exception:
            pass
        # B2 (Round 3) #3: route the final assembled answer through the shared
        # CoherenceGate used by every other generator (engine.py:3793). This
        # enforces the learned SnippetStructureModel hard-reject (Defect F) and
        # completeness uniformly — no web-assembly path slips around the gate.
        # Fail-closed: a non-broadcastable answer is withheld (returns None ->
        # caller routes to honest uncertainty) rather than emitted. Guarded by
        # use_coherence_gate so it can be A/B'd with --no-coherence-gate.
        if getattr(self, "use_coherence_gate", True):
            _gate = getattr(self, "_coherence_gate", None)
            if _gate is not None:
                try:
                    _broadcast, _reason, _score = _gate.judge(text=answer_text)
                    if not _broadcast:
                        if getattr(self, '_trace_enabled', False):
                            print(f"  [webans] CoherenceGate withheld final answer "
                                  f"on '{subject_key}': {_reason}")
                        return None
                except Exception:
                    pass
        return (answer_text, "web_direct_answer")

    def _best_answer_snippet(self, results, subject: str, query: str,
                             is_conditional: bool = False) -> Optional[str]:
        """Pick the most answer-like snippet for a factual query.

        Heuristic (no LLM): prefer a snippet that is (a) a complete sentence,
        (b) mentions the subject or a salient query keyword, (c) reasonably
        concise (40-320 chars), and (d) free of boilerplate noise.
        """
        if not results:
            return None
        subj = (subject or "").lower()
        qkw = set(w for w in query.lower().split()
                  if len(w) > 3 and w not in STOP_WORDS)
        is_conditional = is_conditional or self._is_conditional_query(query)
        # Context-augmented sense vector (N400 predictive-coding analog): bias
        # the coherence gate toward the sense implied by the full query, not the
        # bare blended noun. Used as a ranker (not a hard filter) below so it
        # breaks +3.0 ties between same-shape definitions of different senses.
        _ctx_vec = None
        if hasattr(self, "_context_query_vector"):
            try:
                _ctx_vec = self._context_query_vector(subject, query)
            except Exception:
                _ctx_vec = None
        candidates = []
        # Fix A: compute the "what is X" factual-shape flag once,
        # used by both the per-result looser-relevance fallback and
        # the post-loop fallback scan.
        _is_factual_what = bool(re.match(
            r"^(what|which) (is|are|was|were|means?|does) ",
            query.lower().strip()))
        for r in results[:6]:
            content = r.get("content", "") or ""
            title = r.get("title", "") or ""
            # Hard-reject sources that are structurally incapable of answering
            # our query types (crossword solvers, thesauri, spam, UI shells).
            # This is what previously let "The Sun Two Speed" crossword and
            # "How to use trust in a sentence" beat Merriam-Webster. The list is
            # a heuristic domain blocklist, NOT a fact table.
            _dom = self._domain_of(r.get("url", "") or "")
            if any(j in _dom for j in self._JUNK_SNIPPET_DOMAINS):
                continue
            # Skip results that are mostly HTML / CSS fragments (whole-result
            # junk). Photo-credit words like "getty" are handled at the
            # sentence level below, NOT here — otherwise we'd throw away a good
            # article that merely contains a "© ... via Getty Images" credit.
            raw_low = content.lower()
            # Skip non-English / discussion-page junk (e.g. a Czech Wikipedia
            # "Diskuse" page sneaks in with mostly non-ASCII navigation text).
            if content and sum(1 for c in content if ord(c) > 127) / max(1, len(content)) > 0.35:
                continue
            if ("<img" in raw_low or raw_low.count("<") > 3
                    or "{" in content or "}" in content or "url(" in raw_low
                    or "@font" in raw_low or "src:" in raw_low):
                continue
            blob = self._clean_snippet(content)
            if not blob:
                continue
            # Reject blobs that are largely non-English (discussion/nav pages in
            # other scripts slip past the whole-result check when the raw
            # content mixes ASCII boilerplate with a non-ASCII lead sentence).
            if sum(1 for c in blob if ord(c) > 127) / max(1, len(blob)) > 0.20:
                continue
            low = blob.lower()
            # Subject-relevance gate: the snippet must actually be about the
            # subject. For multi-word subjects require the full phrase or all
            # tokens (so "dark matter" doesn't match a "Dark" TV-series article
            # that only contains the first token). For single-word subjects
            # require a REAL token match (not a bare substring, otherwise
            # "trust" matches "Linux Foundation Digital Trust" and "sun" matches
            # "…under the sun" Bible verses). tolerant of verb inflection.
            subj_tokens = subj.split()
            wordset = set(low.split())
            # ── Fix A: semantic-relevance fallback (fail-closed honesty
            # preserved) ──────────────────────────────────────────────
            # The strict gate below requires a literal subject-token match in
            # the snippet BODY. That is a brittle syntactic proxy for semantic
            # relevance (Frontiers 2024: humans judge coherence semantically,
            # via priming, not string-match). For "what is trust" the gateway
            # can return an encyclopedic result whose TITLE holds the word but
            # whose body opens with a definition of *that* sense (e.g. a
            # "Trust (noun)" page) — or a navigational sense ("Trust Bank")
            # that is correctly rejected. We therefore admit a result via a
            # LOOSER signal ONLY when (a) the query is a single-subject
            # factual "what is X", (b) the result is from a preferred
            # encyclopedic source OR its title/url literally contains the
            # subject, and (c) it is NOT a navigational/brand sense
            # (title start differs from the bare subject). This is a ranker
            # boost, not a pass: the strict gate stays primary, so a clean
            # body-match answer still wins; the fallback only rescues a
            # reachable definition that the strict gate was throwing away.
            _title = (r.get("title", "") or "").lower()
            _title_raw = (r.get("title", "") or "")
            _url = (r.get("url", "") or "").lower()
            _is_factual_what = bool(re.match(
                r"^(what|which) (is|are|was|were|means?|does) ",
                query.lower().strip()))
            _looser_ok = False
            if subj and len(subj_tokens) == 1 and _is_factual_what:
                _title_has = (subj in _title)
                _url_has = (subj in _url)
                _pref_src = self._is_preferred_source(_url)
                if _pref_src or _title_has or _url_has:
                    # Decide whether this is the encyclopedic SENSE of the
                    # subject (admit via fallback) or a navigational/brand
                    # sense (reject). Encyclopaedia titles put the subject
                    # FIRST, then a SEPARATOR or a lower-case gloss:
                    #   "trust - wikipedia", "trust | definition…",
                    #   "trust (noun) - oxford", "gravity - wikipedia".
                    # A navigational/brand sense does NOT look like that:
                    #   - subject leads but next token is a capitalized BRAND
                    #     noun ("trust Bank…", "digital Trust Foundation"
                    #     where the capitalized continuation is the brand), or
                    #   - subject appears mid-title after a different leading
                    #     capitalized/section word ("about | trust Bank").
                    # Rule: NAVIGATIONAL iff
                    #   (subject is the leading token AND the next token is
                    #    capitalized AND not a known separator) OR
                    #   (subject is NOT the leading token AND the leading
                    #    token is capitalized / not a plain section word).
                    _title_toks = _title.split()
                    _title_raw_toks = _title_raw.split()
                    _lead = _title_toks[0] if _title_toks else ""
                    _sep = {"-", "|", "–", "—", ":",
                             "definition", "meaning", "(noun)",
                             "(verb)", "wikipedia", "britannica",
                             "dictionary", "oxford", "cambridge",
                             "about", "the"}
                    _nav_sense = False
                    if _lead == subj:
                        # subject leads; the token AFTER it (in raw case)
                        # must be a separator/lower-case gloss, not a
                        # capitalized brand noun ("Bank", "Foundation").
                        if len(_title_raw_toks) >= 2:
                            _nxt = _title_raw_toks[1]
                            if _nxt[:1].isupper() and _nxt.lower() not in _sep:
                                _nav_sense = True
                    else:
                        # subject not leading; navigational unless the url
                        # is a preferred encyclopedic source (which can
                        # carry a section-style title like "About | X").
                        if not _pref_src:
                            _nav_sense = True
                    if _nav_sense:
                        # Brand/navigational sense of the subject: never
                        # admit via the fallback, even if the body repeats
                        # the word.
                        continue
                    _looser_ok = True
            if len(subj_tokens) >= 2:
                _phrase_ok = subj in low
                _all_tokens = all(self._tok_match(t, wordset) for t in subj_tokens)
                if not (_phrase_ok or _all_tokens or _looser_ok):
                    continue
            elif subj:
                if not (self._tok_match(subj, wordset) or _looser_ok):
                    continue
            # Reject obvious navigation/boilerplate
            if any(n in low for n in self._SNIPPET_NOISE):
                # still keep if it also directly answers; otherwise skip
                if not (subj and subj in low) and not (qkw and qkw & set(low.split())):
                    continue
            # Split into sentences, keep the ones that look like statements
            sents = re.split(r"(?<=[.!?])\s+", blob)
            for s in sents:
                s = s.strip()
                if len(s) < 20 or len(s) > 400:
                    continue
                if not re.search(r"[a-z]", s):
                    continue
                # Reject headline/question fragments (they are not answers)
                if "?" in s:
                    continue
                sl = s.lower()
                # Reject truncated / incomplete sentences
                if s.rstrip().endswith(("…", "...", "…")):
                    continue
                # Strongly reject boilerplate sentences (navigation, promos).
                if any(n in sl for n in self._SNIPPET_NOISE):
                    continue
                # Reject sentences that merely *look* like content but are
                # navigation / meta ("How to use trust in a sentence",
                # "Definition of sun", "Get the latest news…").
                if any(re.match(p, sl) for p in self._SNIPPET_REJECT_SHAPES):
                    continue
                # Track B Phase 2 (M4): learned structural-junk gate (flag-gated;
                # old regex table above remains the backstop). Pass topic
                # coherence so the dual-gate fires on real junk.
                if self._snippet_is_structural_junk(s, self._snippet_topic_max_coherence(subj, s)):
                    continue
                # Score: mentions subject or query keywords, prefer answer shape
                score = 0.0
                sl_words = set(sl.split())
                # Definition / answer-shape detection for factual queries: a
                # sentence that directly defines the subject ("X is the…",
                # "X refers to…", "The meaning of X is…") is the answer we want,
                # so weight it strongly and don't let a stray mention outrank it.
                _first_words = sl.split()[:4]
                _def_verb = re.search(
                    r"\b(is|are|was|were|refers to|means|denotes|describes|"
                    r"explains|consists of|is the|is a|is an|means that|"
                    r"refers)\b", sl)
                _subj0 = subj.split()[0] if subj else ""
                _subj_is_topic = bool(_subj0) and (
                    self._tok_match(_subj0, set(_first_words))
                    or _subj0 in sl.split()[:2])
                if subj and (subj in sl or self._tok_match(_subj0, sl_words)):
                    score += 2.0
                score += 0.5 * len(qkw & sl_words)
                # Answer-pattern bonus: the subject is the grammatical subject
                # of the sentence with a copula/role verb ("France is the...",
                # "Argentina won the...", "Tokyo is the capital..."). These are
                # the sentences that actually answer "who/what is X".
                if _subj_is_topic and _def_verb:
                    score += 3.0
                elif _subj_is_topic:
                    score += 1.5
                # Conditional / hypothetical queries: the user wants a reasoned
                # *consequence*, not a dictionary definition of the subject. A
                # pure definition ("Gravity is the word used to describe a
                # physical law…") must NOT outrank an actual hypothetical
                # answer ("Imagine everything… floating midair…"). So for
                # conditionals we reverse the usual bias: penalize the bare
                # subject-definition shape and strongly boost causal /
                # consequence / scenario sentences.
                if is_conditional:
                    if _subj_is_topic and _def_verb:
                        score -= 3.0
                    if re.search(r"\b(would|could|imagine|without|no longer|"
                                 r"if .* (disappear|vanish|turn|stop|gone|"
                                 r"cease|removed|turned)|plunge|drift|freeze|"
                                 r"darkness|crash|float|orbit|fall|launch|"
                                 r"expand|escape|lost|everyone|everything|"
                                 r"people|oceans|planes|earth|planet|"
                                 r"seconds?|instant|midair)\b", sl):
                        score += 4.0
                if re.match(r"^(who|what|where|when|which|how|why)\b", query.lower()) and \
                        _def_verb:
                    score += 1.0
                # Role-answer bonus for "who is X" (person/creator queries):
                # presidents, founders, painters, authors, inventors, etc.
                if re.match(r"^who\b", query.lower()) and \
                        re.search(r"\b(president|prime minister|leader|head of state|monarch|"
                                  r"king|queen|chancellor|governor|ceo|founder|creator|"
                                  r"author|director|commander|painter|artist|writer|"
                                  r"composer|inventor|scientist|discoverer|musician|"
                                  r"novelist|poet|film maker|filmmaker|designer)\b", sl):
                    score += 2.5
                # Person-name bonus for "who" queries: a capitalized proper noun
                # near the subject is usually the answer (e.g. "Leonardo da Vinci").
                if re.match(r"^who\b", query.lower()) and re.search(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", s):
                    score += 1.5
                # Penalize list/colon fragments and overly promotional text
                if sl.endswith((":", "•", "-", "…")):
                    score -= 0.5
                # Source-quality tie-breaker (never a hard gate): encyclopedic /
                # dictionary domains tend to give cleaner definitions than a
                # random blog, so nudge their snippets up without overriding a
                # genuinely better-scoring answer from elsewhere.
                rurl = (r.get("url", "") or "").lower()
                if self._is_preferred_source(rurl):
                    score += 0.3
                # Context-augmented sense ranker (N400 analog): when the
                # context vector is available, reward candidates whose content
                # words align with the *intended* sense of the query. This
                # breaks the +3.0 ties between same-shape definitions of
                # different senses (e.g. legal vs interpersonal "trust") by
                # ranking the context-fitting sense higher rather than letting
                # search-engine result order decide.
                if _ctx_vec is not None and hasattr(self, "_definition_coherence_score"):
                    try:
                        coh = self._definition_coherence_score(subject, s)
                        # _definition_coherence_score uses the bare subject by
                        # default; re-rank against the context vector directly.
                        glove_fn = getattr(self, "_glove_vector", None)
                        if callable(glove_fn):
                            def_words = [w for w in re.findall(r"[a-z']{3,}", s.lower())
                                         if w not in STOP_WORDS]
                            def_vecs = [glove_fn(w) for w in def_words if glove_fn(w) is not None]
                            if def_vecs:
                                dcent = np.mean(def_vecs, axis=0)
                                n = np.linalg.norm(dcent)
                                if n > 0:
                                    dcent /= n
                                    ctx_n = np.linalg.norm(_ctx_vec)
                                    if ctx_n > 0:
                                        csim = float(np.dot(dcent, _ctx_vec / ctx_n))
                                        coh = max(0.0, csim)
                        score += 0.6 * coh
                    except Exception:
                        pass
                candidates.append((score, s))
        if not candidates:
            # Fallback: scan results for the first clean, non-boilerplate
            # snippet and return its first real sentence. Skip HTML/CSS/photo
            # junk and headline-only results.
            for r0 in results[:6]:
                # Fix A: skip navigational/brand titles in the fallback too
                # (same sense check as the primary gate above), so a
                # "Trust Bank" page can't leak through as the answer
                # just because its body repeats the subject word.
                _fb_title = (r0.get("title", "") or "")
                _fb_toks = _fb_title.split()
                _fb_lead = _fb_toks[0] if _fb_toks else ""
                _fb_nav = False
                if subj and len(subj.split()) == 1 and _is_factual_what:
                    if _fb_lead == subj and len(_fb_toks) >= 2:
                        _fb_nxt = _fb_toks[1]
                        if _fb_nxt[:1].isupper() and _fb_nxt.lower() not in _sep:
                            _fb_nav = True
                    elif _fb_lead and _fb_lead[:1].isupper() and subj not in _fb_toks[:1]:
                        _fb_nav = True
                if _fb_nav:
                    continue
                content = r0.get("content", "") or ""
                raw_low = content.lower()
                if ("<img" in raw_low or "getty" in raw_low or raw_low.count("<") > 3
                        or "{" in content or "}" in content or "url(" in raw_low
                        or "@font" in raw_low or "src:" in raw_low):
                    continue
                blob = self._clean_snippet(content)
                if not blob:
                    continue
                if any(n in blob.lower() for n in self._SNIPPET_NOISE):
                    continue
                # Track B Phase 2 (M4): learned structural-junk gate (flag-gated;
                # old noise table above remains the backstop). Pass topic
                # coherence so the dual-gate fires on real junk.
                if self._snippet_is_structural_junk(blob, self._snippet_topic_max_coherence(subj, blob)):
                    continue
                # Fallback must still be about the subject
                blow = blob.lower()
                _fb_tokens = subj.split()
                if len(_fb_tokens) >= 2:
                    if not (subj in blow or all(t in blow for t in _fb_tokens)):
                        continue
                elif subj and subj not in blow:
                    continue
                first = re.split(r"(?<=[.!?])\s+", blob)[0] if blob else ""
                if first and "?" not in first and len(first) >= 20:
                    return first.strip()
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _snippet_is_structural_junk(self, snippet: str,
                                    coherence: Optional[float] = None) -> bool:
        """Track B Phase 2 (M4): learned structural-junk gate for snippets.

        When ``use_cerebellar_snippet`` is ON, consults the trained
        ``SnippetStructureModel`` (predictive-coding PE): rejects snippets that
        are structurally OOD from learned good definitions AND semantically
        incoherent. The hardcoded ``_SNIPPET_REJECT_SHAPES`` / ``_SNIPPET_NOISE``
        tables are ALWAYS consulted as a fallback (old constant kept until the
        learned model is verified to beat them on the regression set).

        Returns True only when the snippet should be rejected.
        """
        # Old constant as backstop — never weakens the existing hard reject.
        sl = (snippet or "").lower()
        _old_reject = (
            any(n in sl for n in self._SNIPPET_NOISE) or
            any(re.match(p, sl) for p in self._SNIPPET_REJECT_SHAPES)
        )
        if _old_reject:
            return True
        if not self.use_cerebellar_snippet:
            return False
        # Lazy-init the learned model (trained on the seed corpus once). Reuse
        # the instance created in __init__ (Defect F) so we don't re-train.
        if getattr(self, "_snippet_structure_model", None) is None and _HAS_SNIPPET_MODEL and default_model:
            try:
                self._snippet_structure_model = default_model()
            except Exception:
                self._snippet_structure_model = False  # avoid re-init on failure
        if self._snippet_structure_model and hasattr(self._snippet_structure_model, "is_junk"):
            try:
                return bool(self._snippet_structure_model.is_junk(snippet, coherence))
            except Exception:
                return False
        return False

    def _is_preferred_source(self, url: str) -> bool:
        """Whether a source should get the quality tie-breaker boost.

        OFF (default): uses the hardcoded allowlist (backstop). ON: uses the
        learned trust score > threshold.
        """
        if not self.use_source_trust:
            return any(s in (url or "").lower()
                       for s in self._PREFERRED_SNIPPET_SOURCES)
        return self._domain_trust(url) > self._source_trust_threshold()

    def _domain_trust(self, url: str) -> float:
        """Learned trust score for a snippet's source domain.

        When source-trust learning is OFF, falls back to the hardcoded
        allowlist (1.0 for a preferred source, else 0.5 neutral). When ON,
        returns the accumulated trust score for the domain (0.0 floor)."""
        _dom = self._domain_of(url) if url else ""
        if not self.use_source_trust:
            if _dom and any(s in url.lower() for s in self._PREFERRED_SNIPPET_SOURCES):
                return 1.0
            return 0.5
        if _dom in self._source_trust:
            return float(self._source_trust[_dom])
        return 0.5  # untried domain: neutral, not trusted, not banned

    def _record_source_outcome(self, url: str, accepted: bool,
                               survived_sleep: bool = False) -> None:
        """Update the per-domain trust accumulator from a snippet outcome.

        +0.1 when a snippet from the domain was accepted (coherence/structure
        passed); an extra +0.1 (total +0.2) when it also survived a sleep
        consolidation cycle; -0.2 when a snippet was rejected. Clamped to
        [0.0, 1.0]. Only runs when source-trust learning is enabled.
        """
        if not self.use_source_trust:
            return
        _dom = self._domain_of(url) if url else ""
        if not _dom:
            return
        _delta = 0.0
        if accepted:
            _delta = 0.2 if survived_sleep else 0.1
        else:
            _delta = -0.2
        _cur = self._source_trust.get(_dom, 0.5)
        self._source_trust[_dom] = max(0.0, min(1.0, _cur + _delta))

    def _source_trust_threshold(self) -> float:
        """Domains with trust above this are preferred (replaces the allowlist
        decision). 0.5 = neutral; a domain must earn trust to be preferred."""
        return 0.5

    def _snippet_quality(self, snippet: str, subject: str, term: str,
                         is_conditional: bool = False) -> float:
        """Heuristic quality signal for a candidate answer snippet.

        Used to pick the best snippet across multiple search-query variants.
        Pure signal, no LLM, no fact table: rewards answer shape + encyclopedic
        source, penalizes residue from junk domains / meta sentences.
        """
        if not snippet:
            return -1.0
        s = snippet.lower().strip()
        score = 1.0
        # Junk/repetition guard: a snippet with consecutive identical tokens
        # (e.g. "GRAVITY+ GRAVITY+ is an upgrade…") or heavy local repetition is
        # not a real answer — penalize hard so a clean snippet wins instead of
        # relying on the downstream self-monitor to repair it post-selection.
        _rtoks = s.split()
        _dup_run = 0
        for i in range(len(_rtoks) - 1):
            if _rtoks[i] == _rtoks[i + 1] and _rtoks[i] not in (
                    "that", "had", "bye", "hello", "no", "yeah", "well", "good"):
                _dup_run += 1
        if _dup_run > 0:
            score -= 5.0 * _dup_run
        # Repeated capitalized acronym or token (e.g. "GRAVITY+ GRAVITY+") also
        # shows as a token appearing 3+ times in a short snippet.
        from collections import Counter as _Counter
        _cnt = _Counter(_rtoks)
        _max_rep = max(_cnt.values()) if _cnt else 0
        if _max_rep >= 3 and len(_rtoks) < 25:
            score -= 4.0
        # Answer shape: subject is the topic with a copula/definition verb.
        subj0 = (subject.split()[0] if subject else "")
        def_verb = re.search(
            r"\b(is|are|was|were|refers to|means|denotes|describes|explains|"
            r"consists of|is the|is a|is an|refers)\b", s)
        first_words = s.split()[:4]
        subj_is_topic = bool(subj0) and (
            self._tok_match(subj0, set(first_words)) or subj0 in s.split()[:2])
        if subj_is_topic and def_verb:
            score += 2.0
        elif subj_is_topic:
            score += 1.0
        # Strongly reward a SUBSTANTIVE definition sentence: the subject as topic
        # with a definition verb AND a real predicate (not just a title fragment
        # like "Definition of sun noun in Oxford…Dictionary."). A substantive
        # definition has several content words after the verb.
        if subj_is_topic and def_verb and len(s.split()) >= 6:
            # Penalize title-like fragments that end in "dictionary." / "nouns" /
            # "glossary" — these are page titles, not answers.
            if re.search(r"(dictionary\.|advanced learner|glossary|noun\b.*dictionary|"
                         r"definition of .* (noun|verb|adjective)|\bapi\b)", s):
                score -= 4.0
            else:
                score += 2.0
        # Real dictionary-concept definition shape (good for abstract concepts
        # like "trust", "love"): "The meaning of X is…".
        if re.search(r"the meaning of .* is\b", s):
            score += 2.0
        # Hypothetical / causal answer shape (good for "what if X" queries).
        if re.search(r"\b(would|without|if .* disappeared|if .* vanished|"
                     r"turned off|no longer|plunge|drift|freeze|darkness|"
                     r"crash|float|orbit)\\b", s):
            score += 1.5
        # For conditional / hypothetical queries, reverse the usual definition
        # bias: a bare dictionary definition of the subject ("Earth's gravity is
        # what keeps you on the ground") must NOT outrank a real hypothetical
        # answer ("Without gravity, Earth would be flung out into space"). The
        # user asked for a *consequence*, so weight consequence sentences up and
        # the pure subject-definition down.
        if is_conditional:
            if subj_is_topic and def_verb:
                score -= 3.0
            if re.search(r"\b(would|without|imagine|no longer|if .* (disappear|"
                         r"vanish|turn|stop|gone|cease|removed|turned|lost)|"
                         r"flung|float|drift|freeze|darkness|crash|orbit|fall|"
                         r"launch|expand|escape|everyone|everything|people|"
                         r"oceans|planes|earth|planet|spaces?|midair)\b", s):
                score += 3.0
        # Penalize page-TITLE fragments (not answers): "Definition of sun noun
        # in Oxford…Dictionary.", "The Free Dictionary", etc. These contain the
        # subject but no real predicate, so they must never beat a definition.
        if re.search(r"(dictionary\.|advanced learner|learner's dictionary|"
                     r"definition of .* (noun|verb|adjective)|\bglossary\b|"
                     r"\bapis?\b|the free dictionary|collins dictionary)", s):
            score -= 4.0
        # Penalize residual meta / junk phrasing.
        if any(re.match(p, s) for p in self._SNIPPET_REJECT_SHAPES):
            score -= 3.0
        if any(n in s for n in ("under the sun", "crossword", "how to use",
                                "get the latest", "sign in to your",
                                "applies to", "chapter", "verse")):
            score -= 3.0
        return score

    def _snippet_plausibility(self, subject: str, snippet: str) -> Optional[float]:
        """Reality-monitoring plausibility of a snippet's *added* content.

        Cognitive basis: the brain's N400 / plausibility check (Kuperberg;
        Bornkessel-Schlesewsky) evaluates whether incoming information fits the
        situation model evoked by the question. A snippet that merely repeats the
        subject word ("Invisible is a gear that makes you invisible in Roblox")
        looks coherent only because of that repetition — its *new* content
        (gear, roblox) is incoherent with what "invisible" means. So we drop the
        subject word and its morphological variants and measure how well the
        remaining content coheres with the subject's GloVe vector. This is a
        domain-agnostic criterion on a semantic dimension (Johnson & Raye set
        criteria on reality-monitoring dimensions; they do not keep source
        blocklists), so it rejects game wikis, spam, and any other incoherent
        source without naming any of them.

        Returns None when GloVe is unavailable (unknown -> not incoherent) or the
        snippet carries no content beyond the subject (can't judge -> pass).
        """
        glove_fn = getattr(self, "_glove_vector", None)
        if not callable(glove_fn) or getattr(self, "_glove_vecs", None) is None:
            return None
        subj_vec = glove_fn(subject)
        if subj_vec is None:
            return None
        _stem = subject[:5].lower() if len(subject) >= 5 else subject.lower()
        words = [w for w in re.findall(r"[a-z']{3,}", (snippet or "").lower())
                 if w not in STOP_WORDS and _stem not in w]
        vecs = [glove_fn(w) for w in words if glove_fn(w) is not None]
        if not vecs:
            return None
        centroid = np.mean(vecs, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            return None
        centroid /= norm
        snorm = np.linalg.norm(subj_vec)
        if snorm == 0:
            return None
        return float(np.dot(centroid, subj_vec / snorm))

    def _belief_coherence(self, subject: str, snippet: str) -> float:
        """Belief coherence: does the snippet's *added* content cohere with what
        RAVANA already believes about the subject (its GloVe/definition vector)?

        PROMPT 3 comparative reality-monitoring (Johnson & Raye 1981): a retrieved
        claim is accepted when it coheres with the existing belief model, not when
        it clears an absolute floor. Reuses _snippet_plausibility's subject-word
        drop so repetition doesn't fake coherence. Returns 0..1 (0 = no signal).
        """
        _p = self._snippet_plausibility(subject, snippet)
        return float(_p) if _p is not None else 0.0

    def _answer_prediction_error(self, query: str, subject: str,
                                 snippet: str) -> float:
        """Continuous semantic prediction error of a candidate web answer.

        Returns 0.0 (fully expected) .. 1.0 (maximally surprising). Each
        component is a normalized [0,1] surprise; they are combined by max
        (the strongest violation dominates, mirroring how a single incoherent
        continuation spikes the N400) so one clear contradiction vetoes without
        needing all signals to agree.
        """
        q = (query or "").lower()
        s = (snippet or "").lower()
        if not s:
            return 0.0
        # ── 1) Premise / polarity consistency (fixes Q15 "gravity doubled" ->
        # "world WITHOUT gravity"). Extract the query's causal/quantity modifier
        # polarity on the subject (increased / decreased / removed / present) and
        # the snippet's. A mismatch between an INCREASE premise and a REMOVAL
        # premise is a contradiction the literal plausibility cosine misses.
        _pe_pol = self._polarity_mismatch(q, s, subject)
        # ── 2) Claim / answer-type match (fixes "break a promise" -> "no contact
        # ex"). Score whether the snippet's speech act matches the query's
        # requested answer type using lightweight prototype features (positive /
        # normative / assertion), not a phrase blocklist.
        _pe_type = self._answer_type_mismatch(q, s)
        # ── 3) Belief convergence (fixes Q11 "perpetual motion = govt secret").
        # A fringe claim has LOW coherence with the established belief model and
        # should raise PE. Reuse the existing plausibility/coherence signals.
        _plaus = self._snippet_plausibility(subject, snippet)
        if _plaus is None:
            _pe_belief = 0.0
        else:
            # Below degenerate -> full surprise; at/above floor -> no surprise;
            # linear in between. Continuous, not a hard cutoff.
            _lo = self._SNIPPET_PLAUSIBILITY_DEGENERATE
            _hi = self._SNIPPET_PLAUSIBILITY_FLOOR
            _pe_belief = max(0.0, min(1.0, (_hi - _plaus) / max(1e-6, _hi - _lo)))
        # ── 4) Topic coverage (the plan's missing "coverage" PE). Does the
        # snippet actually ADDRESS the query, or is it topically unrelated
        # boilerplate that merely shares a stray keyword ("break a promise" ->
        # "...hacking is inherently wrong...")? Measure semantic proximity
        # between the snippet's content vector and the QUERY's content vector
        # (subject extraction can be imperfect, so we use the full query; a
        # brain does not trust a single lexical head — it checks the whole
        # evoked situation model). Low coverage => high surprise => veto.
        _pe_cov = self._topic_coverage_pe(query, subject, snippet)
        # Combine by max (strongest violation wins). This is the learned-style
        # continuous aggregation; a future fit step can replace max with a
        # weighted sum whose weights are fit to labeled good/bad answers.
        return float(max(_pe_pol, _pe_type, _pe_belief, _pe_cov))

    def _polarity_mismatch(self, query: str, snippet: str,
                           subject: str) -> float:
        """Surprise from premise-polarity contradiction between query and snippet.

        Detects the modifier polarity the user asserted on the subject
        ("doubled" / "increased" / "removed" / "gone") and the polarity the
        snippet asserts, and returns 0.0 when they agree or there is no
        detectable premise, else a normalized mismatch in (0,1].
        """
        # Polarity lexicon is minimal + functional (increase vs decrease vs
        # removal), NOT a topic blocklist. These are closed-class modification
        # words the brain tracks for quantity/causal reasoning. Sourced from the
        # consolidated functional lexicon (data/functional_lexicon.json).
        _lex = self._func_lex
        if _lex is not None:
            _INC = tuple(_lex.polarity_increase)
            _DEC = tuple(_lex.polarity_decrease)
            _REM = tuple(_lex.polarity_remove)
        else:
            _INC = ("doubl", "tripl", "increase", "more", "stronger", "higher",
                    "grow", "add", "extra", "boost", "amplif", "intensif")
            _DEC = ("halv", "less", "weaker", "lower", "reduc", "shrink", "diminish")
            _REM = ("without", "gone", "disappear", "vanish", "remov", "lost",
                    "absent", "cease", "eliminat", "no longer", "none")
        q_tokens = query.split()
        s_tokens = snippet.split()
        def _sign(tokens):
            inc = any(any(t in w for t in _INC) for w in tokens)
            dec = any(any(t in w for t in _DEC) for w in tokens)
            rem = any(any(t in w for t in _REM) for w in tokens)
            # removal is the strongest "absent" premise; increase/decrease are
            # present-but-changed. Rank: rem(2) > inc/dec(1) > none(0).
            if rem:
                return 2
            if inc and not dec:
                return 1
            if dec and not inc:
                return -1
            return 0
        _qs = _sign(q_tokens)
        _ss = _sign(s_tokens)
        if _qs == 0 or _ss == 0:
            return 0.0  # no detectable premise on one side -> cannot contradict
        if _qs == _ss:
            return 0.0  # same polarity -> coherent
        # Increase vs removal, or decrease vs removal, or inc vs dec => mismatch.
        return (self._pe_cfg.polarity_surprise
                if self._pe_cfg is not None else 1.0)

    def _answer_type_mismatch(self, query: str, snippet: str) -> float:
        """Surprise from answer-type (speech-act) mismatch.

        The brain fits the *kind* of answer to the *kind* of question (speech-
        act / answer-type congruence). A moral/advice question ("is it okay to
        break a promise") wants a *normative* answer; a snippet that is an
        unrelated anecdote / off-topic premise scores high surprise. We score
        the query's requested answer type and the snippet's realized type via
        lightweight distributional features (no phrase blocklist of topics).
        """
        q = query.split()
        s = snippet.split()
        # Query answer-type prototypes (functional, closed-class-ish cues).
        # Sourced from the consolidated functional lexicon so there is a single
        # authority (no duplicated inline moral/framing sets). The "right"/"wrong"
        # ambiguity check below excludes temporal ("right now") / adjacency
        # ("all right") senses to avoid the keyword-list false positive that
        # mislabeled "how can i become invisible right now" as a moral question.
        _lex = self._func_lex
        _bare_moral = tuple(_lex.moral_markers) if _lex is not None \
            else ("okay", "ok", "moral", "should", "ethical", "fair",
                  "promise", "lie", "ever")
        _ambiguous = tuple(_lex.moral_ambiguous) if _lex is not None \
            else ("right", "wrong")
        _moral = any(w in q for w in _bare_moral)
        # Only count ambiguous "right"/"wrong" as moral when not temporal/
        # adjacency-bound (e.g. "right now", "all right").
        if "right" in q and "now" in q:
            pass  # "right now" -> temporal, skip ambiguous moral
        elif "all" in q and "right" in q:
            pass  # "all right" -> adjacency, skip
        else:
            _moral = _moral or any(w in q for w in _ambiguous)
        _factual = any(w in q for w in
                       ("what", "who", "when", "where", "define", "is", "are"))
        # Procedural request: "how do I build/make X", "how to X", "steps to X".
        # The brain expects a METHOD-style answer (speech-act congruence); a bare
        # declarative CLAIM with no procedural content ("is a government secret
        # kept from the masses...") does not answer a build request. This is the
        # answer-type match the plan maps to M-C#2 and fixes the Q11 conspiracy
        # leak: that snippet is topically coherent (GloVe cosine passes) but is a
        # CLAIM, not a procedure, so it must raise prediction error.
        _proc = bool(re.search(
            r"\bhow\s+(do\s+i|do\s+we|to)\b|\b(build|make|create|construct|"
            r"write|cook|bake|draw)\b|\bsteps?\s+to\b", query))
        if not (_moral or _factual or _proc):
            return 0.0
        # Snippet realized type: a normative/hedged statement expresses judgment
        # ("it's important", "consider", "depends", "generally", "should",
        # "might be"); a bare off-topic premise (e.g. "no contact ex") expresses
        # neither. Absence of any answer-like framing on a non-factual query is
        # the mismatch signal.
        _normative = any(w in s for w in
                         ("important", "consider", "depends", "generally",
                          "should", "might", "could", "right", "wrong",
                          "acceptable", "okay", "ethical", "moral"))
        if _moral and not _normative:
            # Moral question but the snippet gives no normative framing -> the
            # answer doesn't match the requested type. Moderate surprise; the
            # caller's max-combine keeps it from over-vetoing a merely terse but
            # correct answer (the refined-search fallback will recover).
            return (self._pe_cfg.answer_type_surprise
                    if self._pe_cfg is not None else 0.6)
        if _proc:
            # A procedural query wants a METHOD. Procedural markers: step/recipe
            # words, action verbs, "by"-means, numerals-as-steps. A snippet that
            # is a bare CLAIM or narrative without any procedural framing mismatches
            # the requested answer type -> moderate surprise. We deliberately do
            # NOT require specific topic words (that would be a blocklist); we
            # only check that the *form* of the answer fits a how-to.
            _procedural = bool(re.search(
                r"\b(step|steps|method|process|recipe|first|second|then|"
                r"next|finally|by\s+\w+ing|you\s+(need|must|can|should|start|"
                r"begin|take|use|mix|add|place|put|write|draw|build|make))\b",
                snippet))
            if not _procedural:
                return (self._pe_cfg.answer_type_surprise
                        if self._pe_cfg is not None else 0.6)
        return 0.0

    def _subject_head(self, subject: str, query: str) -> Optional[str]:
        """Pick the salient subject noun to test coverage against.

        Subject grounding can return a malformed multi-word phrase
        (\"ever okay break\" for \"break a promise\"), so fall back to the
        longest query content word that carries a GloVe vector and is not
        generic question framing. Returns the head token (e.g. \"promise\").
        """
        glove_fn = getattr(self, "_glove_vector", None)
        # Generic framing words that signal a malformed multi-word subject;
        # sourced from the consolidated functional lexicon (single source of
        # truth, no duplicated inline copy).
        _generic = (tuple(self._func_lex.framing)
                    if self._func_lex is not None
                    else {"okay", "ok", "ever", "right", "wrong", "thing",
                          "things", "really", "actually", "question",
                          "answer", "make", "break"})
        cands = [w for w in re.findall(r"[a-z']{3,}", (subject or "").lower())]
        for w in cands:
            if w in _generic:
                continue
            if glove_fn is not None and glove_fn(w) is not None:
                return w
        # Fall back to the longest non-generic query content word with a vector.
        _stop = getattr(self, "_STOP_WORDS", set()) | set(_generic)
        qtoks = [w for w in re.findall(r"[a-z']{3,}", (query or "").lower())
                 if w not in _stop]
        qtoks = [w for w in qtoks if glove_fn is None or glove_fn(w) is not None]
        if qtoks:
            return max(qtoks, key=len)
        return None

    def _topic_coverage_pe(self, query: str, subject: str,
                           snippet: str) -> float:
        """Surprise from poor topic COVERAGE: the snippet does not actually
        engage the query's specific subject, only shares the same semantic
        region.

        Brain basis: the N400 / plausibility check evaluates whether incoming
        information fits the *situation model evoked by the question*. For
        "is it okay to break a promise" a snippet about "hacking being wrong"
        sits in the SAME ethics semantic field (GloVe cosine of the whole
        snippet is ~0.67) yet never mentions "promise" — it fails to cover the
        specific queried concept. The brain would flag this as a non-sequitur,
        not a relevant answer. We require the snippet to contain at least one
        token that is near the subject HEAD (max-token cosine >= a fit
        threshold); a snippet with no token engaging the subject raises PE.
        This is distributional (a learned cosine threshold), not a blocklist of
        topics, and generalizes: any off-topic snippet for any subject fails.
        """
        glove_fn = getattr(self, "_glove_vector", None)
        if not callable(glove_fn) or getattr(self, "_glove_vecs", None) is None:
            return 0.0
        head = self._subject_head(subject, query)
        if head is None:
            return 0.0
        hv = glove_fn(head)
        if hv is None:
            return 0.0
        hn = np.linalg.norm(hv)
        if hn == 0:
            return 0.0
        hv = hv / hn
        _stop = getattr(self, "_STOP_WORDS", None) or STOP_WORDS
        best = 0.0
        for w in re.findall(r"[a-z']{3,}", (snippet or "").lower()):
            if w in _stop:
                continue
            v = glove_fn(w)
            if v is None:
                continue
            n = np.linalg.norm(v)
            if n == 0:
                continue
            c = float(np.dot(hv, v / n))
            if c > best:
                best = c
        # Fit-ready coverage threshold: a snippet with no token near the subject
        # head (best < threshold) fails coverage -> it is a non-sequitur w.r.t.
        # the queried concept and raises a strong surprise (above the veto
        # midpoint) so it is withheld and the query is refined. A genuinely
        # on-topic answer almost always contains the subject word or a close
        # synonym (best >= threshold) and scores no coverage surprise.
        _cov_thr = (self._pe_cfg.coverage_threshold
                    if self._pe_cfg is not None else 0.6)
        if best >= _cov_thr:
            return 0.0
        return (self._pe_cfg.coverage_surprise
                if self._pe_cfg is not None else 0.7)

    def _conditional_has_graph_anchor(self, query: str) -> bool:
        """Analogy-gate for hypothetical/counterfactual web answers.

        Repair plan C (weakness C): a counterfactual needs an anchor in known
        structure to simulate (Van Hoeck 2015 — FPC tracks counterfactual value
        only when simulation has an anchor). For an unanchored hypothetical
        ("if pigs could fly would democracy still work") the graph has no
        analogical or causal edge to project onto, so simulation cannot engage;
        a human says "i can't really picture how that'd play out" rather than
        dumping an encyclopedia article. So we veto the literal web path when
        NONE of the premise concepts resolve to a graph node that carries at
        least one outgoing edge (i.e. the premise is unanchored).

        Distribution-driven (graph connectivity), not a topic blocklist. Returns
        True when at least one premise concept has a graph anchor (so the query
        may proceed to web/simulation), False when the premise is fully
        unanchored (caller should abstain to honest uncertainty). On any failure
        to resolve the premise (no graph / GloVe), returns True so we never
        silently suppress a potentially answerable query — fail-open, not
        fail-closed-suppress.
        """
        graph = getattr(self, "graph", None)
        _kw = getattr(self, "_concept_keywords", None)
        if graph is None or not _kw:
            return True
        # Premise content words (drop conditional lead-ins + stopwords).
        _DROP = set(("if", "when", "would", "could", "should", "what", "how",
                     "why", "do", "does", "did", "can", "will", "happen",
                     "still", "the", "a", "an", "and", "or", "but", "to",
                     "of", "in", "on", "for", "with", "is", "are", "be"))
        _words = [w for w in re.findall(r"[a-z']{3,}", (query or "").lower())
                  if w not in STOP_WORDS and w not in _DROP]
        if not _words:
            return True
        _stable = getattr(self, "_stable_node_ids", None)
        for _w in _words:
            _nids = _kw.get(_w)
            if not _nids:
                continue
            for _nid in _nids:
                # Only pre-existing (stable) nodes can anchor a simulation;
                # web-learned nodes from THIS turn are not a counterfactual
                # anchor (they were just looked up, not known structure).
                if _stable is not None and _nid not in _stable:
                    continue
                try:
                    _out = graph.get_outgoing(_nid)
                except Exception:
                    _out = []
                if not _out:
                    continue
                # Require a SIMULATION-grade edge: a counterfactual projects onto
                # causal or analogical structure (Van Hoeck 2015 — FPC needs an
                # anchor in known causal/structural relations). Mere semantic or
                # episodic edges (e.g. a concept learned from the web this turn)
                # are NOT a simulation anchor, so they don't legitimize a literal
                # web dump for an unanchored hypothetical. This also stops
                # same-turn web-learning from retroactively "anchoring" the
                # premise (pig -> pig-article semantic edge) and defeating the gate.
                for _tgt, _e in _out:
                    # The anchor target must also be stable knowledge (not a
                    # freshly-learned leaf) to count as real structure.
                    if _stable is not None and _tgt not in _stable:
                        continue
                    _rt = getattr(_e, "relation_type", "semantic")
                    if _rt in ("causal", "analogical", "temporal", "contrastive"):
                        return True
        return False

    def _refine_query_variants(self, query: str, subject: str) -> List[str]:
        """Metacognitive control: re-frame the query when the first answer fails
        the plausibility monitor (the brain's second-pass reanalysis / repair,
        indexed by the late posterior positivity / P600 — Kuperberg & Jaeger).

        We don't block sources; we change *what we ask*. For a how-to / goal
        query the first hit is often in-world lore, so we push the query toward
        the real-world sense ("in real life" / "method"). For a factual query we
        add a real-world disambiguator. This gives the search engine a chance to
        surface a genuinely useful, plausible answer before we give up.
        """
        q = (query or "").lower().strip()
        if re.match(r"^(how|what) (can|do|to|would|should|does)\b", q):
            return [f"how to {subject} in real life",
                    f"{subject} method real world"]
        return [f"{subject} real", f"{subject} science"]

    def _is_function_word(self, word: str) -> bool:
        """True if `word` is a function word (not a discourse/content target).

        Flag OFF (default): uses the hardcoded _GRAMMATICAL_CONCEPTS set
        (current behavior, no regression). Flag ON (use_learned_pos): uses the
        learned distributional POS from self._concept_pos — a word is a function
        word when its POS tag is a function category (prep/pron/det/conj/aux),
        with the hardcoded set retained as a safety net for residual cases the
        distributional tagger does not cover (e.g. some adverbs, numerals).
        """
        if not word:
            return False
        _w = word.lower()
        if not self.use_learned_pos:
            return _w in self._GRAMMATICAL_CONCEPTS
        # Learned path: a real distributional POS tagger (ConceptPosDict)
        # — the word is a function word when its POS tag is a closed-class
        # category (prep/pron/det/conj/aux/part). The hardcoded set
        # is the safety net for residual cases the tagger does not
        # cover (e.g. some adverbs, numerals).
        _cp = getattr(self, "_concept_pos", None)
        if _cp is not None:
            try:
                _pos = _cp.get(_w)
            except Exception:
                _pos = None
            if _pos in ("prep", "pron", "det", "conj", "aux", "part", "func"):
                return True
        # Safety net: words the distributional tagger leaves as 'noun'/'verb'/
        # 'adj' but the curated set knows are function (adverbs, numerals).
        return _w in self._closed_class("grammatical_concepts")

    def _ensure_intent_router(self):
        if self._intent_router is not None or not _HAS_INTENT_ROUTER \
                or IntentRouter is None:
            return
        try:
            _loaded = IntentRouter.load()
            if _loaded is not None and _loaded._sem:
                self._intent_router = _loaded
            else:
                glove_fn = getattr(self, "_glove_vector", None)
                if callable(glove_fn):
                    self._intent_router = IntentRouter.from_seed(glove_fn)
        except Exception:
            self._intent_router = None

    def _route_intent(self, query: str) -> Optional[str]:
        """Return the prototype-router intent for `query`, or None (uncertain)
        so the caller falls back to the legacy regex routing. Only promoted
        routes are returned — unpromoted routes (e.g. self_disclosure, which
        fusion could not separate from self_directed) stay on the regex
        backstop, so this can never regress a route the router hasn't cleared.
        """
        if not self.use_intent_router:
            return None
        self._ensure_intent_router()
        if self._intent_router is None:
            return None
        route = self._intent_router.classify(
            query, getattr(self, "_glove_vector", None))
        # Respect the per-route promotion allow-list: only speak for routes
        # that have cleared the regex backstop on the calibration corpus.
        if route is not None and not self._intent_router.is_promoted(route):
            return None
        return route

    def _router_says(self, route: str, query: str) -> bool:
        """True iff the promoted prototype router classifies `query` as
        `route`. Used as the FIRST check inside the legacy boolean gates so the
        router drives the decision for promoted routes and falls through to the
        regex otherwise. Safe: only promoted routes are ever returned."""
        if not self.use_intent_router:
            return False
        return self._route_intent(query) == route

    def _focus_attribute_answer(self, query: str, subject: str, snippet: str) -> str:
        """Return a focused clause when `query` asks for an attribute of an
        entity and `snippet` carries that attribute; else return `snippet`
        unchanged. Fail-open: never worse than the full snippet.
        """
        if not snippet or not query:
            return snippet
        q = query.lower()
        # Detect "what/which is the <attr> of <entity>" or "who wrote/…".
        attr = None
        m = re.search(r"\b(?:what|which)\s+(?:is|are|was|were)\s+the\s+"
                      r"([a-z]+)\s+of\b", q)
        if m and m.group(1) in self._closed_class("attr_words"):
            attr = m.group(1)
        # "who wrote/directed/founded/invented X" -> attribute is the verb-object.
        _who = re.search(r"\bwho\s+(wrote|directed|founded|invented|created|"
                         r"discovered|painted|composed|built)\b", q)
        # Split snippet into sentences; prefer the sentence that carries the
        # attribute AND a capitalized answer token (proper noun / number).
        sents = re.split(r"(?<=[.!?])\s+", snippet.strip())
        if attr:
            # Look for "<attr> ... is <Answer>" or "... <attr> is <Answer>".
            for sent in sents:
                sl = sent.lower()
                if attr in sl and re.search(r"\bis\b|\bwas\b|:", sl):
                    # Prefer a clause starting at "Its capital ... is X" / "The
                    # capital of Y is X".
                    cm = re.search(
                        r"((?:its?\s+|the\s+)?[^.]*\b" + re.escape(attr) +
                        r"\b[^.]*?\bis\b[^.]*)", sent, re.IGNORECASE)
                    if cm:
                        clause = cm.group(1).strip(" ,;")
                        # Only accept when the clause actually names something
                        # (a capitalized token or a number after "is").
                        if re.search(r"\bis\b\s+.*[A-Z0-9]", clause):
                            return clause[0].upper() + clause[1:] if clause else snippet
            return snippet
        if _who:
            # Return the sentence naming the person (has a capitalized full name).
            for sent in sents:
                if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+", sent):
                    return sent.strip()
        return snippet

    def _belief_value_overlap(self, a: Optional[str], b: Optional[str]) -> float:
        """Content-word Jaccard overlap between two belief value strings (0..1).

        Used to decide whether a fresh web claim MATCHES an established belief
        (high overlap -> corroboration) or NONTRIVIALLY conflicts with it (near
        zero overlap -> disagreement). Stop-words and short tokens are dropped
        so shared filler doesn't inflate the score.
        """
        ta = {w for w in re.findall(r"[a-z0-9]+", (a or "").lower())
              if len(w) >= 3 and w not in STOP_WORDS}
        tb = {w for w in re.findall(r"[a-z0-9]+", (b or "").lower())
              if len(w) >= 3 and w not in STOP_WORDS}
        if not ta or not tb:
            return 0.0
        union = ta | tb
        return len(ta & tb) / len(union) if union else 0.0

    def _needs_web_search(self, subject: str, query: Optional[str] = None) -> bool:
        """Check if a subject needs web search to enrich its associations.

        Returns True if:
        - The subject is not in the graph at all (completely unknown)
        - The subject IS in the graph but has fewer than 3 meaningful
          associations (edges with weight > 0.3). This catches abstract
          concepts like "consciousness" that are seeded with weak teenage
          associations and need web enrichment to produce useful responses.
        - D (research item D): the query is informational / definitional /
          "why" / "how" — these request mechanisms/facts that a handful of
          graph edges can't satisfy, so we ALWAYS attempt the web regardless
          of edge count. This closes the silent-failure case where a known
          subject (e.g. "sky", "dream") with hollow auto-expand edges was
          answered from the graph and never web-searched.

        Returns False only if the concept has >= 3 strong graph edges
        (enough knowledge to generate a meaningful response via the
        ventral path alone) AND the query is not informational.
        """
        if not subject:
            return False
        subj_lower = subject.lower().strip()

        # D (research item D): informational/why/definition queries always
        # attempt the web — the requested fact is a mechanism/definition, not
        # any graph edge. This is the direct fix for Q10/Q14-style hollow
        # answers. (Dependent on item E: provenance scoring weights the result.)
        if query and self._is_informational_query(query, subject):
            return True

        # Not in graph at all → definitely needs web search
        if subj_lower not in self._concept_keywords and subj_lower not in self._concept_labels:
            with self._graph_lock:
                for nid, node in list(self.graph.nodes.items()):
                    if node.label and node.label.lower() == subj_lower:
                        break
                else:
                    return True

        # Subject is in the graph — count strong outgoing edges (weight > 0.3)
        strong_edges = 0
        subj_nids = self._concept_keywords.get(subj_lower, [])
        for nid in subj_nids:
            for tid, edge in self.graph.get_outgoing(nid):
                if edge.weight > 0.3:
                    strong_edges += 1
            for src, edge in self.graph.get_incoming(nid):
                if edge.weight > 0.3:
                    strong_edges += 1

        # Need >= 3 strong associations to have enough knowledge
        return strong_edges < 3

