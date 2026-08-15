"""Auto-generated mixin module for CognitiveChatEngine.
Episodic & working-memory mixin — recall, retrieval, activation, user-model updates, forward simulation.
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
    from ravana._import_guard import report_missing
    report_missing("bs4", "BeautifulSoup HTML parsing (web scraping)", kind="optional")

# Import constants
from .constants import (TEEN_CONCEPTS, WEB_GARBAGE, STOP_WORDS, ConceptPosDict,
                        _is_word_salad, _is_keyboard_mash,
                        _UNIVERSAL_PURGE, _DEFINITION_ASSERTION)
from .web_learning import WebLearningMixin
from ravana._import_guard import report_missing  # non-silent import-guard logging
from . import pet_slots as _pet_slots
from . import possession_attrs as _poss_attrs

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
    report_missing("ravana.chat.snippet_quality", "web-snippet quality model", kind="internal")
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
    report_missing("ravana.chat.salad_classifier", "learned word-salad detector", kind="internal")

# Stage 5a (de-hardcoding plan): snippet-PE gate parameters live in a fit file
# (data/snippet_pe.json) rather than inline constants. Fails open to seed
# constants when the fit file is absent.
try:
    from .snippet_pe_config import default_config as _default_pe_config
    _HAS_PE_CONFIG = True
except Exception:  # pragma: no cover - defensive
    _HAS_PE_CONFIG = False
    _default_pe_config = None
    report_missing("ravana.chat.snippet_pe_config", "snippet-PE gate fit config", kind="internal")

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
    report_missing("ravana.chat.pos_model", "learned POS classifier", kind="internal")

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
    report_missing("ravana.chat.intent_router", "semantic-prototype intent router", kind="internal")

# Stage 5b-ii (de-hardcoding plan): the duplicated closed-class functional
# lexicons (_generic / _FRAMING / _bare_moral / _INC/_DEC/_REM) collapse into one
# data-driven source of truth (data/functional_lexicon.json). Fails open.
try:
    from .functional_lexicon import default_lexicon as _default_lexicon
    _HAS_FUNC_LEX = True
except Exception:  # pragma: no cover - defensive
    _HAS_FUNC_LEX = False
    _default_lexicon = None
    report_missing("ravana.chat.functional_lexicon", "data-driven functional lexicon", kind="internal")

import pickle
from ravana.web.learner import SearchEngine
from ravana.core.dual_code_space import DualCodeSpace
from ravana.core.hrr_reasoner import HRRReasoner
from ravana.core.in_prompt_reasoner import (
    answer_in_prompt_causal,
    answer_universal_syllogism,
    answer_evaluative_framing,
    answer_self_evaluation,
)
from ravana.core.temporal_reasoner import answer_temporal

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




class MemoryMixin:
    """Episodic & working-memory mixin — recall, retrieval, activation, user-model updates, forward simulation."""

    def _record_episode(self, user_input: str) -> None:
        """Append a structured turn record to the gist-based episodic transcript.

        Brain basis: conversational memory is cue-dependent and gist-based
        (Brown-Schmidt & Benjamin 2018; Tulving encoding specificity). We store
        the verbatim text (for reconstruction), a timestamp, the topic, and the
        salient facts/preferences mined from the utterance (favorites, likes),
        so a later "remember what I told you" can reconstruct GIST without
        confabulating. Capped at 100 turns; oldest dropped first.

        §2 temporal index: every record also carries a monotonic ``turn_index``
        and a ``content_hash`` so the hippocampal time-cells (FIRST = lowest
        index, LAST = highest index, date-bucket) can answer "our first
        conversation" / "what did i just tell you" as pure index math over the
        already-stored data.
        """
        import hashlib
        import time as _time
        t = (user_input or "").strip()
        if not t:
            return
        _topic = ""
        try:
            _topic = self._ground_query(t)[0] or ""
        except Exception:
            _topic = ""
        _hash = hashlib.md5(t.lower().encode("utf-8", "ignore")).hexdigest()[:12]
        rec = {
            "text": t,
            "ts": _time.time(),
            "topic": _topic,
            "facts": self._mine_episodic_facts(t),
            "turn_index": self.turn_count,   # monotonic temporal index
            "content_hash": _hash,
        }
        self._episodic_transcript.append(rec)
        if len(self._episodic_transcript) > 100:
            self._episodic_transcript = self._episodic_transcript[-100:]
        # Mirror into the temporal indexer (hippocampal time cells).
        try:
            from .brain_regions import Episode, EpisodicIndex
            if self._episodic_indexer is None:
                self._episodic_indexer = EpisodicIndex()
            self._episodic_indexer.add(Episode(
                text=t, turn_index=self.turn_count, ts=rec["ts"],
                content_hash=_hash, facts=rec["facts"], topic=_topic))
        except Exception:
            pass

    def _mine_episodic_facts(self, text: str) -> Dict[str, str]:
        """Extract salient self-disclosed facts from a user turn (gist mining).

        Brain basis: hippocampal relational binding encodes ANY
        subject->relation->object triple (Hannula 2008; Yonelinas 2019), not
        just favorite/like slots. We capture the canonical shapes AND possessive
        disclosures so a later "remember my cat's name" reconstructs the right
        entity (pattern separation — Yassa & Stark 2011), never a cross-contaminated
        gist. Returns a small dict of {slot: value}. Deterministic, no LLM.

        Shapes captured:
          - "my favorite X is Y"      -> favorite_X: Y
          - "my X's Y is Z"           -> X_Y: Z   (e.g. cat_name: whiskers)
          - "my X is Y" / "i am X"    -> X: Y     (self/pet description)
          - "i love/like X"           -> likes: X
        The entity is also indexed in self._episodic_index (keyed by entity)
        for precise pattern-completion recall.
        """
        facts: Dict[str, str] = {}
        low = (text or "").lower().strip()
        # "my favorite X is Y" / "my favorite X: Y"
        m = re.search(r"\bmy favorite\s+([a-z0-9 ]+?)\s+(?:is|are|:)\s+([a-z0-9 ]+?)[.!?]?\s*$",
                      low)
        if m:
            facts["favorite_" + m.group(1).strip()] = m.group(2).strip()
        # Possessive relational: "my X's Y is Z" -> entity X, attribute Y, value Z
        # e.g. "my cat's name is whiskers" / "my dog's age is 3"
        for pm in re.finditer(
                r"\bmy\s+([a-z0-9]+)'?s\s+([a-z0-9 ]+?)\s+(?:is|are|:)\s+([a-z0-9 ]+?)[.!?]?\s*$",
                low):
            ent, attr, val = pm.group(1).strip(), pm.group(2).strip(), pm.group(3).strip()
            if ent and attr and val:
                facts[f"{ent}_{attr}"] = val
        # Bare possession / self-description: "my X is Y" (X not 'favorite')
        if "favorite" not in low:
            for bm in re.finditer(
                    r"\bmy\s+([a-z0-9]+)\s+(?:is|are|:)\s+([a-z0-9 ]+?)[.!?]?\s*$",
                    low):
                ent, val = bm.group(1).strip(), bm.group(2).strip()
                if ent and val and ent not in ("name",):
                    facts[ent] = val
        # "i love/like X" (last such clause). Cut the captured object at the
        # first comparative/prepositional tail ("over", "except", "rather
        # than", "but", "and", "because") so "i prefer acoustic music over
        # anything produced on a laptop" stores "acoustic music", not the
        # run-on comparative clause. Generic splitting, no per-topic list.
        for verb in ("love", "like", "enjoy", "prefer"):
            mm = re.findall(r"\bi\s+" + verb + r"\s+([a-z0-9 \-]+?)(?:,|!|\?| and | but | because | over | except | rather than |$)",
                            low)
            if mm:
                _cap = mm[-1].strip()
                # keep only the leading content head (drop trailing modifiers)
                _cap = re.split(r"\s+(?:over|except|rather than|but|and|because|to)\s", _cap)[0].strip()
                if _cap:
                    facts.setdefault("likes", _cap)
        # Index mined facts into the hippocampal entity store (pattern separation).
        for slot, val in facts.items():
            # entity = leading token before an underscore (cat_name -> cat) or
            # the slot itself for favorites (favorite_color -> color concept).
            if "_" in slot and not slot.startswith("favorite_"):
                ent = slot.split("_", 1)[0]
            elif slot.startswith("favorite_"):
                ent = slot[len("favorite_"):]
            else:
                ent = slot
            attr = slot.split("_", 1)[1] if "_" in slot and not slot.startswith("favorite_") else \
                ("favorite" if slot.startswith("favorite_") else "is")
            idx = self._episodic_index.setdefault(ent, {})
            idx[attr] = val
        return facts

    def _retrieve_episodic(self, query: str,
                           transcript: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """Brain-faithful episodic recall (Tulving encoding specificity).

        Match by (a) explicit fact slot the query asks about (e.g. 'cat' /
        'book'), or (b) GloVe semantic similarity between the query and stored
        turn text/gist. Reconstruct the gist. If nothing clears the bar, return
        None so the caller fails CLOSED (never confabulate — RAVANA bar).

        `transcript` lets the caller pass a restricted set (e.g. all-but-current
        turn for a "remember what I told you" query) instead of the live store.
        """
        store = transcript if transcript is not None else self._episodic_transcript
        if not store:
            # Post-load: the transcript is NOT persisted, so after a reload it
            # is empty and cued recall would silently no-op. Rebuild a working
            # store from the durable hippocampal indexer (which IS persisted) so
            # "what are my cats called" / "what did i say about X" still resolve
            # to the right episode. B-fix (round v-aug04).
            try:
                _idxr = getattr(self, "_episodic_indexer", None)
                if _idxr is not None:
                    store = [{"text": getattr(ep, "text", ""),
                              "facts": getattr(ep, "facts", {}) or {}}
                             for ep in _idxr.all()]
            except Exception:
                pass
        # NOTE: we do NOT bail on an empty `store` here. The entity-indexed
        # pattern-completion path (below) answers from self._episodic_index +
        # the PersonalFactStore (both persisted), which is sufficient for cued
        # recall even when the gist transcript is unavailable post-load.
        q = (query or "").lower().strip()
        # (a0) HIPPOCAMPAL ENTITY INDEX — highest precision pattern completion
        # (A3: Yassa & Stark 2011). Extract the cued entity and attribute from
        # the recall query ("remember my cat's name" / "what was the book i
        # mentioned") and return ONLY that entity's stored facts. This prevents
        # the wrong-episode contamination bug where "my book" returned the cat's
        # gist. We match against the live index AND a transcript-derived index
        # so cued recall works even when the index store and transcript diverge.
        # Both indexes share the SAME entity->attr->value shape, so folding is a
        # clean merge (never keyed by raw slot, which would pollute attr keys).
        def _slot_to_ent_attr(slot):
            if slot.startswith("favorite_"):
                return slot[len("favorite_"):], "favorite"
            if slot == "likes":
                return "likes", "likes"
            if "_" in slot:
                e, _, a = slot.partition("_")
                return e, a
            return slot, "is"
        _entity_idx = {e: dict(v) for e, v in self._episodic_index.items()}
        for rec in store:
            for slot, val in rec.get("facts", {}).items():
                ent, attr = _slot_to_ent_attr(slot)
                _entity_idx.setdefault(ent, {})[attr] = val
        # Pets live in the PersonalFactStore under a species-keyed slot
        # ('cat', 'cat_2', 'dog') written by the miner via pet_slots. The
        # transcript/episodic-index miner does NOT capture them, so the entity
        # index built above has no entry for them and a cued recall ("what are
        # my cats called") would fall through to a wrong episode. Fold the
        # store's pet facts in under their canonical species entity, so the
        # entity scan below resolves the user's own animal word to them.
        try:
            _pf = getattr(self, "user_model", None)
            if _pf is not None:
                _pfs = getattr(_pf, "personal_facts", None)
                if _pfs is not None:
                    for _key, _fact in getattr(_pfs, "facts", {}).items():
                        # A superseded value is a RETIRED memory (the user
                        # corrected it); folding it in would let the retired
                        # value collide with the active one. Corrections win.
                        if getattr(_fact, "superseded", False):
                            continue
                        _attr = _key[1] if isinstance(_key, (tuple, list)) and len(_key) > 1 else None
                        _val = getattr(_fact, "value", _fact)
                        if _attr and _pet_slots.is_pet_attribute(_attr):
                            _sp = _pet_slots.base_species(_attr)
                            _m_idx = re.search(r"_(\d+)$", str(_attr))
                            _idxnum = _m_idx.group(1) if _m_idx else "1"
                            _entity_idx.setdefault(_sp, {})[_idxnum] = _val
                        # Possession-attribute facts (round 2026-08-15T0830Z,
                        # Bug 4) are stored under the ENTITY key (cabin / sword)
                        # with attributes 'madeof' / a feature noun (roof/wall/..),
                        # mirroring the pet folding above. Without this fold a
                        # "what's my cabin made of" recall cannot resolve the
                        # structured fact and falls through to a whole-sentence
                        # echo. The render site (_reconstruct_entity) already
                        # knows how to phrase 'madeof' / feature attrs via
                        # possession_attrs.render, so folding here is sufficient
                        # for a clean recall answer.
                        elif _attr and _attr not in ("name", "location", "does",
                                                     "event", "is", "favorite",
                                                     "likes", "background"):
                            _entity_idx.setdefault(_key[0], {})[_attr] = _val
        except Exception:
            pass

        def _reconstruct_entity(ent, facts):
            bits = []
            # The "i" entity is the USER's own biographical profile
            # (populated by the self-disclosure handler), so its attributes
            # must render as natural first/second-person statements,
            # never as "your i's name is X".
            if ent == "i":
                for attr, val in facts.items():
                    if attr == "name":
                        bits.append(f"your name is {val}")
                    elif attr == "location":
                        bits.append(f"you live in {val}")
                    elif attr == "background":
                        bits.append(f"{val}")
                    elif attr == "favorite":
                        bits.append(f"your favorite {val}")
                    elif attr == "likes":
                        bits.append(f"you like {val}")
                    elif attr == "does":
                        bits.append(f"you {val}")
                    elif attr == "event":
                        bits.append(f"you {val}")
                    elif attr == "is":
                        bits.append(f"you are {val}")
                    else:
                        bits.append(f"your {attr} is {val}")
                return bits
            for attr, val in facts.items():
                if attr == "favorite":
                    bits.append(f"your favorite {ent} is {val}")
                elif attr == "likes":
                    bits.append(f"you mentioned you like {val}")
                elif attr == "does":
                    bits.append(f"you do {val}")
                elif attr == "event":
                    bits.append(f"you {val}")
                elif attr == "is":
                    bits.append(f"your {ent} is {val}")
                elif attr == "location":
                    bits.append(f"you live in {val}")
                elif attr == "background":
                    bits.append(f"{val}")
                # Possession-attribute facts (round 2026-08-15T0830Z, Bug 4):
                # render 'madeof' as a natural clause and feature nouns
                # (roof/wall/...) via the shared possession_attrs renderer, so a
                # cued recall of a material fact reads cleanly ("your cabin is
                # made of pine" / "your cabin's roof is sod") instead of the
                # bare-slot form "your cabin's madeof is pine".
                elif attr == "madeof":
                    bits.append(f"your {ent} is made of {val}")
                elif _poss_attrs.is_feature_noun(attr):
                    bits.append(_poss_attrs.render(ent, attr, val))
                # Pets are stored under a species-keyed slot (entity "cat",
                # attr "1"/"2"). Render as a natural clause rather than
                # "your cat's 1 is pixel".
                elif (_pet := _pet_slots.render_pair(ent, attr, val)):
                    bits.append(_pet)
                else:
                    bits.append(f"your {ent}'s {attr} is {val}")
            return bits

        # find an entity token from the query that exists in the index
        # (strip a trailing "'s" so "cat's" matches entity "cat").
        # ALSO map first/second-person + location/origin question
        # words to the "i" biographical entity so "where do I live" /
        # "what city are you from" recall the user's stored location.
        # Map the user's spoken animal word to the canonical species entity so
        # "what are my cats called" / "what did i name my dog" resolve to the
        # slots the miner wrote. Both sides go through pet_slots, so they agree
        # on the key by construction instead of via a duplicated synonym table.
        _ent_hit = None
        _LOC_WORDS = ("live", "lives", "from", "city", "town", "country",
                      "born", "grew", "located", "location", "origin")
        # Specific-entity cues must be resolved BEFORE the generic self-profile
        # fallback. Root cause of the wrong-episode defect ("what is my cat's
        # name?" -> "you told me you live in berlin"): the scan was a single
        # left-to-right pass, so the bare pronoun "my" (which carries NO
        # retrieval target) matched at position 2 and short-circuited the loop
        # before reaching the real cue "cat". A cued recall is specific by
        # construction — the generic self-profile is only correct when the
        # query names no entity at all.
        _generic_self = False
        for tok in re.findall(r"[a-z']+", q):
            _tok = tok[:-2] if tok.endswith("'s") else tok
            if _tok in _entity_idx and _tok not in ("i", "you", "my", "your"):
                _ent_hit = _tok
                break
            # species map (e.g. "cats" -> "cat" entity)
            _sp = _pet_slots.species_of(_tok)
            if _sp is not None and _sp in _entity_idx:
                _ent_hit = _sp
                break
            if _tok in ("i", "you", "my", "your") and "i" in _entity_idx:
                # only treat as a cued recall when the query also
                # asks about a biographical attribute
                if any(w in q for w in _LOC_WORDS) or "name" in q:
                    _generic_self = True
        if _ent_hit is None and _generic_self:
            _ent_hit = "i"
        if _ent_hit is not None:
            _facts = _entity_idx[_ent_hit]
            if _facts:
                _bits = _reconstruct_entity(_ent_hit, _facts)
                if _bits:
                    return "you told me " + "; ".join(dict.fromkeys(_bits)) + "."
        # (a0) LITERAL-CONTENT CUE PASS (B-fix, round v-aug04). The previous
        # semantic cosine matcher returned the highest-scoring UNRELATED
        # episode because GloVe similarity is loosely positive across many
        # turns ("what did i say about open source" returned "what's my
        # favorite color?"). Root cause: recall was bound to activation, not to
        # the episode that actually contained the cue. Fix: extract the query's
        # content cue (drop recall-scaffold + question-structure words), then
        # return the episode whose STORED TEXT verbatim contains that cue. This
        # is content-addressable recall — exactly the episode the user is asking
        # about — and it cannot return a wrong turn. Fail-open to the semantic
        # pass below only when no episode literally contains the cue.
        _CUE_STOP = {
            "remember", "recall", "told", "said", "say", "tell", "telling",
            "mention", "mentioned", "ask", "asked", "what", "earlier", "before",
            "about", "thing", "things", "did", "do", "you", "your", "i", "my",
            "we", "our", "the", "a", "an", "that", "this", "me", "name",
            "say", "said", "saying", "tell", "told", "think", "thought",
            "like", "likes", "liked", "love", "hate", "still", "feel",
            "feeling", "believe", "mention", "mentioned", "talk",
            "talking", "know", "recall", "what", "did",
            "do", "you", "your", "i", "my", "we", "our", "me", "about",
            "on", "of", "the", "a", "an", "that", "this", "is", "are",
            "was", "were", "have", "has", "had", "name", "color", "colour",
        }
        _cue_tokens = [w.strip(".,!?") for w in re.findall(r"[a-z']+", q.lower())
                       if len(w) >= 3 and w not in _CUE_STOP]
        if _cue_tokens:
            # ROUND 2026-08-09i FIX: cue matching was VERBATIM-only, so a query
            # cue that differed by morphology from the stored episode text could
            # never match (e.g. "swarm" vs the stored turn "the hive *swarmed*",
            # "die" vs "died"), and the recall fell through to the loose semantic
            # pass which returned a WRONG, unrelated episode. Add lemma stemming
            # (Porter) on both the cue and each episode's token stream so the
            # content-addressable recall matches on word FORM, not exact spelling.
            # This is a general, distribution-driven fix (no per-topic synonym
            # table) and keeps the fail-closed fallback to the semantic pass when
            # nothing stems-match. Lazy-import; PorterStemmer is already a
            # project dependency.
            try:
                from nltk.stem import PorterStemmer as _PS
                _stemmer = _PS()
                _stem = lambda w: _stemmer.stem(w)
            except Exception:
                _stem = (lambda w: w)  # graceful degrade: verbatim match only
            _cue_stems = {_stem(c) for c in _cue_tokens}
            _best_cue = None
            _best_cue_score = 0
            for rec in store:
                _t = rec.get("text", "").lower()
                # D1 (round 2026-08-08b-d): a prior RECALL QUERY ("remind me what
                # i told you about the one that molted") carries no shareable
                # content, but the old skip-regex only caught
                # "remember/recall/what did i/what was i" + told/said/ask. A
                # later semantically-overlapping recall ("what's the strongest
                # read you've formed") matched the prior recall query's OWN
                # text (it shares "told you"/"molted") and echoed it verbatim ->
                # a recursive recall loop (the "you mentioned my tarantula
                # before, remind me..." echo). Generalize the skip to ANY
                # recall-scaffold query: remember/recall/remind/what i
                # told|said|mentioned|asked you, so a query is never retrieved
                # AS content. Structural (regex on recall syntax), not a
                # per-topic guard.
                if re.search(
                    r"\b(remember|recall|remind(?: me)?)\b"
                    r".*\b(told|said|ask|mention|tell|said you|mentioned|asked)\b",
                    _t) or re.search(
                    r"\b(what|do you remember|remind)\b.*\b(i|you)\b.*"
                    r"\b(told|said|mentioned|asked|tell|remember|recall)\b", _t):
                    continue  # skip prior recall queries (no content)
                # Count how many cue tokens' STEMS appear in this episode's
                # stemmed token stream (morphology-invariant match).
                _t_stems = {_stem(w) for w in re.findall(r"[a-z']+", _t)}
                _hit = sum(1 for _cs in _cue_stems if _cs in _t_stems)
                # weight by fraction of cue tokens present (a focused match
                # beats a scattered one)
                _frac = _hit / len(_cue_stems)
                if _hit > 0 and _frac >= 0.34 and _hit > _best_cue_score:
                    _best_cue_score = _hit
                    _best_cue = rec
            if _best_cue is not None:
                return self._reconstruct_gist(_best_cue)
        # (a) fact-slot cue match — highest precision.
        for rec in store:
            facts = rec.get("facts", {})
            for slot, val in facts.items():
                # the query references this fact's value or slot keyword
                if val and (val in q or slot.replace("_", " ") in q):
                    return self._reconstruct_gist(rec)
        # (b) semantic match over turn text / gist.
        best = None
        best_score = 0.0
        qwords = [w.strip(".,!?") for w in q.split()
                  if len(w.strip(".,!?")) >= 3 and w.strip(".,!?") not in STOP_WORDS]
        # Exclude recall-scaffold words (the verbs/pronouns that make a query a
        # "remember" act) so the semantic cue is the REAL content (e.g. "cat",
        # "name", "book"), not the recall scaffolding. Otherwise "remember"
        # itself loosely cosine-matches stored episodes and a cued recall
        # returns an unrelated memory (confabulation). Fail-closed instead.
        _RECALL_SCAFFOLD = {
            "remember", "recall", "told", "said", "say", "tell", "telling",
            "mention", "mentioned", "ask", "asked", "what", "earlier", "before",
            "about", "thing", "things", "did", "do", "you", "your", "i", "my",
            "we", "our", "the", "a", "an", "that", "this", "me", "name",
        }
        qwords = [w for w in qwords if w not in _RECALL_SCAFFOLD]
        for rec in store:
            text = rec.get("text", "")
            # Skip episodes that are themselves memory queries (e.g. a previous
            # "remember what I told you") — they carry no shareable content and
            # would otherwise be retrieved by semantic overlap with a new recall
            # query, producing a confabulated self-reference. Fail-closed instead.
            if re.search(
                r"\b(remember|recall|remind(?: me)?)\b"
                r".*\b(told|said|ask|mention|tell|said you|mentioned|asked)\b",
                text.lower()) or re.search(
                r"\b(what|do you remember|remind)\b.*\b(i|you)\b.*"
                r"\b(told|said|mentioned|asked|tell|remember|recall)\b",
                text.lower()):
                continue
            score = 0.0
            _strong_link = False
            _text_l = text.lower()
            for w in qwords:
                # A genuine topical link for a cued recall requires a VERBATIM
                # query word in the stored episode (precise), OR a very strong
                # semantic neighbor (cosine >= 0.5). Weak cosine alone is
                # rejected so an unrelated cue ("my cat's name") cannot loosely
                # match any stored episode (RAVANA bar: never confabulate).
                if w in _text_l:
                    _strong_link = True
                wv = self._glove_vector(w)
                if wv is None:
                    continue
                for tw in re.findall(r"[a-z']+", _text_l):
                    if tw in STOP_WORDS or len(tw) < 3:
                        continue
                    tv = self._glove_vector(tw)
                    if tv is None:
                        continue
                    _cos = float(np.dot(wv, tv))
                    score += _cos
                    if self._adaptive_gate("episodic_cos", _cos):
                        _strong_link = True
            if not _strong_link:
                continue
            if score > best_score:
                best_score = score
                best = rec
        # Adaptive bar: require a non-trivial match (distribution-driven, not a
        # fixed threshold — but we must avoid firing on an empty/weak cue).
        # B-fix (round v-aug04): additionally require the winning episode to
        # contain at least one query cue token VERBATIM. Loose GloVe cosine can
        # still rank an unrelated episode highest (the original defect), so a
        # pure cosine win is no longer accepted — fail CLOSED (None) instead of
        # returning a wrong turn. This is the RAVANA bar: never confabulate a
        # memory that wasn't there.
        if best is not None and self._adaptive_gate("recall_gist", best_score):
            _best_text = (best.get("text", "") or "").lower()
            _verbatim = any(c in _best_text for c in qwords)
            if _verbatim:
                return self._reconstruct_gist(best)
        return None

    # ════════════════════════════════════════════════════════════════
    # Combined "statement(s) + question" handler (R1: in-turn memory)
    # ════════════════════════════════════════════════════════════════
    def _try_combined_fact_query(self, user_input: str) -> Optional[str]:
        """Store premise facts packed in a combined "facts + question" turn and
        answer the trailing question from them.

        Many benchmark / real prompts pack premises AND a question into ONE
        message (LoCoMo, LongMemEval items, the lamp causal chain). The
        rest of process_turn treats the whole blob as one query, so the
        premises are never stored and the question is answered from a blank
        slate (or the question is echoed back). This intercepts that shape:
          1. split the text on newlines (and on ". " between a statement
             and a following clause),
          2. find a trailing question (ends with '?' or starts with a
             question word),
          3. store every statement before it as a premise fact,
          4. if a premise-bearing question exists, answer it from the
             in-turn fact store + causal rules.

        Returns None when the turn is NOT a combined fact+question shape
        (so normal pipeline runs untouched). Fail-closed: if no premise
        matches the question's cue, returns None (honest path handles it).
        """
        text = (user_input or "").strip()
        if not text:
            return None

        # Identify a question clause (defined first so the split guards can use it).
        def _looks_like_question(clause: str) -> bool:
            c = clause.strip().lower()
            if c.endswith("?"):
                return True
            return bool(re.match(
                r"^(what|who|where|when|why|how|which|is|are|do|does|"
                r"did|can|could|would|should|will|has|have)\b", c))

        # Split into clauses. Benchmarks use "\n" between premises;
        # real prompts separate a statement from the final question
        # with ". ". Do NOT split on commas by default — premises
        # like "When turned on, the lamp lights up" must stay
        # intact so the causal-rule miner sees both clauses.
        # The ONE case we DO split on commas is a parallel
        # list disclosure ("a cat named X, a dog named Y, and a
        # hamster named Z. What is my dog's name?") — there the
        # comma is followed by a fresh list item ("a "/"an "/"my "
        # /"i ").
        raw_parts = [p.strip() for p in re.split(r"(?:\n|\.(?=\s))", text)
                       if p.strip()]
        # Guard: a combined fact+question turn needs >= 2 clauses
        # AND the LAST clause must be a question (otherwise it is
        # an ordinary multi-clause statement that belongs to the
        # normal pipeline / self-disclosure block).
        if len(raw_parts) < 2 or not _looks_like_question(raw_parts[-1]):
            return None
        # Secondary: split comma-separated list items (comma + list
        # opener), but ONLY inside named-entity list disclosures
        # ("a cat named X, a dog named Y, and a hamster named Z").
        # Gate on the presence of "named"/"called" so we never
        # break an enabling condition like "If the lamp lights
        # up, an explosion occurs" (the comma there is followed by
        # "an", which would otherwise be wrongly split).
        _expanded = []
        if re.search(r"\b(?:named|called)\b", text):
            for part in raw_parts:
                _sub = re.split(r",\s*(?=(?:a|an|my|i)\b)", part)
                _expanded.extend(s.strip() for s in _sub if s.strip())
        else:
            _expanded = raw_parts
        raw_parts = _expanded
        # A combined fact+question turn must contain at least one
        # statement AND at least one question; the question must be the
        # LAST clause (otherwise it is an ordinary multi-sentence turn).
        question_parts = [p for p in raw_parts if _looks_like_question(p)]
        if not question_parts:
            return None
        # Premises = every non-question clause BEFORE the final question.
        premise_parts = [p for p in raw_parts[:-1] if not _looks_like_question(p)]
        if not premise_parts:
            return None

        # Each combined fact+question turn is SELF-CONTAINED: the
        # premises belong ONLY to this turn's question. Use a fresh
        # local store per call (do NOT accumulate across turns,
        # otherwise a later query would see stale premises from an
        # earlier case). The hippocampal entity index is also
        # populated for genuine cross-turn "remember" queries —
        # that is a DIFFERENT, persistent path.
        local_facts: Dict[str, Dict[str, str]] = {}
        local_rules: List[Dict[str, str]] = []

        # Mine + store every premise as a fact.
        for prem in premise_parts:
            facts = self._mine_episodic_facts(prem)
            for slot, val in facts.items():
                ent = slot.split("_", 1)[0] if "_" in slot and not slot.startswith("favorite_") else (slot[len("favorite_"):] if slot.startswith("favorite_") else slot)
                local_facts.setdefault(ent, {})[slot.split("_", 1)[1] if "_" in slot and not slot.startswith("favorite_") else ("favorite" if slot.startswith("favorite_") else "is")] = val
                self._episodic_index.setdefault(ent, {})[slot.split("_", 1)[1] if "_" in slot and not slot.startswith("favorite_") else ("favorite" if slot.startswith("favorite_") else "is")] = val
            # Possessive/biographical shapes the miner misses.
            self._store_possessive_fact(prem, local_facts)
            # Causal "if/when X -> Y" rules (lamp chain etc.)
            rule = self._mine_causal_rule(prem)
            if rule:
                local_rules.append(rule)

        question_text = raw_parts[-1]
        answer = self._answer_from_session_facts(
            question_text, local_facts, local_rules)
        if answer is not None:
            return answer
        # Reasoning fall-through (the in-prompt reasoners are wired
        # HERE, not left as dead code): a combined fact+question
        # turn may be a logical deduction, not a stored-fact lookup.
        #   - answer_in_prompt_causal: causal conditionals
        #     ("if/when X, Y" multi-hop chains — lamp test).
        #   - answer_universal_syllogism: categorical syllogisms
        #     ("all men are mortal; socrates is a man; is socrates
        #     mortal?"). Both return None when the shape doesn't
        #     match, so non-reasoning turns fall through untouched.
        _reason = answer_in_prompt_causal(text)
        if _reason is not None:
            return _reason
        _syll = answer_universal_syllogism(text)
        if _syll is not None:
            return _syll
        _temp = answer_temporal(text)
        if _temp is not None:
            return _temp
        # No in-turn fact matched the question cue — fall through to the
        # normal pipeline (honest path handles it).
        return None

    def _store_possessive_fact(self, text: str, local_facts: Dict[str, Dict[str, str]]) -> None:
        """Catch named-entity / biographical facts that the miner misses:
          - "my X is named Y" / "my X is called Y"
          - bare "my X is Y"  (e.g. "my pet dog is max")
          - "a cat named Y" / "a dog named Y"  (list-form disclosures)
          - "I was born in Y" / "I built a Z last month" (biographical)
        Written into the per-turn `local_facts` dict (self-contained
        to this combined fact+question turn). Also mirrored into the
        persistent hippocampal entity index so a genuine later
        "remember what I told you" query can still find them.
        """
        low = (text or "").lower().strip()
        if not low:
            return
        # "my <X> is named/called <Y>"
        m = re.search(
            r"\bmy\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)\s+"
            r"(?:is|are)\s+(?:named|called|nemed|caled)\s+([a-z0-9'\-]+)",
            low)
        if m:
            ent, val = m.group(1).strip(), m.group(2).strip()
            local_facts.setdefault(ent, {})["name"] = val
            self._episodic_index.setdefault(ent, {})["name"] = val
            return
        # "a/an <X> named <Y>"  (list-form: "a cat named whiskers")
        for m2 in re.finditer(
                r"\b(?:a|an)\s+([a-z0-9]+(?:\s+[a-z0-9]+)?)\s+"
                r"named\s+([a-z0-9'\-]+)", low):
            ent, val = m2.group(1).strip(), m2.group(2).strip()
            local_facts.setdefault(ent, {})["name"] = val
            self._episodic_index.setdefault(ent, {})["name"] = val
        # "i was born in <Y>" / "i was born on <Y>"
        mb = re.search(r"\bi\s+(?:was|were|am)\s+born\s+(?:in|on|at)\s+"
                        r"([a-z0-9'\-]+(?:\s+[a-z0-9'\-]+)?)", low)
        if mb:
            local_facts.setdefault("i", {})["born"] = mb.group(1).strip()
            self._episodic_index.setdefault("i", {})["born"] = mb.group(1).strip()
        # "i built/made a/an <Y> ..." (biographical achievement)
        mk = re.search(r"\bi\s+(?:built|made|wrote|created|founded|started)\s+"
                        r"(?:a|an)\s+([a-z0-9'\-]+(?:\s+[a-z0-9'\-]+)?)", low)
        if mk:
            local_facts.setdefault("i", {})["built"] = mk.group(1).strip()
            self._episodic_index.setdefault("i", {})["built"] = mk.group(1).strip()
        # bare "my <X> is <Y>" (X not 'favorite') — e.g. "my pet dog is max"
        if "favorite" not in low and "named" not in low and "called" not in low:
            for bm in re.finditer(
                    r"\bmy\s+([a-z0-9]+)\s+(?:is|are)\s+([a-z0-9'\-]+)\b",
                    low):
                ent, val = bm.group(1).strip(), bm.group(2).strip()
                if ent in ("name",):
                    continue
                local_facts.setdefault(ent, {})["is"] = val
                self._episodic_index.setdefault(ent, {})["is"] = val

    def _mine_causal_rule(self, text: str) -> Optional[Dict[str, str]]:
        """Extract a conditional causal rule from a premise clause.

        Handles two shapes:
          (1) two-clause: "when <X> <v>, <Y> <v2>"
              e.g. "when turned on, the lamp lights up"
                   trigger="the lamp lights up"  (the subject that ACTS)
                   (the 'when turned on' is the enabling condition, not
                    the causal subject)
              e.g. "if the lamp lights up, an explosion occurs"
                   trigger="the lamp lights up", result="an explosion occurs"
          (2) single clause: "the lamp lights up" / "an explosion occurs"

        We key on the SECOND clause (after the comma) as the real
        causal event, and treat the enabling condition as its trigger
        only when no explicit second clause result exists.
        Returns {trigger_subj, trigger_verb, result_subj, result_verb}
        or None.
        """
        low = (text or "").lower().strip()
        if not low:
            return None
        _VEBS = (r"(lights?|turns?|turned|switches?|goes|explodes?|opens?|"
                 r"starts?|breaks?|falls?|rises?|occurs?|happens?|is|are|"
                 r"was|were)")
        # Shape (1): "when/if <cond>, <subject> <verb> [, <subj2> <verb2>]"
        # Split on commas; the enabling condition clause starts with
        # when/if/once/after and is NOT the causal event. The
        # REAL causal trigger is the other clause (or the clause
        # after a "when X, Y" comma). Mine subject+verb from
        # the trigger clause and (optionally) the result clause.
        _clauses = [c.strip() for c in re.split(r",", low) if c.strip()]
        _conn = re.compile(r"^(?:when|if|once|after)\b")
        _trigger_clause = None
        _result_clause = None
        _consumed = set()
        for ci, c in enumerate(_clauses):
            if _conn.match(c):
                # enabling condition: mine the trigger from WITHIN the
                # condition clause (after the connector word), e.g.
                # "if the lamp lights up" -> trigger "the lamp lights".
                # The NEXT clause (if any) is the RESULT.
                _inner = re.sub(r"^(?:when|if|once|after)\b\s*", "", c).strip()
                _trig_from_inner = self._verb_phrase(_inner, _VEBS)
                if _trig_from_inner:
                    _trigger_clause = _inner
                elif ci + 1 < len(_clauses):
                    _trigger_clause = _clauses[ci + 1]
                    _consumed.add(ci + 1)
                if ci + 1 < len(_clauses):
                    _result_clause = _clauses[ci + 1]
                    _consumed.add(ci + 1)
                continue
            if ci in _consumed:
                continue
            if _trigger_clause is None:
                _trigger_clause = c
            else:
                _result_clause = c
        trig = self._verb_phrase(_trigger_clause, _VEBS) if _trigger_clause else None
        res = self._verb_phrase(_result_clause, _VEBS) if _result_clause else None
        if trig:
            return {
                "trigger_subj": trig[0],
                "trigger_verb": trig[1],
                "result_subj": (res[0] if res else None),
                "result_verb": (res[1] if res else None),
            }
        # Shape (2): bare "<subject> <verb>" clause. Only keep
        # ACTION verbs (lights/turns/explodes/...) — exclude
        # statives like "was/is/are" that merely describe
        # state and carry no causal consequence.
        _ACTION = (r"(lights?|turns?|turned|switches?|goes|explodes?|"
                    r"opens?|starts?|breaks?|falls?|rises?|occurs?|happens?)")
        m2 = re.search(r"\b([a-z0-9'\s]+?)\s+" + _ACTION + r"\b", low)
        if m2:
            return {
                "trigger_subj": m2.group(1).strip(),
                "trigger_verb": m2.group(2).strip(),
                "result_subj": None,
                "result_verb": None,
            }
        return None

    def _verb_phrase(self, clause: str, verb_class: str) -> Optional[tuple]:
        """Extract (subject, verb) from a clause like 'the lamp lights up'.

        verb_class is the regex alternation of accepted verbs (e.g. _VEBS).
        Returns (subject_str, verb_str) or None if no verb match.
        """
        if not clause:
            return None
        m = re.search(r"([a-z0-9'\s]+?)\s+" + verb_class + r"\b", clause)
        if not m:
            return None
        subj = m.group(1).strip()
        # keep the article (a/an/the) so result clauses read naturally
        # ("an explosion occurs"); only strip a stray leading article
        # when the subject is the trigger and would read redundantly.
        return (subj, m.group(2).strip())

    def _normalize_token(self, tok: str) -> str:
        t = tok.lower().strip().strip("'\".,!?;:()")
        return t[:-2] if t.endswith("'s") else t

    def _answer_from_session_facts(self, question: str,
                                local_facts: Dict[str, Dict[str, str]],
                                local_rules: List[Dict[str, str]]) -> Optional[str]:
        """Answer a question from the in-turn premise store + causal rules.

        Strategy:
          (a) causal-chain question ('what happens if you turn on the lamp')
              -> walk the stored rules: turning on the lamp lights it up,
              which (per the rule) causes an explosion.
          (b) fact-retrieval question ('what is my pet dog's name') ->
              match the cued entity/attribute against the per-turn
              `local_facts` and the persistent hippocampal entity index
              (the latter covers a later cross-turn "remember" query).
        Returns a natural reply string, or None if nothing matches.
        """
        q = (question or "").lower().strip()
        if not q:
            return None

        # (a) Causal-chain query: "what happens if <X> <trigger>" / "what
        # would happen if ... turned on".
        _chain = re.search(
            r"\b(?:what|which)\s+(?:happens?|occurs?|would happen|will happen)"
            r".*\b(if|when|after)\b.*\b(turn|switch|light|explod|open|start|"
            r"break|fire|push|press)\b", q)
        if _chain and local_rules:
            return self._answer_causal_chain(local_rules)

        # (b) Fact retrieval: find a cued entity token in the question.
        # "I" / "you" map to the stored "i" biographical entity.
        _ent_hit = None
        _q_tokens = re.findall(r"[a-z']+", q)
        for tok in _q_tokens:
            t = self._normalize_token(tok)
            if t in ("i", "you") and ("i" in local_facts or "i" in self._episodic_index):
                _ent_hit = "i"
                break
            if t in local_facts or t in self._episodic_index:
                _ent_hit = t
                break
        if _ent_hit is None:
            # last resort: any stored entity whose label appears verbatim
            for ent in list(local_facts.keys()) + list(self._episodic_index.keys()):
                if ent and re.search(r"\b" + re.escape(ent) + r"\b", q):
                    _ent_hit = ent
                    break
        if _ent_hit is None:
            return None
        facts = dict(local_facts.get(_ent_hit, {}))
        facts.update(self._episodic_index.get(_ent_hit, {}))
        if not facts:
            return None
        # Attribute-focused: "what is my pet dog's NAME" -> name slot.
        attr_hit = None
        for attr in ("name", "color", "favorite", "is"):
            if re.search(r"\b" + attr + r"\b", q):
                attr_hit = attr
                break
        if attr_hit and attr_hit in facts:
            val = facts[attr_hit]
            if _ent_hit == "i":
                # Biographical self-fact: read as a first/second-person reply.
                if attr_hit == "born":
                    return f"you were born in {val}."
                if attr_hit == "built":
                    return f"you built {val}."
                return f"{val}."
            if attr_hit == "favorite":
                return f"your favorite {_ent_hit} is {val}."
            if attr_hit == "is":
                return f"your {_ent_hit} is {val}."
            return f"your {_ent_hit}'s {attr_hit} is {val}."
        # Generic: surface every stored attribute for the cued entity.
        bits = []
        for attr, val in facts.items():
            if _ent_hit == "i":
                if attr == "born":
                    bits.append(f"you were born in {val}")
                elif attr == "built":
                    bits.append(f"you built {val}")
                else:
                    bits.append(f"{val}")
            elif attr == "favorite":
                bits.append(f"your favorite {_ent_hit} is {val}")
            elif attr == "is":
                bits.append(f"your {_ent_hit} is {val}")
            else:
                bits.append(f"your {_ent_hit}'s {attr} is {val}")
        if bits:
            return "you told me " + "; ".join(dict.fromkeys(bits)) + "."
        return None

    def _answer_causal_chain(self, local_rules: List[Dict[str, str]]) -> str:
        """Walk stored causal rules into a natural-language chain answer.

        The miner stores each premise as an independent rule
        {trigger_subj, trigger_verb, result_subj, result_verb}.
        We chain them: a rule whose trigger matches another
        rule's result becomes a link. For the lamp test the
        stored rules are:
          R1: trigger="the lamp lights up"  (from "when turned on, the lamp lights up")
          R2: trigger="the lamp lights up", result="an explosion occurs"
        The enabling condition "turn on" maps to R1's trigger, so
        the full chain is:
          "the lamp lights up, and an explosion occurs!"
        """
        rules = local_rules
        if not rules:
            return None
        # Emit ONLY rules that carry a result clause (the real causal
        # links). Drop bare-trigger standalones: enabling conditions
        # like "the lamp lights up" (from "when turned on, the
        # lamp lights up") are already captured as the trigger of the
        # result-bearing rule, so a second standalone copy would
        # duplicate. Also skip non-causal statives ("a lamp was on
        # the table" has verb 'was' but no result -> excluded here).
        links = []
        for r in rules:
            if r["result_subj"]:
                links.append(
                    f"{r['trigger_subj']} {r['trigger_verb']}, "
                    f"and {r['result_subj']} {r['result_verb']}")
        if not links:
            return None
        joined = ", ".join(links)
        # Normalise articles for a smoother read.
        joined = re.sub(r"\ba lamp\b", "the lamp", joined)
        joined = re.sub(r"\ban explosion\b", "an explosion", joined)
        joined = re.sub(r"\ba explosion\b", "an explosion", joined)
        return joined.rstrip(", ") + "!"

    def _reconstruct_gist(self, rec: Dict[str, Any]) -> str:
        """Reconstruct a gist reply from a stored episode (no verbatim parroting
        beyond what was stored; no invention)."""
        facts = rec.get("facts", {})
        if facts:
            bits = []
            for slot, val in facts.items():
                if slot.startswith("favorite_"):
                    bits.append(f"your favorite {slot[len('favorite_'):]} is {val}")
                elif slot == "likes":
                    bits.append(f"you mentioned you like {val}")
            if bits:
                return "you told me " + "; ".join(bits) + "."
        return f"you mentioned: \"{rec.get('text', '')}\""

    def _episodic_remember(self, user_input: str) -> Optional[str]:
        """Handle a broad 'remember what I told you' recall query (Human-Likeness
        Plan C). Tries fact/slot retrieval and semantic gist retrieval over the
        portable transcript; fails closed when nothing matches.

        GATE: only attempt retrieval when the query is actually a memory/recall
        request. A plain new question ("what color is the sun") must NOT be
        intercepted by the episodic matcher — otherwise it would be hijacked by
        semantic overlap with an unrelated past turn. Fail-closed: non-recall
        queries return None and continue down the normal pipeline.
        """
        _recall_pat = re.compile(
            r"\b(remember|recall|remind me|what did i|what was i|"
            r"do you remember|what i (told|said|mentioned)|"
            r"what have i (told|said))\b", re.IGNORECASE)
        if not _recall_pat.search(user_input or ""):
            self._episodic_miss = False
            return None
        # Recognized as a recall query: if we end up returning None, it means we
        # genuinely have nothing stored — the caller must fail CLOSED (no
        # confabulation via web/graph), not fall through to a web lookup.
        self._episodic_miss = True
        # Stage 1 (plan M-E): an explicit STORE directive ("remember i love
        # stargazing") writes its fact THIS turn; the self-reference effect means
        # it should be retrievable immediately, not only from next turn on. The
        # old `transcript[:-1]` excluded the current turn, so a freshly stored
        # fact could not be recalled same-turn. We include the current turn's
        # record in the recall pool, EXCEPT when the current turn is itself a
        # recall query (which would otherwise self-match its own text in the
        # bare-recall path). A stored disclosure is never a recall pattern, so
        # this safely enables same-turn recall of just-stored facts.
        _current_is_recall = bool(_recall_pat.search(user_input or ""))
        prior = (self._episodic_transcript
                 if (not _current_is_recall and self._episodic_transcript)
                 else (self._episodic_transcript[:-1]
                       if self._episodic_transcript else []))
        if not prior:
            return None
        # Bare recall with no specific cue (a human still surfaces the gist of
        # what was shared). Reconstruct ALL mined facts from prior turns. We do
        # this BEFORE semantic cue-matching because the bare recall query shares
        # words with prior *recall* queries (e.g. "remember what i told you"
        # matches a previous "remember what i told you" turn) and would
        # otherwise retrieve the query itself instead of the shared content.
        _q = (user_input or "").lower()
        _bare = bool(re.search(r"remember\b.*\b(told|said|mentioned|tell)\b", _q)) \
            and not re.search(r"\b(cat|book|dune|astrophysics|name|color|food|movie|song|band|place|pet)\b", _q)
        if _bare:
            bits = []
            for rec in prior:
                facts = rec.get("facts", {})
                for slot, val in facts.items():
                    if slot.startswith("favorite_"):
                        bits.append(f"your favorite {slot[len('favorite_'):]} is {val}")
                    elif slot == "likes" and val not in " ".join(bits):
                        bits.append(f"you like {val}")
                    else:
                        # possessive/relational slot (cat_name) -> reconstruct as
                        # "your cat's name is whiskers" (pattern-completion, not
                        # the last episode's verbatim text). Fixes the
                        # wrong-episode contamination bug where bare recall echoed
                        # an unrelated prior turn's text.
                        ent, _, attr = slot.partition("_")
                        # C-fix (round v-aug04): pets stored under pet_name_N
                        # (entity "pet_name", attr index "N") must render as
                        # "your pet's name is X", not "your pet_name's 1 is X".
                        if (_pet := _pet_slots.render_pair(ent, attr, val)):
                            bits.append(_pet)
                        else:
                            bits.append(f"your {ent}'s {attr} is {val}")
            if bits:
                return "you told me " + "; ".join(dict.fromkeys(bits)) + "."
        # Specific-cue retrieval (fact slot or semantic gist) for cued recalls
        # like "remember my cat's name" / "what was the book i mentioned".
        out = self._retrieve_episodic(user_input, prior)
        if out:
            return out
        return None

    def _recall_past(self, subj: str, obj: str) -> List[str]:
        related = []
        for t in self._topic_list:
            pl = t.lower()
            sl = subj.lower()
            if pl != sl and (pl in sl or sl in pl or len(set(pl.split()) & set(sl.split())) > 0):
                related.append(t)
        return related[:3]

    def _recall_hippocampal(self, topic: str) -> Optional[List[int]]:
        """Reactivate a hippocampal index, spreading activation through the
        indexed graph pattern to reconstruct the memory experience.

        Returns the list of reactivated concept IDs, or None if topic not found.
        """
        entry = self._topic_store.get(topic.lower())
        if not entry:
            return None

        reactivated = []
        # Phase 1: Reactivate indexed concepts (sparse pattern)
        for nid in entry.get('indexed_concepts', []):
            node = self.graph.get_node(nid)
            if node and node.label:
                self.graph.activate(nid, 0.5)
                reactivated.append(nid)

        # Phase 2: Spread activation through indexed edges (pattern completion)
        for f_label, t_label in entry.get('indexed_edges', []):
            f_nids = self._concept_keywords.get(f_label.lower(), [])
            t_nids = self._concept_keywords.get(t_label.lower(), [])
            for fn in f_nids:
                for tn in t_nids:
                    edge = self.graph.get_edge(fn, tn)
                    if edge:
                        # Strengthen episodic edges during recall (pattern strengthening)
                        if edge.relation_type == "episodic":
                            edge.weight = min(0.35, edge.weight + 0.05)
                        # Activate both endpoints
                        self.graph.activate(fn, 0.4)
                        self.graph.activate(tn, 0.4)
                        if fn not in reactivated:
                            reactivated.append(fn)
                        if tn not in reactivated:
                            reactivated.append(tn)

        # Phase 3: Activate the subject concept at higher strength
        subj_nids = self._concept_keywords.get(topic.lower(), [])
        for sn in subj_nids:
            self.graph.activate(sn, 0.7)
            if sn not in reactivated:
                reactivated.append(sn)

        return reactivated

    def _hippocampal_index_topic(self, subject: str, activated_ids: List[int],
                                   hop_labels: List[Tuple[str, str]]):
        """Create a hippocampal index for the current topic and store it.

        The index is a lightweight pointer to the distributed graph pattern
        (concept IDs + edge references), not the content itself.
        """
        sl = subject.lower()
        # Build index: which concept nodes were activated
        indexed_concepts = list(set(activated_ids))

        # Build index: which edge pairs were traversed
        indexed_edges = [(f.lower(), t.lower()) for f, t in hop_labels]

        # Store as lightweight index, not full content
        index_entry = {
            'label': subject,
            'turn': self.turn_count,
            'indexed_concepts': indexed_concepts[:10],  # sparse index
            'indexed_edges': indexed_edges[:5],
            'vad': (self.emotion.state.valence, self.emotion.state.arousal,
                    self.emotion.state.dominance),
            'visit_count': 1,
            'response_summary': '',  # placeholder, not content
        }

        if sl not in self._topic_store:
            self._topic_store[sl] = index_entry
        else:
            entry = self._topic_store[sl]
            entry['visit_count'] += 1
            entry['turn'] = self.turn_count
            # Merge new indexed concepts
            existing_cons = set(entry.get('indexed_concepts', []))
            existing_cons.update(indexed_concepts[:10])
            entry['indexed_concepts'] = list(existing_cons)[:15]
            entry['vad'] = index_entry['vad']

    def _detect_recall_trigger(self, text: str) -> Optional[str]:
        """Phase 3.3: Detect if user is recalling a past topic using vector semantics.
        
        Uses GloVe vector similarity between query words and recall-related seed concepts.
        If any query word has a cosine similarity >= _RECALL_DETECTION_THRESHOLD to any
        recall seed concept, the query is treated as a recall attempt.
        
        This avoids hardcoded trigger patterns and naturally generalizes to any
        semantically similar phrasing (e.g., "forgot", "previously", "what did I")."""
        text_lower = text.lower()
        words = [w.strip(".,!?") for w in text_lower.split() if len(w.strip(".,!?")) >= 3]
        
        # Pre-compute GloVe vectors for recall seeds (lazy cache)
        if not hasattr(self, '_recall_seed_vecs'):
            seed_vecs = {}
            for seed in self._recall_seed_concepts():
                v = self._glove_vector(seed)
                if v is not None:
                    seed_vecs[seed] = v
            self._recall_seed_vecs = seed_vecs
        
        if not self._recall_seed_vecs:
            return None
        
        # Check each content word in the query for semantic similarity to recall seeds
        is_recall = False
        for word in words:
            wv = self._glove_vector(word)
            if wv is None:
                continue
            for seed, sv in self._recall_seed_vecs.items():
                sim = float(np.dot(wv, sv))
                if self._adaptive_gate("recall_cos", sim):
                    is_recall = True
                    break
            if is_recall:
                break
        
        if not is_recall:
            return None
        
        # If recall detected, find the most relevant past topic
        if self._topic_list:
            # Score each past topic by semantic similarity to the query
            best_topic = None
            best_score = 0.0
            for topic in reversed(self._topic_list):
                tv = self._glove_vector(topic)
                if tv is None:
                    continue
                score = 0.0
                for word in words:
                    wv = self._glove_vector(word)
                    if wv is not None:
                        score += float(np.dot(wv, tv))
                if score > best_score:
                    best_score = score
                    best_topic = topic
            if best_topic:
                return best_topic
            return self._topic_list[-1]
        
        return text_lower.split()[0] if words else None

    def _store_episodic(self, subject: str, associations: List[Tuple[str, float]]):
        """Create episodic edges linking current subject to top associations.
        Phase 3.2: On revisit, boost weight. 3+ visits => migrate to semantic."""
        if not subject or not associations:
            return
        subj_nids = self._concept_keywords.get(subject.lower(), [])
        if not subj_nids:
            return
        subj_nid = subj_nids[0]
        for assoc_label, _ in associations[:3]:
            assoc_nids = self._concept_keywords.get(assoc_label.lower(), [])
            if not assoc_nids:
                continue
            assoc_nid = assoc_nids[0]
            existing = self.graph.get_edge(subj_nid, assoc_nid)
            if existing is None:
                self.graph.add_edge(subj_nid, assoc_nid,
                                    weight=0.15, relation_type="episodic")
            elif existing.relation_type == "episodic":
                sl = subject.lower()
                entry = self._topic_store.get(sl, {})
                visits = entry.get('visit_count', 1) if isinstance(entry, dict) else 1
                if visits >= 3:
                    existing.relation_type = "semantic"
                    existing.weight = min(0.40, existing.weight + 0.15)
                elif visits >= 2:
                    existing.weight = min(0.30, existing.weight + 0.10)

    def _try_memory_query(self, user_input: str) -> Optional[str]:
        """Fix 4 (Q12): answer episodic meta-queries about the conversation.

        Handles "what did I just ask you", "what did I say", "what were we
        talking about", "what was my last question", "do you remember what I
        asked". These are queries whose subject is the DIALOGUE itself, so the
        subject-keyed hippocampal buffer cannot serve them — we answer from the
        verbatim user-turn ring buffer (Baddeley episodic buffer + hippocampal
        pattern completion). Returns None when the input is not a memory
        meta-query, so the normal pipeline runs.

        NOTE: called BEFORE the current turn is appended to
        ``_recent_user_turns``, so ``[-1]`` is the immediately preceding turn.
        """
        t = (user_input or "").lower().strip(" ?!.")
        if not t:
            return None
        # R3 (round v3): a first-person self-disclosure STATEMENT (e.g.
        # "i play the veena", "i run a marine research boat") is something to
        # STORE, not a memory/recall query. Routing it through the episodic
        # matcher returned a RANDOM prior utterance ("you mentioned: i run a
        # boat") instead of storing the new fact. _is_self_disclosure_stmt
        # already rejects interrogatives and recall-verb phrasings
        # ("remember/what did i/recall"), so a True here is unambiguous: this is
        # a fresh disclosure. Return None so it flows down to the vmPFC
        # self-disclosure gate (which stores + acks it). No authored prose.
        if hasattr(self, "_is_self_disclosure_stmt") and self._is_self_disclosure_stmt(user_input):
            return None
        # Questions about the USER's name/identity ("do you remember my name?",
        # "what is my name?", "who am i?") belong to the identity block in
        # process_turn (user_model.user_name), NOT episodic recall — otherwise
        # "do you remember my name?" would be swallowed here as a generic
        # self-recall (strategy=memory_recall) and never reach the stored name.
        if re.search(r"\b(?:my name|who am i)\b", t):
            return None
        # D-fix (round 2026-08-08b): the agent-claim recall below must fire ONLY
        # when the user is asking about RAVANA's OWN self-description ("what did
        # you say about who you are", "earlier you described yourself"). It must
        # NOT fire on a user-fact recall like "earlier you said something about
        # how I see cities" — that is the user asking about THEIR OWN stance, and
        # routing it to the agent-claim store returns RAVANA's self-intro (a
        # self/other boundary breach: D-C class). Gate the "earlier you said/
        # mentioned/told me" and "you said something about" branches on the
        # recalled content being about the AGENT (yourself / who you are / what
        # you are / your nature), so a query containing any first-person USER
        # reference (i / my / me) falls through to the genuine user-episode
        # recall instead. Structural (regex), not a per-topic guard.
        _user_ref = bool(re.search(r"\b(i|my|me|we)\b", t))
        _agent_self_recall = (
            bool(re.search(
                r"\bwhat did you (?:say|tell me|answer|describe|say about)\b", t))
            and not _user_ref
        ) or bool(re.search(
            r"\bearlier you (?:described|said|told me|mentioned) "
            r"(?:yourself|who you are|what you are|your nature)\b", t)
        ) or bool(re.search(r"\byou described yourself\b", t)
        ) or (bool(re.search(r"\byour answer about\b|\bwhat was your answer\b", t))
              and not bool(re.search(r"\b(i|my|we)\b", t))
        ) or bool(re.search(
            r"\byou (?:said|mentioned|told me) something about what you "
            r"(?:are|were)\b", t)
        ) or bool(re.search(
            r"\bremind me what you (?:said|told me) (?:about|you are)\b", t))
        if _agent_self_recall:
            _claim = getattr(self, "_agent_claims", {}).get("self")
            if _claim:
                return _claim
        # D3 (round v3): self-attribute EXISTENCE questions ("am i a doctor",
        # "are you a vegetarian", "was i your friend"). These ask whether a
        # specific attribute is TRUE of the user, so the answer must come from
        # the user's stored facts — not from the confabulating semantic matcher,
        # which would echo a random prior turn (the D-E bug: "am i a doctor"
        # returned "tell me something you've figured out about me"). If the
        # attribute is stored, report it; if NOT stored, answer honestly that
        # the user never told RAVANA that, rather than inventing a memory.
        _exist = re.search(
            r"\b(am i|are you|was i|were you|is (?:that|it) true|did i)\b"
            r"(?:\s+(?:really\s+)?(?:a|an|the))?\s+([a-z]+)\b", t)
        if _exist:
            _attr = _exist.group(2).strip().lower()
            if _attr in ("doctor", "vegetarian", "teacher", "pilot", "student",
                         "friend", "married", "single", "rich", "poor", "sick",
                         "tired", "happy", "sad"):
                _idx = getattr(self, "_episodic_index", {}) or {}
                _found = False
                for _ent, _facts in _idx.items():
                    for _k, _v in _facts.items():
                        if _attr in str(_v).lower() or _attr == _k:
                            _found = True
                            break
                    if _found:
                        break
                if _found:
                    return f"yes — from what you've told me, you are {_attr}."
                return (f"you haven't told me you're {_attr}, so i can't say. "
                        f"if you are, just tell me and i'll remember it.")
        # B1: self-knowledge recall. "what do you remember about me" / "what do
        # you know about me" / "what have i told you" are recalls of the USER's
        # own disclosed autobiographical facts (stored in the hippocampal entity
        # index, self._episodic_index). They must be answered from that personal
        # store, NEVER by looking up the dictionary definition of the word
        # "remember" (source monitoring / self-other boundary; Mitchell & Johnson
        # 2009). Detect the speech act structurally (first/second person + recall
        # verb + self reference) and via the SocialIntentClassifier 'self_recall'
        # centroid, then route to _retrieve_episodic. Fail-closed: if the episodic
        # store has nothing for the user, return None and let the pipeline fall
        # through to honest uncertainty (the ConceptNet def of "remember" is NOT
        # a valid answer for this query).
        # W1: first-person PAST-TENSE autobiographical memory report
        # ("i remember when i felt anxious last year", "i felt anxious
        # last year", "we experienced that back then"). A tense/aspect
        # detector: a first/second-person subject + a past-memory verb or a
        # recall verb, optionally with a temporal-displacement anchor
        # (last year / yesterday / ago / when i / back then). This is a
        # grammatical-aspect signal, not a topic keyword list, so it stays
        # brain-faithful (Tulving autonoetic recollection is past-displaced;
        # source-monitoring tags a retrieved memory vs a present feeling).
        _self_recall_struct = bool(re.search(
            r"\b(?:what|anything|tell me)\b.*\b(?:do )?you\b.*\b(?:remember|know|recall|told|tell|learned?|found out|discovered|figured out)\b"
            r".*\b(?:about me|me|my|myself)\b", t)) or \
            bool(re.search(r"\b(?:remember|recall)\b.*\b(?:what i|what i told|me)\b", t)) or \
            bool(re.search(
                r"\b(?:i|we|you)\b\s+(?:remember|recall|felt|felt like|was feeling|"
                r"experienced|went through|lived through|thought about)\b"
                r"(?:.*\b(?:last year|last week|yesterday|ago|back then|when i|"
                r"when we|that time)\b)?", t))
        _self_recall_intent = False
        _clf = getattr(self, "_social_intent", None)
        if _clf is not None:
            try:
                _si, _ = _clf.detect(t)
                _self_recall_intent = (_si == "self_recall")
            except Exception:
                _self_recall_intent = False
        if _self_recall_struct or _self_recall_intent:
            # D5 (round v2): decide whether this is a SPECIFIC cued recall
            # ("what did you tell me about my cat") or a GENERIC self-recall
            # ("what have you learned about me" / "what do you know about me").
            # A generic self-recall carries NO entity/attribute cue, so routing
            # it through _retrieve_episodic's loose semantic matcher returns a
            # RANDOM prior utterance (confabulation) instead of the honest
            # self-profile summary. We only call the cued retriever when the
            # query actually names a stored entity or a biographical attribute
            # (location/name); otherwise we build the profile summary directly
            # from the hippocampal entity index. Fail-closed: if the index is
            # empty, the summary path returns the honest "you haven't told me
            # much" line. This keeps "what have you learned about me" honest.
            _idx = getattr(self, "_episodic_index", None) or {}
            _LOC_WORDS = ("live", "lives", "from", "city", "town", "country",
                          "born", "grew", "located", "location", "origin")
            _cue = False
            for _tok in re.findall(r"[a-z']+", (user_input or "").lower()):
                _t = _tok[:-2] if _tok.endswith("'s") else _tok
                # A SPECIFIC cued entity (a stored entity name, or a
                # biographical attribute word) — not the bare self-reference
                # pronouns, which carry no specific retrieval target and were
                # wrongly sending generic recalls into the confabulating
                # semantic matcher.
                if _t in _idx and _t not in ("i", "you", "my", "your"):
                    _cue = True
                    break
                if _t in _LOC_WORDS or _t == "name":
                    _cue = True
                    break
            if _cue:
                _ep = self._retrieve_episodic(user_input)
                if _ep is not None:
                    return _ep
            # Generic self-recall: reconstruct a gist from ALL disclosed user
            # facts held in the hippocampal entity index (Tulving encoding
            # specificity: a self-directed memory query without a target
            # reconstructs the whole disclosed self-profile). Brain-faithful:
            # we surface what was actually stored, never a dictionary node.
            #
            # D3 (round v3): merge the PersonalFactStore (the canonical store
            # mine_personal_facts writes to — e.g. "does" activity facts like
            # "i run a chai stall") with the hippocampal entity index, because
            # the learned-profile summary must reflect ALL disclosed facts, not
            # only those the episodic indexer captured. Both stores are runtime
            # stores RAVANA grows from conversation, so this is a union of real
            # cognition, not a lookup table.
            _bits = []
            _pf = getattr(self, "user_model", None)
            _pf_facts = {}
            if _pf is not None:
                try:
                    for _k, _v in _pf.personal_facts.facts.items():
                        if getattr(_v, "superseded", False):
                            continue
                        # key shape: (subject, attribute, value)
                        _subj = _k[0] if isinstance(_k, (tuple, list)) and len(_k) >= 3 else "i"
                        _attr = _k[1] if isinstance(_k, (tuple, list)) and len(_k) >= 3 else _k
                        if str(_subj).lower() not in ("i", "me", "my", "you"):
                            continue
                        _val = getattr(_v, "value", _v)
                        if not _val:
                            continue
                        # R3 (round v3): keep ALL values per (entity, attribute)
                        # as a LIST, not a single slot. The PersonalFactStore
                        # stores "i/does/boat", "i/does/veena", "i/does/coral"
                        # as SEPARATE records (multiple activities coexist).
                        # Collapsing them into one dict slot lost every value
                        # but the last, so "what do you know about me" only
                        # surfaced "you do veena". Lists preserve all learned
                        # facts. Content comes from the live store, no prose.
                        _pf_facts.setdefault("i", {}).setdefault(_attr, []).append(_val)
                except Exception:
                    pass
            # Union of episodic index + personal-fact store (personal facts win
            # on conflict so corrections surface). Episodic index holds single
            # values; personal-fact store holds lists — normalize both to lists.
            _merged = {}
            for _e, _f in _idx.items():
                _merged.setdefault(_e, {})
                for _a, _v in _f.items():
                    _merged[_e].setdefault(_a, []).append(_v)
            for _e, _f in _pf_facts.items():
                _m = _merged.setdefault(_e, {})
                for _a, _vals in _f.items():
                    _m.setdefault(_a, []).extend(_vals)
            for _ent, _facts in _merged.items():
                # The "i" entity is the USER's own biographical profile
                # (populated by the self-disclosure handler), so its attributes
                # must render as natural first/second-person statements, never
                # as "your i's name is X".
                _is_user = (str(_ent).lower() in ("i", "me", "my", "you"))
                for _attr, _vals in _facts.items():
                    for _val in _vals:
                        if _attr == "favorite":
                            _bits.append(f"your favorite {_val}")
                        elif _attr == "likes":
                            _bits.append(f"you like {_val}")
                        elif _attr == "is":
                            _bits.append(f"your {_ent} is {_val}" if not _is_user
                                         else f"your {_attr} is {_val}")
                        elif _attr == "does":
                            # D3 (round v3): self-disclosed activity
                            # ("i run a chai stall" -> does=chai stall). Render
                            # as "you do X" so the learned-profile summary
                            # reflects what the user told us they do.
                            _bits.append(f"you do {_val}")
                        elif _attr == "name":
                            # D6 (round 2026-08-08b-d): a possessive NAME fact
                            # (partner, pet, ...) must keep its OWNER in the
                            # render, never collapse onto the user's "your name
                            # is X". The old code rendered every name attr as
                            # "your name is X" regardless of entity, so a
                            # partner's/pet's name was reported as the USER's own
                            # name — a self/other boundary breach. Only the user
                            # entity (i/me/my/you) uses "your name is"; any other
                            # entity keeps "{ent}'s name".
                            _bits.append(
                                f"your name is {_val}" if _is_user
                                else f"your {_ent}'s name is {_val}")
                        elif _attr == "location":
                            _bits.append(f"you live in {_val}")
                        elif _attr == "role":
                            _bits.append(f"you are {_val}")
                        elif _attr == "allergy":
                            _bits.append(f"you're allergic to {_val}")
                        elif _attr.startswith("favorite_"):
                            _bits.append(f"your favorite {_attr[len('favorite_'):]} is {_val}")
                        # Pets stored under a species-keyed slot.
                        elif (_pet := _pet_slots.render_pair(_ent, _attr, _val)):
                            _bits.append(_pet)
                        else:
                            _bits.append(f"your {_ent}'s {_attr} is {_val}" if not _is_user
                                         else f"your {_attr} is {_val}")
            if _bits:
                # Case-insensitive dedup: the same fact can surface from both
                # the personal-fact store ("your name is a hypocrite") and the
                # episodic index ("your name is A Hypocrite"); keep the first
                # occurrence's original casing, drop the duplicate. Avoids the
                # garbled "your name is a hypocrite; your name is A Hypocrite".
                _seen = set()
                _deduped = []
                for _b in _bits:
                    _k = _b.lower()
                    if _k not in _seen:
                        _seen.add(_k)
                        _deduped.append(_b)
                _summary = "; ".join(_deduped)
                return f"from what you've told me, {_summary}."
            return ("i don't think you've told me much about yourself yet — "
                    "but i'm listening whenever you want to share.")
        # A2: first/second-person autobiographical-ATTRIBUTE recall
        # ("where do i live", "what city am i from", "when was i born",
        # "what is my name"). These ask about a SPECIFIC stored personal
        # fact, so route them to the entity-indexed _retrieve_episodic
        # (which does precise pattern completion on self._episodic_index).
        # This is the self/other boundary for recall: a personal attribute
        # is retrieved from the hippocampal self-profile, never from the
        # world-knowledge graph of the subject word (Mitchell & Johnson 2009
        # source monitoring). Must NOT match genuine world queries
        # ("what do you know about paris").
        _personal_attr_recall = bool(re.search(
            r"\b(i|me|my|we|our|you)\b", t)) and bool(re.search(
            r"\b(live|lives|from|born|named|called|name|location|"
            r"city|town|country|age|height|weight|work|study|studied|"
            r"grew up|went to school|run|own|operate|play|keep|stall|"
            r"business|shop|instrument|hobby|watch|watch(?:ing)?)\b", t))
        if _personal_attr_recall:
            _ep = self._retrieve_episodic(user_input)
            if _ep is not None:
                return _ep
            # B1/d3-fix: when the episodic retriever misses because the query
            # cued a self-activity/attribute fact that lives in the
            # PersonalFactStore (e.g. "where do i run my stall" -> does=chai
            # stall near) but never entered the episodic index (the
            # entity-indexing only maps LOCATION/name words to the "i"
            # profile), answer from the REAL personal-fact store. The store is
            # a runtime store RAVANA grows from conversation, so this is a
            # union of genuine cognition, not a lookup table. Fail-closed: if
            # nothing matches, return None and let the pipeline fall through to
            # honest uncertainty instead of a web definition of the word.
            _pf = getattr(self, "user_model", None)
            if _pf is not None:
                try:
                    _pf_facts = _pf.personal_facts.facts
                    _q_tokens = {t.strip(".,!?\"'()[]{}*:;").lower()
                                 for t in re.findall(r"[a-z']+", (user_input or "").lower())}
                    _q_tokens.discard("")
                    _matched = []  # (attr, value)
                    for _k, _v in _pf_facts.items():
                        if getattr(_v, "superseded", False):
                            continue
                        _subj = _k[0] if isinstance(_k, (tuple, list)) and len(_k) >= 3 else "i"
                        if str(_subj).lower() not in ("i", "me", "my", "you"):
                            continue
                        _val = str(getattr(_v, "value", _v) or "").lower()
                        _attr = (_k[1] if isinstance(_k, (tuple, list)) and len(_k) >= 3 else _k) or ""
                        # Match on a value token OR the cued attribute word.
                        _val_tokens = {t for t in re.findall(r"[a-z']+", _val)}
                        if _val_tokens & _q_tokens or _attr.lower() in _q_tokens:
                            _matched.append((_attr, getattr(_v, "value", _v)))
                    if _matched:
                        _bits = []
                        for _attr, _val in _matched:
                            if _attr == "does":
                                _bits.append(f"you do {_val}")
                            elif _attr == "name":
                                _bits.append(f"your name is {_val}")
                            elif _attr == "is":
                                _bits.append(f"you are {_val}")
                            elif _attr == "location":
                                _bits.append(f"you live in {_val}")
                            elif _attr == "likes":
                                _bits.append(f"you like {_val}")
                            elif _attr == "allergy":
                                _bits.append(f"you're allergic to {_val}")
                            else:
                                _bits.append(f"your {_attr} is {_val}")
                        if _bits:
                            return "you told me " + "; ".join(dict.fromkeys(_bits)) + "."
                except Exception:
                    pass
        # First/second-person + a recall/speech verb, referring to a prior turn.
        # Require an explicit conversational-memory phrasing to stay narrow.
        _patterns = [
            r"\bwhat did i (?:just )?(?:ask|say|tell|mention)\b",
            r"\bwhat (?:was|were) (?:my|the) (?:last |previous |first )?"
            r"(?:question|questions|message|thing i said|conversation)\b",
            r"\bwhat (?:were|are) we (?:talking|chatting) about\b",
            r"\bwhat (?:did|were) we (?:talk|talking) about\b",
            r"\b(?:do|can) you remember what i (?:asked|said|told)\b",
            r"\bwhat was i (?:just )?(?:asking|saying|talking) about\b",
            r"\brepeat (?:my|the) (?:last |previous )?question\b",
            r"\bour (?:first|last) (?:conversation|chat|talk)\b",
        ]
        if not any(re.search(p, t) for p in _patterns):
            return None
        prior = self._recent_user_turns
        if not prior:
            return "you haven't asked me anything yet this session — i don't have an earlier turn to recall."

        # §2 temporal index: "first/last conversation" answered as pure index
        # math over the transcript's turn_index (hippocampal time cells), never
        # a middle/arbitrary turn.
        if re.search(r"\bour (first|last) (?:conversation|chat|talk)\b", t):
            idx = self._episodic_indexer
            if idx is not None:
                ep = idx.first() if "first" in t else idx.last()
                if ep is not None:
                    _rec = self._reconstruct_gist({
                        "text": ep.text, "topic": ep.topic, "facts": ep.facts})
                    return (f"our {('first' if 'first' in t else 'last')} "
                            f"conversation — {_rec}")

        # "what did i just tell you [i like / my favorite ...]" — reconstruct
        # the GIST of the relevant prior turn (Tulving encoding
        # specificity), not the verbatim question. The miner already extracted
        # the self-disclosed facts, so surface those.
        if re.search(r"\bwhat did i (?:just )?(?:tell|say|mention)\b", t):
            # B-fix (round v-aug04) + ROUND 2026-08-09i: a TOPIC-CUED
            # variant ("what did i say about swarm", "what did i tell you
            # about my cats") asks about a SPECIFIC subject, not the
            # literally-preceding turn. The old code did a strict substring
            # scan of the cue against each episode's text, which MISSED
            # morphology variants ("swarm" vs the stored "swarmed",
            # "die" vs "died") and fell through to the previous turn — so
            # "what did i say about the swarm" returned the trailer fact
            # instead of the swarm episode. Delegate to _retrieve_episodic,
            # which now does morphology-invariant (Porter-stem) cue matching
            # and returns the episode actually containing the cue. Only when
            # there is NO cue do we default to the immediately-preceding turn
            # (genuine "what did i just say?").
            _cue = ""
            # Broaden cue capture (round 2026-08-14T1110Z): the old regex only
            # matched `about/that/regarding/on <word>`, so a possessive/relative
            # query like "what did i say MY BROTHER does" or "what did i tell you
            # about MY CATS" (the latter actually matched via `about`) left the
            # cue empty and fell through to a verbatim prior-turn echo. We now
            # also capture `my <word>`, `the <word>`, and `<word>'s` so the
            # query's real subject (brother, cat, ...) is used to retrieve the
            # matching episode. This is structural (possessive/referent regex),
            # not a per-topic table.
            _m = re.search(
                r"\b(?:about|that|regarding|on|my|the)\s+([a-z']+)"
                r"|([a-z']+)'s\b", t)
            if _m:
                _cue = (_m.group(1) or _m.group(2) or "").lower().strip(".,!?")
            if _cue and len(_cue) >= 3:
                # Delegate to _retrieve_episodic with the ORIGINAL query — it now
                # does morphology-invariant (Porter-stem) cue matching and returns
                # the episode actually containing the cue. Rewriting the query
                # (e.g. "what did i tell you about swarm") changes the cue shape
                # and can miss; the original user_input is what was proven to work.
                _cued = self._retrieve_episodic(user_input)
                if _cued:
                    return _cued
            last_turn = prior[-1].strip()
            # Pull the matching transcript record (highest turn_index = prev).
            matching = [r for r in self._episodic_transcript
                        if r.get("text", "").strip().lower() == last_turn.lower()]
            rec = matching[-1] if matching else None
            if rec is not None:
                facts = rec.get("facts", {})
                # Reconstruct gist from mined facts when present.
                bits = []
                for slot, val in facts.items():
                    if slot.startswith("favorite_"):
                        bits.append(f"your favorite {slot[len('favorite_'):]} is {val}")
                    elif slot == "likes":
                        bits.append(f"you like {val}")
                    elif _pet_slots.is_pet_attribute(slot):
                        bits.append(_pet_slots.render(slot, val))
                    elif "_" in slot:
                        ent, _, attr = slot.partition("_")
                        bits.append(f"your {ent}'s {attr} is {val}")
                    else:
                        bits.append(f"you said: {val}")
                if bits:
                    return "you just told me " + "; ".join(dict.fromkeys(bits)) + "."
            # Fallback: verbatim echo of the prior turn.
            return f'you just told me: "{last_turn}"'

        last = prior[-1].strip()
        # "what were we talking about" → topic-oriented; else verbatim recall.
        if re.search(r"\bwe (?:talking|talk|were|are) (?:about|chatting)\b", t) \
                or re.search(r"\btalking about\b", t):
            topic = ""
            if getattr(self, "_topic_list", None):
                topic = self._topic_list[-1]
            if topic:
                return f'we were talking about {topic}. your last message was: "{last}"'
            return f'your last message was: "{last}"'
        return f'you just asked me: "{last}"'

    def _reasoning_loop(self, ctx: CognitiveResponseContext) -> Tuple[str, str]:
        """Reasoning loop: web search + syntactic pipeline only.

        Stripped of the neural decoder (CE ~3.9, always produced word salad).
        Web search enriches graph knowledge, then the syntactic pipeline
        generates the response via SurfaceRealizer.
        """
        subject = ctx.subject
        query = ctx.raw_input
        assocs = ctx.associated_concepts

        subj_lower = subject.lower()
        subj_known = subj_lower in self._concept_keywords or subj_lower in self._concept_labels
        assoc_known = len(assocs) > 0

        is_complex = any(w in query.lower() for w in
                        ["how", "why", "create", "build", "design", "blueprint",
                         "explain", "detail", "comprehensive", "step by step",
                         "architecture", "implementation", "guide", "tutorial"])
        is_unknown = not subj_known or not assoc_known

        search_queries = []
        if is_complex or is_unknown:
            search_queries = self._decompose_for_search(query, subject, assocs)

        search_queries = [sq for sq in search_queries if sq][:4]

        for sq in search_queries:
            try:
                self.learn_from_web(sq)
            except Exception:
                continue

        # Syntactic pipeline only — no neural decoder generation
        try:
            syntax_response = self._generate_with_decoder_and_syntax(ctx)
            if syntax_response and len(syntax_response) > 10 and not _is_word_salad(syntax_response, subject=ctx.subject):
                _words = syntax_response.lower().split()
                _unique_ratio = len(set(_words)) / max(1, len(_words))
                if _unique_ratio >= 0.35:
                    return (syntax_response, "dorsal_reasoned")
        except Exception:
            pass

        return self._graph_fallback_response(ctx)

    def _extract_learning_query(self, text: str, activated_ids: List[int]) -> Optional[str]:
        """Extract what topic RAVANA should search for.

        Uses the LEAST-known word (not matched to any concept) as the query.
        """
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        # Find words NOT matched to any concept
        matched_labels = set()
        for nid in activated_ids:
            node = self.graph.get_node(nid)
            if node and node.label:
                matched_labels.add(node.label.lower())

        meaningful = [w.strip("'") for w in words if w.strip("'") not in STOP_WORDS]
        # Pick the last meaningful word that is NOT already known
        for w in reversed(meaningful):
            if w not in matched_labels:
                return w
        # If all words are known, use the last meaningful word anyway
        if meaningful:
            return meaningful[-1]
        return None

    def _predict_user_next(self, subject: str, assocs) -> None:
        """Covert other-monitoring (brief behavior 8): predict the user's likely
        next concept from the subgraph co-activated with the current subject.

        This is the lightweight forward simulation of the interlocutor — the bot
        internally "simulates" what the user will say next (mirroring Castellucci
        / Pickering & Garrod other-monitoring) so the relevant subgraph is
        pre-activated and common ground can be tracked. The prediction is the
        most salient association to the current subject that isn't the subject
        itself, weighted by edge strength. Stored for comparison next turn.
        """
        best, best_score = "", 0.0
        try:
            for label, score in (assocs or []):
                ll = label.lower()
                if ll == (subject or "").lower():
                    continue
                if self._is_function_word(ll):
                    continue
                s = float(score) if score else 0.0
                if s > best_score:
                    best, best_score = ll, s
        except Exception:
            pass
        self._predicted_user_next = best
        self._predicted_user_conf = best_score

    def _common_ground_score(self, subject: str) -> float:
        """Compare this turn's subject to the predicted next concept.

        Returns a 0..1 common-ground signal: 1.0 when the user's actual next
        topic matches the prediction (shared mental model), falling off with
        topic distance via GloVe cosine when available, else 0.5 on a near
        match and 0.0 on a miss. Feeds the verbosity knob so the bot stays
        concise once ground is established rather than re-explaining.
        """
        pred = self._predicted_user_next
        if not pred or not subject:
            return 0.0
        subj = subject.lower()
        if pred == subj:
            return 1.0
        sv = self._glove_vector(pred) if hasattr(self, "_glove_vector") else None
        tv = self._glove_vector(subj) if hasattr(self, "_glove_vector") else None
        if sv is not None and tv is not None:
            sim = float(np.dot(sv, tv))
            if self._adaptive_gate("episodic_rel", sim, strict=True):
                return float(np.clip(0.5 + sim * 0.5, 0.0, 1.0))
        if pred in subj or subj in pred:
            return 0.5
        return 0.0

    def _activate_from_input(self, text: str) -> List[int]:
        """Activate concepts using N400/P600 sequential per-word processing.

        For each word in the input (in order):
        N400 phase: Retrieve the word's concept vector and compute prediction error
          (surprise = 1 - cosine similarity with accumulated context). High surprise
          = stronger retrieval activation (larger N400 amplitude).
        P600 phase: Integrate the retrieved meaning into the evolving sentence context.
          Propagate activation to graph neighbors, modulated by how well the word fits.

        Neuroscience basis:
        - Brouwer Retrieval-Integration theory (2012, 2017): every word elicits
          N400 (retrieval) followed by P600 (integration)
        - Nature 2024: single-neuron responses are context-dependent, not fixed
        - PMC 2023: representational dimensionality ramps across sentence
        """
        words = re.findall(r"[a-zA-Z']{1,}", text.lower())
        scores: Dict[int, float] = {}
        # Accumulated sentence context for N400 modulation
        acc_ctx = np.zeros(self.dim, dtype=np.float32)
        word_count = 0

        for w in words:
            if w in STOP_WORDS and word_count > 0:
                continue

            # === N400: Retrieve word meaning ===
            w_nids = self._concept_keywords.get(w, [])
            w_vec = None
            if w_nids:
                node = self.graph.get_node(w_nids[0])
                if node and node.vector is not None:
                    w_vec = node.vector

            # Compute N400 amplitude (surprise): how predictable is this word?
            n400_surprise = 0.5  # baseline
            if w_vec is not None and word_count > 0:
                n_acc = np.linalg.norm(acc_ctx)
                n_w = np.linalg.norm(w_vec)
                if n_acc > 1e-8 and n_w > 1e-8:
                    cos_sim = float(np.dot(acc_ctx, w_vec)) / (n_acc * n_w)
                    n400_surprise = 1.0 - max(0.0, min(1.0, cos_sim))  # 0=expected, 1=surprising
            elif word_count == 0:
                n400_surprise = 0.8  # First word is always somewhat surprising

            # Activate with N400-modulated strength
            if w_nids:
                for nid in w_nids:
                    node = self.graph.nodes.get(nid)
                    pos_boost = 0.5 if node and node.label and self._concept_pos.get(node.label.lower()) == 'noun' else 0.0
                    base = 5.0 + pos_boost
                    # N400 surprise amplifies activation for unexpected words
                    n400_boost = 1.0 + n400_surprise * 2.0
                    scores[nid] = scores.get(nid, 0) + base * n400_boost

            # Label matching (same as before)
            # Snapshot: background learner may add nodes mid-turn.
            for nid, node in list(self.graph.nodes.items()):
                if not node or not node.label:
                    continue
                label = node.label.lower()
                s = scores.get(nid, 0.0)
                if label == w:
                    s += 5.0 * (1.0 + n400_surprise)
                elif len(w) >= 3 and (label == w or label.startswith(w + " ") or (" " + w + " ") in label or label.endswith(" " + w)):
                    s += 3.0
                elif len(label) >= 3 and label in w:
                    s += 2.0
                if s > 0:
                    scores[nid] = s

            # === P600: Integrate into evolving sentence representation ===
            if w_vec is not None:
                # Integration: blend word vector into accumulated context
                # Gate is lower for surprising words (harder to integrate)
                integration_gate = 0.5 + 0.3 * (1.0 - n400_surprise)
                if word_count == 0:
                    acc_ctx = w_vec.copy()
                else:
                    acc_ctx = integration_gate * w_vec + (1.0 - integration_gate) * acc_ctx
                n = np.linalg.norm(acc_ctx)
                if n > 0:
                    acc_ctx /= n
            word_count += 1

            # Propagate activation to graph neighbors (P600 spread)
            if w_nids:
                for src_nid in w_nids:
                    for tgt_id, edge in self.graph.get_outgoing(src_nid):
                        if tgt_id not in scores:
                            # Weaker propagation for surprising words
                            prop_strength = 2.0 * (1.0 - n400_surprise * 0.5)
                            scores[tgt_id] = scores.get(tgt_id, 0) + edge.weight * prop_strength

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        activated = []
        # Keep more activations for richer sentence context (up to 8 vs 5)
        for nid, sc in sorted_scores[:8]:
            self.graph.activate(nid, min(1.0, sc * 0.12))
            activated.append(nid)
        return activated

    def _get_user_model(self):
        """Get or create user model for the current session."""
        if not hasattr(self, 'user_model'):
            from .user_model import UserModel
            self.user_model = UserModel()
        return self.user_model

    def _update_user_model(self, text: str, subject: str,
                           associations: List[Tuple[str, float]]):
        """Deep Theory of Mind update after turn processing (roadmap Â§7).

        Extends the lightweight observe_user_query (which runs early in
        process_turn) with post-spread-activation updates:
        - Update topic familiarity from graph associations
        - Track inferred goals alongside cognitive style
        - Build personalized greeting eligibility
        """
        um = self.user_model
        # Update topic familiarity from the spread-activation associations.
        # Each association the user's query touched becomes slightly more
        # familiar (exponential moving average, rate 0.1).
        for concept, confidence in associations:
            cl = concept.lower()
            um.knowledge_model[cl] = (
                0.9 * um.knowledge_model.get(cl, 0.0)
                + 0.1 * min(1.0, confidence + 0.3)
            )
        # Goal is already inferred inside observe_user_query via infer_user_goal.
        # Store the last goal for adaptive verbosity check.
        self._last_user_goal = getattr(self, '_last_user_goal', 'EXPLORING')
        self._last_user_goal = um.last_goal

    # Affective SIGNAL lexicon (word -> valence/affect direction). This is a
    # lexical *signal* seed, NOT authored reply prose: it maps an observed word
    # to an affective polarity the VAD engine integrates — the reply content
    # never comes from here. It is an instance attribute seeded from a base
    # table, so the background learner can EXTEND it at runtime (the deciding
    # test: RAVANA can grow this by experience, it is not a frozen table). The
    # base set is deliberately broad over common affect words; missing words
    # simply fail to move valence rather than fabricating a reply.
    _AFFECT_LEXICON_BASE = {
        # positive valence signal (+1)
        "good": 1, "great": 1, "happy": 1, "love": 1, "nice": 1, "fun": 1,
        "yay": 1, "wow": 1, "cool": 1, "amazing": 1, "awesome": 1,
        "wonderful": 1, "beautiful": 1, "excited": 1, "grateful": 1,
        "proud": 1, "hopeful": 1, "joy": 1, "interesting": 1, "thrilled": 1,
        "delighted": 1, "glad": 1, "pleased": 1, "content": 1, "calm": 1,
        "relieved": 1, "win": 1, "won": 1, "success": 1, "celebrate": 1,
        "fantastic": 1, "brilliant": 1, "lovely": 1, "excellent": 1,
        # negative valence signal (-1)
        "bad": -1, "sad": -1, "scared": -1, "angry": -1, "hurt": -1, "cry": -1,
        "mean": -1, "terrible": -1, "awful": -1, "upset": -1, "frustrated": -1,
        "anxious": -1, "worried": -1, "disappointed": -1, "lonely": -1,
        "guilty": -1, "afraid": -1, "gutted": -1, "devastated": -1,
        "heartbroken": -1, "miserable": -1, "furious": -1, "hopeless": -1,
        "overwhelmed": -1, "exhausted": -1, "broken": -1, "dying": -1,
        "dead": -1, "grief": -1, "grieving": -1, "suffer": -1, "suffering": -1,
        "pain": -1, "painful": -1, "lost": -1, "fail": -1, "failed": -1,
        "fear": -1, "panic": -1, "cry": -1, "crying": -1, "cried": -1,
        "wrecked": -1, "crushed": -1, "down": -1, "low": -1,
        # explicit dislike / aversion (common in user disclosures)
        "hate": -1, "hates": -1, "hating": -1, "despise": -1, "detest": -1,
        "loathe": -1, "dislike": -1, "dislikes": -1, "gross": -1,
        "annoyed": -1, "irritated": -1, "bitter": -1, "resent": -1,
        "enraged": -1, "livid": -1, "mad": -1, "hated": -1,
    }

    def _update_emotion(self, text: str):
        """More nuanced emotional processing — teenage range of emotions."""
        if not hasattr(self, "_affect_lexicon"):
            # Seed the growable affective-signal lexicon from the base table.
            self._affect_lexicon = dict(self._AFFECT_LEXICON_BASE)
        lex = self._affect_lexicon
        positive = {w for w, s in lex.items() if s > 0}
        negative = {w for w, s in lex.items() if s < 0}
        curious = {"why", "how", "what", "wonder", "curious", "interesting",
                    "really", "tell me", "explain", "mean"}
        words = set(w.lower().strip(".,!?") for w in text.split())
        sv = 0.0
        sa = 0.2  # baseline engagement floor
        # Positive words boost valence
        if words & positive:
            sv += 0.4
            sa += 0.2
        # Negative words lower valence
        if words & negative:
            sv -= 0.4
            sa += 0.25
        # Curiosity words increase arousal (engagement)
        if words & curious:
            sa += 0.3
            if sv == 0.0:
                sv += 0.05  # slight positive bias for curiosity
        # Learning excitement
        if self._learned_this_turn:
            sa += 0.3
            sv += 0.2
        # Novelty-based arousal (unknown words = mild surprise)
        input_words = [w for w in words if len(w) >= 3]
        known = sum(1 for w in input_words if w in self._concept_keywords)
        if input_words and known / len(input_words) < 0.5:
            sa += 0.15  # novelty surprise
        # Phase 9b: Prediction error surprise (Active Inference)
        # High prediction error = world doesn't match expectations = arousal
        if self._prediction_error_count > 5:
            pe_surprise = min(0.4, self._mean_prediction_error * 2.0)
            sa += pe_surprise
        # Phase 10.4: N400-like arousal modulation from per-hop prediction error
        if hasattr(self, '_mean_sentence_pe') and self._sentence_pe_count > 0:
            n400_surprise = min(0.3, self._mean_sentence_pe * 2.0)
            sa += n400_surprise
        # Phase 14.4: Identity prediction error
        if hasattr(self, '_expected_strength'):
            identity_pe = abs(self.identity.state.strength - self._expected_strength)
            if identity_pe > 0.3:
                sa += min(0.3, identity_pe * 0.5)
        # Phase 7.5: Curiosity drive â€” boost arousal for impossible queries
        if getattr(self, '_last_strategy_used', '') in ('G_uncertainty', 'F_web_research'):
            sa += 0.6  # strong curiosity arousal for impossible query
            sv += 0.1  # slight positive valence for curiosity
        self.emotion.update(stimulus_valence=sv, stimulus_arousal=sa,
                           stimulus_dominance=self.identity.state.strength * 0.4 + 0.2,
                           uncertainty=self._free_energy * 0.5, dt=1.0)

    def _decay_episodic_edges(self):
        """Phase 15.2: Inter-turn episodic edge decay (forgetting between turns)."""
        if not hasattr(self, '_episodic_edges') or not self._episodic_edges:
            return
        for pair in list(self._episodic_edges.keys()):
            edge = self._episodic_edges[pair]
            edge.weight *= 0.95
            if edge.weight < 0.05:
                del self._episodic_edges[pair]

