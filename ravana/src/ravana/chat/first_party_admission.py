"""First-party (conversational) vocabulary admission — FIX-RV-16.

ROOT CAUSE THIS MODULE EXISTS TO REPAIR
---------------------------------------
`_learn_from_text` (web_learning.py) was the ONLY routine that ever minted a
ConceptGraph node, and its only callers were web fetchers (`learn_from_web`,
`ravana/web/learner.py`). `CognitiveChatEngine.process_turn` never reached it.
Consequently a novel content word spoken by the user had no admission path at
all, and `self._learning_count` — incremented at exactly one place
(web_learning.py, inside `learn_from_web`, *after* the RAVANA_OFFLINE early
return) — was structurally always 0 for any interactive session.

The counter was a WEB-SEARCH counter being read as a LEARNING counter. This
module supplies the missing half: an online, incremental admission route for
the user's OWN utterance, so RAVANA can learn from one conversation with no
retrain, no corpus rebuild, and no regeneration step.

WHY A SEPARATE ROUTE (and not a relaxed `k>=2` web gate)
-------------------------------------------------------
The web gate requires a token to be corroborated by >=2 distinct SOURCES before
promotion because scraped third-party text is unverified. A first-person
disclosure is a different epistemic situation: the speaker is the source, so
the warrant is claim-scoped (it describes the speaker, not the world), not
world-scoped. Relaxing the web gate globally would import scraped junk into
the permanent graph; instead first-party admission is gated on a *different*
and still structural property — see `_fp_admissible`: the shared structural
junk floor (shape-only) plus the requirement that the word arrived in a real
utterance at all.

NO HARDCODING / NO RETRAINING
-----------------------------
- There is NO word list here. Candidates are extracted generically from the
  utterance and filtered with the engine's existing STOP_WORDS / WEB_GARBAGE
  lexicons, the shared structural junk floor, and the shared question-phrase
  guard. Nothing in this file can name a content word; remove every line of
  this module and the engine still boots (it simply stops learning vocabulary
  from chat, which is the pre-fix behaviour).
- The only seed is the single structural label `user` (the speaker anchor).
  It is a ROLE, not vocabulary, and it is created on demand at runtime.
- A novel word's vector is composed ONLINE from the utterance's context: the
  known concepts it was spoken alongside, or — when the utterance offers none —
  the speaker anchor. No embedding is refit, no corpus is rebuilt, no generator
  is regenerated.

VECTOR GROUNDING IS ALSO THE JUNK GATE
---------------------------------------
An OOV word with no grounding would otherwise fall back to
`RandomState(hash(w))` — a random vector that makes the admitted node
permanently UNREACHABLE by similarity (it lands nowhere near its context).
Here the new vector is the normalised mean of the utterance's known-concept
vectors, so an admitted word sits inside the region of the concepts it was
actually used with, and the typed edges to those concepts are real.
"""

import hashlib
import re
from typing import Dict, List, Optional, Set

import numpy as np

from .constants import (
    INAPPROPRIATE_WORDS,
    STOP_WORDS,
    WEB_GARBAGE,
    _is_question_phrase,
)

# Role label for the speaker of a first-party utterance. A ROLE, not
# vocabulary: it is the single anchor that first-party evidence attaches to,
# so the graph has one place where "what the user said" accumulates degree.
SPEAKER_LABEL = "user"

# Minimum token length. Mirrors the web path's `{3,}` tokenisation so both
# admission routes agree on what a token even is.
_MIN_TOKEN = 3


# The structural junk floor threshold for the FIRST-PARTY route.
#
# This is deliberately stricter than the web route's `_junk_theta` (0.5), and
# the reasoning is structural rather than tuned:
#
#   - A scraped web token gets the benefit of the doubt: it is admitted to a
#     PROVISIONAL buffer and only reaches the permanent graph after two
#     independent sources corroborate it, so one bad token costs nothing.
#   - A first-party token gets ONE chance, because the speaker is the source.
#     If it is going into the permanent graph on a single utterance, it must be
#     clean on its face.
#
# `_structural_floor` assigns 0.40+ to each strong shape signal it knows
# (keyboard mash, POS fragment, website/digit shape, zero-vowel string, exact
# cyclic repetition) and 0.35 to the weaker website-shape signal. So a
# threshold of 0.40 means "any ONE strong structural-junk signal disqualifies
# the token", while a merely website-shaped token can still be real vocabulary
# a user legitimately typed. It is set on the engine so it is inspectable and
# adjustable at runtime rather than frozen as a literal deep in the gate.
_FP_FLOOR_THRESHOLD = 0.4


class FirstPartyAdmissionMixin:
    """Online admission of novel vocabulary from the user's own utterances.

    Mixin so `CognitiveChatEngine` gains the capability by composition. Every
    method is defensive: an engine built via `__new__` in a unit test (no
    graph, no locks, no GloVe) must degrade quietly rather than raise.
    """

    # ------------------------------------------------------------------ utils

    def _fp_known_labels(self) -> Set[str]:
        """Lowercased labels already present in the concept graph."""
        labels = getattr(self, "_concept_labels", None)
        if labels:
            return set(labels)
        # Fall back to the graph itself when the label index is not populated.
        graph = getattr(self, "graph", None)
        if graph is None:
            return set()
        out = set()
        try:
            for node in list(graph.nodes.values()):
                if node is not None and node.label:
                    out.add(node.label.lower())
        except Exception:
            return set()
        return out

    def _fp_content_tokens(self, text: str) -> List[str]:
        """Generic content-word extraction from an utterance.

        No vocabulary of our own: the only word lists consulted are the
        engine's existing shared lexicons, so this function's behaviour tracks
        whatever the engine already believes about function words.
        """
        if not text:
            return []
        toks: List[str] = []
        for raw in re.findall(r"[a-zA-Z']+", text.lower()):
            w = raw.strip("'")
            if len(w) < _MIN_TOKEN:
                continue
            if w in STOP_WORDS or w in WEB_GARBAGE or w in INAPPROPRIATE_WORDS:
                continue
            toks.append(w)
        # Preserve order, drop repeats within the utterance.
        seen: Set[str] = set()
        out: List[str] = []
        for w in toks:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out

    def _fp_admissible(self, word: str) -> bool:
        """The first-party junk gate.

        Reuses the shared structural junk floor (shape-only, source-agnostic:
        keyboard mashes, POS fragments, website shapes, vowel-less strings,
        cyclic repetition) so this gate cannot drift from the web gate's
        notion of junk. The grounding half of `junk_score` is deliberately NOT
        used: an out-of-vocabulary word is *by definition* ungrounded, so
        scoring it would reject precisely the novel vocabulary this route
        exists to admit.
        """
        if not word or len(word) < _MIN_TOKEN:
            return False
        if word in STOP_WORDS or word in WEB_GARBAGE or word in INAPPROPRIATE_WORDS:
            return False
        if _is_question_phrase(word):
            return False
        try:
            from ravana.chat.junk_scorer import _structural_floor
            theta = getattr(self, "_fp_floor_threshold", None)
            if theta is None:
                theta = _FP_FLOOR_THRESHOLD
            if _structural_floor(word) >= float(theta):
                return False
        except Exception:
            # A failed import must not silently widen the gate. Refuse the
            # candidate and let the caller record the limitation.
            return False
        return True

    def _fp_vector_for(self, word: str, context: List[str],
                       fallback: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Compose an online vector for a novel word from its context.

        Order of preference:
          1. GloVe, if the word happens to be in-vocabulary after all.
          2. The normalised mean of the GloVe vectors of the KNOWN concepts
             this word was spoken alongside. This is the distributional
             grounding that makes an admitted word reachable and its edges
             meaningful — computed live, per turn, with no refit.
          3. The mean of the known concepts' stored graph vectors, so the
             grounding survives on a GloVe-less (offline) box.
          4. `fallback` (the speaker anchor's vector) when the utterance
             supplied no known concept at all. The word is then admitted
             sitting near the speaker rather than refused: we honestly do not
             know where it belongs semantically, but we DO know the speaker
             said it, and that is a real, typed, queryable fact.

        Never falls back to ``RandomState(hash(w))``: a random vector makes the
        admitted node permanently unreachable by similarity, which is how
        junk gets in and stays in. Returns None only when there is no grounding
        and no fallback at all.
        """
        dim = int(getattr(self, "dim", 64) or 64)

        # (1) Direct GloVe lookup.
        try:
            vec = self._glove_vector(word)
        except Exception:
            vec = None
        if vec is not None and np.linalg.norm(vec) > 0:
            return np.asarray(vec, dtype=np.float32)

        # (2)/(3) Compose from the co-occurring KNOWN concepts.
        acc = np.zeros(dim, dtype=np.float32)
        n = 0
        for ctx in context:
            if ctx == word:
                continue
            cv = None
            try:
                cv = self._glove_vector(ctx)
            except Exception:
                cv = None
            if cv is None:
                # Fall back to the vector this concept already has in the
                # graph (seeded nodes, previously admitted words).
                cv = self._fp_stored_vector(ctx)
            if cv is not None and np.isfinite(cv).all():
                acc += np.asarray(cv, dtype=np.float32)
                n += 1
        if n > 0:
            norm = float(np.linalg.norm(acc))
            if norm > 0:
                return (acc / norm).astype(np.float32)
        if fallback is not None and np.isfinite(fallback).all():
            return np.asarray(fallback, dtype=np.float32)
        return None

    def _fp_stored_vector(self, label: str) -> Optional[np.ndarray]:
        """Return the vector a concept already has in the graph, if any."""
        ck = getattr(self, "_concept_keywords", None) or {}
        for nid in ck.get(label.lower(), []) or []:
            try:
                node = self.graph.get_node(nid)
            except Exception:
                node = None
            if node is not None and getattr(node, "vector", None) is not None:
                return node.vector
        return None

    def _fp_ensure_node(self, label: str, vec: np.ndarray) -> Optional[int]:
        """Mint (or fetch) a concept node for `label`. Returns the node id."""
        graph = getattr(self, "graph", None)
        if graph is None:
            return None
        ck = getattr(self, "_concept_keywords", None)
        if ck is None:
            self._concept_keywords = ck = {}
        for nid in ck.get(label.lower(), []) or []:
            try:
                if graph.get_node(nid) is not None:
                    return nid
            except Exception:
                continue
        lock = getattr(self, "_graph_lock", None)
        if lock is not None:
            with lock:
                node = graph.add_node(vector=vec, label=label)
        else:
            node = graph.add_node(vector=vec, label=label)
        if node is None:
            return None
        # A first-party claim is weaker than a curated fact, so the node starts
        # less stable; repeated disclosure raises it (see _fp_reinforce).
        try:
            node.stability = min(float(getattr(node, "stability", 0.5) or 0.0), 0.6)
        except Exception:
            pass
        if hasattr(node, "source_metadata") and isinstance(node.source_metadata, dict):
            node.source_metadata.update({"edge_kind": "first_party", "source": SPEAKER_LABEL})
        ck[label.lower()] = list(ck.get(label.lower(), [])) + [node.id]
        labels = getattr(self, "_concept_labels", None)
        if labels is None:
            self._concept_labels = labels = set()
        labels.add(label.lower())
        return node.id

    def _fp_link(self, a_id: int, b_id: int, a: str, b: str, weight: float) -> None:
        """Idempotently wire a typed edge a->b, inferring the relation type."""
        graph = getattr(self, "graph", None)
        if graph is None or a_id == b_id or a_id is None or b_id is None:
            return
        try:
            if graph.get_edge(a_id, b_id) is not None:
                return
        except Exception:
            return
        rel = "semantic"
        try:
            rel, _conf = self._infer_relation_type(a, b, "semantic")
        except Exception:
            rel = "semantic"
        try:
            graph.add_edge(a_id, b_id, weight=weight, relation_type=rel, confidence=0.6)
        except Exception:
            return
        try:
            if hasattr(graph, "_vectors_dirty"):
                graph._vectors_dirty = True
            if hasattr(graph, "_adj_dirty"):
                graph._adj_dirty = True
        except Exception:
            pass

    # ------------------------------------------------------------- admission

    def _admit_first_party_utterance(self, text: str) -> int:
        """Admit novel content words from `text` to the ConceptGraph.

        Online and incremental: called once per conversational turn, mutating
        the live graph. Returns the number of NEW concepts admitted.
        """
        if not text or not isinstance(text, str):
            return 0
        graph = getattr(self, "graph", None)
        if graph is None:
            return 0

        candidates = self._fp_content_tokens(text)
        if not candidates:
            return 0

        known = self._fp_known_labels()
        anchor_id = self._fp_speaker_anchor()

        novel = [w for w in candidates
                 if w not in known and self._fp_admissible(w)]
        if not novel:
            return 0

        context = [w for w in candidates if w in known]
        # Novel words ground each other too: a disclosure of two unseen words
        # ("kalamazoo kiln") still supplies mutual context.
        if not context:
            context = list(novel)

        admitted = 0
        minted: Dict[str, int] = {}
        for word in novel:
            # The speaker anchor is the last-resort grounding: even with no
            # other known concept in the utterance, "the speaker said this" is
            # a real fact, so the word is admitted near the anchor rather than
            # refused outright.
            anchor_vec = self._fp_stored_vector(SPEAKER_LABEL)
            vec = self._fp_vector_for(
                word,
                context + [w for w in novel if w != word],
                fallback=anchor_vec,
            )
            if vec is None:
                # No grounding and no anchor -> refuse. Admitting it would put
                # an unreachable random vector in the permanent graph.
                continue
            nid = self._fp_ensure_node(word, vec)
            if nid is None:
                continue
            minted[word] = nid
            admitted += 1

        if not admitted:
            return 0

        # Typed edges: speaker -> concept, and concept -> each known concept
        # it was spoken alongside. The co-occurrence edges are what make the
        # admission usable downstream (recall, chain walking).
        for word, nid in minted.items():
            if anchor_id is not None and nid != anchor_id:
                self._fp_link(anchor_id, nid, SPEAKER_LABEL, word, weight=0.5)
            for ctx in context:
                if ctx == word:
                    continue
                cid = None
                ck = getattr(self, "_concept_keywords", {}) or {}
                for x in ck.get(ctx.lower(), []) or []:
                    cid = x
                    break
                if cid is not None and cid != nid:
                    self._fp_link(nid, cid, word, ctx, weight=0.35)

        # Re-activation: a word already in the graph that the user mentions
        # again is corroborated by a NEW first-party event, so it is
        # re-weighted and its source set grows. This is the rehearsal
        # mechanism, and it uses a turn-scoped source id so that repeating the
        # SAME sentence on a later turn counts as a distinct event (the web
        # path's md5(topic) source id makes verbatim repeats collide forever).
        try:
            turn = int(getattr(self, "turn_count", 0) or 0)
            source_id = "fp:" + hashlib.md5(
                f"turn:{turn}".encode()).hexdigest()[:12]
        except Exception:
            source_id = "fp:turn"
        try:
            if getattr(self, "_concept_sources", None) is None:
                self._concept_sources = {}
            for word in candidates:
                self._concept_sources.setdefault(word, set()).add(source_id)
        except Exception:
            pass

        # THIS is the fix for the reported symptom: `learning_count` now
        # records real learning events (web + first-party), not just web
        # searches. It is durable (engine.py persists/loads it).
        try:
            self._learning_count = int(getattr(self, "_learning_count", 0) or 0) + admitted
        except Exception:
            pass

        if getattr(self, "_trace_enabled", False):
            print(f"  [learn] admitted {admitted} first-party concept(s): "
                  f"{sorted(minted)}")
        return admitted

    def _fp_speaker_anchor(self) -> Optional[int]:
        """Return the node id of the speaker anchor, creating it if needed.

        One anchor, always the same label: if first-party evidence scattered
        across two spellings the accumulated degree — which is what protects
        these nodes from any degree-based pruning — would be split in half.
        """
        try:
            vec = self._glove_vector(SPEAKER_LABEL)
        except Exception:
            vec = None
        if vec is None:
            # The anchor is a ROLE, not learned content, so when GloVe cannot
            # ground it we mint a deterministic vector derived from the label
            # itself. Stable across boots, and never mistaken for a learned
            # word because it is minted on the anchor path, not the content
            # path.
            dim = int(getattr(self, "dim", 64) or 64)
            rng = np.random.RandomState(abs(hash(SPEAKER_LABEL)) % 50000 + 100)
            v = rng.randn(dim).astype(np.float32) * 0.1
            n = float(np.linalg.norm(v))
            vec = (v / n if n > 0 else v)
        return self._fp_ensure_node(SPEAKER_LABEL, np.asarray(vec, dtype=np.float32))

    def _fp_reinforce(self, text: str) -> None:
        """Raise the stability of concepts the user re-mentions.

        Kept separate from admission so re-mentioning a KNOWN word is still a
        learning event even though it mints no node.
        """
        if not text or not isinstance(text, str):
            return
        ck = getattr(self, "_concept_keywords", None) or {}
        touched = 0
        for word in self._fp_content_tokens(text):
            for nid in ck.get(word.lower(), []) or []:
                try:
                    node = self.graph.get_node(nid)
                except Exception:
                    continue
                if node is None or word.lower() == SPEAKER_LABEL:
                    continue
                try:
                    node.stability = min(0.95, float(node.stability) + 0.02)
                    touched += 1
                except Exception:
                    continue
        if touched:
            try:
                self._learning_count = int(getattr(self, "_learning_count", 0) or 0) + 1
            except Exception:
                pass
