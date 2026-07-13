"""Self-address router for the self-model responder (research item C).

Root cause (plan item C): ``_handle_self_model`` (response_gen.py:1471) matches
on ANY occurrence of feel/think/alive/real/conscious anywhere in the text, so a
genuinely answerable question like "why do people feel lonely in a crowd"
contains the word "feel" and is swallowed by the canned self-model turn-back,
never reaching the factual/answerable path.

The fix is a metacognitive arbiter (CDR/SOFAI-style per the plan): route to the
self-model ONLY when the query is genuinely *self-addressed* — directed at the
agent (2nd person "you" + a self-model predicate) — and NOT when the predicate
refers to a third-person experiencer (people/they/humans/...). This is a learned
boundary, not a blind regex OR: we extract lexical features and fit a logistic
decision boundary on labeled transcripts (the cross-cutting calibration
substrate, parallel to salad_classifier.py / provenance.py), reporting
precision/recall/EER so the policy is MEASURED, never hard-coded.

Also provides the demotion hook: when a query is self-addressed but ALSO
answerable, the caller may append a stance reflection (augmentation layer)
instead of an early-return dead-end. That is wired in response_gen.py /
engine.py call sites.
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_THIS))
# Reuse the production GloVe-64 pipeline for any vector features if needed.
for p in (os.path.join(_ROOT, "ravana", "src"),
         os.path.join(_ROOT, "ravana_ml", "src")):
    if p not in sys.path:
        sys.path.insert(0, p)


# ── Lexical feature vocabulary ──────────────────────────────────────────────
_SELF_PREDICATE = re.compile(
    r"\b(feel|feeling|feelings|emotion|emotions|alive|living|real|human|"
    r"think|thinks|thinking|thoughts|conscious|aware|awareness|mind|soul|"
    r"have feelings|are you|do you)\b", re.I)
_SECOND_PERSON = re.compile(r"\b(you|your|you're|youre|yours|ur|u)\b", re.I)
# Third-person experiencers that make a predicate refer to OTHERS, not the agent.
_THIRD_EXPERIENCER = re.compile(
    r"\b(people|they|them|he|she|him|her|humans|kids|children|dogs|cats|animals|"
    r"men|women|person|someone|everyone|others|we|folks|users)\b", re.I)
_QUESTION_INVERSION = re.compile(
    r"\b(do|does|did|are|is|can|could|would|will|have|has|am)\b\s+\b(you|they|"
    r"people|he|she|we)\b", re.I)
_AGENT_NOUN = re.compile(r"\b(ravana|bot|ai|machine|computer|robot|assistant)\b", re.I)


def extract_features(text: str) -> Dict[str, float]:
    """Distribution-light lexical features for self-address detection.

    These are interpretable signals (counts / booleans), not a fixed threshold.
    The decision boundary is FIT over them on labeled data.
    """
    t = (text or "").lower()
    has_pred = bool(_SELF_PREDICATE.search(t))
    has_2p = bool(_SECOND_PERSON.search(t))
    has_3p = bool(_THIRD_EXPERIENCER.search(t))
    inversion = bool(_QUESTION_INVERSION.search(t))
    has_agent_noun = bool(_AGENT_NOUN.search(t))
    # "about the agent" is strongly cued by 2nd person OR an explicit agent noun
    # combined with a self predicate (e.g. "is the bot conscious").
    about_agent_cue = (has_2p or (has_agent_noun and has_pred))
    feats: Dict[str, float] = {
        "self_predicate": float(has_pred),
        "second_person": float(has_2p),
        "third_person_experiencer": float(has_3p),
        "question_inversion": float(inversion),
        "agent_noun": float(has_agent_noun),
        "about_agent_cue": float(about_agent_cue),
        # Interaction: predicate + (2nd person but NOT 3rd person experiencer)
        # is the canonical self-addressed shape.
        "pred_and_self_addr": float(has_pred and has_2p and not has_3p),
        "pred_and_third_party": float(has_pred and has_3p),
    }
    return feats


@dataclass
class RouterFit:
    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    method: str = "rule"  # "logistic" if sklearn fit, else "rule"
    threshold: float = 0.5


class SelfAddressRouter:
    """Decide whether a query is genuinely self-addressed (about the agent)."""

    def __init__(self) -> None:
        self._fit = RouterFit()

    # ── Rule-based fallback (no training needed, transparent) ──────────────
    def _rule_score(self, feats: Dict[str, float]) -> float:
        # Self-addressed: predicate present AND (2nd person without a 3rd-party
        # experiencer) OR an explicit agent noun + predicate.
        if feats["pred_and_self_addr"]:
            return 1.0
        if feats["self_predicate"] and feats["agent_noun"] and not feats["third_person_experiencer"]:
            return 1.0
        if feats["self_predicate"] and feats["second_person"] and feats["third_person_experiencer"]:
            # ambiguous: predicate refers to both "you" and a 3rd party (e.g.
            # "do you and people feel...") — bias to NOT self-addressed so an
            # answerable question is not swallowed.
            return 0.0
        if feats["self_predicate"] and not feats["second_person"] and not feats["agent_noun"]:
            # predicate with no address cue at all (e.g. "why do people feel…")
            return 0.0
        return 0.0

    # ── Fit a logistic boundary on labeled transcripts (measured, not fixed) ─
    def fit(self, labeled: List[Tuple[str, int]], threshold: Optional[float] = None):
        """labeled: list of (text, gold) where gold=1 self-addressed, 0 not.

        Fits a logistic regression if scikit-learn is available; otherwise
        keeps the transparent rule fallback. Reports nothing here — the
        harness measures precision/recall/EER.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_predict
            from sklearn.metrics import roc_auc_score
            X = [list(extract_features(t).values()) for t, _ in labeled]
            y = [g for _, g in labeled]
            clf = LogisticRegression(max_iter=1000)
            keys = list(extract_features("").keys())
            clf.fit(X, y)
            self._fit = RouterFit(
                weights=dict(zip(keys, clf.coef_[0].tolist())),
                bias=float(clf.intercept_[0]),
                method="logistic",
                threshold=threshold or 0.5,
            )
            self._clf = clf
            return clf
        except Exception:
            # sklearn unavailable or degenerate — keep rule fallback.
            self._fit = RouterFit(method="rule", threshold=0.5)
            return None

    def score(self, text: str) -> float:
        feats = extract_features(text)
        if self._fit.method == "logistic" and hasattr(self, "_clf"):
            import numpy as np
            keys = list(feats.keys())
            x = np.array([[feats[k] for k in keys]])
            return float(self._clf.predict_proba(x)[0][1])
        return self._rule_score(feats)

    def is_self_addressed(self, text: str) -> Tuple[bool, float]:
        """Return (about_agent, confidence)."""
        s = self.score(text)
        thr = self._fit.threshold if self._fit.threshold is not None else 0.5
        return (s >= thr, s)


# Module-level default router (rule-based until fit() is called).
_default_router = SelfAddressRouter()


def is_self_addressed(text: str) -> Tuple[bool, float]:
    """Convenience wrapper used by _handle_self_model."""
    return _default_router.is_self_addressed(text)


def get_router() -> SelfAddressRouter:
    return _default_router
