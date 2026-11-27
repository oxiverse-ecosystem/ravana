"""Default-OFF lexical-superordinate transfer for unseen query subjects.

This module is a pure, engine-free implementation of the scratch finding in
``docs/investigations/LEXICAL_SUPERORDINATE_TRANSFER.md``. It does NOT boot an
engine, write a store, import a benchmark, read an answer key, or change any
RSI/TRA gate, threshold or item. Nothing happens unless
``RAVANA_LEXICAL_SUPERORDINATE=1`` is set explicitly.

Mechanism (brain-grounded; see the doc for the graded citation ledger):

* a structured lexical hypothesis space (WordNet 3.0 noun hypernyms) supplies
  candidate SUBJECT-SPECIFIC type evidence -- Miller 1995, Xu & Tenenbaum 2007;
* the criterion for admitting a candidate is calibrated from the item's own
  publicly taught demonstration pairs, never from a gold key -- Tse et al.
  2007, McClelland et al. 1995;
* a schema may only propose; the subject's own lexical evidence admits --
  Johnson et al. 1993, and the RAVANA "schemas propose; sources admit" rule.

Deployable inputs are: the query subject, the displayed options, and the
public teaching pairs. ``gold``/``type_sets``/``world_model``/domain tags are
never parameters of any scorer here.

WordNet 3.0 licence notice (Princeton University, 1995-2006):
Copyright (c) 1995-2006 Princeton University. All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that this notice, this list of
conditions and the following disclaimer are retained. THIS SOFTWARE IS
PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS". See the full
LICENSE shipped with the WordNet 3.0 corpus. Citation: Miller (1995),
CACM 38(11):39-41, doi:10.1145/219717.219748; Fellbaum (ed.) (1998),
WordNet: An Electronic Lexical Database, MIT Press.
"""
from __future__ import annotations

import importlib
import os
import statistics
from typing import Any, Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "ENV_FLAG", "DELTA", "MARGIN", "enabled", "wordnet_available",
    "path_similarity", "schema_criterion", "subject_scores", "demo_scores",
    "decide", "pick",
]

ENV_FLAG = "RAVANA_LEXICAL_SUPERORDINATE"
WORDNET_PATH_FLAG = "RAVANA_WORDNET_DATA"
DELTA = 0.10
MARGIN = 0.05


def enabled() -> bool:
    """Inert by default. The only way to turn this path on is the flag."""
    return os.environ.get(ENV_FLAG, "") == "1"


def _wordnet_module():
    """Lazy import: no ontology is opened unless this path is explicitly used."""
    data_dir = os.environ.get(WORDNET_PATH_FLAG, "")
    if data_dir:
        try:
            import nltk
            if data_dir not in nltk.data.path:
                nltk.data.path.insert(0, data_dir)
        except Exception:
            return None
    try:
        nltk = importlib.import_module("nltk")
        return importlib.import_module("nltk.corpus").wordnet
    except Exception:
        return None


def wordnet_available() -> bool:
    return _wordnet_module() is not None


def path_similarity(word_a: str, word_b: str) -> Optional[float]:
    """Max WordNet noun path similarity across sense pairs; None is fail-closed."""
    wn = _wordnet_module()
    if wn is None or not word_a or not word_b:
        return None
    try:
        vals = [wn.path_similarity(a, b)
                for a in wn.synsets(word_a, pos=wn.NOUN)
                for b in wn.synsets(word_b, pos=wn.NOUN)]
    except Exception:
        return None
    vals = [v for v in vals if v is not None]
    return max(vals) if vals else None


def schema_criterion(teaching_pairs: Iterable[Tuple[str, str]],
                     fallback_theta: Optional[float]) -> Optional[float]:
    """Calibrate the admission bar from publicly taught pairs only."""
    sims = [path_similarity(a, b) for a, b in teaching_pairs]
    sims = [s for s in sims if s is not None]
    if sims:
        return statistics.median(sims) + DELTA
    return fallback_theta


def subject_scores(subject: str, options: Sequence[str]) -> dict:
    """Subject-specific evidence: query subject -> each displayed candidate."""
    return {c: path_similarity(subject, c) for c in options}


def demo_scores(teaching_pairs: Sequence[Tuple[str, str]],
                options: Sequence[str]) -> dict:
    """Subject-INDEPENDENT comparator: taught r2 words -> displayed options."""
    words = sorted({r2 for _r1, r2 in teaching_pairs})
    out = {}
    for c in options:
        vals = [path_similarity(c, w) for w in words]
        vals = [v for v in vals if v is not None]
        out[c] = max(vals) if vals else None
    return out


def decide(scores: dict, theta: Optional[float]) -> Tuple[Optional[str], str]:
    """Fail-closed unique winner; no floor/margin tuning at query time."""
    ranked = sorted(((c, s) for c, s in scores.items() if s is not None),
                    key=lambda cs: (-cs[1], cs[0]))
    if not ranked or theta is None or ranked[0][1] < theta:
        return None, "below_criterion"
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    if ranked[0][1] - second < MARGIN:
        return None, "no_margin"
    return ranked[0][0], "emit"


def pick(subject: str, options: Sequence[str],
         teaching_pairs: Sequence[Tuple[str, str]],
         fallback_theta: Optional[float] = None) -> Optional[str]:
    """Return the admitted option, or None. Inert unless the flag is on."""
    if not enabled():
        return None
    theta = schema_criterion(teaching_pairs, fallback_theta)
    choice, _reason = decide(subject_scores(subject, options), theta)
    return choice
