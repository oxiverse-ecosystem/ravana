#!/usr/bin/env python3
"""Regression tests for the reverse-name + activity recall fix (defect E, round 2026-08-17).

A reverse-name lookup ("who is <Name> to me") must render the FULL combined-attr
relationship AND, when the query also asks what the person does, the stored
activity value (copula dropped for verb-phrase values via is_activity_verb).
Previously it returned only the kinship head ("your sister.") and dropped both
the name and the activity.
"""
import os
import sys
import pytest

os.environ.setdefault("RAVANA_OFFLINE", "1")
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for p in (PROJ, os.path.join(PROJ, "ravana_ml", "src"),
          os.path.join(PROJ, "ravana", "src"), os.path.join(PROJ, "ravana-v2", "src")):
    sys.path.insert(0, p)

from ravana.chat.engine import CognitiveChatEngine

SUFFIX = "test_revname_20260817"


def _seed():
    e = CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX)
    for s in [
        "my sister Lena paints huge murals on the sides of barns",
        "my aunt Mei grows shiitake mushrooms in a damp cellar",
    ]:
        e.process_turn(s)
    return e


def test_reversename_includes_activity_when_asked():
    e = _seed()
    out = e.process_turn("who is Lena to me and what does she do?")
    assert "sister" in out and "paints huge murals" in out, out
    assert out.strip().endswith("murals."), out  # copula dropped for verb phrase


def test_reversename_relationship_only_when_not_asked_activity():
    e = _seed()
    out = e.process_turn("who is Lena to me?")
    assert "sister" in out, out
    # when not asked what she does, the activity need not appear
    assert "paints" not in out or "sister" in out


def test_reversename_aunt_activity():
    e = _seed()
    out = e.process_turn("who is Mei to me and what does she do?")
    assert "aunt" in out and "grows shiitake mushrooms" in out, out


def teardown_module(module):
    try:
        CognitiveChatEngine(dim=64, seed=42, baby_mode=True, user_suffix=SUFFIX).stop_background_learning()
    except Exception:
        pass
