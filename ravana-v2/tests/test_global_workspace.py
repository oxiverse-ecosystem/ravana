"""
Tests for the Global Workspace competition cycle.
"""

try:
    from .conftest import import_core
except ImportError:
    from conftest import import_core

GlobalWorkspace, GWConfig = import_core("global_workspace", "GlobalWorkspace", "GWConfig")


def test_below_threshold_bids_are_cleared():
    gw = GlobalWorkspace(GWConfig(broadcast_threshold=0.8, competition_noise=0.0))

    gw.submit_bid("emotion", {"mood": "low"}, urgency=0.4, valence=-0.2, episode=1)
    gw.submit_bid("meaning", {"meaning": 0.2}, urgency=0.6, valence=0.1, episode=1)

    result = gw.compete()

    assert result is None
    assert gw.get_status()["pending_bids"] == 0
    assert gw.get_status()["buffer_size"] == 0


def test_winning_bid_is_broadcast_and_buffered():
    gw = GlobalWorkspace(GWConfig(broadcast_threshold=0.2, competition_noise=0.0))

    gw.submit_bid("emotion", {"mood": "salient"}, urgency=0.9, valence=0.7, episode=2)
    gw.submit_bid("meaning", {"meaning": 0.5}, urgency=0.4, valence=0.2, episode=2)

    result = gw.compete()

    assert result is not None
    assert result.source == "emotion"
    assert gw.get_status()["pending_bids"] == 0
    assert gw.get_status()["buffer_size"] == 1
    assert gw.get_recent(1)[0].source == "emotion"
