"""
Smoke tests for the reality grounding module.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "interface_agent" / "scripts"))

from reality_grounding import NewsItem, RealityGrounding


def test_build_news_mdp():
    rg = RealityGrounding(rss_feeds=[], max_items_per_source=2)
    news_items = [
        NewsItem(
            title="AI safety warning issued",
            url="https://example.com/1",
            source="Example",
            published="now",
            summary="Critical risk found in deployment",
            topic="AI safety",
            relevance_score=0.9,
            published_epoch=2.0,
        ),
        NewsItem(
            title="Research paper shows progress",
            url="https://example.com/2",
            source="Example",
            published="now",
            summary="Study finds improved method",
            topic="machine learning",
            relevance_score=0.7,
            published_epoch=1.0,
        ),
    ]

    scenarios = rg.build_news_mdp(news_items, {"dissonance": 0.4, "identity": 0.6}, max_scenarios=2)

    assert len(scenarios) == 2
    assert scenarios[0]["action"] == "increase scrutiny"
    assert scenarios[1]["action"] == "update beliefs"
    assert 0.0 <= scenarios[0]["next_state"]["dissonance"] <= 1.0
    assert 0.0 <= scenarios[1]["next_state"]["identity"] <= 1.0
    assert "rationale" in scenarios[0]
    assert scenarios[0]["learning_card"]["source"] == "news-mdp"
    assert "correctness" in scenarios[0]["learning_card"]


def test_ingest_news_pipeline():
    rg = RealityGrounding(rss_feeds=[], max_items_per_source=2)
    cycle = rg.ingest_news(
        query="AI safety",
        ravana_state={"dissonance": 0.62, "identity": 0.44},
        topics=["AI safety", "AI ethics"],
        max_items=2,
        max_scenarios=2,
    )

    assert "summary" in cycle
    assert "scenarios" in cycle
    assert "alignment" in cycle
    assert "events" in cycle
    assert "event_cards" in cycle
    assert "workspace_bids" in cycle
    assert isinstance(cycle["events"], list)
    assert isinstance(cycle["event_cards"], list)
    assert isinstance(cycle["workspace_bids"], list)
    assert isinstance(cycle["max_pressure"], float)
    if cycle["workspace_bids"]:
        urgencies = [bid["urgency"] for bid in cycle["workspace_bids"]]
        assert urgencies == sorted(urgencies, reverse=True)
        assert "source" in cycle["workspace_bids"][0]
    if cycle["scenarios"]:
        scenario = cycle["scenarios"][0]
        assert scenario.pressure >= 0.0
        assert isinstance(scenario.entities, list)
        assert isinstance(scenario.source_url, str)


if __name__ == "__main__":
    test_build_news_mdp()
    test_ingest_news_pipeline()
    print("reality grounding smoke test passed")
