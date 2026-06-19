"""
Bootstrap Consolidation - Unified concept seeding logic.
Consolidates: _auto_expand_concepts, _seed_from_graph_curiosity, _bootstrap_domain_concepts
"""
from typing import Dict, List, Set, Tuple, Optional
import numpy as np

from ..graph import GraphEngine
from ..web import WebLearner


class BootstrapManager:
    """Unified concept bootstrapping and expansion manager."""

    def __init__(self, graph_engine: GraphEngine, web_learner: Optional[WebLearner] = None):
        self.graph_engine = graph_engine
        self.web_learner = web_learner

    def bootstrap_all(self):
        """Run all bootstrapping steps in order."""
        # 1. Seed core teen concepts
        self.graph_engine.seed_concepts()

        # 2. Bootstrap domain concepts
        self.graph_engine.bootstrap_domain_concepts()

        # 3. Prime background learning queue from graph curiosity
        if self.web_learner and self.web_learner._curiosity_drive_enabled:
            self.web_learner._seed_from_graph_curiosity(max_topics=8)

    def auto_expand_from_input(self, text: str) -> int:
        """Phase 1: Auto-expand from every user message."""
        return self.graph_engine.auto_expand_concepts(text)

    def curiosity_bootstrap(self, max_topics: int = 5) -> int:
        """Bootstrap learning queue from graph's curiosity signals."""
        if self.web_learner:
            return self.web_learner._seed_from_graph_curiosity(max_topics)
        # Fallback to graph's curiosity scores
        curiosity_scores = self.graph_engine.get_curiosity_scores(max_topics=max_topics * 2)
        candidates = [label for label, score in curiosity_scores if score > 0.1]
        return len(candidates)

    def bootstrap_domain(self):
        """Bootstrap domain-specific concepts."""
        self.graph_engine.bootstrap_domain_concepts()