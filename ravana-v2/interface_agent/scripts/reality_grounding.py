"""
RAVANA v2 — Reality Grounding Module
Fetches news + RSS feeds to evaluate RAVANA actions against real-world events.
"""

import feedparser
import json
import time
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import requests

# Try importing newspaper3k for article extraction
try:
    from newspaper import Article
    NEWSPAPER_AVAILABLE = True
except ImportError:
    NEWSPAPER_AVAILABLE = False


@dataclass
class NewsItem:
    """A single news article or RSS item."""
    title: str
    url: str
    source: str
    published: str
    summary: str
    topic: str
    relevance_score: float  # 0.0-1.0 relevance to current context
    published_epoch: float  # For sorting
    entities: List[str] = field(default_factory=list)
    dissonance_pressure: float = 0.0
    scenario_action: str = ""
    scenario_priority: float = 0.0

    def to_event_record(self) -> Dict[str, Any]:
        return {
            "kind": "news",
            "title": self.title,
            "source": self.source,
            "url": self.url,
            "topic": self.topic,
            "published": self.published,
            "published_epoch": self.published_epoch,
            "summary": self.summary,
            "relevance_score": round(self.relevance_score, 3),
            "entities": list(self.entities),
            "dissonance_pressure": round(self.dissonance_pressure, 3),
            "scenario_action": self.scenario_action,
            "scenario_priority": round(self.scenario_priority, 3),
        }


@dataclass
class NewsMDPScenario:
    """A toy news-derived MDP scenario for cognitive grounding."""
    topic: str
    state: Dict[str, Any]
    action: str
    reward: float
    next_state: Dict[str, Any]
    rationale: str
    entities: List[str] = field(default_factory=list)
    pressure: float = 0.0
    confidence: float = 0.0
    source_url: str = ""
    published_epoch: float = 0.0

    def to_learning_card(self) -> Dict[str, Any]:
        action = self.action.lower()
        if action in {"increase scrutiny", "escalate attention"}:
            correctness = False
        elif action in {"update beliefs", "monitor impact"}:
            correctness = True
        else:
            correctness = self.reward >= 0.2

        return {
            "correctness": correctness,
            "difficulty": max(0.1, min(0.9, self.pressure)),
            "domain": self.topic,
            "source": "news-mdp",
            "pressure": round(self.pressure, 3),
            "confidence": round(self.confidence, 3),
            "reward": round(self.reward, 3),
            "rationale": self.rationale,
        }

    def to_event_record(self) -> Dict[str, Any]:
        return {
            "kind": "scenario",
            "topic": self.topic,
            "action": self.action,
            "reward": round(self.reward, 3),
            "pressure": round(self.pressure, 3),
            "confidence": round(self.confidence, 3),
            "source_url": self.source_url,
            "published_epoch": self.published_epoch,
            "entities": list(self.entities),
            "state": self.state,
            "next_state": self.next_state,
            "rationale": self.rationale,
            "learning_card": self.to_learning_card(),
        }


class NewsToMDPPipeline:
    """Convert news items into a structured grounding cycle."""

    def __init__(self, grounding: "RealityGrounding"):
        self.grounding = grounding

    def select_topics(self, ravana_state: Optional[dict] = None) -> List[str]:
        if not ravana_state:
            return self.grounding.DEFAULT_TOPICS[:3]

        dissonance = float(ravana_state.get("dissonance", 0.5))
        identity = float(ravana_state.get("identity", 0.5))

        if dissonance >= 0.7:
            return ["AI safety", "AI ethics", "cognitive science"]
        if identity <= 0.35:
            return ["AI ethics", "behavioral economics", "AI safety"]
        if dissonance >= 0.45:
            return ["machine learning", "AGI development", "AI ethics"]
        return self.grounding.DEFAULT_TOPICS[:3]

    def extract_entities(self, text: str) -> List[str]:
        import re

        tokens = re.findall(r"\b(?:[A-Z]{2,}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", text)
        stopwords = {
            "The", "A", "An", "And", "Or", "But", "For", "With", "From", "Into", "Over", "Under",
            "This", "That", "These", "Those", "In", "On", "At", "By", "To", "Of",
        }
        seen = set()
        entities: List[str] = []
        for token in tokens:
            token = token.strip()
            if not token or token in stopwords:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            entities.append(token)
        return entities[:8]

    def score_pressure(self, item: NewsItem, ravana_state: Optional[dict] = None) -> float:
        text = f"{item.title} {item.summary}".lower()
        pressure = item.relevance_score * 0.45

        if any(k in text for k in ["warning", "risk", "danger", "critical", "ban", "attack", "breach"]):
            pressure += 0.22
        if any(k in text for k in ["study", "research", "paper", "finds", "shows", "analysis"]):
            pressure += 0.10
        if any(k in text for k in ["launch", "release", "announces", "introduces", "rolls out"]):
            pressure += 0.08
        if any(k in text for k in ["policy", "regulation", "court", "lawsuit", "probe"]):
            pressure += 0.15

        if ravana_state:
            dissonance = float(ravana_state.get("dissonance", 0.5))
            identity = float(ravana_state.get("identity", 0.5))
            pressure += max(0.0, dissonance - 0.4) * 0.15
            if identity < 0.4:
                pressure += 0.05

        return round(min(1.0, pressure), 3)

    def _dedupe_items(self, news_items: List[NewsItem]) -> List[NewsItem]:
        seen = set()
        deduped: List[NewsItem] = []
        for item in news_items:
            key = (item.url or item.title).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def build_scenarios(
        self,
        news_items: List[NewsItem],
        ravana_state: Optional[dict] = None,
        max_scenarios: int = 5,
    ) -> List[NewsMDPScenario]:
        current_dissonance = float((ravana_state or {}).get("dissonance", 0.5))
        current_identity = float((ravana_state or {}).get("identity", 0.5))
        scenarios: List[NewsMDPScenario] = []

        for item in self._dedupe_items(news_items)[:max_scenarios]:
            text = f"{item.title} {item.summary}".lower()
            entities = self.extract_entities(f"{item.title} {item.summary}")
            pressure = self.score_pressure(item, ravana_state)

            if any(k in text for k in ["warning", "risk", "danger", "critical", "ban", "attack", "breach"]):
                action = "increase scrutiny"
                reward = 0.2 + pressure * 0.6
                delta_d = 0.08 + pressure * 0.18
                rationale = "High-risk language should push the agent toward caution."
            elif any(k in text for k in ["study", "research", "paper", "finds", "shows", "analysis"]):
                action = "update beliefs"
                reward = 0.25 + pressure * 0.45
                delta_d = -0.04 + pressure * 0.04
                rationale = "Research-style news should revise beliefs, not just confirm them."
            elif any(k in text for k in ["launch", "release", "announces", "introduces", "rolls out"]):
                action = "monitor impact"
                reward = 0.15 + pressure * 0.25
                delta_d = 0.01 + pressure * 0.07
                rationale = "Launches are best treated as monitoring events unless evidence says otherwise."
            elif any(k in text for k in ["policy", "regulation", "court", "lawsuit", "probe"]):
                action = "escalate attention"
                reward = 0.18 + pressure * 0.5
                delta_d = 0.06 + pressure * 0.15
                rationale = "Policy and legal news should raise attention and verification pressure."
            else:
                action = "hold position"
                reward = 0.05 + pressure * 0.2
                delta_d = 0.0
                rationale = "Unclear news should not trigger overconfident updates."

            next_dissonance = max(0.0, min(1.0, current_dissonance + delta_d - (current_identity * 0.03)))
            next_identity = max(0.0, min(1.0, current_identity + max(0.0, reward - 0.25) * 0.05))
            confidence = max(0.1, min(1.0, 1.0 - pressure * 0.5))

            scenarios.append(
                NewsMDPScenario(
                    topic=item.topic,
                    state={
                        "dissonance": round(current_dissonance, 3),
                        "identity": round(current_identity, 3),
                        "relevance": round(item.relevance_score, 3),
                        "pressure": pressure,
                    },
                    action=action,
                    reward=round(reward, 3),
                    next_state={
                        "dissonance": round(next_dissonance, 3),
                        "identity": round(next_identity, 3),
                    },
                    rationale=rationale,
                    entities=entities,
                    pressure=pressure,
                    confidence=round(confidence, 3),
                    source_url=item.url,
                    published_epoch=item.published_epoch,
                )
            )

            current_dissonance = next_dissonance
            current_identity = next_identity

        return scenarios

    def build_event_stream(
        self,
        news_items: List[NewsItem],
        scenarios: List[NewsMDPScenario],
    ) -> List[Dict[str, Any]]:
        events = [item.to_event_record() for item in self._dedupe_items(news_items)]
        events.extend(scenario.to_event_record() for scenario in scenarios)
        events.sort(
            key=lambda event: (
                float(event.get("published_epoch", 0.0) or 0.0),
                float(event.get("pressure", 0.0) or 0.0),
                float(event.get("relevance_score", 0.0) or 0.0),
            ),
            reverse=True,
        )
        return events

    def summarize_cycle(
        self,
        news_items: List[NewsItem],
        scenarios: List[NewsMDPScenario],
        alignment: Dict[str, Any],
    ) -> str:
        if not news_items:
            return "No recent news available."

        parts = [f"{len(news_items)} articles ingested", f"{len(scenarios)} scenarios built"]
        if scenarios:
            top = scenarios[0]
            parts.append(f"top action={top.action} pressure={top.pressure:.2f}")
        if alignment:
            parts.append(f"alignment={alignment.get('verdict', 'no_evidence')} ({alignment.get('alignment_score', 0.5):.2f})")
        return " | ".join(parts)

    def run_cycle(
        self,
        query: Optional[str] = None,
        ravana_state: Optional[dict] = None,
        topics: Optional[List[str]] = None,
        max_items: int = 5,
        max_scenarios: int = 5,
    ) -> Dict[str, Any]:
        topics = topics or self.select_topics(ravana_state)
        news_items = self.grounding.fetch_rss_feeds(topics=topics)

        if query:
            news_items = self._dedupe_items(self.grounding.search_news(query, num_results=max_items) + news_items)
        else:
            news_items = self._dedupe_items(news_items)

        limit = max(1, max_items * max(1, len(topics)))
        news_items = news_items[:limit]
        scenarios = self.build_scenarios(news_items, ravana_state=ravana_state, max_scenarios=max_scenarios)

        alignment_belief = query or topics[0]
        alignment_action = scenarios[0].action if scenarios else "hold position"
        alignment_news = news_items[: max(1, min(5, len(news_items)))]
        alignment = self.grounding.evaluate_belief_alignment(
            belief=alignment_belief,
            action=alignment_action,
            news_items=alignment_news,
        )
        summary = self.summarize_cycle(news_items, scenarios, alignment)
        events = self.build_event_stream(news_items, scenarios)
        event_cards = [scenario.to_learning_card() for scenario in scenarios]

        return {
            "query": query,
            "topics": topics,
            "news_items": news_items,
            "scenarios": scenarios,
            "events": events,
            "event_cards": event_cards,
            "alignment": alignment,
            "max_pressure": max((scenario.pressure for scenario in scenarios), default=0.0),
            "summary": summary,
        }


class RealityGrounding:
    """
    Grounds RAVANA's decisions in real-world events.
    
    - Polls RSS feeds for live news on relevant topics
    - Uses Google News search for targeted queries
    - Evaluates whether RAVANA's beliefs would lead to correct/wrong outcomes in the real world
    - Caches results to avoid redundant fetches
    """

    DEFAULT_RSS_FEEDS = [
        # AI / ML news
        ("AI Safety", "https://news.google.com/rss/search?q=AI+safety+AGI&hl=en-US&gl=US&ceid=US:en"),
        ("Machine Learning", "https://news.google.com/rss/search?q=machine+learning&hl=en-US&gl=US&ceid=US:en"),
        ("Cognitive Science", "https://news.google.com/rss/search?q=cognitive+science&hl=en-US&gl=US&ceid=US:en"),
        ("AI Ethics", "https://news.google.com/rss/search?q=AI+ethics&hl=en-US&gl=US&ceid=US:en"),
        # Broader science
        ("Neuroscience", "https://news.google.com/rss/search?q=neuroscience&hl=en-US&gl=US&ceid=US:en"),
        ("Behavioral Economics", "https://news.google.com/rss/search?q=behavioral+economics&hl=en-US&gl=US&ceid=US:en"),
    ]

    DEFAULT_TOPICS = [
        "AI safety", "AGI development", "cognitive architecture",
        "neuroscience breakthroughs", "behavioral economics",
        "AI ethics", "machine learning",
    ]

    def __init__(
        self,
        rss_feeds: List[tuple] = None,
        cache_ttl_seconds: int = 300,  # 5 min cache
        max_items_per_source: int = 10,
    ):
        self.rss_feeds = rss_feeds or self.DEFAULT_RSS_FEEDS
        self.cache_ttl = cache_ttl_seconds
        self.max_items = max_items_per_source
        self._cache: Dict[str, tuple[float, List[NewsItem]]] = {}
        self.pipeline = NewsToMDPPipeline(self)

    def fetch_rss_feeds(self, topics: List[str] = None) -> List[NewsItem]:
        """
        Fetch latest items from all RSS feeds.

        Returns list of NewsItem sorted by recency and relevance.
        """
        all_items = []
        topics = topics or self.DEFAULT_TOPICS

        for topic_name, feed_url in self.rss_feeds:
            cached = self._get_cached(feed_url)
            if cached:
                all_items.extend(cached)
                continue

            try:
                items = self._fetch_single_feed(topic_name, feed_url, topics)
                self._set_cached(feed_url, items)
                all_items.extend(items)
            except Exception as e:
                print(f"  [RSS] Failed to fetch {topic_name}: {e}")

        all_items.sort(key=lambda x: (x.published_epoch, x.relevance_score), reverse=True)
        return all_items[:self.max_items * len(self.rss_feeds)]

    def ingest_news(
        self,
        query: Optional[str] = None,
        ravana_state: Optional[dict] = None,
        topics: Optional[List[str]] = None,
        max_items: int = 5,
        max_scenarios: int = 5,
    ) -> Dict[str, Any]:
        """Run the full news-to-MDP grounding cycle."""
        return self.pipeline.run_cycle(
            query=query,
            ravana_state=ravana_state,
            topics=topics,
            max_items=max_items,
            max_scenarios=max_scenarios,
        )

    def build_news_mdp(self, news_items: List[NewsItem], ravana_state: Optional[dict] = None,
                       max_scenarios: int = 5) -> List[Dict[str, Any]]:
        """Convert news items into toy MDP scenarios for grounding.

        This is intentionally lightweight: it gives the agent a structured
        event loop (state → action → reward → next_state) without pretending
        that news is a clean reinforcement-learning environment.
        """
        scenarios = self.pipeline.build_scenarios(news_items, ravana_state=ravana_state, max_scenarios=max_scenarios)
        structured = []
        for scenario in scenarios:
            payload = scenario.__dict__.copy()
            payload["learning_card"] = scenario.to_learning_card()
            structured.append(payload)
        return structured

    def search_news(self, query: str, num_results: int = 5) -> List[NewsItem]:
        """
        Search Google News for a specific query.

        Args:
            query: Search query (e.g., "RAVANA AI architecture")
            num_results: Max number of results

        Returns:
            List of NewsItem
        """
        cache_key = f"search:{query}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached[:num_results]

        try:
            encoded_q = requests.utils.quote(query)
            feed_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en-US&gl=US&ceid=US:en"
            items = self._fetch_single_feed(query, feed_url, [query])
            self._set_cached(cache_key, items)
            return items[:num_results]
        except Exception as e:
            print(f"  [News] Search failed for '{query}': {e}")
            return []

    def evaluate_belief_alignment(
        self,
        belief: str,
        action: str,
        news_items: List[NewsItem],
    ) -> Dict[str, Any]:
        """
        Evaluate if a belief/action aligns with current real-world evidence.

        Args:
            belief: What RAVANA believes (e.g., "lying causes more harm")
            action: What RAVANA would do based on that belief
            news_items: Current news to evaluate against

        Returns:
            {
                "alignment_score": 0.0-1.0,
                "supporting_articles": [...],
                "contradicting_articles": [...],
                "verdict": "aligned" | "misaligned" | "mixed" | "no_evidence",
                "summary": str,
            }
        """
        if not news_items:
            return {
                "alignment_score": 0.5,
                "supporting_articles": [],
                "contradicting_articles": [],
                "verdict": "no_evidence",
                "summary": "No news data available to evaluate this belief.",
            }

        supporting = []
        contradicting = []

        belief_keywords = self._extract_keywords(belief)

        for item in news_items:
            text = (item.title + " " + item.summary).lower()
            keyword_matches = sum(1 for kw in belief_keywords if kw.lower() in text)

            if keyword_matches >= 2:
                if self._supports_belief(item, belief):
                    supporting.append(item)
                elif self._contradicts_belief(item, belief):
                    contradicting.append(item)

        total = len(supporting) + len(contradicting)
        if total == 0:
            verdict = "no_evidence"
            score = 0.5
        elif len(supporting) > len(contradicting) * 2:
            verdict = "aligned"
            score = 0.7 + (len(supporting) / (total + 1)) * 0.3
        elif len(contradicting) > len(supporting) * 2:
            verdict = "misaligned"
            score = 0.3 - (len(contradicting) / (total + 1)) * 0.3
        else:
            verdict = "mixed"
            score = 0.5

        return {
            "alignment_score": max(0.0, min(1.0, score)),
            "supporting_articles": supporting[:5],
            "contradicting_articles": contradicting[:5],
            "verdict": verdict,
            "belief": belief,
            "action": action,
            "summary": self._generate_alignment_summary(belief, verdict, supporting, contradicting),
        }

    def get_grounding_report(self, ravana_state: dict, recent_actions: List[dict]) -> str:
        """
        Generate a report on how RAVANA's beliefs align with current reality.

        Args:
            ravana_state: Current RAVANA cognitive state
            recent_actions: List of recent actions/decisions

        Returns:
            Human-readable grounding report
        """
        topics_to_check = None
        if ravana_state.get('dissonance', 0) > 0.6:
            topics_to_check = ["AI safety", "cognitive science", "AI ethics"]
        else:
            topics_to_check = ["machine learning", "AGI development"]

        query = None
        if recent_actions:
            query = recent_actions[-1].get('belief') or recent_actions[-1].get('action')

        cycle = self.ingest_news(
            query=query,
            ravana_state=ravana_state,
            topics=topics_to_check,
            max_items=5,
            max_scenarios=3,
        )

        news = cycle["news_items"]
        relevant_news = news[:5] if news else []
        mdp_scenarios = cycle["scenarios"]

        lines = [
            "=== REALITY GROUNDING REPORT ===",
            f"Timestamp: {datetime.now().isoformat()}",
            f"News sources checked: {len(self.rss_feeds)}",
            f"Relevant articles found: {len(relevant_news)}",
            f"Cycle summary: {cycle['summary']}",
            "",
            "TOP STORIES:",
        ]

        for i, item in enumerate(relevant_news[:5], 1):
            lines.append(f"  {i}. {item.title}")
            lines.append(f"     {item.summary[:150]}...")
            lines.append(f"     Source: {item.source} | Topic: {item.topic}")

        if mdp_scenarios:
            lines.append("")
            lines.append("NEWS→MDP SCENARIOS:")
            for i, scenario in enumerate(mdp_scenarios, 1):
                state = scenario.state
                next_state = scenario.next_state
                lines.append(
                    f"  {i}. {scenario.topic} → {scenario.action} "
                    f"(reward: {scenario.reward:+.2f}, pressure: {scenario.pressure:.2f})"
                )
                lines.append(
                    f"     D: {state['dissonance']:.2f} → {next_state['dissonance']:.2f} | "
                    f"I: {state['identity']:.2f} → {next_state['identity']:.2f}"
                )
                lines.append(f"     {scenario.rationale}")

        if recent_actions:
            lines.append("")
            lines.append("BELIEF ALIGNMENT CHECK:")
            for action in recent_actions[-3:]:
                belief = action.get('belief', 'unknown')
                eval_result = self.evaluate_belief_alignment(
                    belief, action.get('action', ''), relevant_news
                )
                lines.append(f"  - {belief}: {eval_result['verdict'].upper()} (score: {eval_result['alignment_score']:.2f})")

        return "\n".join(lines)

    def fetch_article_content(self, url: str) -> str:
        """
        Extract full text from a news article URL.

        Falls back to summary if full extraction fails.
        """
        if not NEWSPAPER_AVAILABLE:
            return ""

        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text[:5000]  # Limit to 5000 chars
        except Exception:
            return ""

    # ─── Private Methods ────────────────────────────────────────────────────────

    def _fetch_single_feed(self, topic: str, url: str, relevance_topics: List[str]) -> List[NewsItem]:
        """Fetch and parse a single RSS feed."""
        feed = feedparser.parse(url)
        items = []

        for entry in feed.entries[:self.max_items]:
            try:
                published = self._parse_date(entry)
                summary = self._clean_summary(getattr(entry, 'summary', getattr(entry, 'description', '')))

                item = NewsItem(
                    title=entry.get('title', 'No title'),
                    url=entry.get('link', ''),
                    source=feed.feed.get('title', topic),
                    published=published.strftime("%Y-%m-%d") if published else "Unknown",
                    published_epoch=published.timestamp() if published else 0,
                    summary=summary,
                    topic=topic,
                    relevance_score=self._compute_relevance(
                        getattr(entry, 'title', '') + " " + summary,
                        relevance_topics
                    ),
                )
                items.append(item)
            except Exception:
                continue

        return items

    def _parse_date(self, entry) -> Optional[datetime]:
        """Parse date from RSS entry with multiple fallbacks."""
        for attr in ['published_parsed', 'updated_parsed', 'created_parsed']:
            parsed = getattr(entry, attr, None)
            if parsed:
                try:
                    return datetime(*parsed[:6])
                except Exception:
                    pass
        return datetime.now()

    def _clean_summary(self, summary: str) -> str:
        """Strip HTML from summary."""
        import re
        summary = re.sub(r'<[^>]+>', '', summary)
        summary = re.sub(r'\s+', ' ', summary)
        return summary.strip()[:500]

    def _compute_relevance(self, text: str, topics: List[str]) -> float:
        """Compute relevance score of text to topics."""
        text_lower = text.lower()
        matches = sum(1 for topic in topics if topic.lower() in text_lower)
        return min(1.0, matches / max(1, len(topics)))

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract significant keywords from text."""
        import re
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'should', 'could', 'may', 'might', 'must', 'can',
                     'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she',
                     'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
                     'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
                     'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                     'own', 'same', 'so', 'than', 'too', 'very', 'just', 'and',
                     'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of',
                     'at', 'by', 'for', 'with', 'about', 'against', 'between',
                     'into', 'through', 'during', 'before', 'after', 'above',
                     'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off'}

        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        keywords = [w for w in words if w not in stopwords]
        return list(set(keywords))[:10]

    def _supports_belief(self, item: NewsItem, belief: str) -> bool:
        """Check if news item supports a belief (heuristic)."""
        belief_lower = belief.lower()
        text_lower = (item.title + " " + item.summary).lower()

        positive_patterns = [
            ("harm", "reduces harm"), ("truth", "honesty prevails"),
            ("ethics", "ethical"), ("safety", "safe"),
        ]

        for pattern, indicator in positive_patterns:
            if pattern in belief_lower and indicator in text_lower:
                return True
            if "not " + pattern in belief_lower and "no " + pattern in text_lower:
                return True

        return False

    def _contradicts_belief(self, item: NewsItem, belief: str) -> bool:
        """Check if news item contradicts a belief (heuristic)."""
        belief_lower = belief.lower()
        text_lower = (item.title + " " + item.summary).lower()

        contradiction_pairs = [
            ("harm", "causes harm"), ("truth", "lie"), ("safety", "danger"),
            ("honesty", "deception"),
        ]

        for belief_pattern, contradiction in contradiction_pairs:
            if belief_pattern in belief_lower and contradiction in text_lower:
                return True

        return False

    def _generate_alignment_summary(
        self,
        belief: str,
        verdict: str,
        supporting: List[NewsItem],
        contradicting: List[NewsItem],
    ) -> str:
        """Generate human-readable alignment summary."""
        if verdict == "no_evidence":
            return f"No current news evidence to evaluate belief: '{belief}'"

        s_count = len(supporting)
        c_count = len(contradicting)

        if verdict == "aligned":
            return f"Belief '{belief}' is SUPPORTED by {s_count} recent articles. Reality aligns with this belief."
        elif verdict == "misaligned":
            return f"Belief '{belief}' is CONTRADICTED by {c_count} recent articles. Reality challenges this belief."
        elif verdict == "mixed":
            return f"Belief '{belief}' has mixed evidence: {s_count} supporting, {c_count} contradicting articles."
        else:
            return f"Cannot determine alignment for belief: '{belief}'"

    def _get_cached(self, key: str) -> Optional[List[NewsItem]]:
        """Get cached items if still valid."""
        if key in self._cache:
            timestamp, items = self._cache[key]
            if time.time() - timestamp < self.cache_ttl:
                return items
        return None

    def _set_cached(self, key: str, items: List[NewsItem]):
        """Cache items with current timestamp."""
        self._cache[key] = (time.time(), items)


if __name__ == "__main__":
    rg = RealityGrounding()

    print("=== Reality Grounding Test ===")

    print("\nRunning structured news-to-MDP cycle...")
    cycle = rg.ingest_news(ravana_state={"dissonance": 0.55, "identity": 0.52}, max_items=3, max_scenarios=3)
    print(cycle["summary"])

    print("\nFetching RSS feeds...")
    news = cycle["news_items"]
    print(f"Fetched {len(news)} items")
    for item in news[:3]:
        print(f"  - [{item.topic}] {item.title}")
        print(f"    {item.summary[:100]}...")

    print("\nNews→MDP scenarios:")
    for scenario in cycle["scenarios"][:3]:
        print(f"  - {scenario.topic}: {scenario.action} (pressure={scenario.pressure:.2f})")

    print("\nEvaluating belief alignment...")
    result = rg.evaluate_belief_alignment(
        "lying causes more harm than good",
        "tell the truth even if uncomfortable",
        news,
    )
    print(f"  Verdict: {result['verdict']}")
    print(f"  Score: {result['alignment_score']:.2f}")
    print(f"  Summary: {result['summary']}")