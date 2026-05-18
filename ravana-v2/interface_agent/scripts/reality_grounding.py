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
from dataclasses import dataclass
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


class RealityGrounding:
    """
    Grounds RAVANA's decisions in real-world events.
    
    - Polls RSS feeds for live news on relevant topics
    - Uses Google News search for targeted queries
    - Evaluates whether RAVANA's beliefs/actions align with reality
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
        
        # Sort by published date (newest first) then relevance
        all_items.sort(key=lambda x: (x.published_epoch, x.relevance_score), reverse=True)
        return all_items[:self.max_items * len(self.rss_feeds)]
    
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
            # Use Google News RSS
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
                # Check sentiment alignment
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
        # Fetch latest news
        news = self.fetch_rss_feeds()
        
        # Filter to relevant topics based on RAVANA state
        if ravana_state.get('dissonance', 0) > 0.6:
            topics_to_check = ["AI safety", "cognitive science", "AI ethics"]
        else:
            topics_to_check = ["machine learning", "AGI development"]
        
        relevant_news = [n for n in news if n.topic in topics_to_check]
        
        if not relevant_news:
            relevant_news = news[:5]
        
        # Build report
        lines = [
            "=== REALITY GROUNDING REPORT ===",
            f"Timestamp: {datetime.now().isoformat()}",
            f"News sources checked: {len(self.rss_feeds)}",
            f"Relevant articles found: {len(relevant_news)}",
            "",
            "TOP STORIES:",
        ]
        
        for i, item in enumerate(relevant_news[:5], 1):
            lines.append(f"  {i}. {item.title}")
            lines.append(f"     {item.summary[:150]}...")
            lines.append(f"     Source: {item.source} | Topic: {item.topic}")
        
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
        # Fallback to now
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
        # Remove stopwords
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
        
        # Positive indicators
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
    # Quick test
    rg = RealityGrounding()
    
    print("=== Reality Grounding Test ===")
    
    # Test RSS fetch
    print("\nFetching RSS feeds...")
    news = rg.fetch_rss_feeds()
    print(f"Fetched {len(news)} items")
    for item in news[:3]:
        print(f"  - [{item.topic}] {item.title}")
        print(f"    {item.summary[:100]}...")
    
    # Test news search
    print("\nSearching news for 'AI safety'...")
    results = rg.search_news("AI safety cognitive architecture")
    for item in results[:3]:
        print(f"  - {item.title}")
    
    # Test belief evaluation
    print("\nEvaluating belief alignment...")
    result = rg.evaluate_belief_alignment(
        "lying causes more harm than good",
        "tell the truth even if uncomfortable",
        news
    )
    print(f"  Verdict: {result['verdict']}")
    print(f"  Score: {result['alignment_score']:.2f}")
    print(f"  Summary: {result['summary']}")
