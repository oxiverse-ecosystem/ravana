"""
Web Learner - RAVANA's web search and autonomous learning.
Contains: SearchEngine (multi-API with circuit breaker), background learning,
curiosity-driven topic selection, article fetching.
"""
import sys
import os
import time
import json
import re
import hashlib
import urllib.request
from urllib.parse import quote
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


# Standard English stop words used throughout the WebLearner
STOP_WORDS: Set[str] = {
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other', 'some',
    'such', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
    'also', 'not', 'no', 'nor', 'is', 'are', 'was', 'were', 'be', 'been',
    'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
    'will', 'would', 'can', 'could', 'shall', 'should', 'may', 'might',
    'must', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'myself',
    'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs',
    'themselves', 'what', 'which', 'who', 'whom', 'whose', 'any', 'some',
    'many', 'much', 'one', 'two', 'three', 'get', 'got', 'make', 'made',
    'take', 'took', 'say', 'said', 'know', 'knew', 'think', 'thought',
    'see', 'saw', 'come', 'came', 'go', 'went', 'give', 'gave', 'find',
    'found', 'tell', 'told', 'become', 'became', 'leave', 'left', 'feel',
    'felt', 'put', 'set', 'bring', 'brought', 'begin', 'began', 'keep',
    'kept', 'hold', 'held', 'write', 'wrote', 'stand', 'stood', 'hear',
    'heard', 'let', 'mean', 'meant', 'run', 'ran', 'move', 'moved', 'live',
    'lived', 'believe', 'believed', 'hold', 'held', 'bring', 'brought',
}

# Optional BeautifulSoup
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


@dataclass
class SearchConfig:
    """Configuration for search engine."""
    max_results: int = 10
    timeout: int = 5
    cooldown: int = 60
    max_failures: int = 3


class SearchError(Exception):
    """Raised when all search APIs fail."""
    pass


class SearchEngine:
    """Multi-API search engine with circuit breaker fallback."""

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.apis = [
            ("oxiverse", "https://api.oxiverse.com/search?q={}", 3, 10),
            ("duckduckgo", "https://html.duckduckgo.com/html/?q={}", 5, 10),
        ]
        self._api_failure_counts = {name: 0 for name, _, _, _ in self.apis}
        self._api_last_failure_time = {name: 0 for name, _, _, _ in self.apis}
        self._headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 RAVANA/1.0'
        }

    def _is_api_available(self, api_name: str) -> bool:
        if self._api_failure_counts[api_name] < self.config.max_failures:
            return True
        elapsed = time.time() - self._api_last_failure_time[api_name]
        if elapsed > self.config.cooldown:
            self._api_failure_counts[api_name] = 0
            return True
        return False

    def _record_success(self, api_name: str):
        self._api_failure_counts[api_name] = 0

    def _record_failure(self, api_name: str):
        self._api_failure_counts[api_name] += 1
        self._api_last_failure_time[api_name] = time.time()

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search with automatic fallback across APIs."""
        query_encoded = quote(query)
        errors = []

        for api_name, url_template, timeout, api_max_results in self.apis:
            if not self._is_api_available(api_name):
                continue

            try:
                url = url_template.format(query_encoded)
                results = self._call_api(api_name, url, timeout, max_results)
                if results:
                    self._record_success(api_name)
                    return results[:max_results]
            except Exception as e:
                self._record_failure(api_name)
                errors.append(f"{api_name}: {e}")
                continue

        raise SearchError(f"All search APIs failed: {'; '.join(errors)}")

    def _call_api(self, api_name: str, url: str, timeout: int, max_results: int) -> List[Dict[str, Any]]:
        req = urllib.request.Request(url, headers=self._headers)
        resp = urllib.request.urlopen(req, timeout=timeout)
        content = resp.read().decode('utf-8', errors='replace')

        if api_name == "oxiverse":
            data = json.loads(content)
            results = data.get('results', [])
            return [
                {'title': r.get('title', ''), 'url': r.get('url', ''),
                 'content': r.get('content', r.get('snippet', ''))}
                for r in results if r.get('url')
            ]
        elif api_name == "duckduckgo":
            return self._parse_duckduckgo_html(content, max_results)
        return []

    def _parse_duckduckgo_html(self, html: str, max_results: int) -> List[Dict[str, Any]]:
        results = []
        if HAS_BS4:
            soup = BeautifulSoup(html, 'html.parser')
            for result in soup.select('.result__snippet, .result__title, a.result__url')[:max_results * 3]:
                container = result.find_parent('div', class_='result')
                if container:
                    title_elem = container.select_one('.result__title a, h2 a')
                    url_elem = container.select_one('.result__url, a.result__url')
                    snippet_elem = container.select_one('.result__snippet')
                    title = title_elem.get_text(strip=True) if title_elem else ''
                    url = title_elem.get('href', '') if title_elem else (url_elem.get_text(strip=True) if url_elem else '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else title
                    if url and not url.startswith('/'):
                        results.append({'title': title, 'url': url, 'content': snippet})
                    if len(results) >= max_results:
                        break
        else:
            import re
            snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>([^<]+)</a>', html)
            titles = re.findall(r'<a[^>]*class="result__title"[^>]*>([^<]+)</a>', html)
            urls = re.findall(r'<a[^>]*class="result__url"[^>]*>([^<]+)</a>', html)
            for i in range(min(len(snippets), len(titles), len(urls), max_results)):
                results.append({'title': titles[i], 'url': urls[i], 'content': snippets[i]})
        return results


class WebLearner:
    """Web learning engine: search, fetch articles, extract concepts, train decoder."""

    def __init__(self, graph_engine, decoder_engine, glove_vector_fn,
                 data_dir: Optional[str] = None, trace_enabled: bool = False):
        self.graph_engine = graph_engine
        self.decoder_engine = decoder_engine
        self._glove_vector = glove_vector_fn
        self._trace_enabled = trace_enabled

        # Network state
        self._network_available: Optional[bool] = None
        self._network_retry_turn: int = 0
        self._turns_since_last_search: int = 0

        # Data paths
        _proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if data_dir:
            self._glove_cache_path = os.path.join(data_dir, "ravana_glove_cache.npz")
        else:
            os.makedirs(os.path.join(_proj_root, "data"), exist_ok=True)
            self._glove_cache_path = os.path.join(_proj_root, "data", "ravana_glove_cache.npz")

        # Search engine
        self.search_engine = SearchEngine()

        # Background learning
        self._bg_learning_active: bool = False
        self._bg_learning_thread: Optional[threading.Thread] = None
        self._bg_learning_queue: List[str] = []
        self._bg_lock = threading.Lock()
        self._bg_idle_event = threading.Event()
        self._bg_search_count: int = 0
        self._bg_multi_search_max: int = 3
        self._bg_idle_search_count: int = 0

        # Phase 18: Curiosity Drive
        self._curiosity_drive_enabled: bool = True
        self._curiosity_cycles_this_session: int = 0
        self._concept_visit_count: Dict[str, int] = {}
        self._concept_learning_progress: Dict[str, float] = {}
        self._concept_pe_delta: Dict[str, float] = {}
        self._curiosity_topics_queue: List[str] = []
        self._last_auto_learn_turn: int = 0
        self._curiosity_urgency: float = 0.0
        self._user_query_topics: List[str] = []
        self._user_last_topic: str = ""
        self._concept_sources: Dict[str, Set[str]] = {}
        self._explored_contradictions: Set[Tuple[str, str]] = set()

        # Deferred learning queue
        self._pending_learning_queue: List[str] = []

    def learn_from_web(self, query: str) -> str:
        """Search web, fetch articles, extract concepts, learn."""
        # Check network availability
        if self._network_available is False:
            if self._network_retry_turn > 0 and self.graph_engine._topic_list.__len__() >= self._network_retry_turn:
                self._network_available = None
                self._network_retry_turn = 0
            elif self._network_retry_turn == 0:
                self._network_retry_turn = len(self.graph_engine._topic_list) + 20
                return "offline - already knew about query"

        try:
            results = self.search_engine.search(query, max_results=10)
            self._network_available = True
        except SearchError:
            self._network_available = False
            return self._offline_learn(query)

        if not results:
            return self._learn_from_snippets(query, [])

        # Fetch articles
        snippets = []
        def _fetch_single(r):
            snippet = r.get("content", "") or ""
            title = r.get("title", "") or ""
            url = r.get("url", "") or ""
            text = f"{title}. {snippet}"
            if url:
                try:
                    article_text = self._fetch_article_text(url)
                    if article_text and len(article_text) > len(text):
                        text = article_text[:5000]
                except Exception:
                    pass
            return text

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_fetch_single, r) for r in results[:3]]
            for future in as_completed(futures):
                snippets.append(future.result())

        combined_text = " ".join(snippets)
        new_concepts = self._learn_from_text(combined_text, query, source_url=results[0].get("url", "") if results else query)

        if new_concepts > 0:
            return f"learned {new_concepts} new things about {query}"
        else:
            return f"read about {query} but already knew the words"

    def _fetch_article_text(self, url: str) -> Optional[str]:
        try:
            req = urllib.request.Request(url, headers=self.search_engine._headers)
            resp = urllib.request.urlopen(req, timeout=8)
            html = resp.read().decode('utf-8', errors='replace')

            if HAS_BS4:
                soup = BeautifulSoup(html, 'html.parser')
                for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                main = soup.find('article') or soup.find('main') or soup.find('body')
                text = main.get_text(separator=' ', strip=True) if main else soup.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text)[:5000]
            else:
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:3000]
        except Exception:
            return None

    def _learn_from_text(self, text: str, topic: str, source_url: str = "") -> int:
        """Extract keywords, add concepts, form connections, train decoder."""
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        word_counts = {}
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and len(wc) >= 3:
                word_counts[wc] = word_counts.get(wc, 0) + 1

        if not word_counts:
            return 0

        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        important_words = [w for w, c in sorted_words[:30]]

        topic_lower = topic.lower().strip()
        topic_words = [tw for tw in re.findall(r"[a-zA-Z']{3,}", topic_lower) if tw not in STOP_WORDS]
        important_words = topic_words + [w for w in important_words if w not in topic_words]

        existing_labels = set(n.label.lower() for n in self.graph_engine.graph.nodes.values() if n.label)
        new_count = 0
        existing_labels_before = existing_labels.copy()
        label_to_id = {n.label: nid for nid, n in self.graph_engine.graph.nodes.items() if n.label}

        source_id = hashlib.md5(source_url.encode()).hexdigest()[:16] if source_url else hashlib.md5(topic.encode()).hexdigest()[:16]

        for word in important_words:
            if word in existing_labels:
                self._concept_sources.setdefault(word, set()).add(source_id)
                if len(self._concept_sources[word]) >= 2:
                    nids = self.graph_engine._concept_keywords.get(word, [])
                    for nid in nids:
                        for tid, edge in self.graph_engine.graph.get_outgoing(nid):
                            edge.confidence = min(0.8, edge.confidence + 0.05)
                continue

            vec = self._glove_vector(word)
            if vec is None:
                h = hash(word) % 50000
                vr = np.random.RandomState(h + 100)
                vec = vr.randn(self.graph_engine.dim).astype(np.float32) * 0.1
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            node = self.graph_engine.graph.add_node(vector=vec, label=word)
            label_to_id[word] = node.id
            self.graph_engine._concept_keywords[word] = self.graph_engine._concept_keywords.get(word, []) + [node.id]
            self.graph_engine._concept_labels.add(word.lower())
            existing_labels.add(word)
            new_count += 1
            self._concept_sources[word] = {source_id}

        # Form connections between co-occurring words
        article_words = [w for w in re.findall(r"[a-zA-Z']{3,}", text.lower())
                        if w not in STOP_WORDS and len(w.strip("'")) >= 3]
        present_important = list(set(w for w in article_words if w in important_words))

        for i in range(len(present_important)):
            for j in range(i + 1, len(present_important)):
                w1, w2 = present_important[i], present_important[j]
                nid1 = label_to_id.get(w1)
                nid2 = label_to_id.get(w2)
                if nid1 is not None and nid2 is not None:
                    existing = self.graph_engine.graph.get_edge(nid1, nid2)
                    if existing is None:
                        weight = max(0.3, min(0.7, 0.2 + word_counts.get(w1, 1) * word_counts.get(w2, 1) * 0.001))
                        self.graph_engine.graph.add_edge(nid1, nid2, weight=weight, relation_type="semantic")
                    else:
                        existing.weight = min(0.9, existing.weight + 0.05)

        # Connect new words to existing via vector similarity
        for word in important_words[:10]:
            nid = label_to_id.get(word)
            if nid is None:
                continue
            node = self.graph_engine.graph.get_node(nid)
            if node is None:
                continue
            if self.graph_engine.graph._vectors_dirty or self.graph_engine.graph._vector_matrix_normed is None:
                self.graph_engine.graph._rebuild_vector_matrix()
            if self.graph_engine.graph._vector_matrix_normed is not None and len(self.graph_engine.graph._node_id_order) > 0:
                vec_norm = node.vector / (np.linalg.norm(node.vector) + 1e-15)
                all_sims = self.graph_engine.graph._vector_matrix_normed @ vec_norm.astype(np.float32)
                for idx in np.where(all_sims > 0.3)[0]:
                    existing_nid = self.graph_engine.graph._node_id_order[idx]
                    if existing_nid == nid or existing_nid not in self.graph_engine.graph.nodes:
                        continue
                    existing_node = self.graph_engine.graph.get_node(existing_nid)
                    if existing_node is None or existing_node.vector is None:
                        continue
                    if existing_node.label and existing_node.label in important_words:
                        continue
                    sim = float(all_sims[idx])
                    if self.graph_engine.graph.get_edge(nid, existing_nid) is None:
                        weight = max(0.25, min(0.5, sim * 0.5))
                        inf_type, _ = self.graph_engine._infer_relation_type(word, existing_node.label, "semantic")
                        self.graph_engine.graph.add_edge(nid, existing_nid, weight=weight, relation_type=inf_type)

        # Train neural decoder
        if self.decoder_engine.neural_decoder and self.decoder_engine._decoder_vocab_built:
            cond_embs = self._build_conditioning(topic, article_words)
            new_labels = [n.label for nid, n in self.graph_engine.graph.nodes.items()
                          if n.label and n.label.lower() not in self.decoder_engine._decoder_word_to_idx
                          and n.label.lower() not in ('<pad>', '?', '<bos>', '<eos>')]
            if new_labels:
                self.decoder_engine._expand_vocab(new_labels)
            self.decoder_engine.train_on_text(text, topic, self.graph_engine, self._glove_vector, cond_embs, passes=10)

        return new_count

    def _build_conditioning(self, topic: str, text_words: List[str]) -> Optional[np.ndarray]:
        """Build graph walk conditioning embeddings."""
        concept_embs = []
        seen = set()

        tl = topic.lower().strip()
        if tl in self.graph_engine._concept_keywords:
            nids = self.graph_engine._concept_keywords[tl]
            node = self.graph_engine.graph.get_node(nids[0])
            if node and node.vector is not None:
                concept_embs.append(node.vector.copy())
                seen.add(tl)

        counted = {}
        for w in text_words:
            wl = w.lower().strip(".,!?\"' ")
            if wl not in STOP_WORDS and len(wl) >= 3:
                counted[wl] = counted.get(wl, 0) + 1
        for w, _ in sorted(counted.items(), key=lambda x: x[1], reverse=True)[:5]:
            if w not in seen and w in self.graph_engine._concept_keywords:
                nids = self.graph_engine._concept_keywords[w]
                node = self.graph_engine.graph.get_node(nids[0])
                if node and node.vector is not None:
                    concept_embs.append(node.vector.copy())
                    seen.add(w)

        if len(concept_embs) < 1:
            tl = topic.lower().strip()
            if tl in self.decoder_engine._decoder_word_to_embed:
                concept_embs.append(self.decoder_engine._decoder_word_to_embed[tl].copy())
                seen.add(tl)
            for w, _ in sorted(counted.items(), key=lambda x: x[1], reverse=True)[:5]:
                if w not in seen and w in self.decoder_engine._decoder_word_to_embed:
                    concept_embs.append(self.decoder_engine._decoder_word_to_embed[w].copy())
                    seen.add(w)

        if len(concept_embs) < 1:
            return None
        return np.stack(concept_embs, axis=0).astype(np.float32)

    def _offline_learn(self, query: str) -> str:
        """Learn from query text alone when offline."""
        known = self._learn_from_text(query + " " + query, query, source_url=query)
        if known > 0:
            return f"learned {known} things about {query}"
        return f"offline - already knew about {query}"

    def _learn_from_snippets(self, query: str, snippets: List[str]) -> str:
        combined = f"{query} " + " ".join(snippets[:3])
        count = self._learn_from_text(combined, query, source_url=query)
        if count > 0:
            return f"learned {count} new things about {query} from search snippets"
        return f"read about {query} but already knew those words"

    # ─── Background Learning ───

    def start_background_learning(self):
        if self._bg_learning_active and self._bg_learning_thread and self._bg_learning_thread.is_alive():
            return
        self._bg_learning_active = True
        self._bg_learning_thread = threading.Thread(target=self._bg_learn_loop, daemon=True)
        self._bg_learning_thread.start()
        if self._trace_enabled:
            print('  [bg] background learning thread started')

    def stop_background_learning(self):
        self._bg_learning_active = False
        self._bg_idle_event.set()
        if self._bg_learning_thread and self._bg_learning_thread.is_alive():
            self._bg_learning_thread.join(timeout=5)
        if self._trace_enabled:
            print(f'  [bg] background learning stopped (performed {self._bg_search_count} searches)')

    def _bg_learn_loop(self):
        while self._bg_learning_active:
            self._bg_idle_event.wait(timeout=30)
            self._bg_idle_event.clear()
            if not self._bg_learning_active:
                break
            queries_to_process = []
            with self._bg_lock:
                queries_to_process = list(self._bg_learning_queue)
                self._bg_learning_queue.clear()
            with self._bg_lock:
                deferred = list(self._pending_learning_queue)
                self._pending_learning_queue.clear()
            all_queries = queries_to_process + deferred
            if not all_queries and self._curiosity_drive_enabled:
                if self._curiosity_cycles_this_session < 3:
                    self._curiosity_cycles_this_session += 1
                    self._bg_idle_search_count = 0
                    if self._trace_enabled:
                        print(f'  [bg] idle cycle {self._curiosity_cycles_this_session}/3 - selecting curiosity topics...')
                    self._auto_select_curiosity_topics(max_topics=2)
                    with self._bg_lock:
                        all_queries = list(self._bg_learning_queue)
                        self._bg_learning_queue.clear()
            try:
                for query in all_queries:
                    if not self._bg_learning_active:
                        break
                    self._bg_idle_search_count += 1
                    if self._trace_enabled:
                        print(f'  [bg] ({self._bg_idle_search_count}) researching: {query}')
                    self._bg_multi_search(query)
                if all_queries:
                    try:
                        self._save()
                    except Exception:
                        pass
            except Exception:
                pass

    def _bg_multi_search(self, query: str):
        if not self._bg_learning_active:
            return
        try:
            result = self.learn_from_web(query)
            self._bg_search_count += 1
            if self._trace_enabled:
                print(f'  [bg]   + {result}')
        except Exception:
            if self._trace_enabled:
                print(f'  [bg]   ! failed to research: {query}')
            return
        related = self._extract_related_queries(query)
        for rq in related[:self._bg_multi_search_max]:
            if not self._bg_learning_active:
                break
            time.sleep(1)
            try:
                result = self.learn_from_web(rq)
                self._bg_search_count += 1
                if self._trace_enabled:
                    print(f'  [bg]   + {result}')
            except Exception:
                continue

    def _extract_related_queries(self, query: str) -> List[str]:
        related = []
        words = [w for w in query.lower().split() if w not in STOP_WORDS and len(w) >= 3]
        for w in words:
            if w not in self.graph_engine._concept_labels and w not in related:
                related.append(w)
        for w in words:
            nids = self.graph_engine._concept_keywords.get(w, [])
            for nid in nids:
                for tid, edge in self.graph_engine.graph.get_outgoing(nid):
                    tn = self.graph_engine.graph.nodes.get(tid)
                    if tn and tn.label:
                        neighbor = tn.label.lower()
                        if neighbor not in STOP_WORDS and len(neighbor) >= 3 and neighbor not in related:
                            related.append(f'{w} {neighbor}')
                for src, edge in self.graph_engine.graph.get_incoming(nid):
                    sn = self.graph_engine.graph.nodes.get(src)
                    if sn and sn.label:
                        neighbor = sn.label.lower()
                        if neighbor not in STOP_WORDS and len(neighbor) >= 3 and neighbor not in related:
                            related.append(f'{neighbor} {w}')
        if len(words) >= 2:
            related.append(f'{words[0]} explained')
            related.append(f'how does {words[0]} work')
        return related[:self._bg_multi_search_max + 2]

    def queue_background_search(self, query: str):
        with self._bg_lock:
            if query not in self._bg_learning_queue:
                self._bg_learning_queue.append(query)
        self._bg_idle_event.set()

    def notify_user_active(self):
        self._bg_idle_event.clear()

    def notify_user_idle(self):
        self._bg_idle_event.set()
        self._curiosity_cycles_this_session = 0

    def _save(self):
        """Save graph state - placeholder for integration with main engine."""
        pass

    # ─── Curiosity Drive (Phase 18) ───

    def _update_concept_learning_progress(self):
        if not self._curiosity_drive_enabled:
            return
        concept_edges: Dict[str, List[float]] = {}
        for (src, tgt), edge in self.graph_engine.graph.edges.items():
            src_node = self.graph_engine.graph.get_node(src)
            tgt_node = self.graph_engine.graph.get_node(tgt)
            pe = getattr(edge, 'prediction_free_energy', 0.0)
            if src_node and src_node.label:
                concept_edges.setdefault(src_node.label.lower(), []).append(pe)
            if tgt_node and tgt_node.label:
                concept_edges.setdefault(tgt_node.label.lower(), []).append(pe)

        alpha = 0.3
        for label, pe_list in concept_edges.items():
            if not pe_list:
                continue
            mean_pe = sum(pe_list) / len(pe_list)
            prev = self._concept_learning_progress.get(label, mean_pe)
            delta = prev - mean_pe
            smoothed = (1 - alpha) * prev + alpha * mean_pe
            self._concept_learning_progress[label] = smoothed
            self._concept_pe_delta[label] = delta

    def _compute_curiosity_urgency(self) -> float:
        if not self._curiosity_drive_enabled:
            self._curiosity_urgency = 0.0
            return 0.0
        urgency = 0.0
        if hasattr(self, '_prediction_error_count') and self._prediction_error_count > 5:
            pe_factor = min(0.4, self._mean_prediction_error * 2.0)
            urgency += pe_factor
        unresolved = sum(1 for iq in self._impossible_queries if not iq.resolved) if hasattr(self, '_impossible_queries') else 0
        if unresolved > 0:
            urgency += min(0.5, unresolved * 0.05)
        arousal_gain = 0.5 + 0.5 * self.emotion.state.arousal if hasattr(self, 'emotion') else 0.75
        urgency *= arousal_gain
        identity_curiosity = (1.0 - self.identity.state.strength) * 0.15 if hasattr(self, 'identity') else 0.0
        urgency += identity_curiosity
        low_conf = sum(1 for c in self._concept_confidence.values() if c < 0.3) if hasattr(self, '_concept_confidence') else 0
        if low_conf > 0:
            urgency += min(0.2, low_conf * 0.01)
        self._curiosity_urgency = min(1.0, urgency)
        return self._curiosity_urgency

    def _auto_select_curiosity_topics(self, max_topics: int = 3) -> List[str]:
        if not self._curiosity_drive_enabled or not self._bg_learning_active:
            return []
        candidates: List[Tuple[str, float]] = []
        seen = set()

        # 1. Unresolved impossible queries
        if hasattr(self, '_impossible_queries'):
            for iq in self._impossible_queries:
                if iq.resolved:
                    continue
                topic = iq.subject.lower().strip()
                if topic and topic not in seen and len(topic) >= 3:
                    candidates.append((topic, 5.0))
                    seen.add(topic)

        # 2. High prediction error
        high_pe = []
        for nid, node in self.graph_engine.graph.nodes.items():
            if node.label and node.prediction_free_energy > 0.3:
                label = node.label.lower()
                if label not in seen and len(label) >= 3:
                    high_pe.append((label, node.prediction_free_energy))
        high_pe.sort(key=lambda x: -x[1])
        for label, pe in high_pe[:5]:
            candidates.append((label, 3.0 * min(1.0, pe)))
            seen.add(label)

        # 3. Contradiction pairs
        for concept, antonyms in self.graph_engine._contradiction_map.items():
            c_conf = self._get_concept_confidence(concept)
            for ant in antonyms:
                a_conf = self._get_concept_confidence(ant)
                mismatch = abs(c_conf - a_conf) * 2.0
                conf_level = (c_conf + a_conf) / 2.0
                if conf_level > 0.6:
                    weight = 3.0 + mismatch * 0.5
                else:
                    weight = 2.0 + mismatch * 0.75
                if concept not in seen and len(concept) >= 3:
                    candidates.append((concept, max(2.0, weight)))
                    seen.add(concept)
                if ant not in seen and len(ant) >= 3:
                    candidates.append((ant, max(1.5, weight * 0.8)))
                    seen.add(ant)

        # 4. Dormant edges (novelty)
        dormant_ratios = self._compute_dormant_edge_ratio()
        all_labels = {n.label.lower() for n in self.graph_engine.graph.nodes.values() if n.label}
        unvisited = [l for l in all_labels if l not in seen and len(l) >= 3]
        unvisited.sort(key=lambda l: dormant_ratios.get(l, 0), reverse=True)
        for label in unvisited[:3]:
            dr = dormant_ratios.get(label, 0)
            candidates.append((label, 0.5 + dr * 0.5))
            seen.add(label)

        # 5. Random walk from hubs (serendipity)
        if len(self.graph_engine.graph.nodes) > 0:
            degree_counts = {}
            for src, tgt in self.graph_engine.graph.edges:
                degree_counts[src] = degree_counts.get(src, 0) + 1
                degree_counts[tgt] = degree_counts.get(tgt, 0) + 1
            if degree_counts:
                top_hub = max(degree_counts, key=degree_counts.get)
                hub_node = self.graph_engine.graph.get_node(top_hub)
                if hub_node and hub_node.label:
                    current = top_hub
                    for _ in range(2):
                        edges = list(self.graph_engine.graph.get_outgoing(current))
                        if not edges:
                            break
                        next_id = self.graph_engine.rng.choice([e[0] for e in edges]) if len(edges) > 1 else edges[0][0]
                        next_node = self.graph_engine.graph.get_node(next_id)
                        if next_node and next_node.label:
                            lbl = next_node.label.lower()
                            if lbl not in seen and len(lbl) >= 3:
                                candidates.append((lbl, 0.8))
                                seen.add(lbl)
                            current = next_id

        # Boost user-primed topics
        if self._user_query_topics:
            for i, (topic, weight) in enumerate(candidates):
                for uq in self._user_query_topics[-3:]:
                    if uq in topic or topic in uq:
                        candidates[i] = (topic, min(2.0, weight * 1.5))
                        break
                    uq_nids = self.graph_engine._concept_keywords.get(uq, [])
                    topic_nids = self.graph_engine._concept_keywords.get(topic, [])
                    if uq_nids and topic_nids:
                        for un in uq_nids:
                            for tn in topic_nids:
                                if self.graph_engine.graph.get_edge(un, tn) or self.graph_engine.graph.get_edge(tn, un):
                                    candidates[i] = (topic, min(1.5, weight * 1.3))
                                    break

        candidates.sort(key=lambda x: -x[1])
        selected = [topic for topic, _ in candidates[:max_topics]]

        if selected and self._trace_enabled:
            weights_str = ", ".join(f"{t}({w:.1f})" for t, w in candidates[:max_topics])
            print(f"  [curiosity] auto-selected: {weights_str} (urgency={self._curiosity_urgency:.2f})")

        for topic in selected:
            self.queue_background_search(topic)

        self._last_auto_learn_turn = len(self.graph_engine._topic_list)
        return selected

    def _get_concept_confidence(self, concept_label: str) -> float:
        nids = self.graph_engine._concept_keywords.get(concept_label.lower(), [])
        if not nids:
            return 0.0
        confidences = []
        for nid in nids:
            for tid, edge in self.graph_engine.graph.get_outgoing(nid):
                confidences.append(getattr(edge, 'confidence', 0.0))
            for src, edge in self.graph_engine.graph.get_incoming(nid):
                confidences.append(getattr(edge, 'confidence', 0.0))
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    def _compute_dormant_edge_ratio(self) -> Dict[str, float]:
        ratios: Dict[str, float] = {}
        for nid, node in self.graph_engine.graph.nodes.items():
            if not node.label:
                continue
            total = 0
            dormant = 0
            for tid, edge in self.graph_engine.graph.get_outgoing(nid):
                total += 1
                if (nid, tid) in self.graph_engine._dormant_edges:
                    dormant += 1
            for src, edge in self.graph_engine.graph.get_incoming(nid):
                if src == nid:
                    continue
                total += 1
                if (src, nid) in self.graph_engine._dormant_edges:
                    dormant += 1
            if total > 0:
                ratios[node.label.lower()] = dormant / total
        return ratios

    # Properties that need to be set from main engine
    @property
    def _trace_enabled(self):
        return False  # Will be set from main

    @_trace_enabled.setter
    def _trace_enabled(self, val):
        pass

    # These need to be accessed from main engine
    _impossible_queries = []
    _prediction_error_count = 0
    _mean_prediction_error = 0.0
    _concept_confidence = {}
    emotion = None
    identity = None

    def _seed_from_graph_curiosity(self, max_topics: int = 8) -> int:
        """Seed background learning queue from graph's curiosity signals."""
        if not self._curiosity_drive_enabled:
            return 0
        selected = self._auto_select_curiosity_topics(max_topics=max_topics)
        if self._trace_enabled:
            print(f"  [seed] Seeded {len(selected)} topics from curiosity: {selected}")
        return len(selected)
    identity = None