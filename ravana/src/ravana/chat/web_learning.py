"""Auto-generated mixin module for CognitiveChatEngine."""
from __future__ import annotations
import sys, os, time, random, json, re, threading, hashlib
import urllib.request
import socket
socket.setdefaulttimeout(4.0)
from urllib.error import URLError
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Set
from collections import deque, Counter


from .constants import STOP_WORDS
from .response_gen import ResponseGenMixin
# Compute project root (same logic as engine.py)
_proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from ravana.web.learner import SearchError

# Optional bs4 (same as engine.py)
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False





class WebLearningMixin(ResponseGenMixin):
    """Mixin providing web search, background learning, and curiosity drive methods."""

    def learn_from_web(self, query: str, max_results: int = 3) -> Tuple[str, str]:
        """Search the web, fetch articles, extract concepts, and learn from them.

        Phase 5: Auto offline fallback. If the network API fails (timeout, DNS,
        HTTP error), sets _network_available = False and falls back silently.
        No error messages leak to the user — just returns a short summary.
        """
        self._learned_this_turn = True
        self._learning_count += 1

        # Phase 5: Skip API call if we already know the network is down
        # (unless it's time for a retry — check every 20 turns)
        if self._network_available is False:
            if self._network_retry_turn > 0 and self.turn_count >= self._network_retry_turn:
                self._network_available = None  # try again
                self._network_retry_turn = 0
            elif self._network_retry_turn == 0:
                # First time being offline — schedule a retry
                self._network_retry_turn = self.turn_count + 20
                known_count = self._learn_from_text(query + " " + query, query, source_url=query)
                if known_count > 0:
                    return f"learned {known_count} things about {query}", ""
                return f"offline - already knew about {query}", ""
            else:
                # Still try GloVe-only learning from the query text itself
                known_count = self._learn_from_text(query + " " + query, query, source_url=query)
                if known_count > 0:
                    return f"learned {known_count} things about {query}", ""
                return f"offline - already knew about {query}", ""

        query_clean = quote(query)

        try:
            # Step 1: Search with fallback engines (circuit breaker)
            try:
                results = self.search_engine.search(query, max_results=max_results)
                if self._network_available is None:
                    self._network_available = True
            except SearchError:
                # All search engines failed
                raise URLError("All search APIs failed")

            if not results:
                return self._learn_from_snippets(query, [])

            # Step 2: Get snippets and try to fetch article text
            snippets = []

            # Parallel article fetching (ThreadPoolExecutor)
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
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(_fetch_single, r) for r in results[:3]]
                for future in as_completed(futures):
                    snippets.append(future.result())

            # Step 3: Extract and learn concepts from all gathered text
            combined_text = " ".join(snippets)
            first_url = results[0].get("url", "") if results else ""
            new_concepts_added = self._learn_from_text(combined_text, query, source_url=first_url if first_url else query)

            # --- Extract definitional knowledge (ATL convergence zone) ---
            # Scan article text for "X is a Y" / "X refers to Y" patterns
            # to populate the neocortical definition store.
            # Neuroscience: Binder & Desai (2011) — ATL convergence zones
            # store category membership as stable representations.
            if combined_text:
                try:
                    defn_patterns = [
                        # Capitalized concepts: "Google is a search engine"
                        re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an|the)\s+(.+?)\.', re.IGNORECASE),
                        re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:refers?|means?|describes?)\s+to\s+(.+?)\.', re.IGNORECASE),
                        # Lowercase concepts: "hacking is a form of unauthorized access"
                        re.compile(r'\b([a-z]{3,})\s+is\s+(?:a|an|the)\s+(.+?)\.'),
                        re.compile(r'\b([a-z]{3,})\s+(?:refers?|means?)\s+to\s+(.+?)\.'),
                    ]
                    query_lower = query.lower().strip()
                    for pat in defn_patterns:
                        for m in pat.finditer(combined_text):
                            concept = m.group(1).strip().lower()
                            definition = m.group(2).strip()
                            # Only store if related to query or known concept
                            # Skip very short or very long definitions
                            if len(definition) < 10 or len(definition) > 200:
                                continue
                            if (concept in query_lower or query_lower in concept
                                    or concept in self._concept_keywords):
                                existing = self._definitions.get(concept, '')
                                if concept not in self._definitions or len(definition) > len(existing):
                                    self._definitions[concept] = definition[:200]
                                    if self._trace_enabled:
                                        print(f"  [definition] Stored: {concept} -> {definition[:80]}...")
                except Exception:
                    pass

            if new_concepts_added > 0:
                return f"learned {new_concepts_added} new things about {query}", combined_text
            else:
                return f"read about {query} but already knew the words", combined_text

        except (urllib.request.URLError, urllib.request.HTTPError,
                ConnectionError, TimeoutError, OSError, json.JSONDecodeError) as e:
            # Phase 5: Network failure — mark as unavailable, fall back silently
            if self._trace_enabled:
                print(f"  [bg] Network error in learn_from_web: {type(e).__name__}")
            self._network_available = False
            # Still try GloVe-only learning from the query text
            known_count = self._learn_from_text(query + " " + query, query, source_url=query)
            if known_count > 0:
                return f"learned {known_count} things about {query}", ""
            return f"offline - already knew about {query}", ""
        except Exception as e:
            # Any other error — also fall back silently
            if self._trace_enabled:
                print(f"  [bg] Unexpected error in learn_from_web: {type(e).__name__}: {e}")
            self._network_available = False
            return "offline", ""



    def _fetch_article_text(self, url: str) -> Optional[str]:
        """Fetch a URL and extract readable article text."""
        # Skip video/social media URLs that don't have extractable article content
        skip_domains = ('youtube.com', 'youtu.be', 'facebook.com', 'twitter.com', 'x.com',
                        'instagram.com', 'tiktok.com', 'reddit.com', 'linkedin.com',
                        'pinterest.com', 'vimeo.com', 'dailymotion.com', 'twitch.tv')
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if any(skip_domain in domain for skip_domain in skip_domains):
                return None
        except Exception:
            pass

        try:
            req = urllib.request.Request(url, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            resp = urllib.request.urlopen(req, timeout=8)
            html = resp.read().decode('utf-8', errors='replace')

            if HAS_BS4:
                soup = BeautifulSoup(html, 'html.parser')
                # Remove script/style elements
                for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                    tag.decompose()
                # Try to find main content
                main = soup.find('article') or soup.find('main') or soup.find('body')
                if main:
                    text = main.get_text(separator=' ', strip=True)
                else:
                    text = soup.get_text(separator=' ', strip=True)
                return re.sub(r'\s+', ' ', text)[:5000]
            else:
                # Simple regex-based tag stripping
                text = re.sub(r'<[^>]+>', ' ', html)
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:3000]
        except Exception:
            return None



    def _learn_from_text(self, text: str, topic: str, source_url: str = "") -> int:
        """Extract important keywords from text, add them as concepts, form connections.

        Returns number of new concepts added.
        """
        # Tokenize and count word frequencies
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        word_counts = {}
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and len(wc) >= 3:
                word_counts[wc] = word_counts.get(wc, 0) + 1

        if not word_counts:
            return 0

        # Sort by frequency, take top 30
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        important_words = [w for w, c in sorted_words[:30]]

        # Always add the topic word
        topic_lower = topic.lower().strip()
        topic_words = [tw for tw in re.findall(r"[a-zA-Z']{3,}", topic_lower) if tw not in STOP_WORDS]
        important_words = topic_words + [w for w in important_words if w not in topic_words]

        # Check which words are new vs known (lock graph reads)
        with self._graph_lock:
            existing_labels = set()
            for nid, node in self.graph.nodes.items():
                if node.label:
                    existing_labels.add(node.label.lower())

            new_count = 0
            existing_labels_before = existing_labels.copy()
            label_to_id = {}
            for nid, node in self.graph.nodes.items():
                if node.label:
                    label_to_id[node.label] = nid

        # Phase 18c: Track source of this learning - use URL when available
        if source_url:
            source_id = hashlib.md5(source_url.encode()).hexdigest()[:16]
        else:
            source_id = hashlib.md5((source_url or topic).encode()).hexdigest()[:16]

        # Add new concepts (locked to prevent graph mutation during save iteration)
        for word in important_words:
            if word in existing_labels:
                # Concept already exists - reinforce by tracking additional source
                self._concept_sources.setdefault(word, set()).add(source_id)
                # Boost edge confidence if concept confirmed by multiple sources
                if len(self._concept_sources[word]) >= 2:
                    nids = self._concept_keywords.get(word, [])
                    for nid in nids:
                        for tid, edge in self.graph.get_outgoing(nid):
                            edge.confidence = min(0.8, edge.confidence + 0.05)
                continue
            # GloVe vector if available, else hash-based random
            vec = self._glove_vector(word)
            if vec is None:
                h = hash(word) % 50000
                vr = np.random.RandomState(h + 100)
                vec = vr.randn(self.dim).astype(np.float32) * 0.1
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec /= norm
            with self._graph_lock:
                node = self.graph.add_node(vector=vec, label=word)
            label_to_id[word] = node.id
            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]
            self._concept_labels.add(word.lower())
            existing_labels.add(word)
            new_count += 1
            # Phase 18c: Track source for new concepts
            self._concept_sources[word] = {source_id}
            # New concepts tracked by source (confidence set naturally by prediction error)

        # Form connections between co-occurring important words
        # Words that appear together in the article get edges
        article_words_lower = [w for w in re.findall(r"[a-zA-Z']{3,}", text.lower())
                              if w not in STOP_WORDS and len(w.strip("'")) >= 3]
        # Get unique important words that appear in the text
        present_important = list(set(w for w in article_words_lower if w in important_words))

        with self._graph_lock:
            for i in range(len(present_important)):
                for j in range(i + 1, len(present_important)):
                    w1, w2 = present_important[i], present_important[j]
                    nid1 = label_to_id.get(w1)
                    nid2 = label_to_id.get(w2)
                    if nid1 is not None and nid2 is not None:
                        # Don't duplicate existing edges, just boost weight
                        existing = self.graph.get_edge(nid1, nid2)
                        if existing is None:
                            weight = max(0.3, min(0.7, 0.2 + word_counts.get(w1, 1) * word_counts.get(w2, 1) * 0.001))
                            self.graph.add_edge(nid1, nid2, weight=weight, relation_type="semantic")
                        else:
                            existing.weight = min(0.9, existing.weight + 0.05)

        # Connect new words to existing related concepts via vector similarity
        # Only do this for the topic word and first 2 important words (others
        # rely on co-occurrence edges, which are more topically coherent)
        with self._graph_lock:
            for word in important_words[:10]:
                nid = label_to_id.get(word)
                if nid is None:
                    continue
                node = self.graph.get_node(nid)
                if node is None:
                    continue
                edge_count = 0
                # FAISS/vectorized: matrix multiply instead of per-node loop
                if self.graph._vectors_dirty or self.graph._vector_matrix_normed is None:
                    self.graph._rebuild_vector_matrix()
                if self.graph._vector_matrix_normed is not None and len(self.graph._node_id_order) > 0:
                    vec_norm = node.vector / (np.linalg.norm(node.vector) + 1e-15)
                    all_sims = self.graph._vector_matrix_normed @ vec_norm.astype(np.float32)
                    for idx in np.where(all_sims > 0.3)[0]:
                        existing_nid = self.graph._node_id_order[idx]
                        if existing_nid == nid or existing_nid not in self.graph.nodes:
                            continue
                        existing_node = self.graph.get_node(existing_nid)
                        if existing_node is None or existing_node.vector is None:
                            continue
                        if existing_node.label and existing_node.label in important_words:
                            continue
                        sim = float(all_sims[idx])
                        if self.graph.get_edge(nid, existing_nid) is None:
                            weight = max(0.25, min(0.5, sim * 0.5))
                            inf_type, _ = self._infer_relation_type(word, existing_node.label, "semantic")
                            self.graph.add_edge(nid, existing_nid, weight=weight,
                                                relation_type=inf_type)
                            edge_count += 1
                # Guarantee at least one connection for the topic word
                if edge_count == 0 and word == important_words[0]:
                    best_sim = 0.0
                    best_nid = None
                    best_label = None
                    for existing_nid, existing_node in list(self.graph.nodes.items()):
                        if existing_nid == nid or existing_node.label in important_words:
                            continue
                        if existing_node.vector is not None and node.vector is not None:
                            sim = float(np.dot(node.vector, existing_node.vector))
                            if sim > best_sim:
                                best_sim = sim
                                best_nid = existing_nid
                                best_label = existing_node.label
                    if best_nid is not None and best_label is not None:
                        weight = max(0.25, min(0.4, best_sim * 0.5))
                        inf_type, _ = self._infer_relation_type(word, best_label, "semantic")
                        self.graph.add_edge(nid, best_nid, weight=weight, relation_type=inf_type)

        # Train neural decoder on article text (unsupervised next-word prediction)
        if self.neural_decoder is not None and self._decoder_vocab_built:
            with self._vocab_lock:
                # Build graph walk conditioning from the article's important words
                article_words = re.findall(r"[a-zA-Z']{3,}", text.lower())
                cond_embs = self._build_conditioning_for_text(topic, article_words)

                # Expand decoder vocab with newly added concepts (lock graph read)
                with self._graph_lock:
                    new_labels = [n.label for nid, n in self.graph.nodes.items()
                                  if n.label and n.label.lower() not in self._decoder_word_to_idx
                                  and n.label.lower() not in ('<pad>', '<unk>', '<bos>', '<eos>')]
                if new_labels:
                    self._expand_decoder_vocab(new_labels[:500])

                # FIX: Deduplicate web text before training — remove repeated sentences
                sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
                seen_sents = set()
                unique_sents = []
                for s in sentences:
                    key = s.lower().strip()[:60]
                    if key not in seen_sents:
                        seen_sents.add(key)
                        unique_sents.append(s)
                clean_text = '. '.join(unique_sents)

                # FIX: Expand vocab selectively — only add words appearing >=3 times
                with self._graph_lock:
                    new_labels_all = [n.label for nid, n in self.graph.nodes.items()
                                      if n.label and n.label.lower() not in self._decoder_word_to_idx
                                      and n.label.lower() not in ('<pad>', '<unk>', '<bos>', '<eos>')]
                    if new_labels_all:
                        # Only add words seen >=3 times in the original text
                        text_counts = {}
                        for w in re.findall(r"[a-zA-Z']{3,}", text.lower()):
                            wc = w.strip("'")
                            text_counts[wc] = text_counts.get(wc, 0) + 1
                        frequent_labels = [l for l in new_labels_all if text_counts.get(l.lower(), 0) >= 3]
                        self._expand_decoder_vocab(frequent_labels[:200])

                # FIX: Use freeze_core=True for web learning — only update word embeddings
                # and output_proj, not the core GRU/attention. This prevents web noise
                # from corrupting the core language model.
                err, trained = self.neural_decoder.train_on_text(
                    clean_text, self._decoder_word_to_embed, self._decoder_word_to_idx,
                    conditioning_embs=cond_embs, freeze_core=True)
                if trained > 0:
                    self._decoder_web_training_count += trained
                    self._decoder_training_count += trained
                # Single additional pass (was 30, already reduced to 2, now 1 with freeze_core)
                if trained > 0:
                    for _ in range(1):
                        err2, trained2 = self.neural_decoder.train_on_text(
                            clean_text, self._decoder_word_to_embed, self._decoder_word_to_idx,
                            conditioning_embs=cond_embs, freeze_core=True)
                        self._decoder_web_training_count += trained2
                        self._decoder_training_count += trained2
                    self.neural_decoder.sleep_cycle()

        # Extract definitional knowledge (ATL convergence zone)
        # Scan article text for "X is a Y" / "X refers to Y" patterns
        # to populate the neocortical definition store.
        if text:
            try:
                defn_patterns = [
                    # Capitalized concepts: "Google is a search engine"
                    re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+is\s+(?:a|an|the)\s+(.+?)\.', re.IGNORECASE),
                    re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:refers?|means?|describes?)\s+to\s+(.+?)\.', re.IGNORECASE),
                    # Lowercase concepts: "hacking is a form of unauthorized access"
                    re.compile(r'\b([a-z]{3,})\s+is\s+(?:a|an|the)\s+(.+?)\.'),
                    re.compile(r'\b([a-z]{3,})\s+(?:refers?|means?)\s+to\s+(.+?)\.'),
                ]
                query_lower = topic.lower().strip()
                for pat in defn_patterns:
                    for m in pat.finditer(text):
                        concept = m.group(1).strip().lower()
                        definition = m.group(2).strip()
                        # Skip very short or very long definitions
                        if len(definition) < 10 or len(definition) > 200:
                            continue
                        if (concept in query_lower or query_lower in concept
                                or concept in self._concept_keywords):
                            existing = self._definitions.get(concept, '')
                            if concept not in self._definitions or len(definition) > len(existing):
                                self._definitions[concept] = definition[:200]
            except Exception:
                pass

        return new_count


        return new_count
    def _learn_from_snippets(self, query: str, snippets: List[str]) -> str:
        """Learn from search snippet text when article fetch fails."""
        combined = f"{query} " + " ".join(snippets[:3])
        count = self._learn_from_text(combined, query, source_url=query)
        if count > 0:
            return f"learned {count} new things about {query} from search snippets"
        return f"read about {query} but already knew those words"

    # ─── Core Response Pipeline ───



    def _bg_learn_loop(self):
        """Background learning thread: processes pending queue and related searches when idle."""
        while self._bg_learning_active:
            # Wake periodically (30s timeout) — event is set when user goes idle
            self._bg_idle_event.wait(timeout=30)
            self._bg_idle_event.clear()
            if not self._bg_learning_active:
                break

            # Periodic curiosity cycle reset for continuous exploration in background mode
            self._bg_idle_search_count += 1
            if self._bg_idle_search_count % 10 == 0:
                self._curiosity_cycles_this_session = 0
                if self._trace_enabled:
                    print('  [bg] curiosity budget refreshed for new exploration cycle')

            # Process pending queue items in background
            all_queries = []
            with self._bg_lock:
                queries_to_process = list(self._bg_learning_queue)
                self._bg_learning_queue.clear()
                deferred = list(self._pending_learning_queue)
                self._pending_learning_queue.clear()
                all_queries = queries_to_process + deferred
            # Phase 18: Autonomously select curiosity topics when queue runs low
            # Run curiosity selection if queue is small (<=3) or periodically every idle period
            queue_size = len(all_queries)
            should_run_curiosity = (not all_queries and self._curiosity_drive_enabled) or \
                                   (queue_size <= 3 and self._curiosity_drive_enabled and \
                         self._bg_idle_search_count % 2 == 0)  # Every other idle period
            
            if should_run_curiosity:
                if self._curiosity_cycles_this_session < 5:  # Increased from 3
                    self._curiosity_cycles_this_session += 1
                    self._bg_idle_search_count = 0
                    if self._trace_enabled:
                        reason = "queue empty" if not all_queries else "queue low"
                        print(f'  [bg] idle cycle {self._curiosity_cycles_this_session}/5 ({reason}) - selecting curiosity topics...')
                    self._auto_select_curiosity_topics(max_topics=2)
                    with self._bg_lock:
                        all_queries = list(self._bg_learning_queue)
                        self._bg_learning_queue.clear()
                elif self._trace_enabled:
                    if not all_queries:
                        print('  [bg] idle: waiting for more user input (curiosity budget used)')
                    else:
                        print(f'  [bg] idle: queue has {queue_size} items, processing...')
            try:
                for query in all_queries:
                    if not self._bg_learning_active:
                        break
                    self._bg_idle_search_count += 1
                    if self._trace_enabled:
                        print(f'  [bg] ({self._bg_idle_search_count}) researching: {query}')
                    self._bg_multi_search(query)
                # Save periodically after learning
                if all_queries:
                    try:
                        path = self.save()
                        if self._trace_enabled:
                            print(f'  [bg] saved to {os.path.basename(path)}')
                    except Exception:
                        pass
            except Exception:
                # Prevent background thread from dying silently
                pass



    def _bg_multi_search(self, query: str):
        """Perform multiple related searches for a single query.
        1. Search the original query
        2. Extract related terms from results
        3. Search top 2-3 related terms
        All done synchronously within the background thread."""
        if not self._bg_learning_active:
            return
        # Step 1: Search the original query
        try:
            result_summary = self.learn_from_web(query)
            self._bg_search_count += 1
            if self._trace_enabled:
                print(f'  [bg]   + {result_summary}')
        except Exception:
            if self._trace_enabled:
                print(f'  [bg]   ! failed to research: {query}')
            return
        # Step 2: Extract related search terms from the query itself
        related = self._extract_related_queries(query)
        # Step 3: Search top related terms
        for related_query in related[:self._bg_multi_search_max]:
            if not self._bg_learning_active:
                break
            time.sleep(1)  # polite delay between searches
            try:
                result_summary = self.learn_from_web(related_query)
                self._bg_search_count += 1
                if self._trace_enabled:
                    print(f'  [bg]   + {result_summary}')
            except Exception:
                continue



    def _extract_related_queries(self, query: str) -> List[str]:
        """Extract related search queries from the original query.
        Uses graph knowledge and word associations to generate related searches."""
        related = []
        words = [w for w in query.lower().split() if w not in STOP_WORDS and len(w) >= 3]
        # Strategy 1: Individual word deep-dives
        for w in words:
            if w not in self._concept_labels and w not in related:
                related.append(w)
        # Strategy 2: Graph neighbors - what concepts are linked to the query concepts?
        for w in words:
            nids = self._concept_keywords.get(w, [])
            for nid in nids:
                for tid, edge in self.graph.get_outgoing(nid):
                    tn = self.graph.nodes.get(tid)
                    if tn and tn.label:
                        neighbor = tn.label.lower()
                        if neighbor not in STOP_WORDS and len(neighbor) >= 3 and neighbor not in related:
                            related.append(f'{w} {neighbor}')
                for src, edge in self.graph.get_incoming(nid):
                    sn = self.graph.nodes.get(src)
                    if sn and sn.label:
                        neighbor = sn.label.lower()
                        if neighbor not in STOP_WORDS and len(neighbor) >= 3 and neighbor not in related:
                            related.append(f'{neighbor} {w}')
        # Strategy 3: Compound queries from multi-word topics
        if len(words) >= 2:
            related.append(f'{words[0]} explained')
            related.append(f'how does {words[0]} work')
        return related[:self._bg_multi_search_max + 2]



    def queue_background_search(self, query: str):
        """Queue a query for background research. Thread-safe."""
        with self._bg_lock:
            if query not in self._bg_learning_queue:
                self._bg_learning_queue.append(query)
        self._bg_idle_event.set()  # Wake the background thread



    def notify_user_active(self):
        """Signal that the user is actively chatting (pause background learning)."""
        self._bg_idle_event.clear()  # thread will block on wait



    def notify_user_idle(self):
        """Signal that the user is idle (resume background learning)."""
        self._bg_idle_event.set()  # thread will wake up and process
        self._curiosity_cycles_this_session = 0  # fresh curiosity budget per idle period

    # --- Phase 18: Curiosity Drive - Autonomous Topic Selection ---
    # Based on: Berlyne epistemic curiosity, Loewenstein information-gap theory,
    # Friston Active Inference (prediction error minimization), Oudeyer learning progress.



    def _update_concept_learning_progress(self):
        """Aggregate edge-level prediction error to concept-level learning progress.

        Uses EWMA with rolling window to track PE delta per concept.
        Positive delta = genuine learning (PE dropping).
        Flat/negative delta = stuck (curiosity should spike).
        """
        if not self._curiosity_drive_enabled:
            return

        # Build concept -> incident edges map
        concept_edges: Dict[str, List[float]] = {}
        for (src, tgt), edge in list(self.graph.edges.items()):
            src_node = self.graph.get_node(src)
            tgt_node = self.graph.get_node(tgt)
            pe = getattr(edge, 'prediction_free_energy', 0.0)
            if src_node and src_node.label:
                concept_edges.setdefault(src_node.label.lower(), []).append(pe)
            if tgt_node and tgt_node.label:
                concept_edges.setdefault(tgt_node.label.lower(), []).append(pe)

        alpha = 0.3  # EWMA smoothing factor
        for label, pe_list in concept_edges.items():
            if not pe_list:
                continue
            mean_pe = sum(pe_list) / len(pe_list)
            prev = self._concept_learning_progress.get(label, mean_pe)
            # Compute delta: positive = PE dropping = genuine learning
            # Negative = PE rising = getting more confused = curiosity spike
            delta = prev - mean_pe  # positive = improvement
            # Store smoothed PE value for next comparison
            smoothed = (1 - alpha) * prev + alpha * mean_pe
            # Store delta as learning progress (positive = learning, negative = stuck/confused)
            self._concept_learning_progress[label] = smoothed
            # Also track the delta separately for curiosity scoring

            self._concept_pe_delta[label] = delta



    def _compute_curiosity_urgency(self) -> float:
        """Compute overall curiosity urgency from multiple signals.

        Returns (0-1) urgency score. High urgency = strong drive to explore."""
        if not self._curiosity_drive_enabled:
            self._curiosity_urgency = 0.0
            return 0.0

        urgency = 0.0

        # 1. Prediction error urgency (Active Inference: surprise drives exploration)
        if self._prediction_error_count > 5:
            pe_factor = min(0.4, self._mean_prediction_error * 2.0)
            urgency += pe_factor

        # 2. Information gap urgency (unresolved impossible queries)
        unresolved = sum(1 for iq in self._impossible_queries if not iq.resolved)
        if unresolved > 0:
            gap_urgency = min(0.5, unresolved * 0.05)
            urgency += gap_urgency

        # 3. Arousal modulation (multiplicative, coupling dopamine and norepinephrine)
        # Neuroscience: arousal amplifies exploration non-linearly
        # Even at minimal arousal (0.1), 55% of curiosity remains
        # At max arousal (1.0), curiosity is at 100%
        arousal_gain = 0.5 + 0.5 * self.emotion.state.arousal
        urgency *= arousal_gain

        # 4. Identity uncertainty (low identity = more exploration)
        identity_curiosity = (1.0 - self.identity.state.strength) * 0.15
        urgency += identity_curiosity

        # 5. Low-confidence concepts in the graph
        low_conf_count = sum(1 for c in self._concept_confidence.values() if c < 0.3)
        if low_conf_count > 0:
            urgency += min(0.2, low_conf_count * 0.01)

        self._curiosity_urgency = min(1.0, urgency)
        return self._curiosity_urgency



    def _get_concept_confidence(self, concept_label: str) -> float:
        """Get the confidence level of a concept from its incident edges."""
        nids = self._concept_keywords.get(concept_label.lower(), [])
        if not nids:
            return 0.0
        confidences = []
        for nid in nids:
            for tid, edge in self.graph.get_outgoing(nid):
                confidences.append(getattr(edge, 'confidence', 0.0))
            for src, edge in self.graph.get_incoming(nid):
                confidences.append(getattr(edge, 'confidence', 0.0))
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)

    # Phase C: Epistemic hedging - produce hedges based on concept confidence and prediction error



        for nid, node in self.graph.nodes.items():
            if not node.label:
                continue
            total = 0
            dormant = 0
            for tid, edge in self.graph.get_outgoing(nid):
                total += 1
                if (nid, tid) in self._dormant_edges:
                    dormant += 1
            for src, edge in self.graph.get_incoming(nid):
                if src == nid:
                    continue
                pair = (src, nid)
                total += 1
                if pair in self._dormant_edges:
                    dormant += 1
            if total > 0:
                ratios[node.label.lower()] = dormant / total
        return ratios



    def _compute_dormant_edge_ratio(self) -> Dict[str, float]:
        """Compute dormant edge ratio per concept.

        Returns dict mapping concept label -> dormant_ratio (0-1).
        High ratio = most edges are dormant = high novelty.
        """
        ratios: Dict[str, float] = {}
        for nid, node in list(self.graph.nodes.items()):
            if not node.label:
                continue
            total = 0
            dormant = 0
            for tid, edge in self.graph.get_outgoing(nid):
                total += 1
                if (nid, tid) in self._dormant_edges:
                    dormant += 1
            for src, edge in self.graph.get_incoming(nid):
                if src == nid:
                    continue
                pair = (src, nid)
                total += 1
                if pair in self._dormant_edges:
                    dormant += 1
            if total > 0:
                ratios[node.label.lower()] = dormant / total
        return ratios





    def _generate_curiosity_query(self, topic: str, source_type: str = "general", antonym: str = "") -> str:
        """Generate a targeted search query for a curiosity-driven topic.

        Transforms a plain topic into a more specific search query based on
        the source type of the curiosity signal.

        Args:
            topic: The base topic to research.
            source_type: The kind of curiosity signal.
            antonym: For 'contradiction' source, the opposing concept.

        Returns:
            A targeted search query string.
        """
        topic_clean = topic.lower().strip()
        if source_type == "contradiction" and antonym:
            ant = antonym.lower().strip()
            targeted = f"{topic_clean} versus {ant} comparison explained"
            if targeted == topic_clean:
                return topic_clean
            return targeted
        elif source_type == "prediction_error":
            return f"{topic_clean} what is it explained"
        elif source_type == "unknown":
            return f"{topic_clean} definition meaning"
        else:
            return f"{topic_clean} explained overview"

    def _auto_select_curiosity_topics(self, max_topics: int = 3) -> List[str]:

        """Autonomously select topics for background research based on curiosity signals.

        Priority order:
        1. Unresolved impossible queries (direct knowledge gaps) - weight 5x
        2. High prediction error concepts (surprising) - weight 3x
        3. Contradiction pairs (cognitive dissonance) - weight 3x
        4. Least-visited concepts (novelty) - weight 1x
        5. Random graph walk from high-degree hubs (serendipity) - weight 1x

        Returns list of topic strings to research."""
        if not self._curiosity_drive_enabled or not self._bg_learning_active:
            return []

        candidates: List[Tuple[str, float]] = []
        seen_topics = set()

        # Source 1: Unresolved impossible queries (weight 5x)
        for iq in self._impossible_queries:
            if iq.resolved:
                continue
            topic = iq.subject.lower().strip()
            if topic and topic not in seen_topics and len(topic) >= 3:
                candidates.append((topic, 5.0))
                seen_topics.add(topic)

        # Source 2: High prediction error concepts (weight 3x, targeted query)
        high_pe_nodes = []
        for nid, node in list(self.graph.nodes.items()):
            if node.label and node.prediction_free_energy > 0.3:
                label = node.label.lower()
                if label not in seen_topics and len(label) >= 3:
                    high_pe_nodes.append((label, node.prediction_free_energy))
        high_pe_nodes.sort(key=lambda x: -x[1])
        for label, pe in high_pe_nodes[:5]:
            candidates.append((label, 3.0 * min(1.0, pe)))
            seen_topics.add(label)

        # Source 3: Contradiction pairs with confidence-mismatch scoring
        # Higher confidence mismatch = more uncertain = more curious to resolve
        for concept, antonyms in self._contradiction_map.items():
            # Look up edge confidence for both sides of contradiction
            concept_conf = self._get_concept_confidence(concept)
            for ant in antonyms:
                ant_conf = self._get_concept_confidence(ant)
                # Mismatch score: how uncertain are we which side is correct?
                mismatch = abs(concept_conf - ant_conf) * 2.0  # 0 if equal, up to 2.0
                # Base weight: higher for high-mismatch (ambiguity) AND low-mismatch (both confident but opposing)
                # This captures both "don't know which is right" and "both sides claim truth"
                conf_level = (concept_conf + ant_conf) / 2.0
                if conf_level > 0.6:
                    # Both confident but contradictory - interesting!
                    weight = 3.0 + mismatch * 0.5
                else:
                    # At least one side uncertain - knowledge gap
                    weight = 2.0 + mismatch * 0.75
                if concept not in seen_topics and len(concept) >= 3:
                    candidates.append((concept, max(2.0, weight)))
                    seen_topics.add(concept)
                if ant not in seen_topics and len(ant) >= 3:
                    candidates.append((ant, max(1.5, weight * 0.8)))
                    seen_topics.add(ant)

        # Source 4: Novel concepts via dormant edge ratio (high dormant = unexplored)
        all_labels = set()
        dormant_ratios = self._compute_dormant_edge_ratio()
        for nid, node in list(self.graph.nodes.items()):
            if node.label:
                all_labels.add(node.label.lower())
        unvisited = [l for l in all_labels if l not in seen_topics and len(l) >= 3]
        # Sort by dormant ratio descending (most unexplored first)
        unvisited.sort(key=lambda l: dormant_ratios.get(l, 0), reverse=True)
        for label in unvisited[:3]:
            dr = dormant_ratios.get(label, 0)
            novelty_weight = 0.5 + dr * 0.5  # range: 0.5 (fully explored) to 1.0 (fully dormant)
            candidates.append((label, novelty_weight))
            seen_topics.add(label)

        # Generate targeted queries for top contradiction candidates
        # Skip already-explored pairs to prevent "good vs bad debate" repetition
        contradiction_queries = []
        for topic, _ in candidates:
            if topic in self._contradiction_map:
                for ant in self._contradiction_map[topic]:
                    pair = (topic, ant) if topic < ant else (ant, topic)
                    if pair in self._explored_contradictions:
                        continue
                    targeted = self._generate_curiosity_query(
                        topic, source_type="contradiction", antonym=ant)
                    if targeted != topic:
                        contradiction_queries.append((targeted, pair))
        # Queue contradiction-specific queries directly
        for cq, pair in contradiction_queries[:2]:
            if self._bg_learning_active and cq not in self._bg_learning_queue:
                self.queue_background_search(cq)
                self._explored_contradictions.add(pair)

        # Source 5: Random graph walk from high-degree hubs (serendipity)
        if len(self.graph.nodes) > 0:
            degree_counts: Dict[int, int] = {}
            for src, tgt in list(self.graph.edges):
                degree_counts[src] = degree_counts.get(src, 0) + 1
                degree_counts[tgt] = degree_counts.get(tgt, 0) + 1
            if degree_counts:
                top_hub_id = max(degree_counts, key=degree_counts.get)
                hub_node = self.graph.get_node(top_hub_id)
                if hub_node and hub_node.label:
                    current_id = top_hub_id
                    for _ in range(2):
                        edges = list(self.graph.get_outgoing(current_id))
                        if not edges:
                            break
                        next_id = self.rng.choice([e[0] for e in edges]) if len(edges) > 1 else edges[0][0]
                        next_node = self.graph.get_node(next_id)
                        if next_node and next_node.label:
                            lbl = next_node.label.lower()
                            if lbl not in seen_topics and len(lbl) >= 3:
                                candidates.append((lbl, 0.8))
                                seen_topics.add(lbl)
                            current_id = next_id

        # Phase 18b: Boost candidates matching user's recent interests (priming)
        if self._user_query_topics:
            for i in range(len(candidates)):
                topic, weight = candidates[i]
                # Check if topic or any part matches user's recent queries
                for uq in self._user_query_topics[-3:]:  # last 3 topics
                    if uq in topic or topic in uq:
                        candidates[i] = (topic, min(2.0, weight * 1.5))  # 1.5x boost (capped at 2.0)
                        break
                    # Also check if the topic is connected to user's interests via graph
                    uq_nids = self._concept_keywords.get(uq, [])
                    topic_nids = self._concept_keywords.get(topic, [])
                    if uq_nids and topic_nids:
                        for un in uq_nids:
                            for tn in topic_nids:
                                if self.graph.get_edge(un, tn) or self.graph.get_edge(tn, un):
                                    candidates[i] = (topic, min(1.5, weight * 1.3))  # 1.3x boost (capped at 1.5)
                                    break

        # Initialize diversity tracking
        if not hasattr(self, '_recent_curiosity_selections'):
            self._recent_curiosity_selections = []  # list of (topic, turn_count)
        if not hasattr(self, '_curiosity_selection_cooldown'):
            self._curiosity_selection_cooldown = 10  # suppress selected topics for N turns (increased from 5)
        if not hasattr(self, '_bg_learning_cycles'):
            self._bg_learning_cycles = 0
        # Clear old selections from different turn domains (chat vs background)
        # This prevents old chat turns from permanently blocking background selections
        current_domain_turn = self._bg_learning_cycles if self._bg_learning_cycles > 0 else self.turn_count
        if self._recent_curiosity_selections:
            # Check if we're in a different turn domain than existing entries
            latest_turn = self._recent_curiosity_selections[-1][1]
            # If the latest turn is in the "other" domain (e.g., chat turns 40+ vs background 0+),
            # clear the history to allow fresh selections
            domain_threshold = 20  # turns threshold to consider different domains
            if latest_turn > domain_threshold and current_domain_turn <= domain_threshold:
                self._recent_curiosity_selections = []

        # ----- DIVERSITY: Penalize recently selected topics -----
        # Use background cycle count if turn_count isn't advancing
        current_turn = self._bg_learning_cycles if self._bg_learning_cycles > 0 else self.turn_count
        self._bg_learning_cycles += 1
        # Clean old entries
        self._recent_curiosity_selections = [
            (t, turn) for t, turn in self._recent_curiosity_selections
            if current_turn - turn < self._curiosity_selection_cooldown
        ]
        recently_selected = {t for t, _ in self._recent_curiosity_selections}
        
        # Penalize recently selected topics
        penalized_candidates = []
        for topic, weight in candidates:
            if topic in recently_selected:
                # Decay weight exponentially based on recency
                for t, turn in self._recent_curiosity_selections:
                    if t == topic:
                        recency = current_turn - turn
                        # Stronger penalty: 0.2^recency instead of 0.3^recency
                        penalty = 0.2 ** recency  # 0.2, 0.04, 0.008...
                        weight = weight * penalty
                        break
            penalized_candidates.append((topic, weight))
        candidates = penalized_candidates

        # EPSILON-GREEDY: 35% chance to pick random unvisited concept (pure exploration)
        epsilon = 0.35
        unvisited_labels = [l for l in all_labels if l not in seen_topics and len(l) >= 3]
        if unvisited_labels and self.rng.random() < epsilon:
            random_topic = self.rng.choice(unvisited_labels)
            candidates.append((random_topic, 0.5))  # low weight but could be picked
            if self._trace_enabled:
                print(f"  [curiosity] epsilon-exploration: {random_topic}")

        candidates.sort(key=lambda x: -x[1])
        selected = [topic for topic, _ in candidates[:max_topics]]

        # Track selections
        for topic in selected:
            self._recent_curiosity_selections.append((topic, current_turn))

        # For sync mode (max_topics=0), still track the top candidate for diversity
        if max_topics == 0 and candidates:
            top_topic = candidates[0][0]
            self._recent_curiosity_selections.append((top_topic, current_turn))

        if selected and self._trace_enabled:
            weights_str = ", ".join(f"{t}({w:.1f})" for t, w in candidates[:max_topics])
            print(f"  [curiosity] auto-selected: {weights_str} (urgency={self._curiosity_urgency:.2f})")

        self._last_auto_learn_turn = self.turn_count
        return selected


