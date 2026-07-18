# Web Learner - RAVANA's web search and autonomous learning.
# Contains: SearchEngine (multi-API with circuit breaker), background learning,
# curiosity-driven topic selection, article fetching.
# M0 crash-hardening: pin BLAS/OpenMP threads to 1 BEFORE numpy import (numpy
# #27989) so this module's fetch workers can't race the main-thread decoder
# inside BLAS and trigger a Windows access violation. Must precede `import
# numpy as np` below.
import ravana._numpy_threading  # noqa: F401  (side-effect: thread + faulthandler setup)
import sys
import os
import time
import json
import re
import hashlib
import urllib.request
from urllib.parse import quote
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set, Tuple
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed


# Research item E: provenance population helper (TrustGraph / PROV-O style).
try:
    from ..chat.provenance import populate_provenance
except Exception:  # pragma: no cover - defensive
    populate_provenance = None


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

# Optional trafilatura for structured web extraction (roadmap #11)
try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False


@dataclass
class SearchConfig:
    """Configuration for search engine."""
    max_results: int = 10
    timeout: int = 5
    # Circuit breaker: with local_api as the SOLE backend, a hard 60s
    # blackout after a few transient blips would black out the whole session
    # (nothing to fall back to). Tune for resilience: tolerate a small cluster
    # of transient failures (max_failures) and recover FAST via the half-open
    # probe (short cooldown) so a tripped breaker self-heals in seconds, not a
    # minute. Per-call deadline-bounded retry (local_retries) absorbs most
    # one-off blips before they ever reach the breaker.
    cooldown: int = 8
    max_failures: int = 5
    # Phase 19e: ALWAYS prefer the local engine (your SearXNG on localhost:4000).
    # It may be slow at times, but it's the authoritative source and should be
    # awaited up to local_timeout seconds. Remote APIs are only consulted if the
    # local engine is genuinely UNAVAILABLE (circuit breaker / exception / stall
    # past local_timeout) — never merely because it was slow-but-returned, and
    # never as a supplement when local already answered (even with empty results).
    local_prefer: bool = True
    local_timeout: int = 10
    # Transient-retry count for the preferred local_api. local_api's
    # intermittent stall/remote-close is almost always a one-off; a single
    # immediate retry absorbs the blip (verified: 15/15 sequential calls
    # succeed) so it doesn't trip the circuit breaker. 1 = one retry (2 total
    # attempts). Set 0 to disable retries.
    local_retries: int = 1
    # D (research item D): fail-closed retrieval. When local returns an EMPTY
    # result (e.g. SearXNG up but returning junk/zero hits), fall through to the
    # remote fallbacks (DDG/oxiverse) instead of treating empty as authoritative.
    # This closes the silent-failure root cause where a known subject with hollow
    # graph edges is answered from the graph instead of being web-searched.
    fallback_on_empty: bool = True


class SearchError(Exception):
    """Raised when all search APIs fail."""
    pass


class SearchEngine:
    """Multi-API search engine with circuit breaker fallback."""

    def __init__(self, config: Optional[SearchConfig] = None):
        self.config = config or SearchConfig()
        self.apis = [
            # Local search engine is the ONLY configured backend: awaited up to
            # local_timeout seconds (Phase 19e). It is the authoritative source.
            # duckduckgo + oxiverse were removed: duckduckgo is unreachable in
            # this environment (times out, burning ~5s per dead call) and
            # oxiverse returns 401 (no auth token wired in). Re-add them here
            # only once they are reachable / authenticated — until then they
            # only poison the circuit breaker and waste per-turn latency.
            ("local_api", "http://localhost:4000/search?q={}", self.config.local_timeout, 10),
        ]
        self._api_failure_counts = {name: 0 for name, _, _, _ in self.apis}
        self._api_last_failure_time = {name: 0 for name, _, _, _ in self.apis}
        # Half-open probe slots for the circuit breaker (see _is_api_available).
        # True while a single post-cooldown probe is outstanding for an API.
        self._api_half_open = {name: False for name, _, _, _ in self.apis}
        self._headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 RAVANA/1.0'
        }
        # Per-session search cache: the generation pipeline (decomposition +
        # web-direct-answer) issues the SAME query many times per turn
        # (sub-questions x variants x retry). Caching identical searches within
        # a session collapses ~18 network round-trips into a handful. Keyed by
        # (normalized term, local_only, max_results). Bounded; cleared per call
        # to self.clear_search_cache() (call at the start of each process_turn).
        self._search_cache: Dict[tuple, List[Dict[str, Any]]] = {}
        self._search_cache_max = 64

    def _is_api_available(self, api_name: str) -> bool:
        """Circuit-breaker availability with a HALF-OPEN state.

        Closed:   failure_count < max_failures  -> available.
        Open:     failure_count >= max_failures AND cooldown not elapsed -> unavailable.
        Half-open: failure_count >= max_failures AND cooldown elapsed ->
                   allow EXACTLY ONE probe through (consume the half-open slot).
                   A success resets the breaker; a failure re-opens it (resets
                   the cooldown clock). This prevents the old behaviour where a
                   few transient blips hard-disabled the only working backend
                   for the full 60s cooldown, blacking out the whole session.
        """
        if self._api_failure_counts[api_name] < self.config.max_failures:
            return True
        elapsed = time.time() - self._api_last_failure_time[api_name]
        if elapsed > self.config.cooldown:
            # Cooldown elapsed -> grant a single half-open probe. Mark it
            # consumed so a second concurrent check in the same window doesn't
            # also slip through; the probe's own success/failure re-decides.
            if not self._api_half_open.get(api_name, False):
                self._api_half_open[api_name] = True
                return True
            # A half-open probe is already outstanding -> treat as still open
            # until that probe reports.
            return False
        return False

    def _record_success(self, api_name: str):
        self._api_failure_counts[api_name] = 0
        self._api_half_open[api_name] = False

    def _record_failure(self, api_name: str):
        self._api_failure_counts[api_name] += 1
        self._api_last_failure_time[api_name] = time.time()
        # A failed half-open probe re-opens the breaker (cooldown clock resets);
        # a failed closed-state attempt just increments toward the threshold.
        self._api_half_open[api_name] = False

    def clear_search_cache(self):
        """Drop cached search results. Call at the start of each user turn so
        the cache reflects only the current turn's queries (not stale results
        from many turns ago)."""
        self._search_cache.clear()

    def _threaded_fetch(self, api_name: str, url: str, timeout: int, max_results: int):
        """Fetch one API in a daemon thread and join with a hard ``timeout``.

        Phase 19c: urllib's own ``urlopen(timeout=...)`` can be defeated on some
        stacks (a connection that connects but then stalls on the response
        status line), leaving ``socket.readinto`` blocked far past the requested
        timeout. Running the fetch in a thread and ``join(timeout)`` guarantees
        the call can never exceed ``timeout`` seconds, so the turn can't hang.

        Returns ``(thread, result_dict)`` where ``result_dict`` is either
        ``{'v': <list>}`` on success or ``{'err': <exception>}`` on failure.
        If the thread is still alive after the join, the fetch stalled — the
        caller should treat it as a failure and consult the next API.
        """
        _fetch_res = {}

        def _do_fetch():
            try:
                _fetch_res['v'] = self._call_api(api_name, url, timeout, max_results)
            except Exception as _fe:  # noqa: BLE001 - we re-raise via dict
                _fetch_res['err'] = _fe

        _ft = threading.Thread(target=_do_fetch, daemon=True)
        _ft.start()
        _ft.join(timeout)
        return _ft, _fetch_res

    def search(self, query: str, max_results: int = 10,
                local_only: bool = False) -> List[Dict[str, Any]]:
        """Search with automatic fallback across APIs.

        When ``local_only`` is True (used when remote connectivity is down),
        only the local search engine is consulted — remote APIs are skipped so
        a working local engine is never blocked by the offline circuit breaker.
        """
        # Per-session cache: identical (term, local_only, max_results) requests
        # within a turn return the cached list instead of hitting the network.
        _cache_key = (query.strip().lower(), bool(local_only), int(max_results))
        if _cache_key in self._search_cache:
            return self._search_cache[_cache_key]

        # Phase 19c: Hard wall-clock deadline for the ENTIRE search() call.
        # The per-API `urlopen(timeout=...)` can be defeated on some stacks
        # (e.g. a connection that connects but then stalls on the response
        # status line) — observed as a 40s+ hang inside socket.readinto even
        # though the per-call timeout was 5s. A turn-level deadline guarantees
        # search() can NEVER block the user turn longer than this, regardless
        # of per-API timeout quirks. On deadline we return whatever we have
        # (possibly empty) and let the caller fall back to stored knowledge.
        _search_deadline = time.time() + float(getattr(self.config, 'search_deadline', 8.0))

        errors = []

        # Phase 19e (supersedes 19b): local is ALWAYS preferred. The block below
        # consults local_api first and returns its answer verbatim — including an
        # EMPTY list — without ever falling through to remote as a "supplement".
        # Remote fallbacks are only reached if local_api is genuinely UNAVAILABLE
        # (circuit breaker / exception / stall past local_timeout). So an empty
        # local result is treated as authoritative and remote is not consulted.
        # This is exactly the "always prefer local, even when slow" rule.

        # Definitional suffixes (e.g. " definition meaning", " explained overview")
        # improve recall on remote engines (DuckDuckGo/oxiverse) but pollute the

        # definition meaning" returns pages about "explanation"). Strip them when
        # the query is destined for the local engine so it gets a clean subject.
        _LOCAL_SUFFIXES = (
            " definition meaning explained", " definition meaning",
            " explained overview", " explained with examples", " explained",
        )

        # ── Phase 19e: ALWAYS PREFER LOCAL ──────────────────────────────────
        # The local engine (your SearXNG on localhost:4000) is the authoritative
        # source. We await it up to local_timeout seconds even when it's slow.
        # We only consult remote fallbacks if the local engine is GENUINELY
        # unavailable (circuit breaker / exception / stall past local_timeout).
        # If local answers — even with an empty result — we commit to it and
        # never query remote as a "supplement". Remote is a last resort, not a
        # parallel/backup source.
        if self.config.local_prefer and not local_only:
            # local_only callers already force local; this branch handles the
            # general case where remote COULD be used but we prefer local.
            _local_available = self._is_api_available("local_api")
            if not _local_available:
                # Preferred-local path skipped because the breaker is open.
                # Record a clear reason so a fully-breaker-tripped session
                # reports "circuit-breaker open" instead of a misleading empty
                # "All search APIs failed: ".
                if self._api_failure_counts["local_api"] >= self.config.max_failures:
                    _reason = (f"local_api: circuit-breaker open "
                               f"({self._api_failure_counts['local_api']} failures; "
                               f"retry after cooldown)")
                    if _reason not in errors:
                        errors.append(_reason)
                # fall through to the remote fallback loop below (which will
                # also skip local_api, but the reason is now recorded).

            if _local_available:
                api_query = query
                ql = query.lower()
                for suf in _LOCAL_SUFFIXES:
                    if ql.endswith(suf):
                        api_query = query[: len(query) - len(suf)].strip()
                        break
                query_encoded = quote(api_query)
                url = "http://localhost:4000/search?q={}".format(query_encoded)
                # Transient-retry: local_api's intermittent stall/remote-close
                # (~15-20% of requests under load) is almost always a one-off —
                # the very next attempt usually succeeds (verified: 15/15
                # sequential calls succeed). A bounded retry absorbs these
                # blips so they don't trip the circuit breaker and black out
                # the only working backend. CRITICAL: the retry must respect the
                # turn-level deadline — split the REMAINING budget across
                # attempts so 2 tries can never exceed the deadline (otherwise a
                # single stall costs 2x local_timeout and blows the whole turn).
                _fetch_res = None
                _ft = None
                _n_attempts = 1 + self.config.local_retries
                for _attempt in range(_n_attempts):
                    _budget_left = _search_deadline - time.time()
                    if _budget_left <= 0.5:
                        break  # no time left for another attempt
                    # Each attempt gets an equal slice of the remaining budget
                    # (capped at local_timeout), so the loop self-terminates.
                    _att_timeout = min(self.config.local_timeout,
                                       max(1.0, _budget_left / (_n_attempts - _attempt)))
                    _ft, _fetch_res = self._threaded_fetch(
                        "local_api", url, int(_att_timeout), max_results)
                    if not _ft.is_alive() and 'err' not in _fetch_res:
                        break  # success or empty — no need to retry
                    # transient (stall or error): retry if budget + attempts remain
                if _ft is not None and _ft.is_alive():
                    self._record_failure("local_api")
                    errors.append("local_api: fetch stalled past "
                                  f"{self.config.local_timeout}s")
                elif _fetch_res is not None and 'err' in _fetch_res:
                    self._record_failure("local_api")
                    errors.append(f"local_api: {_fetch_res['err']}")
                else:
                    results = _fetch_res.get('v')
                    self._record_success("local_api")
                    out = (results or [])[:max_results]
                    # D (research item D): fail-closed retrieval.
                    # If local returned EMPTY and fallback_on_empty is set,
                    # do NOT treat empty as authoritative — fall through to
                    # the remote fallbacks below (respecting the circuit
                    # breaker and the turn-level deadline). local_only
                    # callers still commit to local even when empty.
                    if out or not self.config.fallback_on_empty or local_only:
                            if len(self._search_cache) < self._search_cache_max:
                                self._search_cache[_cache_key] = out
                            return out
                            # NOTE: we return local's answer (incl. empty) as
                            # final. Remote is NOT consulted — local is
                            # authoritative — UNLESS fallback_on_empty is set and
                            # the result was empty.
                # If we reach here, local genuinely failed → fall through to
                # remote fallbacks below.

        for api_name, url_template, timeout, api_max_results in self.apis:
            # Phase 19c: bail out if we've exceeded the turn-level deadline.
            if time.time() > _search_deadline:
                break
            if api_name == "local_api":
                # Already handled (and preferred) above when local_prefer is on
                # and we're allowing remote. But when local_only=True, local_api
                # is the ONLY allowed source — so it must NOT be skipped here.
                if self.config.local_prefer and not local_only:
                    continue
                if local_only:
                    # fall through to consult local_api below
                    pass
                else:
                    continue
            if local_only and api_name != "local_api":
                continue

            if not self._is_api_available(api_name):
                # Diagnostic: record WHY this API was skipped so a fully
                # breaker-tripped session reports "circuit-breaker open" instead
                # of a misleading empty "All search APIs failed: ".
                if self._api_failure_counts[api_name] >= self.config.max_failures:
                    _reason = (f"{api_name}: circuit-breaker open "
                               f"({self._api_failure_counts[api_name]} failures; "
                               f"retry after cooldown)")
                    if _reason not in errors:
                        errors.append(_reason)
                continue

            api_query = query
            query_encoded = quote(api_query)

            try:
                url = url_template.format(query_encoded)
                # Phase 19c: thread+join hard cap on a stalled fetch.
                _ft, _fetch_res = self._threaded_fetch(
                    api_name, url, timeout, max_results)
                if _ft.is_alive():
                    self._record_failure(api_name)
                    errors.append(f"{api_name}: fetch stalled past {timeout}s")
                    continue
                if 'err' in _fetch_res:
                    raise _fetch_res['err']
                results = _fetch_res.get('v')
                if results:
                    self._record_success(api_name)
                    out = results[:max_results]
                    if len(self._search_cache) < self._search_cache_max:
                        self._search_cache[_cache_key] = out
                    return out
            except Exception as e:
                self._record_failure(api_name)
                errors.append(f"{api_name}: {e}")
                continue

        raise SearchError(f"All search APIs failed: {'; '.join(errors)}")

    def _call_api(self, api_name: str, url: str, timeout: int, max_results: int) -> List[Dict[str, Any]]:
        # HTTPS endpoints (e.g. the oxiverse ecosystem API) target a host whose
        # TLS cert chain Python's default ssl context can't verify in this
        # environment (CERTIFICATE_VERIFY_FAILED). For a first-party aggregator
        # API this is a trust-store gap, not a MITM — open an unverified
        # context so the request can complete. HTTP (local_api, duckduckgo html)
        # is untouched.
        import ssl
        _ctx = None
        if url.startswith("https"):
            try:
                _ctx = ssl._create_unverified_context()
            except Exception:
                _ctx = None
        req = urllib.request.Request(url, headers=self._headers)
        if _ctx is not None:
            resp = urllib.request.urlopen(req, timeout=timeout, context=_ctx)
        else:
            resp = urllib.request.urlopen(req, timeout=timeout)
        content = resp.read().decode('utf-8', errors='replace')

        if api_name == "local_api":
            data = json.loads(content)
            results = data if isinstance(data, list) else data.get('results', [])
            return [
                {'title': r.get('title', ''), 'url': r.get('url', ''),
                 'content': r.get('content', r.get('snippet', ''))}
                for r in results[:max_results] if r.get('url')
            ]
        elif api_name == "oxiverse":
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
        # C-lite: structured fact acquisition (web -> OpenIE -> typed graph)
        self._web_to_graph = None

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

            # trafilatura: statistical heuristics, best quality (roadmap #11)
            if HAS_TRAFILATURA:
                text = trafilatura.extract(html, output_format='txt', favor_precision=True)
                if text and len(text) > 50:
                    return text[:5000]

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

    # Web garbage words that should never become graph concepts
    # These come from HTML/CSS/JS, URLs, and programming artifacts
    WEB_GARBAGE: Set[str] = {
        # HTML/CSS artifacts
        'html', 'css', 'js', 'div', 'span', 'class', 'style', 'script', 'meta',
        'href', 'src', 'img', 'br', 'hr', 'head', 'body', 'font', 'nav',
        'header', 'footer', 'section', 'article', 'aside', 'main',
        'width', 'height', 'color', 'margin', 'padding', 'border',
        'background', 'display', 'position', 'float', 'clear', 'overflow',
        'block', 'inline', 'flex', 'grid', 'align', 'justify', 'content',
        # URL/web artifacts
        'http', 'https', 'www', 'com', 'org', 'net', 'io', 'gov', 'edu',
        'url', 'uri', 'link', 'href', 'src', 'domain', 'cookie',
        'query', 'params', 'token', 'oauth', 'api', 'json', 'xml',
        # JavaScript/programming
        'var', 'let', 'const', 'func', 'function', 'return', 'import',
        'export', 'module', 'require', 'console', 'log', 'debug',
        'undefined', 'null', 'true', 'false', 'typeof', 'instanceof',
        'array', 'object', 'string', 'number', 'boolean', 'promise',
        'async', 'await', 'callback', 'event', 'listener', 'handler',
        'prototype', 'constructor', 'length', 'index', 'value',
        # Common tech/platform names that are not general English words
        'gform', 'wordpress', 'joomla', 'drupal', 'shopify', 'squarespace',
        'wix', 'webflow', 'github', 'gitlab', 'bitbucket', 'heroku',
        'netlify', 'vercel', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
        'react', 'angular', 'vue', 'svelte', 'jquery', 'bootstrap',
        'tailwind', 'sass', 'less', 'webpack', 'vite', 'eslint', 'prettier',
        # Analytics/tracking
        'analytics', 'tracking', 'pixel', 'gtag', 'gaq', 'analytics',
        'utm', 'campaign', 'click', 'impression', 'conversion',
        # Date/time patterns that appear as words
        'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
        'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
    }

    def _learn_from_text(self, text: str, topic: str, source_url: str = "") -> int:
        """Extract keywords, add concepts, form connections, train decoder."""
        words = re.findall(r"[a-zA-Z']{3,}", text.lower())
        word_counts = {}
        for w in words:
            wc = w.strip("'")
            if wc not in STOP_WORDS and len(wc) >= 3 and wc not in self.WEB_GARBAGE:
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
                        e = self.graph_engine.graph.add_edge(nid1, nid2, weight=weight, relation_type="semantic")
                        # Provenance (item E): tag co-occurrence edges so pruning
                        # can tell noisy web co-occurrence from verified web facts.
                        if e is not None and populate_provenance is not None:
                            populate_provenance(e, edge_kind="co_occurrence",
                                                source=source_id, method="web_cooccurrence")
                        elif e is not None and hasattr(e, "source_metadata"):
                            e.source_metadata.update({"source": source_id,
                                                      "edge_kind": "co_occurrence",
                                                      "relation": "semantic"})
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
                        e2 = self.graph_engine.graph.add_edge(nid, existing_nid, weight=weight, relation_type=inf_type)
                        if e2 is not None and populate_provenance is not None:
                            populate_provenance(e2, edge_kind="co_occurrence",
                                                source=source_id, method="web_cooccurrence")
                        elif e2 is not None and hasattr(e2, "source_metadata"):
                            e2.source_metadata.update({"source": source_id,
                                                       "edge_kind": "co_occurrence",
                                                       "relation": inf_type})

        # Train neural decoder
        if self.decoder_engine.neural_decoder and self.decoder_engine._decoder_vocab_built:
            cond_embs = self._build_conditioning(topic, article_words)
            new_labels = [n.label for nid, n in self.graph_engine.graph.nodes.items()
                          if n.label and n.label.lower() not in self.decoder_engine._decoder_word_to_idx
                          and n.label.lower() not in ('<pad>', '?', '<bos>', '<eos>')]
            if new_labels:
                self.decoder_engine._expand_vocab(new_labels)
            self.decoder_engine.train_on_text(text, topic, self.graph_engine, self._glove_vector, cond_embs, passes=10)

        # C-lite: structured fact acquisition (web -> OpenIE -> typed graph).
        # Additive pass AFTER the co-occurrence edges above: facts become typed
        # is_a/has_property/causes/located_in edges with provenance, NOT vectors.
        # No dimensionality change (the graph is already a KG). The thin EFE
        # knowledge-gap interface is exercised here to feed the future E spine.
        try:
            if self._web_to_graph is None:
                from ravana.web.web_to_graph import WebToGraph
                self._web_to_graph = WebToGraph(self.graph_engine, source=source_url or topic)
            facts = self._web_to_graph.learn_text(text, source_url=source_url)
            if facts:
                # record the topic as a curiosity-resolved node (gap shrinks)
                self._web_to_graph._topic_edges[topic.lower().strip()] = \
                    self._web_to_graph._topic_edges.get(topic.lower().strip(), 0) + facts
        except Exception:
            # Fact acquisition must never break web learning.
            facts = 0

        return new_count

    def knowledge_gap(self, topic: str):
        """Thin EFE interface (sketch for E): how uncertain are we about `topic`?

        Returns a KnowledgeGap; high efe => curiosity target. Built at C-time,
        consumed by the full active-inference control spine later (E).
        """
        if self._web_to_graph is None:
            from ravana.web.web_to_graph import WebToGraph
            self._web_to_graph = WebToGraph(self.graph_engine)
        return self._web_to_graph.knowledge_gap(topic)

    def last_fact_count(self) -> int:
        """Number of structured facts written by the C-lite pass (this session)."""
        return self._web_to_graph.fact_count() if self._web_to_graph else 0

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

    def curiosity_e_step(self, candidate_topics: Optional[List[str]] = None) -> Optional[str]:
        """E: active-inference control step.

        Builds an ActiveInferenceController from the thin C-time KnowledgeGap
        EFE interface (C-lite) + the existing node prediction_free_energy, and
        returns the argmax-EFE topic = the best epistemic action (curious web
        read). ADDITIVE: this does not replace _auto_select_curiosity_topics;
        it is the principled unification of the uncertainty signals that engine
        already gathers. Returns None if there is nothing uncertain to ask.

        The epistemic-action loop is closed by the caller: after reading about
        the returned topic, its EFE (via knowledge_gap) drops, confirming the
        uncertainty was reduced (Friston 2015; Schmidhuber 2010).
        """
        try:
            from ravana.core.active_inference import ActiveInferenceController

            def gap_fn(topic):
                return self.knowledge_gap(topic)

            def pfe_fn(topic):
                nid = self.graph_engine._all_labels.get(topic.lower().strip())
                if nid is None:
                    return 0.0
                node = self.graph_engine.graph.get_node(nid)
                if node is None:
                    return 0.0
                return float(getattr(node, "prediction_free_energy", 0.0) or 0.0)

            cands = candidate_topics
            if not cands:
                # default candidate pool: all known labels + user-query topics
                cands = list(self.graph_engine._all_labels.keys())
                if self._user_query_topics:
                    cands = cands + [t for t in self._user_query_topics if t]
            if not cands:
                return None

            ctrl = ActiveInferenceController(gap_fn=gap_fn, pfe_fn=pfe_fn)
            target = ctrl.select_target(cands)
            if target is None or ctrl.score(target).total <= 0.0:
                return None
            return target
        except Exception:
            # E must never break the existing curiosity engine.
            return None

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

    def _seed_from_graph_curiosity(self, max_topics: int = 8) -> int:
        """Seed background learning queue from graph's curiosity signals."""
        if not self._curiosity_drive_enabled:
            return 0
        selected = self._auto_select_curiosity_topics(max_topics=max_topics)
        if self._trace_enabled:
            print(f"  [seed] Seeded {len(selected)} topics from curiosity: {selected}")
        return len(selected)