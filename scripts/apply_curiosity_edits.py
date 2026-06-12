#!/usr/bin/env python3
"""Apply Curiosity Drive edits to scripts/ravana_chat.py."""
import re

with open('scripts/ravana_chat.py', 'r', encoding='utf-8') as f:
    code = f.read()

# 1. Add curiosity fields after _bg_multi_search_max
old1 = '        self._bg_multi_search_max: int = 3  # related searches to expand per query\n\n\n        if os.path.exists(self._save_path):'
new1 = '''        self._bg_multi_search_max: int = 3  # related searches to expand per query

        # Phase 18: Curiosity Drive - autonomous topic selection for background learning
        self._curiosity_drive_enabled: bool = True  # can be disabled via --no-curiosity
        self._concept_visit_count: Dict[str, int] = {}  # how many times each concept was visited
        self._concept_learning_progress: Dict[str, float] = {}  # rate of prediction error decrease per concept
        self._curiosity_topics_queue: List[str] = []  # topics autonomously selected for research
        self._last_auto_learn_turn: int = 0  # turn when we last autonomously selected topics
        self._curiosity_urgency: float = 0.0  # overall curiosity drive (0-1)


        if os.path.exists(self._save_path):'''

if old1 in code:
    code = code.replace(old1, new1, 1)
    print('1. Added curiosity fields to __init__')
else:
    print('1. FAILED - Could not find anchor for curiosity fields')
    idx = code.find('_bg_multi_search_max')
    if idx >= 0:
        print(f'   Found at position {idx}')
        print(f'   Context: {repr(code[idx:idx+200])}')

# 2. Add curiosity methods after notify_user_idle
old2 = '    def notify_user_idle(self):\n        """Signal that the user is idle (resume background learning)."""\n        self._bg_idle_event.set()  # thread will wake up and process'
new2 = '''    def notify_user_idle(self):
        """Signal that the user is idle (resume background learning)."""
        self._bg_idle_event.set()  # thread will wake up and process

    # --- Phase 18: Curiosity Drive - Autonomous Topic Selection ---
    # Based on: Berlyne epistemic curiosity, Loewenstein information-gap theory,
    # Friston Active Inference (prediction error minimization), Oudeyer learning progress.

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

        # 3. Arousal modulation (emotional engagement amplifies curiosity)
        arousal_boost = self.emotion.state.arousal * 0.2
        urgency += arousal_boost

        # 4. Identity uncertainty (low identity = more exploration)
        identity_curiosity = (1.0 - self.identity.state.strength) * 0.15
        urgency += identity_curiosity

        # 5. Low-confidence concepts in the graph
        low_conf_count = sum(1 for c in self._concept_confidence.values() if c < 0.3)
        if low_conf_count > 0:
            urgency += min(0.2, low_conf_count * 0.01)

        self._curiosity_urgency = min(1.0, urgency)
        return self._curiosity_urgency

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

        # Source 2: High prediction error concepts (weight 3x)
        high_pe_nodes = []
        for nid, node in self.graph.nodes.items():
            if node.label and node.prediction_free_energy > 0.3:
                label = node.label.lower()
                if label not in seen_topics and len(label) >= 3:
                    high_pe_nodes.append((label, node.prediction_free_energy))
        high_pe_nodes.sort(key=lambda x: -x[1])
        for label, pe in high_pe_nodes[:5]:
            candidates.append((label, 3.0 * min(1.0, pe)))
            seen_topics.add(label)

        # Source 3: Contradiction pairs (weight 3x)
        for concept, antonyms in self._contradiction_map.items():
            if concept not in seen_topics and len(concept) >= 3:
                candidates.append((concept, 3.0))
                seen_topics.add(concept)
            for ant in antonyms:
                if ant not in seen_topics and len(ant) >= 3:
                    candidates.append((ant, 2.5))
                    seen_topics.add(ant)

        # Source 4: Least-visited concepts (novelty, weight inversely proportional to visits)
        all_labels = set()
        for nid, node in self.graph.nodes.items():
            if node.label:
                all_labels.add(node.label.lower())
        unvisited = [l for l in all_labels if l not in seen_topics and len(l) >= 3]
        unvisited.sort(key=lambda l: self._concept_visit_count.get(l, 0))
        for label in unvisited[:3]:
            visits = self._concept_visit_count.get(label, 0)
            novelty_weight = 1.0 / (1.0 + visits)
            candidates.append((label, novelty_weight))
            seen_topics.add(label)

        # Source 5: Random graph walk from high-degree hubs (serendipity)
        if len(self.graph.nodes) > 0:
            degree_counts: Dict[int, int] = {}
            for src, tgt in self.graph.edges:
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

        candidates.sort(key=lambda x: -x[1])
        selected = [topic for topic, _ in candidates[:max_topics]]

        if selected and self._trace_enabled:
            weights_str = ", ".join(f"{t}({w:.1f})" for t, w in candidates[:max_topics])
            print(f"  [curiosity] auto-selected: {weights_str} (urgency={self._curiosity_urgency:.2f})")

        for topic in selected:
            self.queue_background_search(topic)

        self._last_auto_learn_turn = self.turn_count
        return selected'''

if old2 in code:
    code = code.replace(old2, new2, 1)
    print('2. Added curiosity methods after notify_user_idle')
else:
    print('2. FAILED - Could not find notify_user_idle anchor')

# 3. Modify _bg_learn_loop to call _auto_select_curiosity_topics when queue is empty
old3 = '''            all_queries = queries_to_process + deferred
            try:
                for query in all_queries:
                    if not self._bg_learning_active:
                        break
                    self._bg_multi_search(query)
                # Save periodically after learning
                if all_queries:
                    try:
                        self.save()
                    except Exception:
                        pass'''

new3 = '''            all_queries = queries_to_process + deferred
            # Phase 18: If queue is empty, autonomously select curiosity topics
            if not all_queries and self._curiosity_drive_enabled:
                curiosity_topics = self._auto_select_curiosity_topics(max_topics=2)
                with self._bg_lock:
                    all_queries = list(self._bg_learning_queue)
                    self._bg_learning_queue.clear()
            try:
                for query in all_queries:
                    if not self._bg_learning_active:
                        break
                    self._bg_multi_search(query)
                # Save periodically after learning
                if all_queries:
                    try:
                        self.save()
                    except Exception:
                        pass'''

if old3 in code:
    code = code.replace(old3, new3, 1)
    print('3. Modified _bg_learn_loop for curiosity topic selection')
else:
    print('3. FAILED - Could not find _bg_learn_loop anchor')

# 4. Modify process_turn to track concept visits and compute curiosity urgency
old4 = '        self.notify_user_idle()  # wake background thread after response\n\n        return response'
new4 = '''        # Phase 18: Track concept visits for curiosity/novelty scoring
        if subject:
            sl = subject.lower()
            self._concept_visit_count[sl] = self._concept_visit_count.get(sl, 0) + 1
        for label, _ in ctx.associated_concepts[:5]:
            ll = label.lower()
            self._concept_visit_count[ll] = self._concept_visit_count.get(ll, 0) + 1

        # Phase 18: Compute curiosity urgency for autonomous exploration
        self._compute_curiosity_urgency()

        self.notify_user_idle()  # wake background thread after response

        return response'''

if old4 in code:
    code = code.replace(old4, new4, 1)
    print('4. Modified process_turn for concept visit tracking')
else:
    print('4. FAILED - Could not find process_turn anchor')
    idx = code.find('notify_user_idle()  # wake background thread')
    if idx >= 0:
        print(f'   Found at position {idx}')
        print(f'   Context: {repr(code[idx-50:idx+150])}')

# 5. Add curiosity state to save()
old5 = "            'calibration_error': self._calibration_error,\n            'metacognitive_review_turn': self._metacognitive_review_turn,\n            # Background learning"
new5 = """            'calibration_error': self._calibration_error,
            'metacognitive_review_turn': self._metacognitive_review_turn,
            # Curiosity Drive state
            'curiosity_drive_enabled': self._curiosity_drive_enabled,
            'concept_visit_count': self._concept_visit_count,
            'concept_learning_progress': self._concept_learning_progress,
            'curiosity_topics_queue': self._curiosity_topics_queue,
            'last_auto_learn_turn': self._last_auto_learn_turn,
            'curiosity_urgency': self._curiosity_urgency,
            # Background learning"""

if old5 in code:
    code = code.replace(old5, new5, 1)
    print('5. Added curiosity state to save()')
else:
    print('5. FAILED - Could not find save() anchor')

# 6. Restore curiosity state in _load()
old6 = "        self._calibration_error = state.get('calibration_error', 0.0)\n        self._metacognitive_review_turn = state.get('metacognitive_review_turn', 0)"
new6 = """        self._calibration_error = state.get('calibration_error', 0.0)
        self._metacognitive_review_turn = state.get('metacognitive_review_turn', 0)

        # Restore curiosity drive state
        self._curiosity_drive_enabled = state.get('curiosity_drive_enabled', True)
        self._concept_visit_count = state.get('concept_visit_count', {})
        self._concept_learning_progress = state.get('concept_learning_progress', {})
        self._curiosity_topics_queue = state.get('curiosity_topics_queue', [])
        self._last_auto_learn_turn = state.get('last_auto_learn_turn', 0)
        self._curiosity_urgency = state.get('curiosity_urgency', 0.0)"""

if old6 in code:
    code = code.replace(old6, new6, 1)
    print('6. Restored curiosity state in _load()')
else:
    print('6. FAILED - Could not find _load() anchor')

# 7. Add --no-curiosity CLI flag
old7 = "    parser.add_argument('--no-beliefs', action='store_true', help='Disable belief store')\n    parser.add_argument('--trace', action='store_true', help='Enable edge-level chain traces')"
new7 = """    parser.add_argument('--no-beliefs', action='store_true', help='Disable belief store')
    parser.add_argument('--no-curiosity', action='store_true', help='Disable autonomous curiosity-driven learning')
    parser.add_argument('--trace', action='store_true', help='Enable edge-level chain traces')"""

if old7 in code:
    code = code.replace(old7, new7, 1)
    print('7. Added --no-curiosity CLI flag')
else:
    print('7. FAILED - Could not find CLI flag anchor')
    idx = code.find('--no-beliefs')
    if idx >= 0:
        print(f'   Found --no-beliefs at position {idx}')
        print(f'   Context: {repr(code[idx:idx+200])}')

# 8. Apply the --no-curiosity flag in main()
old8 = "    if args.no_beliefs:\n        engine.use_beliefs = False\n    if args.trace:"
new8 = """    if args.no_beliefs:
        engine.use_beliefs = False
    if args.no_curiosity:
        engine._curiosity_drive_enabled = False
        print('  [Curiosity] Autonomous learning disabled')
    if args.trace:"""

if old8 in code:
    code = code.replace(old8, new8, 1)
    print('8. Applied --no-curiosity flag in main()')
else:
    print('8. FAILED - Could not find main() flag application anchor')

with open('scripts/ravana_chat.py', 'w', encoding='utf-8') as f:
    f.write(code)

print('\nAll edits complete!')
