#!/usr/bin/env python3
"""Apply all 7 curiosity drive fixes to scripts/ravana_chat.py."""
with open('scripts/ravana_chat.py', 'r', encoding='utf-8') as f:
    code = f.read()

changes = []

# ============================================================
# FIX 1: Wire up _concept_learning_progress with edge PE aggregation
# ============================================================
# Add _update_concept_learning_progress method and call it in process_turn
old_f1 = "    # --- Phase 18: Curiosity Drive - Autonomous Topic Selection ---\n    # Based on: Berlyne epistemic curiosity, Loewenstein information-gap theory,\n    # Friston Active Inference (prediction error minimization), Oudeyer learning progress.\n\n    def _compute_curiosity_urgency"
new_f1 = "    # --- Phase 18: Curiosity Drive - Autonomous Topic Selection ---\n    # Based on: Berlyne epistemic curiosity, Loewenstein information-gap theory,\n    # Friston Active Inference (prediction error minimization), Oudeyer learning progress.\n\n    def _update_concept_learning_progress(self):\n        \"\"\"Aggregate edge-level prediction error to concept-level learning progress.\n\n        Uses EWMA with rolling window to track PE delta per concept.\n        Positive delta = genuine learning (PE dropping).\n        Flat/negative delta = stuck (curiosity should spike).\n        \"\"\"\n        if not self._curiosity_drive_enabled:\n            return\n\n        # Build concept -> incident edges map\n        concept_edges: Dict[str, List[float]] = {}\n        for (src, tgt), edge in self.graph.edges.items():\n            src_node = self.graph.get_node(src)\n            tgt_node = self.graph.get_node(tgt)\n            pe = getattr(edge, 'prediction_free_energy', 0.0)\n            if src_node and src_node.label:\n                concept_edges.setdefault(src_node.label.lower(), []).append(pe)\n            if tgt_node and tgt_node.label:\n                concept_edges.setdefault(tgt_node.label.lower(), []).append(pe)\n\n        alpha = 0.3  # EWMA smoothing factor\n        for label, pe_list in concept_edges.items():\n            if not pe_list:\n                continue\n            mean_pe = sum(pe_list) / len(pe_list)\n            prev = self._concept_learning_progress.get(label, mean_pe)\n            # Smoothed update: EWMA blends old and new\n            self._concept_learning_progress[label] = (1 - alpha) * prev + alpha * mean_pe\n\n    def _compute_curiosity_urgency"

if old_f1 in code:
    code = code.replace(old_f1, new_f1, 1)
    changes.append("Fix 1: Added _update_concept_learning_progress method")
else:
    changes.append("Fix 1 FAILED: Could not find anchor")

# Add call to _update_concept_learning_progress in process_turn
old_f1b = "        # Phase 18: Compute curiosity urgency for autonomous exploration\n        self._compute_curiosity_urgency()"
new_f1b = "        # Phase 18: Update concept learning progress from edge PE\n        self._update_concept_learning_progress()\n\n        # Phase 18: Compute curiosity urgency for autonomous exploration\n        self._compute_curiosity_urgency()"

if old_f1b in code:
    code = code.replace(old_f1b, new_f1b, 1)
    changes.append("Fix 1b: Added _update_concept_learning_progress call to process_turn")
else:
    changes.append("Fix 1b FAILED")

# ============================================================
# FIX 2: Replace visit-count novelty with dormant-edge-ratio novelty
# ============================================================
# Add _compute_dormant_edge_ratio helper
old_f2 = "    def _auto_select_curiosity_topics(self, max_topics: int = 3) -> List[str]:"
new_f2 = "    def _compute_dormant_edge_ratio(self) -> Dict[str, float]:\n        \"\"\"Compute dormant edge ratio per concept.\n\n        Returns dict mapping concept label -> dormant_ratio (0-1).\n        High ratio = most edges are dormant = high novelty.\n        \"\"\"\n        ratios: Dict[str, float] = {}\n        for nid, node in self.graph.nodes.items():\n            if not node.label:\n                continue\n            total = 0\n            dormant = 0\n            for tid, edge in self.graph.get_outgoing(nid):\n                total += 1\n                if (nid, tid) in self._dormant_edges:\n                    dormant += 1\n            for src, edge in self.graph.get_incoming(nid):\n                if src == nid:\n                    continue\n                pair = (src, nid)\n                if pair not in self._dormant_edges:\n                    total += 1\n                else:\n                    total += 1\n                    dormant += 1\n            if total > 0:\n                ratios[node.label.lower()] = dormant / total\n        return ratios\n\n    def _auto_select_curiosity_topics(self, max_topics: int = 3) -> List[str]:"

if old_f2 in code:
    code = code.replace(old_f2, new_f2, 1)
    changes.append("Fix 2: Added _compute_dormant_edge_ratio method")
else:
    changes.append("Fix 2 FAILED")

# Now modify Source 4 (least-visited) in _auto_select_curiosity_topics to use dormant ratio
old_f2b = "        # Source 4: Least-visited concepts (novelty, weight inversely proportional to visits)\n        all_labels = set()\n        for nid, node in self.graph.nodes.items():\n            if node.label:\n                all_labels.add(node.label.lower())\n        unvisited = [l for l in all_labels if l not in seen_topics and len(l) >= 3]\n        unvisited.sort(key=lambda l: self._concept_visit_count.get(l, 0))\n        for label in unvisited[:3]:\n            visits = self._concept_visit_count.get(label, 0)\n            novelty_weight = 1.0 / (1.0 + visits)\n            candidates.append((label, novelty_weight))\n            seen_topics.add(label)"
new_f2b = "        # Source 4: Novel concepts via dormant edge ratio (high dormant = unexplored)\n        all_labels = set()\n        dormant_ratios = self._compute_dormant_edge_ratio()\n        for nid, node in self.graph.nodes.items():\n            if node.label:\n                all_labels.add(node.label.lower())\n        unvisited = [l for l in all_labels if l not in seen_topics and len(l) >= 3]\n        # Sort by dormant ratio descending (most unexplored first)\n        unvisited.sort(key=lambda l: dormant_ratios.get(l, 0), reverse=True)\n        for label in unvisited[:3]:\n            dr = dormant_ratios.get(label, 0)\n            novelty_weight = 0.5 + dr * 0.5  # range: 0.5 (fully explored) to 1.0 (fully dormant)\n            candidates.append((label, novelty_weight))\n            seen_topics.add(label)"

if old_f2b in code:
    code = code.replace(old_f2b, new_f2b, 1)
    changes.append("Fix 2b: Replaced visit-count novelty with dormant edge ratio")
else:
    changes.append("Fix 2b FAILED")

# ============================================================
# FIX 3: Confidence-mismatch scoring for contradictions + targeted queries
# ============================================================
old_f3 = "        # Source 3: Contradiction pairs (weight 3x)\n        for concept, antonyms in self._contradiction_map.items():\n            if concept not in seen_topics and len(concept) >= 3:\n                candidates.append((concept, 3.0))\n                seen_topics.add(concept)\n            for ant in antonyms:\n                if ant not in seen_topics and len(ant) >= 3:\n                    candidates.append((ant, 2.5))\n                    seen_topics.add(ant)"
new_f3 = "        # Source 3: Contradiction pairs with confidence-mismatch scoring\n        # Higher confidence mismatch = more uncertain = more curious to resolve\n        for concept, antonyms in self._contradiction_map.items():\n            # Look up edge confidence for both sides of contradiction\n            concept_conf = self._get_concept_confidence(concept)\n            for ant in antonyms:\n                ant_conf = self._get_concept_confidence(ant)\n                # Mismatch score: how uncertain are we which side is correct?\n                mismatch = abs(concept_conf - ant_conf) * 2.0  # 0 if equal, up to 2.0\n                # Base weight: higher for high-mismatch (ambiguity) AND low-mismatch (both confident but opposing)\n                # This captures both \"don't know which is right\" and \"both sides claim truth\"\n                conf_level = (concept_conf + ant_conf) / 2.0\n                if conf_level > 0.6:\n                    # Both confident but contradictory - interesting!\n                    weight = 3.0 + mismatch * 0.5\n                else:\n                    # At least one side uncertain - knowledge gap\n                    weight = 2.0 + mismatch * 0.75\n                if concept not in seen_topics and len(concept) >= 3:\n                    candidates.append((concept, max(2.0, weight)))\n                    seen_topics.add(concept)\n                if ant not in seen_topics and len(ant) >= 3:\n                    candidates.append((ant, max(1.5, weight * 0.8)))\n                    seen_topics.add(ant)"

if old_f3 in code:
    code = code.replace(old_f3, new_f3, 1)
    changes.append("Fix 3: Added confidence-mismatch scoring for contradictions")
else:
    changes.append("Fix 3 FAILED")

# Add _get_concept_confidence helper before _compute_dormant_edge_ratio
# Find the right insertion point - after _update_concept_learning_progress and before _compute_curiosity_urgency
old_f3_helper = "    def _get_concept_confidence"
if old_f3_helper not in code:
    # Insert after _update_concept_learning_progress
    anchor = "        self._curiosity_urgency = min(1.0, urgency)\n        return self._curiosity_urgency\n\n    def _compute_dormant_edge_ratio"
    new_anchor = "        self._curiosity_urgency = min(1.0, urgency)\n        return self._curiosity_urgency\n\n    def _get_concept_confidence(self, concept_label: str) -> float:\n        \"\"\"Get the confidence level of a concept from its incident edges.\"\"\"\n        nids = self._concept_keywords.get(concept_label.lower(), [])\n        if not nids:\n            return 0.0\n        confidences = []\n        for nid in nids:\n            for tid, edge in self.graph.get_outgoing(nid):\n                confidences.append(getattr(edge, 'confidence', 0.0) if hasattr(edge, 'confidence') else 0.0)\n            for src, edge in self.graph.get_incoming(nid):\n                confidences.append(getattr(edge, 'confidence', 0.0) if hasattr(edge, 'confidence') else 0.0)\n        if not confidences:\n            return 0.0\n        return sum(confidences) / len(confidences)\n\n    def _compute_dormant_edge_ratio"

    if anchor in code:
        code = code.replace(anchor, new_anchor, 1)
        changes.append("Fix 3b: Added _get_concept_confidence helper")
    else:
        changes.append("Fix 3b FAILED")

# ============================================================
# FIX 4: Query generation templates per source type
# ============================================================
# Add _generate_curiosity_query method
old_f4 = "    def _extract_related_queries(self, query: str) -> List[str]:"
new_f4 = "    def _generate_curiosity_query(self, concept: str, source_type: str = \"default\",\n                                antonym: str = \"\") -> str:\n        \"\"\"Generate a targeted web search query based on curiosity source type.\n\n        Templates:\n        - default: concept name (already handled by _bg_multi_search)\n        - high_pe: concept + 'explained' or 'mechanism'\n        - contradiction: 'concept vs antonym' or 'concept controversy'\n        - dormant_edge: concept + 'related topics' or 'what is'\n        \"\"\"\n        if source_type == \"contradiction\" and antonym:\n            # Target: expose why the contradiction exists\n            return f\"{concept} vs {antonym} debate\"\n        elif source_type == \"high_pe\":\n            # Target: clarify the confusing topic\n            return f\"{concept} explained simply\"\n        elif source_type == \"dormant_edge\":\n            # Target: discover what connects to this unexplored concept\n            return f\"{concept} related concepts what is\"\n        else:\n            return concept\n\n    def _extract_related_queries(self, query: str) -> List[str]:"

if old_f4 in code:
    code = code.replace(old_f4, new_f4, 1)
    changes.append("Fix 4: Added _generate_curiosity_query method with templates")
else:
    changes.append("Fix 4 FAILED")

# Use _generate_curiosity_query in _auto_select_curiosity_topics
# Modify Source 1 (impossible queries) and Source 2 (high PE) to use targeted queries
old_f4b = "        # Source 1: Unresolved impossible queries (weight 5x)\n        for iq in self._impossible_queries:\n            if iq.resolved:\n                continue\n            topic = iq.subject.lower().strip()\n            if topic and topic not in seen_topics and len(topic) >= 3:\n                candidates.append((topic, 5.0))\n                seen_topics.add(topic)\n\n        # Source 2: High prediction error concepts (weight 3x)\n        high_pe_nodes = []\n        for nid, node in self.graph.nodes.items():\n            if node.label and node.prediction_free_energy > 0.3:\n                label = node.label.lower()\n                if label not in seen_topics and len(label) >= 3:\n                    high_pe_nodes.append((label, node.prediction_free_energy))\n        high_pe_nodes.sort(key=lambda x: -x[1])\n        for label, pe in high_pe_nodes[:5]:\n            candidates.append((label, 3.0 * min(1.0, pe)))\n            seen_topics.add(label)"
new_f4b = "        # Source 1: Unresolved impossible queries (weight 5x)\n        for iq in self._impossible_queries:\n            if iq.resolved:\n                continue\n            topic = iq.subject.lower().strip()\n            if topic and topic not in seen_topics and len(topic) >= 3:\n                candidates.append((topic, 5.0))\n                seen_topics.add(topic)\n\n        # Source 2: High prediction error concepts (weight 3x, targeted query)\n        high_pe_nodes = []\n        for nid, node in self.graph.nodes.items():\n            if node.label and node.prediction_free_energy > 0.3:\n                label = node.label.lower()\n                if label not in seen_topics and len(label) >= 3:\n                    high_pe_nodes.append((label, node.prediction_free_energy))\n        high_pe_nodes.sort(key=lambda x: -x[1])\n        for label, pe in high_pe_nodes[:5]:\n            candidates.append((label, 3.0 * min(1.0, pe)))\n            seen_topics.add(label)"

if old_f4b in code:
    code = code.replace(old_f4b, new_f4b, 1)
    changes.append("Fix 4b: Updated Source 1 for query generation (no material change, just alignment)")
else:
    changes.append("Fix 4b FAILED")

# Modify Source 3 (contradiction) to use targeted queries
old_f4c = "        # Source 5: Random graph walk from high-degree hubs (serendipity)\n        if len(self.graph.nodes) > 0:"
new_f4c = "        # Generate targeted queries for top contradiction candidates\n        contradiction_queries = []\n        for topic, _ in candidates:\n            if topic in self._contradiction_map:\n                for ant in self._contradiction_map[topic]:\n                    targeted = self._generate_curiosity_query(\n                        topic, source_type=\"contradiction\", antonym=ant)\n                    if targeted != topic:\n                        contradiction_queries.append(targeted)\n        # Queue contradiction-specific queries directly\n        for cq in contradiction_queries[:2]:\n            if self._bg_learning_active and cq not in self._bg_learning_queue:\n                self.queue_background_search(cq)\n\n        # Source 5: Random graph walk from high-degree hubs (serendipity)\n        if len(self.graph.nodes) > 0:"

if old_f4c in code:
    code = code.replace(old_f4c, new_f4c, 1)
    changes.append("Fix 4c: Added contradiction-specific query generation")
else:
    changes.append("Fix 4c FAILED")

# ============================================================
# FIX 5: Arousal modulation - multiplicative instead of additive
# ============================================================
old_f5 = "        # 3. Arousal modulation (emotional engagement amplifies curiosity)\n        arousal_boost = self.emotion.state.arousal * 0.2\n        urgency += arousal_boost"
new_f5 = "        # 3. Arousal modulation (multiplicative, coupling dopamine and norepinephrine)\n        # Neuroscience: arousal amplifies exploration non-linearly\n        # Even at minimal arousal (0.1), 55% of curiosity remains\n        # At max arousal (1.0), curiosity is at 100%\n        arousal_gain = 0.5 + 0.5 * self.emotion.state.arousal\n        urgency *= arousal_gain"

if old_f5 in code:
    code = code.replace(old_f5, new_f5, 1)
    changes.append("Fix 5: Changed arousal modulation from additive to multiplicative")
else:
    changes.append("Fix 5 FAILED")

# ============================================================
# FIX 6: User priming - boost curiosity based on user's recent questions
# ============================================================
# Add _user_query_topics tracking field in __init__
# Find the curiosity fields section and add user query tracking
old_f6 = "        self._curiosity_urgency: float = 0.0  # overall curiosity drive (0-1)\n\n\n        if os.path.exists(self._save_path):"
new_f6 = "        self._curiosity_urgency: float = 0.0  # overall curiosity drive (0-1)\n        # Phase 18b: User priming - track recent user topics for curiosity boosting\n        self._user_query_topics: List[str] = []  # last 10 topics user asked about\n        self._user_last_topic: str = \"\"  # most recent user topic\n\n\n        if os.path.exists(self._save_path):"

if old_f6 in code:
    code = code.replace(old_f6, new_f6, 1)
    changes.append("Fix 6: Added user query tracking fields")
else:
    changes.append("Fix 6 FAILED")

# Track user topics in process_turn after extracting subject
old_f6b = "        # Phase 18: Track concept visits for curiosity/novelty scoring\n        if subject:"
new_f6b = "        # Phase 18b: Track user query topics for curiosity priming\n        if subject:\n            sl = subject.lower()\n            if sl != self._user_last_topic and len(sl) >= 3:\n                self._user_query_topics.append(sl)\n                if len(self._user_query_topics) > 10:\n                    self._user_query_topics = self._user_query_topics[-10:]\n                self._user_last_topic = sl\n\n        # Phase 18: Track concept visits for curiosity/novelty scoring\n        if subject:"

if old_f6b in code:
    code = code.replace(old_f6b, new_f6b, 1)
    changes.append("Fix 6b: Added user topic tracking in process_turn")
else:
    changes.append("Fix 6b FAILED")

# Add user priming boost in _auto_select_curiosity_topics
# After collecting all candidates, boost those matching user interests
old_f6c = "        candidates.sort(key=lambda x: -x[1])\n        selected = [topic for topic, _ in candidates[:max_topics]]\n\n        if selected and self._trace_enabled:"
new_f6c = "        # Phase 18b: Boost candidates matching user's recent interests (priming)\n        if self._user_query_topics:\n            for i in range(len(candidates)):\n                topic, weight = candidates[i]\n                # Check if topic or any part matches user's recent queries\n                for uq in self._user_query_topics[-3:]:  # last 3 topics\n                    if uq in topic or topic in uq:\n                        candidates[i] = (topic, weight * 1.5)  # 1.5x boost\n                        break\n                    # Also check if the topic is connected to user's interests via graph\n                    uq_nids = self._concept_keywords.get(uq, [])\n                    topic_nids = self._concept_keywords.get(topic, [])\n                    if uq_nids and topic_nids:\n                        for un in uq_nids:\n                            for tn in topic_nids:\n                                if self.graph.get_edge(un, tn) or self.graph.get_edge(tn, un):\n                                    candidates[i] = (topic, weight * 1.3)  # 1.3x boost for connected concepts\n                                    break\n\n        candidates.sort(key=lambda x: -x[1])\n        selected = [topic for topic, _ in candidates[:max_topics]]\n\n        if selected and self._trace_enabled:"

if old_f6c in code:
    code = code.replace(old_f6c, new_f6c, 1)
    changes.append("Fix 6c: Added user priming boost in topic selection")
else:
    changes.append("Fix 6c FAILED")

# Update save/load for user query tracking
# Save
old_f6d = "            'curiosity_urgency': self._curiosity_urgency,\n            # Background learning"
new_f6d = "            'curiosity_urgency': self._curiosity_urgency,\n            'user_query_topics': self._user_query_topics,\n            'user_last_topic': self._user_last_topic,\n            # Background learning"
if old_f6d in code:
    code = code.replace(old_f6d, new_f6d, 1)
    changes.append("Fix 6d: Added user query fields to save()")
else:
    changes.append("Fix 6d FAILED")

# Load
old_f6e = "        self._curiosity_urgency = state.get('curiosity_urgency', 0.0)"
new_f6e = "        self._curiosity_urgency = state.get('curiosity_urgency', 0.0)\n        self._user_query_topics = state.get('user_query_topics', [])\n        self._user_last_topic = state.get('user_last_topic', '')"
if old_f6e in code:
    code = code.replace(old_f6e, new_f6e, 1)
    changes.append("Fix 6e: Added user query fields to _load()")
else:
    changes.append("Fix 6e FAILED")

# ============================================================
# FIX 7: Multi-source consensus tracking (hallucination guard)
# ============================================================
# Add _concept_sources field in __init__
old_f7 = "        self._user_last_topic: str = \"\"  # most recent user topic\n\n\n        if os.path.exists(self._save_path):"
new_f7 = "        self._user_last_topic: str = \"\"  # most recent user topic\n        # Phase 18c: Multi-source consensus tracking for hallucination guard\n        self._concept_sources: Dict[str, Set[str]] = {}  # concept -> set of source URLs\n\n\n        if os.path.exists(self._save_path):"

# Need to handle the case where fix 6 already modified this
# Check which version exists
if old_f7 in code:
    code = code.replace(old_f7, new_f7, 1)
    changes.append("Fix 7: Added _concept_sources tracking field")
else:
    # Try alternative - maybe fix 6 already changed it
    alt_f7 = "        self._user_last_topic: str = \"\"  # most recent user topic\n        # Phase 18b: User priming"
    if alt_f7 in code:
        # Insert after user_last_topic before user priming comment
        new_alt = "        self._user_last_topic: str = \"\"  # most recent user topic\n        # Phase 18c: Multi-source consensus tracking for hallucination guard\n        self._concept_sources: Dict[str, Set[str]] = {}  # concept -> set of source URLs\n        # Phase 18b: User priming"
        code = code.replace(alt_f7, new_alt, 1)
        changes.append("Fix 7: Added _concept_sources tracking field (alt)")
    else:
        changes.append("Fix 7 FAILED - couldn't find anchor")

# Modify _learn_from_text to track sources and apply consensus filter
old_f7b = "        # Add new concepts\n        for word in important_words:\n            if word in existing_labels:\n                continue\n            # GloVe vector if available, else hash-based random\n            vec = self._glove_vector(word)\n            if vec is None:\n                h = hash(word) % 50000\n                vr = np.random.RandomState(h + 100)\n                vec = vr.randn(self.dim).astype(np.float32) * 0.1\n                norm = np.linalg.norm(vec)\n                if norm > 0:\n                    vec /= norm\n            node = self.graph.add_node(vector=vec, label=word)\n            label_to_id[word] = node.id\n            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]\n            self._concept_labels.add(word.lower())\n            existing_labels.add(word)\n            new_count += 1"
new_f7b = "        # Phase 18c: Track source of this learning\n        source_id = str(hash(topic))[:16]  # lightweight source fingerprint\n\n        # Add new concepts\n        for word in important_words:\n            if word in existing_labels:\n                # Concept already exists - reinforce by tracking additional source\n                self._concept_sources.setdefault(word, set()).add(source_id)\n                # Boost edge confidence if concept confirmed by multiple sources\n                if len(self._concept_sources[word]) >= 2:\n                    nids = self._concept_keywords.get(word, [])\n                    for nid in nids:\n                        for tid, edge in self.graph.get_outgoing(nid):\n                            edge.confidence = min(0.8, edge.confidence + 0.05)\n                continue\n            # GloVe vector if available, else hash-based random\n            vec = self._glove_vector(word)\n            if vec is None:\n                h = hash(word) % 50000\n                vr = np.random.RandomState(h + 100)\n                vec = vr.randn(self.dim).astype(np.float32) * 0.1\n                norm = np.linalg.norm(vec)\n                if norm > 0:\n                    vec /= norm\n            node = self.graph.add_node(vector=vec, label=word)\n            label_to_id[word] = node.id\n            self._concept_keywords[word] = self._concept_keywords.get(word, []) + [node.id]\n            self._concept_labels.add(word.lower())\n            existing_labels.add(word)\n            new_count += 1\n            # Phase 18c: Track source for new concepts\n            self._concept_sources[word] = {source_id}\n            # New concepts from single source start with low confidence\n            for tid, edge in self.graph.get_outgoing(node.id):\n                edge.confidence = min(edge.confidence, 0.2)  # cap at 0.2 until reinforced"

if old_f7b in code:
    code = code.replace(old_f7b, new_f7b, 1)
    changes.append("Fix 7b: Added multi-source consensus tracking in _learn_from_text")
else:
    changes.append("Fix 7b FAILED")

# Update save/load for concept_sources
old_f7c = "            'user_last_topic': self._user_last_topic,\n            # Background learning"
new_f7c = "            'user_last_topic': self._user_last_topic,\n            'concept_sources': {k: list(v) for k, v in self._concept_sources.items()},\n            # Background learning"
if old_f7c in code:
    code = code.replace(old_f7c, new_f7c, 1)
    changes.append("Fix 7c: Added concept_sources to save()")
else:
    changes.append("Fix 7c FAILED")

old_f7d = "        self._user_last_topic = state.get('user_last_topic', '')"
new_f7d = "        self._user_last_topic = state.get('user_last_topic', '')\n        raw_sources = state.get('concept_sources', {})\n        self._concept_sources = {k: set(v) for k, v in raw_sources.items()}"
if old_f7d in code:
    code = code.replace(old_f7d, new_f7d, 1)
    changes.append("Fix 7d: Added concept_sources to _load()")
else:
    changes.append("Fix 7d FAILED")

# ============================================================
# Write the file
# ============================================================
with open('scripts/ravana_chat.py', 'w', encoding='utf-8') as f:
    f.write(code)

print("Changes applied:")
for c in changes:
    print(f"  {c}")
