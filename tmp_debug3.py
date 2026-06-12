"""Trace exact edge state after each step."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Monkey-patch add_edge to trace
from scripts.ravana_chat import CognitiveChatEngine, TEEN_CONCEPTS
import numpy as np

# Create engine but stop before seeding
engine = CognitiveChatEngine.__new__(CognitiveChatEngine)
engine.dim = 64
engine.rng = np.random.RandomState(42)
engine.graph = __import__('ravana_ml.graph', fromlist=['ConceptGraph']).ConceptGraph(dim=64, max_nodes=10000)
engine._concept_labels = set()
engine._concept_keywords = {}
engine._dormant_edges = set()
engine._glove_vector_cache = {}
engine._glove_vecs = None
engine._all_labels = {}
engine._contradiction_map = {}
engine._concept_vad = {}
engine._concept_sources = {}
engine._concept_visit_count = {}
engine._concept_learning_progress = {}
engine._concept_pe_delta = {}
engine.user_model = __import__('scripts.ravana_chat', fromlist=['UserModel']).UserModel()
engine.belief_store = __import__('scripts.ravana_chat', fromlist=['BeliefStore']).BeliefStore()
engine._concept_confidence = {}
engine._mean_prediction_error = 0.0
engine._prediction_error_count = 0
engine._sentence_schema = {}
engine._mean_sentence_pe = 0.0
engine._sentence_pe_count = 0
engine._current_context_vector = None
engine._modulated_vectors = {}
engine._state_dependent_boosts = {}
engine._cognitive_state = 'default'
engine._state_duration = 0
engine._cognitive_state_hold = 0
engine._schema_mode = False
engine._activation_fatigue = {}
engine._recent_traversals = []
engine._visited_concepts = set()
engine._dopamine_tone = 0.5
engine._td_error_history = []
engine._expected_strength = 0.25
engine._episodic_edges = {}
engine._semantic_edges = {}
engine._semantic_by_src = {}
engine._episodic_by_src = {}
engine._cerebellar_ngram = {}
engine._cerebellar_depth = {}
engine._calibration_error = 0.0
engine._metacognitive_review_turn = 0
engine._trace_enabled = True
engine._impossible_queries = []
engine._last_strategy_used = ''
engine._last_strategy = ''
engine._last_responses = []
engine._recall_mode = False
engine._last_chain_hops = []
engine._last_hops = []
engine._response_context = []
engine._topic_list = []
engine._topic_store = {}
engine._prefrontal_buffer = []
engine._pfc_gating_enabled = True
engine._pfc_buffer_capacity = 7
engine._consistency_paths = {}
engine._consistency_trace = []
engine.reasoning_mode = 'stochastic'
engine._learning_count = 0
engine._learned_this_turn = False
engine.turn_count = 0
engine.CONTRASTIVE_PAIRS = CognitiveChatEngine.CONTRASTIVE_PAIRS
engine.CAUSAL_PAIRS = CognitiveChatEngine.CAUSAL_PAIRS
engine.IS_A_PAIRS = CognitiveChatEngine.IS_A_PAIRS
engine._network_available = None
engine._network_retry_turn = 0
engine._pending_learning_queue = []
engine._bg_learning_queue = []
engine._bg_learning_active = False
engine._bg_search_count = 0
engine._bg_multi_search_max = 3
engine._bg_idle_search_count = 0
engine._curiosity_drive_enabled = True
engine._curiosity_cycles_this_session = 0
engine._curiosity_topics_queue = []
engine._last_auto_learn_turn = 0
engine._curiosity_urgency = 0.0
engine._user_query_topics = []
engine._user_last_topic = ''
engine._explored_contradictions = set()
engine._sleep_pressure = 0.0
engine._last_sleep_episode = 0
engine.sleep_cycles_completed = 0
engine._cascade_for_quality = False

# Load GloVe
engine._init_glove()

# Manually seed just learn and knowledge to trace
labels_to_test = ['learn', 'knowledge', 'good', 'bad']
for label, keywords in TEEN_CONCEPTS:
    if label not in labels_to_test:
        continue
    vec = engine._glove_vector(label)
    if vec is None:
        h = hash(label) % 10000
        vr = np.random.RandomState(h + 42)
        vec = vr.randn(64).astype(np.float32) * 0.15
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
    node = engine.graph.add_node(vector=vec, label=label)
    engine._concept_labels.add(label.lower())
    for kw in keywords.split():
        kl = kw.lower()
        engine._concept_keywords.setdefault(kl, []).append(node.id)
    engine._concept_keywords.setdefault(label, []).append(node.id)

label_to_id = {n.label: n.id for n in engine.graph.nodes.values() if n.label}

# Add causal edge for learn -> knowledge
sid = label_to_id.get('learn')
tid = label_to_id.get('knowledge')
if sid is not None and tid is not None:
    e = engine.graph.add_edge(sid, tid, weight=0.45, relation_type="causal")
    print(f"After causal add: learn({sid}) -> knowledge({tid}): type={e.relation_type}, w={e.weight:.3f}, conf={e.confidence:.3f}")

# Now simulate auto-wirer
for i_nid in [sid]:
    ni = engine.graph.get_node(i_nid)
    for j_nid in [tid]:
        existing = engine.graph.get_edge(i_nid, j_nid)
        print(f"Auto-wirer check: get_edge({i_nid}, {j_nid}) = {existing}")
        if existing:
            print(f"  EXISTING: type={existing.relation_type}, w={existing.weight:.3f}")
            print("  -> SKIP")
        else:
            print("  -> would create edge")
            
# Also check reverse order
existing_rev = engine.graph.get_edge(tid, sid)
print(f"get_edge({tid}, {sid}) = {existing_rev}")
if existing_rev:
    print(f"  type={existing_rev.relation_type}, w={existing_rev.weight:.3f}")
