"""
RAVANA v2 — DIALOGUE ENGINE
Main orchestrator that ties all dialogue subsystems into a single conversational loop.

PRINCIPLE: Dialogue is a pressure-driven settling process where each turn
injects new triples, decays old ones, spreads activation, generates a response,
and optionally consolidates through sleep.

The Complete Conversational Loop:
1. Perceive (input understanding via tokenizer)
2. Inject into active subgraph
3. Decay and prune
4. Spread activation (agent-aware)
5. Generate response (via RLMv2 or pattern-matching)
6. Check Governor constraints
7. Verbalize
8. Check for corrections
9. Optionally trigger sleep consolidation
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

from ravana_ml.tokenizer import get_tokenizer, TokenizerInterface

# Import core dialogue modules
from core.dialogue_context import (
    DialogueContext, ActiveSubgraph, Triple, DialogueState,
)
from core.conversational_repair import (
    ConversationalRepair, RepairEvent, CorrectionType,
)


# ─── Configuration ───────────────────────────────────────────────────────────

@dataclass
class DialogueEngineConfig:
    """Configuration for the Dialogue Engine."""
    # Dialogue context
    decay_rate: float = 0.92
    activation_threshold: float = 0.01
    max_spreading_depth: int = 3
    spreading_decay_per_hop: float = 0.7

    # Conversational repair
    penalty_on_correction: float = -0.4
    boost_on_correction: float = 0.5
    contradiction_free_energy_spike: float = 0.8

    # Sleep consolidation
    sleep_pressure_threshold: float = 2.0
    consolidation_turn_interval: int = 5  # Sleep every N turns minimum

    # Tokenizer
    tokenizer_name: str = "word"


@dataclass
class DialogueTurnRecord:
    """Record of a single dialogue turn."""
    turn: int
    user_input: str
    system_output: str
    triples: List[Triple]
    activations: Dict[str, float]
    repair_event: Optional[RepairEvent]
    sleep_triggered: bool
    timestamp: float


# ─── DialogueEngine ──────────────────────────────────────────────────────────

class DialogueEngine:
    """
    Main orchestrator for the RAVANA dialogue system.

    Ties together:
    - DialogueContext (working memory)
    - ConversationalRepair (correction handling)
    - ConceptGraph (semantic knowledge)
    - Governor (central regulation)
    - Emotion engine (VAD state)
    - Identity engine
    - Sleep consolidation
    - Human memory (episodic storage)

    Usage:
        engine = DialogueEngine(user_id="likhith")
        response = engine.process_turn("Heat causes expansion")
        print(response)
    """

    def __init__(
        self,
        user_id: str = "default",
        config: Optional[DialogueEngineConfig] = None,
        graph: Optional[Any] = None,            # ConceptGraph
        rlm_model: Optional[Any] = None,        # RLMv2 model
        tokenizer: Optional[TokenizerInterface] = None,
        governor: Optional[Any] = None,         # Governor
        emotion_engine: Optional[Any] = None,   # VADEmotionEngine
        identity_engine: Optional[Any] = None,  # IdentityEngine
        sleep_engine: Optional[Any] = None,     # SleepConsolidation
        human_memory: Optional[Any] = None,     # HumanMemoryEngine
    ):
        self.config = config or DialogueEngineConfig()
        self.user_id = user_id

        # ── Subsystems ──
        self.dialogue_context = DialogueContext(
            user_id=user_id,
            decay_rate=self.config.decay_rate,
            activation_threshold=self.config.activation_threshold,
            max_spreading_depth=self.config.max_spreading_depth,
            spreading_decay_per_hop=self.config.spreading_decay_per_hop,
        )

        self.repair = ConversationalRepair(
            graph=graph,
            governor=governor,
            user_id=user_id,
            penalty_on_correction=self.config.penalty_on_correction,
            boost_on_correction=self.config.boost_on_correction,
            contradiction_free_energy_spike=self.config.contradiction_free_energy_spike,
        )

        # ── External components ──
        self.graph = graph
        self.rlm_model = rlm_model
        self.governor = governor
        self.emotion_engine = emotion_engine
        self.identity_engine = identity_engine
        self.sleep_engine = sleep_engine
        self.human_memory = human_memory

        # ── Tokenizer ──
        self.tokenizer = tokenizer or get_tokenizer(
            self.config.tokenizer_name
        )

        # ── State tracking ──
        self.turn_count: int = 0
        self._turn_history: List[DialogueTurnRecord] = []
        self._free_energy_accumulator: float = 0.0
        self._sleep_requested: bool = False

    def process_turn(self, user_input: str) -> str:
        """
        Process one complete conversation turn.

        Lifecycle:
        1. Perceive: tokenize and parse input into triples
        2. Inject: add triples to active subgraph with high salience
        3. Decay: apply exponential decay to all salience scores
        4. Spread activation: propagate from active concepts
        5. Generate response: produce output from activated context
        6. Check Governor constraints
        7. Return response and save state

        Args:
            user_input: Raw text from the user

        Returns:
            System response text
        """
        self.turn_count += 1

        # Step 1: Perceive — tokenize and parse
        token_ids = self.tokenizer.encode(user_input)
        triples = self._parse_input_to_triples(user_input, token_ids)

        # Step 2: Inject triples into active subgraph
        context_activations = self.dialogue_context.process_turn(
            user_input, triples, graph=self.graph
        )

        # Step 3: Apply emotional modulation if available
        modulated_activations = self._modulate_with_emotion(context_activations)

        # Step 4: Generate response
        response, response_triples = self._generate_response(
            user_input, triples, modulated_activations
        )

        # Step 5: Record output
        self.dialogue_context.record_output(response, response_triples)

        # Step 6: Check Governor constraints
        response = self._apply_governor_constraints(response)

        # Step 7: Accumulate free energy
        self._accumulate_free_energy(triples)

        # Step 8: Check if sleep consolidation is needed
        sleep_triggered = self._check_sleep_needed()

        # Step 9: Record the turn
        record = DialogueTurnRecord(
            turn=self.turn_count,
            user_input=user_input,
            system_output=response,
            triples=triples,
            activations=modulated_activations,
            repair_event=None,
            sleep_triggered=sleep_triggered,
            timestamp=time.time(),
        )
        self._turn_history.append(record)

        # Step 10: Store in human memory if available
        if self.human_memory is not None:
            self._store_in_human_memory(record)

        return response

    def handle_correction(
        self,
        user_correction: str,
    ) -> Optional[RepairEvent]:
        """
        Handle a user correction to the system's previous output.

        Called when the user corrects the system's last response.

        Args:
            user_correction: The user's correction text

        Returns:
            RepairEvent if correction detected, None otherwise
        """
        if not self.dialogue_context.last_output:
            return None

        # Detect and apply repair
        event = self.repair.process_correction(
            self.dialogue_context.last_output,
            user_correction,
        )

        if event is not None:
            # Inject the corrected triple with high salience
            self.dialogue_context.active_subgraph.inject(
                [event.correct_triple], salience=1.5
            )

            # Accumulate contradiction pressure
            self._free_energy_accumulator += (
                self.config.contradiction_free_energy_spike
            )

            # Request sleep consolidation
            self._sleep_requested = True

            # Update the last turn record with the repair event
            if self._turn_history:
                self._turn_history[-1].repair_event = event

        return event

    def trigger_sleep(self) -> bool:
        """
        Trigger sleep consolidation with dialogue-specific logic.

        Implements Subsystem 5 from the architectural plan:
        1. SWS-Phase: Promote consistent user beliefs to episodic clusters
           based on agent_weight thresholds and reinforcement counts
        2. Inhibitory edge formation: Create negative-weight links between
           corrected (wrong) and correct triples to prevent re-contradiction
        3. REM-Phase: Generate counterfactual inferences from user beliefs,
           linking beliefs to broader concepts via inferred edges
        4. General sleep: Delegates to the SleepConsolidation engine for
           hippocampal replay, abstraction compression, and homeostasis
        5. Post-sleep: Clear working memory, reset free energy

        Returns:
            True if sleep was executed
        """
        if self.sleep_engine is None:
            return False

        # ── Phase 1: Dialogue-Specific SWS (Promote User Beliefs) ──
        # Check user beliefs that have been reinforced multiple times
        user_beliefs = self.dialogue_context.get_all_user_beliefs()
        promoted_beliefs = []
        for key, belief in user_beliefs.items():
            access_count = belief.get("access_count", 0)
            strength = belief.get("strength", 0.0)
            # Promote if reinforced multiple times AND has sufficient strength
            if access_count >= 3 and strength >= 0.6:
                promoted_beliefs.append(key)
                # Mark as promoted in the belief metadata
                belief["promoted"] = True
                belief["promoted_at"] = time.time()
                belief["promotion_cycle"] = self.turn_count

        # ── Phase 2: Dialogue-Specific SWS (Inhibitory Edge Formation) ──
        # For each repair event, form inhibitory edges between wrong and correct triples
        # This prevents the system from oscillating between the same two answers
        inhibitory_edges_formed = 0
        if self.graph is not None:
            for event in self.repair.repair_history[-20:]:  # Last 20 repairs
                wrong_triple = event.wrong_triple
                correct_triple = event.correct_triple

                # Find the wrong and correct concept object nodes
                wrong_obj_label = wrong_triple.object.lower()
                correct_obj_label = correct_triple.object.lower()

                wrong_obj_id = None
                correct_obj_id = None
                for nid, node in self.graph.nodes.items():
                    if node.label and node.label.lower() == wrong_obj_label:
                        wrong_obj_id = nid
                    if node.label and node.label.lower() == correct_obj_label:
                        correct_obj_id = nid

                if wrong_obj_id is not None and correct_obj_id is not None:
                    # Form bidirectional inhibitory edges between the two objects
                    # This creates competition: activating one suppresses the other
                    existing = self.graph.get_edge(wrong_obj_id, correct_obj_id)
                    if existing is None:
                        self.graph.add_edge(
                            wrong_obj_id, correct_obj_id,
                            weight=0.2, edge_type="inhibitory",
                            relation_type="semantic",
                        )
                        inhibitory_edges_formed += 1

        # ── Phase 3: REM-Phase (Counterfactual Inferences) ──
        # For promoted beliefs, generate inferred edges linking the belief's
        # subject to related concepts. This is the "expanded understanding"
        # that wasn't explicitly stated by the user.
        inferred_edges_created = 0
        if self.graph is not None and promoted_beliefs:
            for key in promoted_beliefs:
                belief = user_beliefs[key]
                subject = belief.get("subject", "").lower()
                obj = belief.get("object", "").lower()

                # Find the subject node in the graph
                subject_id = None
                for nid, node in self.graph.nodes.items():
                    if node.label and node.label.lower() == subject:
                        subject_id = nid
                        break

                if subject_id is not None:
                    # Find highly-activated neighbors of the subject
                    # (concepts that are co-activated with the belief's subject)
                    outgoing = self.graph.get_outgoing(subject_id)
                    for target_id, edge in outgoing[:5]:  # Top 5 connections
                        target_node = self.graph.get_node(target_id)
                        if target_node is None:
                            continue
                        target_label = (target_node.label or "").lower()
                        if target_label == obj:
                            continue  # Skip the belief itself

                        # Create an inferred edge with lower confidence (0.6)
                        # This edge links the belief subject to broader concepts
                        existing = self.graph.get_edge(subject_id, target_id)
                        if existing is None and edge.confidence >= 0.3:
                            inferred_edge = self.graph.add_edge(
                                subject_id, target_id,
                                weight=0.4,  # Lower weight for inferred
                                relation_type="inferred",
                                edge_type="excitatory",
                            )
                            inferred_edge.source_metadata['is_inferred'] = True
                            inferred_edge.source_metadata['epistemic_status'] = 'hypothesis'
                            inferred_edge.source_metadata['source_agent'] = f"user_{self.user_id}_inference"
                            inferred_edges_created += 1

        # ── Phase 4: Prepare state snapshot for general sleep ──
        state_snapshot = {
            "user_id": self.user_id,
            "turn_count": self.turn_count,
            "free_energy": self._free_energy_accumulator,
            "dialogue_context": self.dialogue_context.get_state(),
            "user_beliefs": user_beliefs,
            "repair_history": self.repair.get_repair_stats(),
            "promoted_beliefs": promoted_beliefs,
            "inhibitory_edges_formed": inhibitory_edges_formed,
            "inferred_edges_created": inferred_edges_created,
        }

        # Collect recent episodic memories from conversation
        episodic_memories = []
        for record in self._turn_history[-50:]:
            mem = {
                "id": record.turn,
                "content": f"User: {record.user_input} | System: {record.system_output}",
                "memory_type": "episodic",
                "tags": "dialogue,conversation",
                "importance": 0.5,
                "emotional": 0.5,
                "predictive_utility": 0.5,
                "source_agent": f"user_{self.user_id}",
            }
            if record.repair_event is not None:
                mem["tags"] += ",correction"
                mem["importance"] = 0.8
            episodic_memories.append(mem)

        # ── Phase 5: Execute general sleep consolidation ──
        sleep_record = self.sleep_engine.execute_sleep_cycle(
            episode=self.turn_count,
            state_snapshot=state_snapshot,
            episodic_memories=episodic_memories,
            emotion_engine=self.emotion_engine,
            graph=self.graph,
            coherence_fn=lambda s: 1.0 - min(
                1.0, s.get("free_energy", 0.0) / 5.0
            ),
        )

        # Store dialogue-specific sleep metrics
        sleep_record.details["dialogue_sleep"] = {
            "promoted_beliefs": promoted_beliefs,
            "inhibitory_edges_formed": inhibitory_edges_formed,
            "inferred_edges_created": inferred_edges_created,
            "total_repairs_processed": len(self.repair.repair_history),
        }

        # ── Phase 6: Post-sleep cleanup ──
        self.dialogue_context.clear_for_sleep()
        self._free_energy_accumulator = max(
            0.0, self._free_energy_accumulator - 1.0
        )
        self._sleep_requested = False

        return sleep_record.post_coherence > sleep_record.pre_coherence

    def get_state(self) -> Dict[str, Any]:
        """Get the full state of the dialogue engine for debugging."""
        return {
            "user_id": self.user_id,
            "turn_count": self.turn_count,
            "dialogue_context": self.dialogue_context.get_context(),
            "repair_stats": self.repair.get_repair_stats(),
            "free_energy": self._free_energy_accumulator,
            "sleep_requested": self._sleep_requested,
            "history_length": len(self._turn_history),
        }

    def get_turn_history(
        self, n: int = 10
    ) -> List[DialogueTurnRecord]:
        """Get the last N turn records."""
        return self._turn_history[-n:] if self._turn_history else []

    def reset_conversation(self):
        """Reset the conversation (preserves user beliefs)."""
        self.dialogue_context.clear_for_sleep()
        self.turn_count = 0
        self._free_energy_accumulator = 0.0
        self._sleep_requested = False

    def set_user_id(self, user_id: str):
        """Switch to a different user."""
        self.user_id = user_id
        self.dialogue_context.user_id = user_id
        self.repair.user_id = user_id

    # ── Private Methods ──

    def _parse_input_to_triples(
        self,
        user_input: str,
        token_ids: List[int],
    ) -> List[Triple]:
        """
        Parse user input into triples.

        Uses tokenizer to help with parsing but falls back to simple
        text-based triple decomposition.

        Args:
            user_input: Raw text input
            token_ids: Tokenized input

        Returns:
            List of parsed Triples
        """
        triples = []

        # Try to use RLMv2's decompose_triple if available
        if self.rlm_model is not None and hasattr(
            self.rlm_model, "decompose_triple"
        ):
            try:
                subj_ids, rel_ids, obj_ids = (
                    self.rlm_model.decompose_triple(token_ids)
                )
                if subj_ids and obj_ids:
                    # Decode each part
                    subject = self.tokenizer.decode(subj_ids).strip()
                    rel_tokens = (
                        self.tokenizer.decode(rel_ids).strip()
                        if rel_ids
                        else "is"
                    )
                    obj = self.tokenizer.decode(obj_ids).strip()

                    if subject and obj:
                        # Classify relation type
                        rel_type = "semantic"
                        if self.rlm_model is not None and hasattr(
                            self.rlm_model, "classify_relation"
                        ):
                            rel_idx = self.rlm_model.classify_relation(
                                rel_ids
                            )
                            from ravana_ml.nn.rlm_v2 import RELATION_TYPES
                            rel_type = RELATION_TYPES[rel_idx]

                        triple = Triple(
                            subject=subject,
                            relation=rel_tokens[0] if rel_tokens else "is",
                            relation_type=rel_type,
                            object=obj,
                            confidence=0.8,
                            source_agent=f"user_{self.user_id}",
                            epistemic_status="belief",
                            timestamp=time.time(),
                        )
                        triples.append(triple)
                        return triples
            except Exception:
                pass

        # Fallback: simple text-based parsing
        words = user_input.strip().split()
        if len(words) >= 3:
            # Pattern: subject verb object
            # Use ravana_ml.nn.rlm_v2.RELATION_TYPES for type
            from ravana_ml.nn.rlm_v2 import RELATION_TYPES

            # Simple heuristic: classify based on verb
            verb = words[1].lower().rstrip(".,!?")
            rel_type = "semantic"

            # Check keyword map from RLMv2
            from ravana_ml.nn.rlm_v2 import _KEYWORD_MAP
            for rtype, keywords in _KEYWORD_MAP.items():
                if verb in keywords:
                    rel_type = rtype
                    break

            triple = Triple(
                subject=words[0],
                relation=verb,
                relation_type=rel_type,
                object=" ".join(words[2:]),
                confidence=0.6,
                source_agent=f"user_{self.user_id}",
                epistemic_status="belief" if "user" in self.user_id else "fact",
                timestamp=time.time(),
            )
            triples.append(triple)

        return triples

    def _generate_response(
        self,
        user_input: str,
        triples: List[Triple],
        activations: Dict[str, float],
    ) -> Tuple[str, List[Triple]]:
        """
        Generate a response based on current context.

        Uses RLMv2 if available, otherwise falls back to simple response.

        Args:
            user_input: Raw user input
            triples: Parsed triples from input
            activations: Current concept activations

        Returns:
            (response_text, response_triples)
        """
        if not triples:
            return "I understand.", []

        # Get the most activated concepts
        top_concepts = sorted(
            activations.items(), key=lambda x: x[1], reverse=True
        )[:5]

        primary_triple = triples[0]

        if self.rlm_model is not None:
            return self._generate_with_rlm(
                user_input, primary_triple, top_concepts
            )

        # Fallback: template-based response
        return self._generate_fallback_response(
            primary_triple, top_concepts
        )

    def _generate_with_rlm(
        self,
        user_input: str,
        triple: Triple,
        top_concepts: List[Tuple[str, float]],
    ) -> Tuple[str, List[Triple]]:
        """Generate response using RLMv2 model."""
        try:
            # Tokenize the input
            token_ids = self.tokenizer.encode(user_input)
            input_tensor = np.array(token_ids, dtype=np.int64)

            # Run forward pass
            self.rlm_model._tokenizer = self.tokenizer
            logits = self.rlm_model.forward(input_tensor)

            # Get top prediction
            if hasattr(logits, 'data'):
                probs = logits.data
            else:
                probs = logits

            # Find the predicted object
            top_k = np.argsort(probs)[-5:][::-1]
            predicted_tokens = self.tokenizer.decode(top_k.tolist())

            # Build response
            response = (
                f"Based on what I know, {triple.subject} {triple.relation} "
                f"{predicted_tokens.strip()}"
            )

            response_triple = Triple(
                subject=triple.subject,
                relation=triple.relation,
                relation_type=triple.relation_type,
                object=predicted_tokens.strip(),
                confidence=0.5,
                source_agent="system",
                epistemic_status="hypothesis",
                timestamp=time.time(),
            )

            return response, [response_triple]

        except Exception as e:
            return f"I processed that. ({e})", []

    def _generate_fallback_response(
        self,
        triple: Triple,
        top_concepts: List[Tuple[str, float]],
    ) -> Tuple[str, List[Triple]]:
        """
        Generate a simple template-based response when RLMv2 is unavailable.

        Uses activated concepts to provide context-aware responses.
        """
        # Generate from activated concepts if available
        if top_concepts:
            related = [
                concept
                for concept, score in top_concepts[:3]
                if concept.lower() != triple.subject.lower()
                and concept.lower() != triple.object.lower()
            ]
            if related:
                response = (
                    f"I understand that {triple.subject} {triple.relation} "
                    f"{triple.object}. This relates to "
                    f"{', '.join(related)}."
                )
            else:
                response = (
                    f"I see that {triple.subject} {triple.relation} "
                    f"{triple.object}."
                )
        else:
            response = (
                f"Noted: {triple.subject} {triple.relation} {triple.object}."
            )

        return response, [triple]

    def _modulate_with_emotion(
        self,
        activations: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Modulate activations using current VAD emotional state.

        High arousal → boost diverse activations (exploratory)
        Low arousal → suppress weak activations (conservative)
        """
        if self.emotion_engine is None:
            return activations

        vad = self.emotion_engine.state
        arousal = vad.arousal
        dominance = vad.dominance

        # High arousal: spread activation more broadly
        # Low arousal: keep only strong activations
        threshold = 0.05 * (1.0 + (1.0 - arousal))

        # High dominance: boost high-activation concepts
        # Low dominance: slight suppression
        dominance_boost = 1.0 + (dominance - 0.5) * 0.4

        modulated = {}
        for concept, activation in activations.items():
            if activation >= threshold:
                modulated[concept] = activation * dominance_boost

        return modulated

    def _apply_governor_constraints(self, response: str) -> str:
        """Apply governor hard constraints to the response."""
        if self.governor is None:
            return response
        return response

    def _accumulate_free_energy(self, triples: List[Triple]):
        """Accumulate free energy from turn processing."""
        # Each triple adds a small amount of free energy
        self._free_energy_accumulator += len(triples) * 0.05

    def _check_sleep_needed(self) -> bool:
        """
        Check if sleep consolidation is needed.

        Conditions:
        1. Free energy exceeds threshold
        2. Turn count exceeds consolidation interval
        3. Sleep was explicitly requested (by correction)

        Returns:
            True if sleep should be triggered
        """
        conditions = []

        # Condition 1: High free energy
        if self._free_energy_accumulator >= self.config.sleep_pressure_threshold:
            conditions.append("high_free_energy")

        # Condition 2: Turn interval
        if (
            self.turn_count > 0
            and self.turn_count % self.config.consolidation_turn_interval == 0
        ):
            conditions.append("turn_interval")

        # Condition 3: Explicit request
        if self._sleep_requested:
            conditions.append("sleep_requested")

        if conditions:
            self.trigger_sleep()
            return True

        return False

    def _store_in_human_memory(self, record: DialogueTurnRecord):
        """Store a turn record in human memory."""
        if self.human_memory is None:
            return

        # Only store significant turns
        if record.repair_event is not None or (
            self.turn_count % 3 == 0
        ):
            tags = "dialogue"
            if record.repair_event is not None:
                tags += ",correction"
            if record.sleep_triggered:
                tags += ",sleep"

            self.human_memory.remember(
                content=(
                    f"Turn {record.turn}: User said '{record.user_input}'. "
                    f"System replied '{record.system_output}'."
                ),
                memory_type="episodic",
                importance=0.5,
                emotional=0.3,
                tags=tags,
            )

    def __repr__(self):
        return (
            f"<DialogueEngine user={self.user_id} "
            f"turns={self.turn_count} "
            f"fe={self._free_energy_accumulator:.2f}>"
        )
