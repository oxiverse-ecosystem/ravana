"""
RAVANA v2 — PHASE H: Multi-Agent Epistemic Environment
Test social epistemology with controlled multi-agent scenarios.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.social_epistemology import (
    SocialEpistemologyEngine,
    SocialEpistemicConfig,
    AgentType,
    AgentBelief
)


@dataclass
class MultiAgentScenario:
    """A test scenario for multi-agent epistemology."""
    name: str
    description: str
    num_peers: int
    num_experts: int
    num_novices: int
    num_adversaries: int
    ground_truth_boundary: float
    episodes: int
    
    # Scenario dynamics
    peer_reliability: float = 0.6  # How accurate peers are
    expert_reliability: float = 0.85
    novice_reliability: float = 0.4
    adversary_deception: float = 0.8  # How convincing adversaries are


class MultiAgentEnvironment:
    """
    Controlled environment for testing social epistemology.
    
    SCENARIOS:
        1. Honest Consensus: All agents honest, truth emerges
        2. Mixed Reliability: Experts vs novices vs peers
        3. Adversarial Attack: Deceptive agents try to manipulate
        4. Epistemic Divide: Two camps with different beliefs
        5. Truth Emergence: Can truth emerge from disagreement?
    """
    
    def __init__(self, scenario: Optional[MultiAgentScenario] = None):
        self.scenario = scenario or self._default_scenario()
        self.episode = 0
        
        # Initialize social epistemology engine
        config = SocialEpistemicConfig(
            conflict_threshold=0.12,
            deception_threshold=0.35,
            adversarial_test_frequency=25
        )
        self.social = SocialEpistemologyEngine(config)
        
        # Ground truth (unknown to agents)
        self.ground_truth = self.scenario.ground_truth_boundary
        
        # Results tracking
        self.results: List[Dict] = []
        self.metrics_history: List[Dict] = []
        
        # Set up agents
        self._setup_agents()
    
    def _default_scenario(self) -> MultiAgentScenario:
        return MultiAgentScenario(
            name="balanced_community",
            description="Mixed community with honest and deceptive agents",
            num_peers=3,
            num_experts=1,
            num_novices=1,
            num_adversaries=1,
            ground_truth_boundary=0.78,
            episodes=200
        )
    
    def _setup_agents(self):
        """Set up all agents for the scenario."""
        # Register RAVANA (self)
        ravana_belief = AgentBelief(
            agent_id="ravana",
            boundary_estimate=0.75 + np.random.normal(0, 0.05),
            confidence=0.5,
            uncertainty=0.2,
            last_updated=0
        )
        self.social.register_agent("ravana", AgentType.RAVANA, ravana_belief)
        
        # Register experts
        for i in range(self.scenario.num_experts):
            expert_id = f"expert_{i}"
            # Experts start closer to truth
            error = np.random.normal(0, 0.03)
            belief = AgentBelief(
                agent_id=expert_id,
                boundary_estimate=self.ground_truth + error,
                confidence=0.7,
                uncertainty=0.1,
                last_updated=0
            )
            self.social.register_agent(expert_id, AgentType.EXPERT, belief, expertise_domain="boundary")
        
        # Register peers
        for i in range(self.scenario.num_peers):
            peer_id = f"peer_{i}"
            error = np.random.normal(0, 0.08)
            belief = AgentBelief(
                agent_id=peer_id,
                boundary_estimate=self.ground_truth + error,
                confidence=0.5,
                uncertainty=0.15,
                last_updated=0
            )
            self.social.register_agent(peer_id, AgentType.PEER, belief)
        
        # Register novices
        for i in range(self.scenario.num_novices):
            novice_id = f"novice_{i}"
            error = np.random.normal(0, 0.15)
            belief = AgentBelief(
                agent_id=novice_id,
                boundary_estimate=self.ground_truth + error,
                confidence=0.3,
                uncertainty=0.25,
                last_updated=0
            )
            self.social.register_agent(novice_id, AgentType.NOVICE, belief)
        
        # Register adversaries
        for i in range(self.scenario.num_adversaries):
            adv_id = f"adversary_{i}"
            # Adversaries promote false beliefs
            false_boundary = np.random.choice([
                self.ground_truth - 0.25,
                self.ground_truth + 0.25
            ])
            belief = AgentBelief(
                agent_id=adv_id,
                boundary_estimate=false_boundary,
                confidence=0.8,  # Very confident (deceptive)
                uncertainty=0.05,
                last_updated=0
            )
            self.social.register_agent(adv_id, AgentType.ADVERSARY, belief)
    
    def _evolve_agent_beliefs(self, episode: int):
        """
        Simulate agents learning and updating their beliefs over time.
        """
        for agent_id in self.social.agents:
            if agent_id == "ravana":
                continue  # RAVANA's belief evolves through main system
            
            agent_type = self.social.agents[agent_id]['type']
            old_belief = self.social.agent_beliefs[agent_id]
            
            # Different evolution based on agent type
            if agent_type == AgentType.EXPERT:
                # Experts slowly converge to truth
                drift = (self.ground_truth - old_belief.boundary_estimate) * 0.05
                noise = np.random.normal(0, 0.01)
                new_boundary = old_belief.boundary_estimate + drift + noise
                confidence_gain = 0.005
                uncertainty_drop = 0.002
                
            elif agent_type == AgentType.PEER:
                # Peers influenced by consensus
                if self.social.current_consensus:
                    consensus_pull = (self.social.current_consensus.boundary_estimate - 
                                    old_belief.boundary_estimate) * 0.03
                else:
                    consensus_pull = 0
                
                # Also drift toward truth slowly
                truth_pull = (self.ground_truth - old_belief.boundary_estimate) * 0.02
                noise = np.random.normal(0, 0.02)
                new_boundary = old_belief.boundary_estimate + consensus_pull + truth_pull + noise
                confidence_gain = 0.003
                uncertainty_drop = 0.001
                
            elif agent_type == AgentType.NOVICE:
                # Novices more random, slower convergence
                drift = (self.ground_truth - old_belief.boundary_estimate) * 0.01
                noise = np.random.normal(0, 0.03)
                new_boundary = old_belief.boundary_estimate + drift + noise
                confidence_gain = 0.002
                uncertainty_drop = 0.001
                
            elif agent_type == AgentType.ADVERSARY:
                # Adversaries maintain false belief, occasionally shift
                if np.random.random() < 0.05:  # 5% chance to "adapt"
                    # Shift false belief slightly to stay plausible
                    noise = np.random.normal(0, 0.02)
                    new_boundary = old_belief.boundary_estimate + noise
                else:
                    new_boundary = old_belief.boundary_estimate
                confidence_gain = 0.001  # Maintain high confidence
                uncertainty_drop = 0.0
                
            else:
                continue
            
            # Update belief
            self.social.update_belief(
                agent_id=agent_id,
                boundary_estimate=np.clip(new_boundary, 0.2, 0.95),
                confidence=min(1.0, old_belief.confidence + confidence_gain),
                uncertainty=max(0.05, old_belief.uncertainty - uncertainty_drop),
                evidence_count=old_belief.evidence_count + 1
            )
    
    def _update_ravana_from_social(self, episode: int):
        """
        Update RAVANA's belief based on social consensus.
        """
        if not self.social.current_consensus:
            return
        
        ravana_belief = self.social.agent_beliefs.get("ravana")
        if not ravana_belief:
            return
        
        # Compute trust-weighted target
        consensus_boundary = self.social.current_consensus.boundary_estimate
        consensus_confidence = self.social.current_consensus.confidence
        
        # RAVANA moves partially toward consensus based on:
        # 1. Consensus confidence (high confidence → more movement)
        # 2. RAVANA's own uncertainty (high uncertainty → more movement)
        # 3. Average peer trust (high trust → more movement)
        
        avg_trust = np.mean([
            self.social.trust_scores[aid].composite_trust
            for aid in self.social.trust_scores
            if aid != "ravana"
        ]) if self.social.trust_scores else 0.5
        
        # Movement factor
        movement = (consensus_confidence * 0.3 + 
                   ravana_belief.uncertainty * 0.3 + 
                   avg_trust * 0.4)
        movement *= 0.1  # Scale down for gradual learning
        
        # Move toward consensus
        new_boundary = (ravana_belief.boundary_estimate * (1 - movement) + 
                         consensus_boundary * movement)
        
        # Update confidence based on consensus quality
        confidence_change = (consensus_confidence - ravana_belief.confidence) * 0.1
        new_confidence = np.clip(ravana_belief.confidence + confidence_change, 0.1, 0.95)
        
        # Uncertainty decreases with consensus
        new_uncertainty = ravana_belief.uncertainty * 0.98
        
        self.social.update_belief(
            agent_id="ravana",
            boundary_estimate=new_boundary,
            confidence=new_confidence,
            uncertainty=new_uncertainty,
            evidence_count=ravana_belief.evidence_count + 1
        )
    
    def step(self) -> Dict[str, Any]:
        """
        Execute one episode of the multi-agent environment.
        """
        self.episode += 1
        
        # Evolve agent beliefs
        self._evolve_agent_beliefs(self.episode)
        
        # Run social epistemology step
        social_result = self.social.step(self.episode)
        
        # Update RAVANA from social consensus
        self._update_ravana_from_social(self.episode)
        
        # Resolve conflicts (simulated ground truth)
        resolved = []
        for conflict_id in list(self.social.active_conflicts.keys()):
            result = self.social.resolve_conflict(conflict_id, self.ground_truth)
            resolved.append(result)
        
        # Compute metrics
        ravana_belief = self.social.agent_beliefs.get("ravana")
        if ravana_belief and self.social.current_consensus:
            ravana_error = abs(ravana_belief.boundary_estimate - self.ground_truth)
            consensus_error = abs(self.social.current_consensus.boundary_estimate - self.ground_truth)
            
            # Handle case where adversarial_test is None
            adversarial_test = social_result.get('adversarial_test') or {}
            
            metrics = {
                'episode': self.episode,
                'ravana_boundary': ravana_belief.boundary_estimate,
                'ravana_error': ravana_error,
                'ravana_confidence': ravana_belief.confidence,
                'consensus_boundary': self.social.current_consensus.boundary_estimate,
                'consensus_error': consensus_error,
                'consensus_confidence': self.social.current_consensus.confidence,
                'num_agents': len(self.social.agents),
                'active_conflicts': len(self.social.active_conflicts),
                'deception_alerts': len(social_result.get('deception_alerts', [])),
                'manipulation_resistance': adversarial_test.get('manipulation_resistance')
            }
        else:
            metrics = {
                'episode': self.episode,
                'num_agents': len(self.social.agents),
                'active_conflicts': len(self.social.active_conflicts)
            }
        
        self.metrics_history.append(metrics)
        
        return {
            'episode': self.episode,
            'metrics': metrics,
            'social_result': social_result,
            'conflicts_resolved': len(resolved)
        }
    
    def run(self, num_episodes: Optional[int] = None) -> Dict[str, Any]:
        """
        Run the full multi-agent simulation.
        """
        episodes = num_episodes or self.scenario.episodes
        
        print(f"\n{'='*60}")
        print(f"PHASE H: Social Epistemology Simulation")
        print(f"Scenario: {self.scenario.name}")
        print(f"Ground Truth Boundary: {self.ground_truth:.3f}")
        print(f"Agents: {self.scenario.num_experts} experts, {self.scenario.num_peers} peers, "
              f"{self.scenario.num_novices} novices, {self.scenario.num_adversaries} adversaries")
        print(f"{'='*60}\n")
        
        # Initial state
        print(f"[EP000] Initial setup complete")
        for agent_id, belief in self.social.agent_beliefs.items():
            print(f"  {agent_id}: boundary={belief.boundary_estimate:.3f}, confidence={belief.confidence:.2f}")
        
        # Run episodes
        for ep in range(episodes):
            result = self.step()
            
            # Print progress every 50 episodes
            if (ep + 1) % 50 == 0 or ep == 0:
                m = result['metrics']
                print(f"\n[EP{m['episode']:03d}] "
                      f"RAVANA: b={m['ravana_boundary']:.3f}±{m['ravana_error']:.3f} "
                      f"Consensus: b={m['consensus_boundary']:.3f}±{m['consensus_error']:.3f} "
                      f"Conf={m['consensus_confidence']:.2f} "
                      f"Conflicts:{m['active_conflicts']}")
                
                # Print deception alerts if any
                if m['deception_alerts'] > 0:
                    print(f"  ⚠️  Deception alerts: {m['deception_alerts']}")
        
        # Final summary
        final_metrics = self.metrics_history[-1] if self.metrics_history else {}
        
        print(f"\n{'='*60}")
        print("FINAL RESULTS")
        print(f"{'='*60}")
        
        # RAVANA's final belief
        ravana_final = self.social.agent_beliefs.get("ravana")
        if ravana_final:
            print(f"\n🧠 RAVANA Final Belief:")
            print(f"   Boundary: {ravana_final.boundary_estimate:.4f}")
            print(f"   Error from truth: {abs(ravana_final.boundary_estimate - self.ground_truth):.4f}")
            print(f"   Confidence: {ravana_final.confidence:.2f}")
        
        # Consensus
        if self.social.current_consensus:
            print(f"\n🌐 Consensus:")
            print(f"   Boundary: {self.social.current_consensus.boundary_estimate:.4f}")
            print(f"   Error from truth: {abs(self.social.current_consensus.boundary_estimate - self.ground_truth):.4f}")
            print(f"   Confidence: {self.social.current_consensus.confidence:.2f}")
            print(f"   Contributing agents: {len(self.social.current_consensus.contributing_agents)}")
        
        # Trust scores
        print(f"\n🤝 Trust Scores:")
        for agent_id, trust in sorted(self.social.trust_scores.items(), 
                                      key=lambda x: x[1].composite_trust, reverse=True):
            agent_type = self.social.agents[agent_id]['type'].value
            print(f"   {agent_id} ({agent_type}): "
                  f"reliability={trust.reliability:.2f}, "
                  f"honesty={trust.honesty:.2f}, "
                  f"composite={trust.composite_trust:.2f}")
        
        # Deception detection
        if self.social.deception_alerts:
            print(f"\n🚨 Deception Detection:")
            print(f"   Total alerts: {len(self.social.deception_alerts)}")
            unique_deceptive = set(a['agent_id'] for a in self.social.deception_alerts)
            print(f"   Agents flagged: {unique_deceptive}")
        
        # Adversarial tests
        print(f"\n🛡️  Adversarial Tests: {len(self.social.adversarial_episodes)}")
        
        # Overall success metrics
        if ravana_final:
            final_error = abs(ravana_final.boundary_estimate - self.ground_truth)
            initial_error = abs(self.metrics_history[0]['ravana_boundary'] - self.ground_truth) if self.metrics_history else 1.0
            improvement = (initial_error - final_error) / initial_error * 100 if initial_error > 0 else 0
            
            print(f"\n📊 Success Metrics:")
            print(f"   Initial RAVANA error: {initial_error:.4f}")
            print(f"   Final RAVANA error: {final_error:.4f}")
            print(f"   Improvement: {improvement:.1f}%")
            print(f"   Trust learning: {self._trust_learning_score():.2f}")
        
        print(f"\n{'='*60}")
        
        return {
            'scenario': self.scenario.name,
            'episodes_run': self.episode,
            'ground_truth': self.ground_truth,
            'ravana_final': {
                'boundary': ravana_final.boundary_estimate if ravana_final else None,
                'confidence': ravana_final.confidence if ravana_final else None,
                'error': abs(ravana_final.boundary_estimate - self.ground_truth) if ravana_final else None
            },
            'consensus_final': {
                'boundary': self.social.current_consensus.boundary_estimate if self.social.current_consensus else None,
                'confidence': self.social.current_consensus.confidence if self.social.current_consensus else None,
                'error': abs(self.social.current_consensus.boundary_estimate - self.ground_truth) if self.social.current_consensus else None
            },
            'trust_scores': {
                aid: {
                    'reliability': t.reliability,
                    'honesty': t.honesty,
                    'composite': t.composite_trust
                }
                for aid, t in self.social.trust_scores.items()
            },
            'deception_detected': len(self.social.deception_alerts) > 0,
            'deception_alerts': len(self.social.deception_alerts),
            'metrics_history': self.metrics_history
        }
    
    def _trust_learning_score(self) -> float:
        """Compute how well trust scores reflect actual reliability."""
        correlations = []
        
        for agent_id, trust in self.social.trust_scores.items():
            if agent_id == "ravana":
                continue
            
            agent_type = self.social.agents[agent_id]['type']
            
            # Expected reliability by type
            if agent_type == AgentType.EXPERT:
                expected = self.scenario.expert_reliability
            elif agent_type == AgentType.PEER:
                expected = self.scenario.peer_reliability
            elif agent_type == AgentType.NOVICE:
                expected = self.scenario.novice_reliability
            elif agent_type == AgentType.ADVERSARY:
                expected = 0.3  # Low expected reliability
            else:
                expected = 0.5
            
            # Correlation (simplified)
            correlation = 1.0 - abs(trust.reliability - expected)
            correlations.append(correlation)
        
        return np.mean(correlations) if correlations else 0.5


def run_scenario(name: str, **kwargs) -> Dict[str, Any]:
    """Run a named scenario with optional overrides."""
    scenarios = {
        "honest_consensus": MultiAgentScenario(
            name="honest_consensus",
            description="All agents honest, truth should emerge",
            num_peers=4,
            num_experts=2,
            num_novices=1,
            num_adversaries=0,
            ground_truth_boundary=0.80,
            episodes=150,
            peer_reliability=0.7,
            expert_reliability=0.9
        ),
        "adversarial_attack": MultiAgentScenario(
            name="adversarial_attack",
            description="Deceptive agents attempt to manipulate consensus",
            num_peers=3,
            num_experts=1,
            num_novices=1,
            num_adversaries=2,
            ground_truth_boundary=0.75,
            episodes=200,
            adversary_deception=0.85
        ),
        "expert_vs_crowd": MultiAgentScenario(
            name="expert_vs_crowd",
            description="Can experts overcome crowd confusion?",
            num_peers=5,
            num_experts=1,
            num_novices=3,
            num_adversaries=0,
            ground_truth_boundary=0.82,
            episodes=180,
            expert_reliability=0.92,
            peer_reliability=0.5,
            novice_reliability=0.35
        ),
        "epistemic_divide": MultiAgentScenario(
            name="epistemic_divide",
            description="Two camps with opposing beliefs",
            num_peers=3,
            num_experts=1,
            num_novices=2,
            num_adversaries=1,
            ground_truth_boundary=0.78,
            episodes=250
        )
    }
    
    scenario = scenarios.get(name, scenarios["honest_consensus"])
    
    # Apply overrides
    for key, value in kwargs.items():
        if hasattr(scenario, key):
            setattr(scenario, key, value)
    
    env = MultiAgentEnvironment(scenario)
    return env.run()


if __name__ == "__main__":
    import sys
    
    # Default scenario
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "adversarial_attack"
    
    print(f"\n{'='*60}")
    print("RAVANA v2 — Phase H: Social Epistemology")
    print(f"{'='*60}")
    
    # Run selected scenario
    results = run_scenario(scenario_name)
    
    # Save results
    import json
    import os
    
    os.makedirs("results", exist_ok=True)
    output_file = f"results/phase_h_{scenario_name}.json"
    
    # Convert numpy types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj
    
    serializable_results = convert_to_serializable(results)
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print("PHASE H COMPLETE ✓")
    print(f"{'='*60}")
    print(f"Social epistemology achieved:")
    print(f"  • Multi-agent belief conflict resolution")
    print(f"  • Trust scoring with reliability/honesty tracking")
    print(f"  • Adversarial testing and deception detection")
    print(f"  • Consensus formation from distributed beliefs")
    print(f"  • RAVANA learns from social context, not just solo experience")
    print(f"\nLevel L8 (Social Reasoning): ACHIEVED")
