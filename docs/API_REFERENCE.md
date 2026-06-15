# RAVANA API Reference

> **Complete function/class reference** for all public APIs across three layers.

---

## Table of Contents

1. [ravana_ml — ML Framework](#ravana_ml--ml-framework)
2. [ravana-v2/core — Cognitive Core](#ravana-v2core--cognitive-core)
3. [ravana — Unified Package](#ravana--unified-package)

---

## ravana_ml — ML Framework

### `ravana_ml` (Package Root)

```python
from ravana_ml import (
    # Tensors
    RawTensor, StateTensor, Parameter, Tensor,
    tensor, zeros, ones, randn, eye, arange, stack, cat, from_numpy,
    # Device
    Device, device, cuda, cuda_is_available,
    # Utils
    is_tensor, no_grad, save, load,
    # Submodules
    nn, graph, propagation, free_energy, plasticity, world, lab,
    currency, currencies,
)
```

**Constants:**
- `__version__ = '0.1.0'`

**Functions:**
```python
def is_tensor(obj) -> bool          # isinstance(obj, (RawTensor, StateTensor))
def no_grad() -> ContextManager     # compatibility context manager (no-op)
def save(obj, path: str) -> None    # save model/state dict
def load(path_or_model, path=None)  # load model/state dict
```

### `ravana_ml.tensor`

```python
from ravana_ml.tensor import RawTensor, StateTensor, Parameter, tensor, zeros, ones, randn, eye, arange, stack, cat, from_numpy

class RawTensor:
    def __init__(self, data: np.ndarray):
        self.data = data                    # np.ndarray
        self.shape = data.shape
        self.ndim = data.ndim
        self.dtype = data.dtype
        self.device = device                # Device('cpu')
    
    # Properties
    @property
    def T(self) -> 'RawTensor':             # transpose
    def reshape(self, *shape) -> 'RawTensor':
    def squeeze(self, dim=None) -> 'RawTensor':
    def unsqueeze(self, dim) -> 'RawTensor':
    
    # Operations (return new RawTensor)
    def __add__(self, other), __sub__, __mul__, __truediv__, __matmul__
    
    # Reductions
    def sum(self, dim=None, keepdim=False) -> 'RawTensor':
    def mean(self, dim=None, keepdim=False) -> 'RawTensor':
    def max(self, dim=None, keepdim=False):
    def min(self, dim=None, keepdim=False):
    def argmax(self, dim=None, keepdim=False):
    def argmin(self, dim=None, keepdim=False):
    
    def numpy(self) -> np.ndarray:
    def __repr__(self) -> str:

class StateTensor(RawTensor):
    def __init__(self, data, salience=1.0, free_energy=0.0, stability=0.5, decay=0.0):
        super().__init__(data)
        self.salience = salience
        self.free_energy = free_energy
        self.stability = stability
        self.decay = decay

class Parameter(StateTensor):
    """Learnable parameter with cognitive metadata."""
    pass

# Factory functions
def tensor(data, *, salience=1.0, free_energy=0.0, stability=0.5, decay=0.0) -> StateTensor
def zeros(*shape, **meta) -> StateTensor
def ones(*shape, **meta) -> StateTensor
def randn(*shape, **meta) -> StateTensor
def eye(n, **meta) -> StateTensor
def arange(n, **meta) -> StateTensor
def stack(tensors, dim=0) -> StateTensor
def cat(tensors, dim=0) -> StateTensor
def from_numpy(arr, **meta) -> StateTensor
```

### `ravana_ml.nn.module`

```python
from ravana_ml.nn.module import Module, Sequential, Linear, Embedding, LayerNorm, Dropout, GRUCell, ConceptAttentionHead

class Module:
    def __call__(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)
    
    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def parameters(self) -> List[Parameter]:
        """Recursively collect all Parameters."""
    
    def named_parameters(self) -> List[Tuple[str, Parameter]]:
    
    def state_dict(self) -> Dict[str, np.ndarray]:
    
    def load_state_dict(self, state_dict: Dict[str, np.ndarray]) -> None:
    
    def register_module(self, name: str, module: 'Module') -> None:
    
    def modules(self) -> Iterator['Module']:
    
    def train(self, mode: bool = True) -> 'Module':
    
    def eval(self) -> 'Module':
    
    def to(self, device) -> 'Module':
        return self  # CPU only
    
    def __repr__(self) -> str:

class Sequential(Module):
    def __init__(self, *modules: Module):
    
    def forward(self, x) -> Any:

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        self.weight = Parameter(...)      # (out, in)
        self.bias = Parameter(...) if bias else None
    
    def forward(self, x: StateTensor) -> StateTensor:
        # x: (*, in) -> (*, out)

class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        self.weight = Parameter(...)      # (vocab, dim)
    
    def forward(self, indices: StateTensor) -> StateTensor:
        # indices: (*,) -> (*, dim)
    
    def embed_raw(self, idx: int) -> np.ndarray:
        """Direct weight access by int index."""

class LayerNorm(Module):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        self.weight = Parameter(...)      # (dim,)
        self.bias = Parameter(...)        # (dim,)
    
    def forward(self, x: StateTensor) -> StateTensor:

class Dropout(Module):
    def __init__(self, p: float = 0.5):
    
    def forward(self, x: StateTensor) -> StateTensor:

class GRUCell(Module):
    def __init__(self, input_size: int, hidden_size: int):
        # Weights: W_ir, W_hr, W_iz, W_hz, W_in, W_hn
        # Biases: b_ir, b_hr, b_iz, b_hz, b_in, b_hn
    def forward(self, x: StateTensor, h: StateTensor) -> StateTensor:

class ConceptAttentionHead(Module):
    def __init__(self, concept_dim: int, vocab_size: int, n_heads: int = 2):
        self.W_q = Linear(concept_dim, concept_dim)
        self.W_k = Linear(concept_dim, concept_dim)
        self.W_v = Linear(concept_dim, concept_dim)
        self.out_proj = Linear(concept_dim, vocab_size)
    
    def forward(self, concept_vecs: StateTensor) -> StateTensor:
        # concept_vecs: (n_concepts, concept_dim) -> (vocab_size,)
```

### `ravana_ml.nn.functional`

```python
from ravana_ml.nn import functional as F

def relu(x: StateTensor) -> StateTensor:
def softmax(x: StateTensor, dim: int = -1) -> StateTensor:
def log_softmax(x: StateTensor, dim: int = -1) -> StateTensor:
def cross_entropy(input: StateTensor, target: StateTensor) -> StateTensor:
def mse_loss(input: StateTensor, target: StateTensor) -> StateTensor:
def cosine_similarity(x1: StateTensor, x2: StateTensor, dim: int = -1) -> StateTensor:
def normalize(x: StateTensor, p: float = 2.0, dim: int = -1) -> StateTensor:
def dropout(x: StateTensor, p: float = 0.5, training: bool = True) -> StateTensor:
def linear(x: StateTensor, weight: StateTensor, bias: Optional[StateTensor] = None) -> StateTensor:
def embedding(indices: StateTensor, weight: StateTensor) -> StateTensor:
def layer_norm(x: StateTensor, weight: StateTensor, bias: StateTensor, eps: float = 1e-5) -> StateTensor:
```

### `ravana_ml.nn.rlm` (RLM v1)

```python
from ravana_ml.nn import RLM

class RLM(Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        concept_dim: int,
        n_concepts: int,
        n_hidden: int,
        n_layers: int = 3,
        max_seq_len: int = 128,
        free_energy_threshold: float = 8.0,
        sleep_interval: int = 100,
        tokenizer=None,
        replay_buffer_max: int = 500,
        replay_n_samples: int = 20,
        anchor_relation_vectors: bool = True,
        gate_concept_creation: bool = True,
        adaptive_downscale: bool = True,
        deep_sleep_every: int = 1,
    ):
    
    def learn(self, input_ids: np.ndarray, target_ids: np.ndarray) -> None:
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Returns logits (seq_len, vocab_size)."""
    
    def sleep_cycle(self) -> Dict[str, Any]:
    
    def accumulate_free_energy(self, error: StateTensor) -> None:
    
    def save(self, path: str) -> None:
    
    def save_zip(self, path: str) -> None:
    
    @classmethod
    def load(cls, path: str) -> 'RLM':
    
    @classmethod
    def load_zip(cls, path: str) -> 'RLM':
    
    def get_top_predictions(self, input_ids: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
```

### `ravana_ml.nn.rlm_v2` (RLM v2)

```python
from ravana_ml.nn import RLMv2

class RLMv2(Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        concept_dim: int,
        n_concepts: int,
        max_seq_len: int = 128,
        sleep_interval: int = 100,
        gate_concept_creation: bool = True,
        anchor_relation_vectors: bool = True,
        latent_dim: int = 96,
        hidden_dim: int = 128,
        **kwargs,
    ):
    
    def learn(self, input_ids: np.ndarray, target_ids: np.ndarray) -> None:
    
    def forward(self, input_ids: np.ndarray) -> np.ndarray:
        """Returns logits (vocab_size,)."""
    
    def sleep_cycle(self) -> Dict[str, Any]:
    
    def save(self, path: str) -> None:
    
    def save_zip(self, path: str) -> None:
    
    @classmethod
    def load(cls, path: str) -> 'RLMv2':
    
    @classmethod
    def load_zip(cls, path: str) -> 'RLMv2':
    
    def get_top_predictions(self, input_ids: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
```

**Key Attributes (RLMv2):**
```python
model.graph                    # ConceptGraph
model.relation_type_embed      # Embedding(n_rel_types, concept_dim)
model.subject_proj             # Linear(embed_dim, concept_dim)
model.concept_to_embed         # Linear(concept_dim, embed_dim)
model.propagation              # PropagationEngine
model.sleep_cycles_completed
model.total_free_energy
model._global_relation_priors  # {verb_token: offset_vector}
```

### `ravana_ml.graph`

```python
from ravana_ml.graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBinding, ConceptBindingMap

class ConceptGraph:
    def __init__(self, dim: int, max_nodes: int, 
                 anchor_relation_vectors: bool = True,
                 adaptive_downscale: bool = True):
    
    # Nodes
    def add_node(self, vector: np.ndarray, label: str = "") -> int:
    def get_node(self, node_id: int) -> Optional[ConceptNode]:
    def remove_node(self, node_id: int) -> bool:
    def find_similar(self, vector: np.ndarray, k: int = 10) -> List[Tuple[int, float]]:
    
    # Edges
    def add_edge(self, source: int, target: int, weight: float = 0.5,
                 shortcut: bool = False, edge_type: str = "excitatory",
                 relation_type: str = "semantic", relation_dim: int = 16,
                 confidence: float = 0.5) -> ConceptEdge:
    def get_edge(self, source: int, target: int) -> Optional[ConceptEdge]:
    def remove_edge(self, source: int, target: int) -> bool:
    
    # Activation
    def activate(self, node_id: int, amount: float = 1.0) -> None:
    def reset_activation(self) -> None:
    def spread_activation(self, steps: int = 3, k_active: int = 5, decay: float = 0.3) -> None:
    
    # Learning
    def hebbian_update(self, source: int, target: int, coactivation: float, lr: float = 0.01) -> None:
    def anti_hebbian_update(self, source: int, target: int, coactivation: float, lr: float = 0.01) -> None:
    
    # Structural
    def structural_step(self) -> Tuple[int, int]:  # (pruned, formed)
    def homeostatic_downscale(self, protection_threshold: float = 0.8, downscale_factor: float = 0.8) -> Tuple[float, float]:
    def reconcile_contradictions(self) -> int:
    def form_inhibitory_edges(self) -> int:
    def create_abstraction_cluster(self, child_ids: List[int], parent_label: str) -> int:
    
    # Input binding
    def bind_input(self, input_vector: np.ndarray, k: int = 5) -> List[int]:
    
    # Properties
    @property
    def nodes(self) -> Dict[int, ConceptNode]:
    @property
    def edges(self) -> Dict[Tuple[int, int], ConceptEdge]:
    @property
    def version(self) -> int:
    
    def __len__(self) -> int:
    def __repr__(self) -> str:

class ConceptNode:
    # See ARCHITECTURE.md for full attribute list
    id: int
    vector: np.ndarray
    core_vector: np.ndarray
    genesis_vector: np.ndarray
    label: str
    activation: float
    salience: float
    prediction_free_energy: float
    stability: float
    confidence: float
    contradiction_count: int
    fatigue: float
    contradiction_free_energy: float
    free_energy_history: List[float]
    free_energy_gradient: float
    level: int
    parent: Optional[int]
    children: Set[int]
    abstraction_degree: float
    last_activated: float
    activation_history: List[float]
    temporal_context: Optional[np.ndarray]
    
    @property
    def effective_activation(self) -> float:
    @property
    def drift_magnitude(self) -> float:
    def age(self) -> float:
    def decay(self, rate=0.01) -> None:
    def record_activation(self, context_vector=None) -> None:
    def recency_score(self, decay_rate=0.1) -> float:
    def frequency_score(self, window_seconds=86400) -> float:
    @property
    def plasticity(self) -> float:

class ConceptEdge:
    source: int
    target: int
    weight: float              # property with clamping
    confidence: float          # property
    edge_type: str             # "excitatory" | "inhibitory"
    relation_type: str         # "semantic" | "causal" | "temporal" | "analogical" | "contextual" | "inferred"
    shortcut: bool
    predicate_token_id: int
    relation_vector: np.ndarray
    prediction_free_energy: float
    stability: float
    prediction_count: int
    forward_pred_count: int
    backward_pred_count: int
    fisher_importance: float
    old_weight: float
    posterior_alpha: float
    posterior_beta: float
    agent_weights: Dict[str, float]
    source_metadata: Dict[str, Any]
    
    @property
    def effective_weight(self) -> float:
    @property
    def posterior_mean(self) -> float:
    @property
    def posterior_uncertainty(self) -> float:
    @property
    def plasticity(self) -> float:
    def get_weight_for_agent(self, agent_id: str) -> float:
    def update_weight_for_agent(self, agent_id: str, delta: float) -> None:

class ConceptBinding:
    token_id: int
    concept_id: int
    confidence: float
    source: str
    reinforcement_count: int
    last_used: float
    created_at: float
    decay_score: float
    ambiguity: float
    
    def reinforce(self, amount: float = 0.05) -> None:
    def decay(self, rate: float = 0.01) -> None:
    @property
    def strength(self) -> float:

class ConceptBindingMap:
    def bind(self, token_id: int, concept_id: int, confidence: float = 0.5, source: str = "learned") -> ConceptBinding:
    def get_concepts(self, token_id: int, min_confidence: float = 0.1) -> List[ConceptBinding]:
    def get_tokens(self, concept_id: int, min_confidence: float = 0.1) -> List[ConceptBinding]:
    def best_concept(self, token_id: int) -> Optional[int]:
    def best_token(self, concept_id: int) -> Optional[int]:
    def is_ambiguous(self, token_id: int, threshold: float = 0.3) -> bool:
    def ambiguity_score(self, token_id: int) -> float:
    def decay_all(self, rate: float = 0.01) -> None:
```

### `ravana_ml.propagation`

```python
from ravana_ml.propagation import PropagationEngine

class PropagationEngine:
    def __init__(self, graph: ConceptGraph):
    
    def get_prediction(self, active_nids: List[int], top_k: int = 5) -> List[int]:
    def get_activation_vector(self, nids: List[int]) -> np.ndarray:
    def measure_coherence(self, active_nids: List[int]) -> float:
```

### `ravana_ml.free_energy`

```python
from ravana_ml.free_energy import FreeEnergyAccumulator

class FreeEnergyAccumulator:
    def __init__(self, graph: ConceptGraph):
    
    def accumulate_semantic(self, error: float, salience: float = 1.0) -> None:
    def accumulate_linguistic(self, error: float, salience: float = 1.0) -> None:
    def accumulate_episodic(self, error: float, salience: float = 1.0) -> None:
    def accumulate_contradiction(self, error: float, salience: float = 1.0) -> None:
    def accumulate_abstraction(self, error: float, salience: float = 1.0) -> None:
    
    def total(self) -> float:
    def decay(self, rate: float = 0.1) -> None:
    def get_node_free_energy(self, node_id: int) -> float:
```

### `ravana_ml.plasticity`

```python
from ravana_ml.plasticity import HebbianPlasticity, AntiHebbianPlasticity, StructuralPlasticity

class HebbianPlasticity:
    def __init__(self, graph: ConceptGraph, lr: float = 0.03):
    def update(self, source: int, target: int, coactivation: float) -> None:

class AntiHebbianPlasticity:
    def __init__(self, graph: ConceptGraph, lr: float = 0.02):
    def update(self, source: int, target: int, coactivation: float) -> None:

class StructuralPlasticity:
    def __init__(self, graph: ConceptGraph, prune_threshold: float = 0.005, form_threshold: float = 0.3):
    def step(self) -> Tuple[int, int]:  # (pruned, formed)
```

### `ravana_ml.tokenizer`

```python
from ravana_ml.tokenizer import (
    TokenizerInterface, WordTokenizer, BPETokenizer, SimpleTokenizer, PixelTokenizer, get_tokenizer
)

class TokenizerInterface:
    def encode(self, text: str) -> List[int]: ...
    def decode(self, token_ids: List[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...

class WordTokenizer(TokenizerInterface):
    def __init__(self):
    def encode(self, text: str) -> List[int]:
    def decode(self, token_ids: List[int]) -> str:
    @property
    def vocab_size(self) -> int:
    # Dynamic vocab built from seen text

class BPETokenizer(TokenizerInterface):
    def __init__(self, encoding_name: str = "gpt2"):
    # Requires tiktoken

class SimpleTokenizer(TokenizerInterface):
    def __init__(self):
    # Char-level, 256 vocab

class PixelTokenizer:
    PIXEL_OFFSET = 0
    LABEL_OFFSET = 256
    N_CLASSES = 10
    
    def encode_image(self, image: np.ndarray) -> np.ndarray:
    def encode_label(self, label: int) -> int:
    def decode_label(self, token_id: int) -> int:
    @property
    def vocab_size(self) -> int:  # 266

def get_tokenizer(name: str = "word") -> TokenizerInterface:
    # "word", "bpe", "gpt2", "simple", or tiktoken encoding name
```

### `ravana_ml.currencies` / `ravana_ml.currency`

```python
from ravana_ml.currencies import CognitiveCurrencies
from ravana_ml.currency import create_rlm_currency

class CognitiveCurrencies:
    identity_strength: float
    identity_momentum: float
    identity_history: List[float]
    valence: float
    arousal: float
    dominance: float
    accumulated_meaning: float
    meaning_history: List[float]
    sleep_pressure: float
    sleep_pressure_threshold: float
    regulation_mode: str
    dissonance_ema: float

def create_rlm_currency() -> CognitiveCurrencies:
    """Returns CognitiveCurrencies with RLM-compatible property aliases."""
```

---

## ravana-v2/core — Cognitive Core

### Phase A: Core Regulation

```python
from core.governor import Governor, GovernorConfig, RegulationMode, CognitiveSignals, RegulatedOutput, ClampEvent, ClampDiagnostics
from core.identity import IdentityEngine, IdentityState
from core.resolution import ResolutionEngine, ResolutionMemory
from core.state import StateManager, CognitiveState

# Governor
class GovernorConfig:  # dataclass
    max_dissonance: float = 0.95
    min_dissonance: float = 0.15
    target_dissonance: float = 0.30
    center_target: float = 0.50
    max_identity: float = 0.95
    soft_limit: float = 0.70
    boundary_k: float = 12.0
    min_pressure: float = 0.2
    min_identity: float = 0.10
    dissonance_target: float = 0.35
    identity_target: float = 0.85
    exploration_threshold: float = 0.25
    resolution_threshold: float = 0.60
    use_smoothed_dissonance: bool = True
    smoothing_alpha: float = 0.2
    recovery_boost: float = 0.15
    crisis_threshold: float = 0.90
    plateau_window: int = 50
    plateau_tolerance: float = 0.02
    clamp_alert_threshold: float = 0.05

class Governor:
    def __init__(self, config: Optional[GovernorConfig] = None):
    def regulate(self, current_dissonance: float, current_identity: float, 
                 signals: CognitiveSignals, episode: int = 0) -> RegulatedOutput:
    def get_clamp_report(self) -> str:
    def get_clamp_metrics(self) -> Dict[str, Any]:
    def get_health_metrics(self) -> Dict[str, Any]:

class RegulationMode(Enum):
    NORMAL = "normal"
    EXPLORATION = "exploration"
    RESOLUTION = "resolution"
    RECOVERY = "recovery"
    PLATEAU = "plateau"

class CognitiveSignals:  # dataclass
    dissonance_delta: float = 0.0
    identity_delta: float = 0.0
    exploration_drive: float = 0.0
    resolution_potential: float = 0.0
    trend: float = 0.0
    predicted_dissonance: float = 0.0
    horizon: int = 3
    source: str = "unknown"
    confidence: float = 0.5

class RegulatedOutput:  # dataclass
    dissonance_delta: float = 0.0
    identity_delta: float = 0.0
    mode: RegulationMode = RegulationMode.NORMAL
    dampened: bool = False
    boosted: bool = False
    capped: bool = False
    reason: str = ""
    raw_input: Optional[CognitiveSignals] = None

# Identity
class IdentityEngine:
    def __init__(self, initial_strength: float = 0.5):
    def step(self, correctness: bool, difficulty: float, stakes: float) -> IdentityState:

class IdentityState:  # dataclass
    strength: float
    momentum: float
    history: List[float]
    wisdom: float

# Resolution
class ResolutionEngine:
    def process(self, correctness: bool, difficulty: float, 
                identity_strength: float, dissonance: float) -> ResolutionResult:

class ResolutionResult:  # dataclass
    wisdom_delta: float
    identity_delta: float
    partial_credit: float

# State Manager
class StateManager:
    def __init__(self, governor, resolution_engine, identity_engine, emotion_engine,
                 sleep_engine, dual_process, meaning_engine, global_workspace, human_memory):
    def step(self, correctness: bool, difficulty: float, novelty: float, 
             stakes: float, effort: float) -> StepResult:
    @property
    def state(self) -> CognitiveState:

class CognitiveState:  # dataclass
    dissonance: float = 0.5
    identity: float = 0.5
    wisdom: float = 0.0
    cycle: int = 0
    episode: int = 0
    valence: float = 0.0
    arousal: float = 0.0
    dominance: float = 0.0
    mode: str = "normal"
    processing_route: str = "system1_fast"
    sleep_cycles: int = 0
    
    def snapshot(self) -> Dict[str, Any]:
```

### Phase B: Adaptation
```python
from core.adaptation import PolicyTweakLayer, AdaptiveGovernorBridge, AdaptationConfig
```

### Phase C: Strategy Learning
```python
from core.strategy import StrategyLayer, StrategyConfig, ExplorationMode, ModeSelection, BehavioralContext
from core.strategy_learning import StrategyLearningLayer, ModeOutcome, LearningConfig, StrategyWithLearning

class ExplorationMode(Enum):
    AGGRESSIVE = "aggressive"
    SAFE = "safe"
    STABILIZE = "stabilize"
    RECOVER = "recover"
```

### Phase D: Intent & Planning
```python
from core.intent import IntentEngine, IntentConfig, IntentAwareStrategy, SystemObjective
from core.planning import MicroPlanner, PlanningConfig, SimulatedFuture
```

### Phase E: Environment
```python
from core.environment import NonStationaryEnvironment, EnvironmentConfig, HiddenDynamics, WorldState
```

### Phase F: World Model
```python
from core.predictive_world import LearnedWorldModel, WorldModelConfig, PredictedState, AnomalyEvent, FalseWorldTester
```

### Phase F.5: Belief Reasoner
```python
from core.belief_reasoner import BeliefReasoner, BeliefConfig, Hypothesis, EvidenceEvent
```

### Phase G: Active Epistemology
```python
from core.active_epistemology import ActiveEpistemology, VoIConfig, InformationGainMethod, HypothesisDrivenActionSelector
```

### Phase G.5: Surgical Probes
```python
from core.surgical_probes import SurgicalProbeSelector, SurgicalProbeConfig, ProbeType, ProbeExperiment, SurgicalProbing
```

### Phase J: Hypothesis & Occam
```python
from core.hypothesis_generation import HypothesisGenerator, GenerationConfig, HypothesisType, GeneratedHypothesis
from core.occam_layer import OccamLayer, OccamConfig, HypothesisScore, DisciplinedBeliefSystem
```

### Phase K: Emotion & Empathy
```python
from core.emotion import VADEmotionEngine, VADConfig, VADState
from core.empathy import EmpathyEngine, EmpathyConfig, OtherMind

class VADState:  # dataclass
    valence: float      # [-1, 1]
    arousal: float      # [0, 1]
    dominance: float    # [0, 1]
    emotional_label: str
```

### Phase L: Sleep & Dual Process
```python
from core.sleep import SleepConsolidation, SleepConfig, SleepStage, SleepRecord, DreamPerturbationType
from core.dual_process import DualProcessController, DualProcessConfig, ProcessingRoute, RouteDecision

class SleepStage(Enum):
    AWAKE = "awake"
    TOPOLOGY_ANALYSIS = "topology_analysis"
    PATTERN_COMPRESSION = "pattern_compression"
    ABSTRACTION_COMPRESSION = "abstraction_compression"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"
    MODEL_UPDATE = "model_update"
    INTEGRATION = "integration"

class DreamPerturbationType(Enum):
    COUNTERFACTUAL_REVERSAL = "counterfactual_reversal"
    EMOTIONAL_FLIP = "emotional_flip"
    FAILURE_OVERSAMPLE = "failure_oversample"
    SYMBOLIC_RECOMBINATION = "symbolic_recombination"

class ProcessingRoute(Enum):
    SYSTEM1_FAST = "system1_fast"
    SYSTEM2_SLOW = "system2_slow"
```

### Phase M: Meaning
```python
from core.meaning import MeaningEngine, MeaningConfig, MeaningRecord
```

### Phase N: Global Workspace
```python
from core.global_workspace import GlobalWorkspace, GWConfig, GWContent
```

### Phase O: Human Memory
```python
from core.human_memory import HumanMemoryEngine, HumanMemoryConfig, HumanMemoryRecord
```

### Phase P: Dialogue
```python
from core.dialogue_context import DialogueContext, ActiveSubgraph, Triple, DialogueState
from core.conversational_repair import ConversationalRepair, RepairEvent, CorrectionType
```

---

## ravana — Unified Package

```python
import ravana as torch
from ravana import nn, cognitive, graph, propagation, world, lab

# Re-exports all ravana_ml symbols
from ravana import (
    RawTensor, StateTensor, Parameter, Tensor,
    tensor, zeros, ones, randn, eye, arange, stack, cat, from_numpy,
    Device, device, cuda, cuda_is_available,
    is_tensor, no_grad, save, load,
)

# Neural modules
from ravana.nn import RLM, Linear, Embedding, LayerNorm, Dropout, GRUCell, ConceptAttentionHead, functional as F

# Cognitive
from ravana.cognitive import CognitiveFramework, FrameworkConfig, FrameworkState

# Graph
from ravana.graph import ConceptGraph, ConceptNode, ConceptEdge, ConceptBindingMap

# Propagation
from ravana.propagation import PropagationEngine

# World & Lab
from ravana.world import GridWorld, ContinuousWorld, SymbolicWorld
from ravana.lab import analyze_concept_graph, plot_activation_dynamics, compute_coherence_trajectory, visualize_sleep_cycle, diagnose_learning
```

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [ML Framework](ML_FRAMEWORK.md)
- [Cognitive Core](COGNITIVE_CORE.md)
- [Unified Package](UNIFIED_PACKAGE.md)
- [Getting Started](GETTING_STARTED.md)