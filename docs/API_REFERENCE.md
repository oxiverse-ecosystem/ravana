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

**Important Configuration Note:** For optimal performance with the entity-specific adapter (enables test-time adaptation for held-out subjects), set `latent_dim=embed_dim`. If they differ, the model automatically projects embeddings to latent space before applying the adapter, adding slight computational overhead.

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

### `ravana.core.situation_model` — SituationModel (DMN Workspace)

```python
from ravana.core.situation_model import SituationModel, SituationState

@dataclass
class SituationState:
    blended_vector: Optional[np.ndarray] = None
    dmn_state: Optional[np.ndarray] = None
    content_vector: Optional[np.ndarray] = None
    context_vector: Optional[np.ndarray] = None
    active_concepts: Dict[str, float] = field(default_factory=dict)
    narrative_theme: str = ""
    coherence: float = 0.5
    dmn_decay: float = 0.6
    blend_temperature: float = 0.7

class SituationModel:
    """Default Mode Network (DMN) continuous cognitive workspace."""
    def __init__(self, dim: int = 64, dmn_decay: float = 0.6):
        self.dim: int
        self.state: SituationState
    
    def update(self,
               concept_embeddings: Dict[str, np.ndarray],
               activations: Dict[str, float],
               graph_get_vector_fn: Optional[Callable] = None,
               sentence_vector: Optional[np.ndarray] = None,
               context_vector_input: Optional[np.ndarray] = None) -> np.ndarray:
        """Update blended vector, DMN decay state, content/context orthogonality, and coherence."""
        
    def get_blended_vector(self) -> np.ndarray: ...
    def get_dmn_state(self) -> np.ndarray: ...
    def get_content_vector(self) -> np.ndarray: ...
    def get_context_vector(self) -> np.ndarray: ...
    def get_coherence(self) -> float: ...
    def get_narrative_theme(self) -> str: ...
    def get_active_concepts(self) -> Dict[str, float]: ...
    def reset(self) -> None: ...
```

### `ravana.core.event_schema` — EventSchemaLibrary (Procedural Scripts)

```python
from ravana.core.event_schema import EventSchemaLibrary, EventSchema, ProcessStep

@dataclass
class ProcessStep:
    concept: str
    verb: str
    description: str = ""
    duration: str = ""
    causal_precedents: List[str] = field(default_factory=list)
    confidence: float = 0.5

@dataclass
class EventSchema:
    name: str
    steps: List[ProcessStep]
    source: str = "discovered"
    context: str = ""
    confidence: float = 0.5

class EventSchemaLibrary:
    """Library of hippocampal schemas and process chains."""
    def __init__(self):
        self._schemas: Dict[str, EventSchema]
        
    def seed_default_schemas(self) -> None:
        """Seed with default universal schemas (trust, love, learning, growth, etc.)."""
        
    def get_schema(self, concept: str) -> Optional[EventSchema]: ...
    def has_schema(self, concept: str) -> bool: ...
    def add_schema(self, schema: EventSchema) -> None: ...
    def discover_schema_from_text(self, text: str, concept: str) -> Optional[EventSchema]: ...
    def blend_schemas(self, s1: EventSchema, s2: EventSchema, name: str) -> EventSchema: ...
```

### `ravana.chat.user_model` — UserModel (Theory of Mind)

```python
from ravana.chat import UserModel

class UserModel:
    """Theory of Mind model tracking user-specific knowledge, preferences, emotional state, and goals."""
    def __init__(self):
        self.edge_reactivations: Dict[Tuple[str, str], int]  # (from, to) -> count
        self.query_concepts: Set[str]
        self.user_name: str
        self.preferences: Dict[str, Any]  # likes, interests, favorites
        self.knowledge_model: Dict[str, float]  # concept -> familiarity
        self.learning_goals: Dict[str, int]  # concept -> exploration count
        self.emotional_rapport: Dict[str, float]  # concept -> valence rapport
        self.cognitive_style: str  # 'curious' | 'skeptical' | 'practical' | 'balanced'
        self.engagement_level: float
        self.conversation_depth: float
        self.interaction_count: int
        self.relationship_depth: float
        self.goals: List[str]  # history of inferred goals
        self.last_goal: str  # 'LEARNING' | 'DEBUGGING' | 'EXPLORING'
        self.emotional_state: Dict[str, float]  # {valence, arousal, dominance}
        self.belief_state: Dict[str, Dict]
        self.interaction_history: List[Dict]  # capped at 100 entries
    
    def observe_chain(self, hops: List[Tuple[str, str]], is_user_query: bool = False) -> None:
        """Record graph traversal hops and update knowledge model."""
    
    def observe_user_query(self, query: str, subject: str, valence: float) -> None:
        """Process user message: infer emotion, update preferences, track topics, infer goals."""
    
    def infer_user_goal(self, query: str) -> str:
        """Classify query into LEARNING, DEBUGGING, or EXPLORING."""
    
    def infer_user_knows(self, concept: str) -> float:
        """Return familiarity score [0-1] for a concept."""
    
    def infer_user_wants_to_learn(self, concept: str) -> float:
        """Return desire-to-learn score [0-1] for a concept."""
    
    def infer_topic_interest(self, topic: str) -> float:
        """Return composite interest score from goals, rapport, and interaction count."""
    
    def get_preferred_relation_types(self) -> List[str]:
        """Return relation types user most frequently activates."""
    
    def activation_boost_for(self, concept: str) -> Dict[str, float]:
        """Return per-label boost multipliers based on edge reactivation history."""
    
    def get_state(self) -> Dict:
        """Serialize full user model state for persistence."""
    
    def set_state(self, state: Dict) -> None:
        """Restore user model state with backward compatibility."""
```

### `ravana.chat.engine` — CognitiveChatEngine (Quality & Learning Loops)

```python
from ravana.chat.engine import CognitiveChatEngine

class CognitiveChatEngine:
    """The central conversational engine orchestrating web learning, spreading activation, and generation."""
    
    def _assess_response_quality(self, response: str, strategy: str, ctx) -> float:
        """
        Evaluate generated response quality (ACC/ERN analog) on [0-1] scale based on
        strategy, length, noun diversity, context associations, stop-word ratio, and template specificity.
        """
        
    def _queue_weak_concept_for_learning(self, subject: str, quality_score: float) -> None:
        """Emergency queue a concept for immediate web learning, waking the background WebLearner thread."""
        
    def _extract_definitions(self, text: str, query: str) -> None:
        """Scan text using regex patterns and heuristic sentence splitting to extract definitional knowledge."""
        
    def _extract_heuristic_definition(self, text: str, subject: str) -> Optional[str]:
        """Heuristic definition parser scanning sentences for copulas and defining verbs (refers to, means, is a)."""
```

---

## ravana-v2 Agent Infrastructure

### ModeOrchestrator (`agent/mode_orchestrator.py`)

```python
from agent.mode_orchestrator import ModeOrchestrator, AgentMode, ModeDecision, InterviewResult

class AgentMode(Enum):
    RESEARCH = "research"      # Web + RSS → new methods
    INTERVIEW = "interview"    # Groq → RAVANA → test/evaluate
    LEARN = "learn"           # info_collector → RAVANA experience events

@dataclass
class ModeDecision:
    mode: AgentMode
    reason: str
    confidence: float
    priority: int  # 1=highest

@dataclass
class InterviewResult:
    card_id: str
    passed: bool
    expected_D: float
    actual_D: float
    expected_I: float
    actual_I: float
    dissonance_consistent: bool
    failure_type: Optional[str]
    notes: str

class ModeOrchestrator:
    def __init__(self, groq_api_key: str, db_path: str,
                 version_manager_cls=None, ravana_factory=None, grounding_factory=None):
    
    def decide_mode(self) -> ModeDecision:
        """Decide which mode to run based on state.
        - RESEARCH: pending_improvements > 3
        - LEARN: active experiments > 0
        - INTERVIEW: recent tests available or default
        """
    
    def run_research_mode(self) -> Dict[str, Any]:
        """Web + RSS → new methods → queue improvements."""
    
    def run_interview_mode(self) -> Dict[str, Any]:
        """Groq → RAVANA → test/evaluate (requires TestHarness)."""
    
    def run_learn_mode(self) -> Dict[str, Any]:
        """Info collector → RAVANA experience events."""
    
    def run_full_cycle(self) -> Dict[str, Any]:
        """Run one full orchestration cycle — decide, run, report."""
    
    def build_telegram_report(self, report: Dict) -> str:
        """Format report for Telegram delivery."""
```

### VersionManager (`agent/version_manager.py`)

```python
from agent.version_manager import VersionManager, ScriptVersion, VersionEntry, ChangeEntry

class VersionManager:
    def __init__(self, db_path: str = None):
    
    # Versions
    def get_current_versions(self) -> Dict[str, Any]:
    def detect_changed_scripts(self, scripts_dir: str) -> List[Dict]:
    def save_version(self, agent_version: str, script_versions: List[Dict], changelog: List[Dict]):
    
    # Changelog
    def add_changelog(self, component: str, change_type: str, description: str, notes: str = "", tested: bool = False):
    def get_recent_changelog(self, limit: int = 20) -> List[Dict]:
    def mark_tested(self, changelog_id: int):
    
    # Context
    def set_context(self, key: str, value: Any):
    def get_context(self, key: str) -> Any:
    
    # Experiments
    def create_experiment(self, name: str, description: str = "") -> int:
    def update_experiment(self, name: str, status: str = None, results: Dict = None):
    def get_active_experiments(self) -> List[Dict]:
    
    # Improvements Queue
    def queue_improvement(self, description: str, source: str = "web_search", priority: int = 5):
    def get_pending_improvements(self, limit: int = 10) -> List[Dict]:
    def mark_improvement(self, improvement_id: int, status: str):
    
    # Test Results
    def record_test(self, test_name: str, status: str, output: str = "", duration_ms: int = 0):
    def get_test_history(self, limit: int = 20) -> List[Dict]:
    def get_last_test_status(self, test_name: str) -> Optional[str]:
    
    # Summary
    def get_summary(self) -> Dict[str, Any]:
```

### TrainingPipeline (`training/pipeline.py`)

```python
from training.pipeline import TrainingPipeline, TrainingConfig

@dataclass
class TrainingConfig:
    total_episodes: int = 100000
    log_interval: int = 100
    checkpoint_interval: int = 1000
    debug_first_n: int = 50
    initial_difficulty: float = 0.3
    max_difficulty: float = 0.9
    difficulty_ramp_episodes: int = 50000

class TrainingPipeline:
    def __init__(self, state_manager, config: TrainingConfig = None):
    
    def train(self) -> Dict[str, Any]:
        """Execute full training run with governor-gated state evolution."""
```

---

## See Also

- [Architecture Overview](ARCHITECTURE.md)
- [ML Framework](ML_FRAMEWORK.md)
- [Cognitive Core](COGNITIVE_CORE.md)
- [Unified Package](UNIFIED_PACKAGE.md)
- [Getting Started](GETTING_STARTED.md)