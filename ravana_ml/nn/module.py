import numpy as np
from ..tensor import StateTensor, Parameter, RawTensor, tensor


# ─── Base Module ─────────────────────────────────────────────────────────

class Module:
    def __init__(self):
        self._parameters = set()
        self._modules = {}
        self._free_energy = 0.0
        self._free_energy_history = []
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def parameters(self):
        for name, param in self.named_parameters():
            yield param

    def named_parameters(self, prefix=''):
        for name in sorted(self._parameters, key=lambda n: n):
            full_name = f"{prefix}{name}" if not prefix else f"{prefix}.{name}"
            yield full_name, getattr(self, name)
        for mod_name in sorted(self._modules.keys()):
            mod = self._modules[mod_name]
            sub_prefix = f"{prefix}{mod_name}" if not prefix else f"{prefix}.{mod_name}"
            yield from mod.named_parameters(prefix=sub_prefix)

    def modules(self):
        yield self
        for m in self._modules.values():
            yield from m.modules()

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)

    def eval(self):
        self.train(False)

    def register_parameter(self, name, param):
        setattr(self, name, param)
        self._parameters.add(name)

    def register_module(self, name, module):
        self._modules[name] = module

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters.add(name)
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)

    # ── RAVANA learning interface ──

    def accumulate_free_energy(self, error):
        err_val = error.data if isinstance(error, RawTensor) else (np.array(error) if hasattr(error, '__iter__') else error)
        free_energy_delta = float(np.mean(np.abs(err_val)))
        self._free_energy += min(100.0, free_energy_delta)
        self._free_energy_history.append(self._free_energy)
        for m in self._modules.values():
            m.accumulate_free_energy(error)

    def sleep_cycle(self):
        for m in self._modules.values():
            m.sleep_cycle()
        self._free_energy = max(0.0, self._free_energy - 0.5 * self._free_energy)

    def reset_free_energy(self):
        self._free_energy = 0.0
        self._free_energy_history = []

    def state_dict(self):
        """Save all parameters with their cognitive metadata."""
        sd = {}
        for name, param in self.named_parameters():
            entry = {"data": param.data.copy()}
            if isinstance(param, StateTensor):
                entry["salience"] = float(param.salience)
                entry["free_energy"] = float(param.free_energy)
                entry["stability"] = float(param.stability)
            sd[name] = entry
        return sd

    def load_state_dict(self, sd):
        """Load parameters with cognitive metadata."""
        for name, param in self.named_parameters():
            if name in sd:
                entry = sd[name]
                if isinstance(entry, dict):
                    param.data = entry["data"].copy()
                    if isinstance(param, StateTensor) and "stability" in entry:
                        param.salience = entry.get("salience", 0.5)
                        param.free_energy = entry.get("free_energy", entry.get("pressure", 0.0))
                        param.stability = entry.get("stability", 0.5)
                else:
                    # Backwards compat: bare numpy array
                    param.data = entry.copy()

    def __repr__(self):
        params = sum(p.data.size for _, p in self.named_parameters())
        return f"{self.__class__.__name__}({params} params, free_energy={self._free_energy:.2f})"


# ─── Sequential ──────────────────────────────────────────────────────────

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        for i, m in enumerate(modules):
            self.register_module(str(i), m)

    def forward(self, x):
        for m in self._modules.values():
            x = m(x)
        return x

    def __getitem__(self, idx):
        return list(self._modules.values())[idx]

    def __len__(self):
        return len(self._modules)


# ─── Linear ──────────────────────────────────────────────────────────────

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        scale = np.sqrt(2.0 / in_features)
        w = StateTensor(np.random.randn(out_features, in_features).astype(np.float32) * scale)
        self.register_parameter('weight', Parameter(w))
        self._weight_free_energy = StateTensor(np.zeros((out_features, in_features), dtype=np.float32))
        if bias:
            b = StateTensor(np.zeros(out_features, dtype=np.float32))
            self.register_parameter('bias', Parameter(b))
            self._bias_free_energy = StateTensor(np.zeros(out_features, dtype=np.float32))
        else:
            self.bias = None
            self._bias_free_energy = None
        self._trace_x = None

    def forward(self, x):
        x_data = x.data if isinstance(x, RawTensor) else np.array(x)
        w_data = self.weight.data if isinstance(self.weight, Parameter) else self.weight
        result = x_data @ w_data.T
        if self.bias is not None:
            b_data = self.bias.data if isinstance(self.bias, Parameter) else self.bias
            result = result + b_data
        self._trace_x = x_data.copy()
        return StateTensor(result)

    def backprop(self, error_data):
        """REMOVED — backprop is the chain rule escape hatch.
        Use predictive coding (local errors via settle loop) instead."""
        raise NotImplementedError(
            "backprop() removed: RLM uses predictive coding, not gradient propagation. "
            "Each layer computes its own local prediction error."
        )

    def accumulate_free_energy(self, error):
        if self._trace_x is None:
            super().accumulate_free_energy(error)
            return 0.0
        error_data = error.data if isinstance(error, RawTensor) else np.array(error)
        x_data = self._trace_x
        # Ensure 2D for correct matmul with .T
        if x_data.ndim == 1:
            x_data = x_data.reshape(1, -1)
        elif x_data.ndim > 2:
            x_data = x_data.reshape(-1, x_data.shape[-1])
        if error_data.ndim == 1:
            error_data = error_data.reshape(1, -1)
        elif error_data.ndim > 2:
            error_data = error_data.reshape(-1, error_data.shape[-1])
        salience = getattr(error, '_salience', 0.3) if isinstance(error, StateTensor) else 0.3
        hebbian = (x_data.T @ error_data) * salience * 0.01
        self._weight_free_energy.data += hebbian.T
        if self.bias is not None:
            self._bias_free_energy.data += error_data.mean(axis=0) * salience * 0.01
        super().accumulate_free_energy(error)
        return float(np.mean(np.abs(hebbian)))

    def sleep_cycle(self):
        plasticity = 1.0 - float(np.mean(self.weight.stability))
        self.weight.data += self._weight_free_energy.data * plasticity * 0.1
        self._weight_free_energy = StateTensor(np.zeros_like(self._weight_free_energy.data))
        self.weight.stability = min(1.0, self.weight.stability + 0.005)
        if self.bias is not None:
            self.bias.data += self._bias_free_energy.data * plasticity * 0.1
            self._bias_free_energy = StateTensor(np.zeros_like(self._bias_free_energy.data))
            self.bias.stability = min(1.0, self.bias.stability + 0.005)

    def __repr__(self):
        return f"Linear(in={self.in_features}, out={self.out_features}, bias={self.bias is not None})"


# ─── Embedding ───────────────────────────────────────────────────────────

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        scale = np.sqrt(2.0 / embedding_dim)
        w = StateTensor(np.random.randn(num_embeddings, embedding_dim).astype(np.float32) * scale)
        if padding_idx is not None:
            w.data[padding_idx] = 0.0
        self.register_parameter('weight', Parameter(w))
        self._weight_free_energy = StateTensor(np.zeros((num_embeddings, embedding_dim), dtype=np.float32))
        self._trace_indices = None

    def forward(self, indices):
        if isinstance(indices, RawTensor):
            idx = indices.data.astype(np.int64)
        else:
            idx = np.array(indices, dtype=np.int64)
        self._trace_indices = idx.copy()
        w = self.weight.data if isinstance(self.weight, Parameter) else self.weight
        return StateTensor(w[idx])

    def accumulate_free_energy(self, error):
        if self._trace_indices is None:
            super().accumulate_free_energy(error)
            return
        error_data = error.data if isinstance(error, RawTensor) else np.array(error)
        salience = getattr(error, '_salience', 0.3) if isinstance(error, StateTensor) else 0.3
        for i, idx in enumerate(self._trace_indices.flatten()):
            self._weight_free_energy.data[idx] += error_data[i] * salience * 0.01
        super().accumulate_free_energy(error)

    def sleep_cycle(self):
        plasticity = 1.0 - float(np.mean(self.weight.stability))
        self.weight.data += self._weight_free_energy.data * plasticity * 0.1
        self._weight_free_energy = StateTensor(np.zeros_like(self._weight_free_energy.data))
        self.weight.stability = min(1.0, self.weight.stability + 0.005)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx] = 0.0

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"


# ─── LayerNorm ──────────────────────────────────────────────────────────

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        self.normalized_shape = normalized_shape if isinstance(normalized_shape, (tuple, list)) else (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            s = StateTensor(np.ones(self.normalized_shape, dtype=np.float32))
            b = StateTensor(np.zeros(self.normalized_shape, dtype=np.float32))
            self.register_parameter('weight', Parameter(s))
            self.register_parameter('bias', Parameter(b))

    def forward(self, x):
        x_data = x.data if isinstance(x, RawTensor) else np.array(x)
        axis = tuple(-i - 1 for i in range(len(self.normalized_shape)))
        mean = np.mean(x_data, axis=axis, keepdims=True)
        var = np.var(x_data, axis=axis, keepdims=True)
        x_norm = (x_data - mean) / np.sqrt(var + self.eps)
        if self.elementwise_affine:
            w = self.weight.data if isinstance(self.weight, Parameter) else self.weight
            b = self.bias.data if isinstance(self.bias, Parameter) else self.bias
            x_norm = x_norm * w + b
        return StateTensor(x_norm)

    def __repr__(self):
        return f"LayerNorm({self.normalized_shape}, eps={self.eps})"


# ─── Dropout ────────────────────────────────────────────────────────────

class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x if isinstance(x, StateTensor) else StateTensor(x)
        x_data = x.data if isinstance(x, RawTensor) else np.array(x)
        mask = np.random.binomial(1, 1.0 - self.p, x_data.shape).astype(x_data.dtype)
        result = x_data * mask / (1.0 - self.p)
        return StateTensor(result)

    def __repr__(self):
        return f"Dropout(p={self.p})"


# ─── GRU Cell ─────────────────────────────────────────────────────────

class GRUCell(Module):
    """Gated Recurrent Unit cell.

    Three gates control information flow:
    - Update gate (z): how much old state to keep vs new candidate
    - Reset gate (r): how much old state to use when computing candidate
    - Candidate (h~): new state proposal

    h_new = (1 - z) * h_prev + z * h_candidate
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined = input_size + hidden_size
        # Three linear projections for the three gates
        self.W_z = Linear(combined, hidden_size)
        self.W_r = Linear(combined, hidden_size)
        self.W_h = Linear(combined, hidden_size)

    def forward(self, x, h_prev):
        """Forward pass.

        Args:
            x: input vector (input_size,) — ndarray
            h_prev: previous hidden state (hidden_size,) — ndarray

        Returns:
            h_new: new hidden state (hidden_size,) — ndarray
        """
        x_data = np.asarray(x, dtype=np.float32)
        h_data = np.asarray(h_prev, dtype=np.float32)
        combined = np.concatenate([x_data, h_data])
        combined_t = StateTensor(combined[np.newaxis, :])

        z = 1.0 / (1.0 + np.exp(-np.clip(
            self.W_z(combined_t).data[0], -100, 100)))  # sigmoid
        r = 1.0 / (1.0 + np.exp(-np.clip(
            self.W_r(combined_t).data[0], -100, 100)))  # sigmoid
        combined_r = np.concatenate([x_data, r * h_data])
        h_candidate = np.tanh(
            self.W_h(StateTensor(combined_r[np.newaxis, :])).data[0])
        h_new = (1.0 - z) * h_data + z * h_candidate

        # Store combined inputs for Hebbian learning in RLM.learn()
        self._last_combined = combined
        self._last_combined_r = combined_r
        self._last_z = z
        self._last_r = r
        self._last_h_prev = h_data
        self._last_x = x_data

        return h_new

    def __repr__(self):
        return f"GRUCell({self.input_size}, {self.hidden_size})"
