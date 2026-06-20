import numpy as np
import time
from typing import Union, List, Tuple, Optional, Any

_dtype_map = {
    'float32': np.float32, 'float': np.float32, 'float64': np.float64,
    'double': np.float64, 'int32': np.int32, 'int': np.int32,
    'int64': np.int64, 'long': np.int64, 'bool': np.bool_,
    'uint8': np.uint8,
}


def _resolve_dtype(dtype) -> Optional[np.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, type):
        return np.dtype(dtype)
    if isinstance(dtype, str):
        return np.dtype(_dtype_map.get(dtype, dtype))
    return np.dtype(dtype)


def tensor(data: Any, dtype=None, device=None, requires_grad=False):
    if isinstance(data, StateTensor):
        return StateTensor(data.data, dtype=dtype)
    if isinstance(data, RawTensor):
        return StateTensor(data.data, dtype=dtype) if requires_grad else RawTensor(data.data, dtype=dtype)
    arr = np.array(data, dtype=_resolve_dtype(dtype))
    if requires_grad:
        return StateTensor(arr)
    return RawTensor(arr)


# ─── Layer 0: RawTensor (thin NumPy wrapper) ─────────────────────────────

class RawTensor:
    __slots__ = ('data',)

    @property
    def device(self):
        from ravana_ml import device
        return device
    def __init__(self, data, dtype=None):
        if isinstance(data, RawTensor):
            data = data.data
        self.data = np.array(data, dtype=_resolve_dtype(dtype))

    # ── shape properties ──
    @property
    def shape(self) -> tuple: return self.data.shape
    @property
    def dtype(self): return self.data.dtype
    @property
    def ndim(self) -> int: return self.data.ndim
    @property
    def size(self): return self.data.size
    def numel(self) -> int: return self.data.size
    def dim(self) -> int: return self.data.ndim

    def item(self):
        return self.data.item()

    def copy(self):
        return RawTensor(self.data.copy())

    def numpy(self) -> np.ndarray: return self.data
    def detach(self): return RawTensor(self.data.copy())
    def to(self, device=None, dtype=None):
        d = _resolve_dtype(dtype) if dtype else None
        arr = self.data.astype(d) if d else self.data.copy()
        return RawTensor(arr)

    def cpu(self): return self
    def cuda(self): return self

    def float(self): return RawTensor(self.data.astype(np.float32))
    def half(self): return RawTensor(self.data.astype(np.float16))
    def double(self): return RawTensor(self.data.astype(np.float64))
    def int(self): return RawTensor(self.data.astype(np.int32))
    def long(self): return RawTensor(self.data.astype(np.int64))
    def bool(self): return RawTensor(self.data.astype(np.bool_))

    # ── view / reshape ──
    def view(self, *shape):
        return RawTensor(self.data.reshape(shape if len(shape) > 1 else shape[0]))

    def reshape(self, *shape):
        return self.view(*shape)

    def transpose(self, *axes):
        return RawTensor(self.data.transpose(*axes))

    def T(self):
        return RawTensor(self.data.T)

    def squeeze(self, axis=None):
        if axis is None:
            return RawTensor(np.squeeze(self.data))
        return RawTensor(np.squeeze(self.data, axis=axis))

    def unsqueeze(self, dim):
        return RawTensor(np.expand_dims(self.data, dim))

    # ── indexing ──
    def __getitem__(self, key):
        return RawTensor(self.data[key])

    def __setitem__(self, key, value):
        if isinstance(value, RawTensor):
            value = value.data
        self.data[key] = value

    # ── arithmetic ──
    def __add__(self, other): return self._op(other, lambda a, b: a + b)
    def __radd__(self, other): return self._op(other, lambda a, b: b + a)
    def __sub__(self, other): return self._op(other, lambda a, b: a - b)
    def __rsub__(self, other): return self._op(other, lambda a, b: b - a)
    def __mul__(self, other): return self._op(other, lambda a, b: a * b)
    def __rmul__(self, other): return self._op(other, lambda a, b: b * a)
    def __truediv__(self, other): return self._op(other, lambda a, b: a / b)
    def __rtruediv__(self, other): return self._op(other, lambda a, b: b / a)
    def __matmul__(self, other): return self._op(other, lambda a, b: a @ b)
    def __neg__(self): return RawTensor(-self.data)
    def __pos__(self): return self
    def __abs__(self): return RawTensor(np.abs(self.data))
    def __pow__(self, other): return self._op(other, lambda a, b: a ** b)

    def _op(self, other, fn):
        other = other.data if isinstance(other, RawTensor) else other
        return RawTensor(fn(self.data, other))

    # ── comparison ──
    def __eq__(self, other): return self._op(other, lambda a, b: a == b)
    def __ne__(self, other): return self._op(other, lambda a, b: a != b)
    def __lt__(self, other): return self._op(other, lambda a, b: a < b)
    def __le__(self, other): return self._op(other, lambda a, b: a <= b)
    def __gt__(self, other): return self._op(other, lambda a, b: a > b)
    def __ge__(self, other): return self._op(other, lambda a, b: a >= b)

    # ── reduction ──
    def sum(self, axis=None, keepdims=False):
        return RawTensor(np.sum(self.data, axis=axis, keepdims=keepdims))

    def mean(self, axis=None, keepdims=False):
        return RawTensor(np.mean(self.data, axis=axis, keepdims=keepdims))

    def std(self, axis=None, keepdims=False):
        return RawTensor(np.std(self.data, axis=axis, keepdims=keepdims))

    def var(self, axis=None, keepdims=False):
        return RawTensor(np.var(self.data, axis=axis, keepdims=keepdims))

    def max(self, axis=None, keepdims=False):
        return RawTensor(np.max(self.data, axis=axis, keepdims=keepdims))

    def min(self, axis=None, keepdims=False):
        return RawTensor(np.min(self.data, axis=axis, keepdims=keepdims))

    def argmax(self, axis=None):
        return RawTensor(np.argmax(self.data, axis=axis))

    def argmin(self, axis=None):
        return RawTensor(np.argmin(self.data, axis=axis))

    def clamp(self, min_val=None, max_val=None):
        return RawTensor(np.clip(self.data, min_val, max_val))

    def abs(self): return RawTensor(np.abs(self.data))

    # ── utility ──
    def __repr__(self):
        return f"<RawTensor shape={self.shape} dtype={self.dtype}>\n{self.data}"

    def __len__(self): return len(self.data)

    def __hash__(self): return id(self)
    def __iter__(self):
        for i in range(len(self.data)):
            yield RawTensor(self.data[i])

    # ── factory methods ──
    @staticmethod
    def zeros(*shape, dtype=None):
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return RawTensor(np.zeros(s, dtype=_resolve_dtype(dtype)))

    @staticmethod
    def ones(*shape, dtype=None):
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return RawTensor(np.ones(s, dtype=_resolve_dtype(dtype)))

    @staticmethod
    def randn(*shape, dtype=None):
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return RawTensor(np.random.randn(*s).astype(_resolve_dtype(dtype) or np.float32))

    @staticmethod
    def rand(*shape, dtype=None):
        s = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return RawTensor(np.random.rand(*s).astype(_resolve_dtype(dtype) or np.float32))

    @staticmethod
    def eye(n, dtype=None):
        return RawTensor(np.eye(n, dtype=_resolve_dtype(dtype)))

    @staticmethod
    def arange(*args, dtype=None):
        return RawTensor(np.arange(*args, dtype=_resolve_dtype(dtype)))

    @staticmethod
    def full(shape, fill_value, dtype=None):
        return RawTensor(np.full(shape, fill_value, dtype=_resolve_dtype(dtype)))

    @staticmethod
    def empty(shape, dtype=None):
        return RawTensor(np.empty(shape, dtype=_resolve_dtype(dtype)))

    @staticmethod
    def stack(tensors: list, dim=0):
        return RawTensor(np.stack([t.data for t in tensors], axis=dim))

    @staticmethod
    def cat(tensors: list, dim=0):
        return RawTensor(np.concatenate([t.data for t in tensors], axis=dim))


# ─── Layer 1: StateTensor (cognitive fields) ─────────────────────────────

DEFAULT_STABILITY = 0.5
DEFAULT_SALIENCE = 0.3
DECAY_RATE = 0.01


class StateTensor(RawTensor):
    __slots__ = ('data', '_salience', '_free_energy', '_stability', '_timestamp', '_decay_rate')

    def __init__(self, data, dtype=None, salience=None, free_energy=None, stability=None):
        super().__init__(data, dtype)
        self._salience = salience if salience is not None else DEFAULT_SALIENCE
        self._free_energy = free_energy if free_energy is not None else 0.0
        self._stability = stability if stability is not None else DEFAULT_STABILITY
        self._timestamp = time.time()
        self._decay_rate = DECAY_RATE

    # ── cognitive properties ──
    @property
    def salience(self): return self._salience
    @salience.setter
    def salience(self, v): self._salience = max(0.0, min(1.0, float(v)))

    @property
    def free_energy(self): return self._free_energy
    @free_energy.setter
    def free_energy(self, v): self._free_energy = max(0.0, float(v))

    @property
    def stability(self): return self._stability
    @stability.setter
    def stability(self, v): self._stability = max(0.0, min(1.0, float(v)))

    @property
    def plasticity(self): return 1.0 - self._stability

    @property
    def timestamp(self): return self._timestamp
    @timestamp.setter
    def timestamp(self, v): self._timestamp = float(v)

    def age(self) -> float:
        return time.time() - self._timestamp

    def decay(self):
        self.data *= (1.0 - self._decay_rate * self.age())
        self._timestamp = time.time()
        return self

    def boost_salience(self, amount=0.1):
        self._salience = min(1.0, self._salience + amount)
        return self

    def apply_free_energy(self, error, salience_weight=1.0):
        self._free_energy += error * self._salience * salience_weight
        self._free_energy = min(100.0, self._free_energy)
        return self._free_energy

    def consolidate(self, rate=0.1):
        delta = self._free_energy * self.plasticity * rate
        self.data += delta
        self._free_energy *= 0.5
        self._stability = min(1.0, self._stability + 0.01)
        return delta

    # ── override arithmetic to return StateTensor ──
    def _op(self, other, fn):
        other = other.data if isinstance(other, RawTensor) else other
        result_data = fn(self.data, other)
        result = StateTensor.__new__(StateTensor)
        RawTensor.__init__(result, result_data)
        result._salience = self._salience
        result._free_energy = self._free_energy
        result._stability = self._stability
        result._timestamp = time.time()
        result._decay_rate = self._decay_rate
        return result

    def __getitem__(self, key):
        result_data = self.data[key]
        result = StateTensor.__new__(StateTensor)
        RawTensor.__init__(result, result_data)
        result._salience = self._salience
        result._free_energy = self._free_energy
        result._stability = self._stability
        result._timestamp = self._timestamp
        result._decay_rate = self._decay_rate
        return result

    def to(self, device=None, dtype=None):
        d = _resolve_dtype(dtype) if dtype else None
        arr = self.data.astype(d) if d else self.data.copy()
        result = StateTensor.__new__(StateTensor)
        RawTensor.__init__(result, arr)
        result._salience = self._salience
        result._free_energy = self._free_energy
        result._stability = self._stability
        result._timestamp = self._timestamp
        result._decay_rate = self._decay_rate
        return result

    def detach(self):
        result = StateTensor.__new__(StateTensor)
        RawTensor.__init__(result, self.data.copy())
        result._salience = self._salience
        result._free_energy = self._free_energy
        result._stability = self._stability
        result._timestamp = self._timestamp
        result._decay_rate = self._decay_rate
        return result

    def copy(self):
        return self.detach()

    def __repr__(self):
        return (f"<StateTensor shape={self.shape} dtype={self.dtype} "
                f"salience={self._salience:.2f} free_energy={self._free_energy:.2f} "
                f"stability={self._stability:.2f}>\n{self.data}")


# ─── Parameter (like torch.nn.Parameter) ────────────────────────────────

class Parameter(StateTensor):
    def __init__(self, data: Optional[StateTensor] = None):
        if data is None:
            data = StateTensor(np.array([]))
        super().__init__(data.data, dtype=data.dtype)
        self._salience = data._salience
        self._free_energy = data._free_energy
        self._stability = data._stability
        self._timestamp = data._timestamp
        self._decay_rate = data._decay_rate

    def __repr__(self):
        return f"Parameter containing:\n{super().__repr__()}"


def eye(n):
    return RawTensor.eye(n)

def arange(*args):
    return RawTensor.arange(*args)

def stack(tensors, dim=0):
    return RawTensor.stack(tensors, dim)

def cat(tensors, dim=0):
    return RawTensor.cat(tensors, dim)

def zeros(*shape):
    return RawTensor.zeros(*shape)

def ones(*shape):
    return RawTensor.ones(*shape)

def randn(*shape):
    return RawTensor.randn(*shape)

def from_numpy(arr):
    return StateTensor(arr)
