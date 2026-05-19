import numpy as np
from ..tensor import RawTensor, StateTensor, tensor


def relu(x):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    return StateTensor(np.maximum(x_data, 0.0))


def sigmoid(x):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    return StateTensor(1.0 / (1.0 + np.exp(-np.clip(x_data, -100, 100))))


def tanh(x):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    return StateTensor(np.tanh(x_data))


def softmax(x, dim=-1):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    x_exp = np.exp(x_data - np.max(x_data, axis=dim, keepdims=True))
    return StateTensor(x_exp / np.sum(x_exp, axis=dim, keepdims=True))


def log_softmax(x, dim=-1):
    return StateTensor(np.log(softmax(x, dim).data + 1e-30))


def gelu(x):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    return StateTensor(0.5 * x_data * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x_data + 0.044715 * x_data**3))))


def silu(x):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    return StateTensor(x_data * (1.0 / (1.0 + np.exp(-x_data))))


def dropout(x, p=0.5, training=True):
    if not training or p == 0:
        return StateTensor(x) if not isinstance(x, StateTensor) else x
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    mask = np.random.binomial(1, 1.0 - p, x_data.shape).astype(x_data.dtype)
    return StateTensor(x_data * mask / (1.0 - p))


def linear(x, weight, bias=None):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    w_data = weight.data if isinstance(weight, RawTensor) else np.array(weight)
    result = x_data @ w_data.T
    if bias is not None:
        b_data = bias.data if isinstance(bias, RawTensor) else np.array(bias)
        result = result + b_data
    return StateTensor(result)


def embedding(indices, weight, padding_idx=None):
    idx = indices.data.astype(np.int64) if isinstance(indices, RawTensor) else np.array(indices, dtype=np.int64)
    w_data = weight.data if isinstance(weight, RawTensor) else np.array(weight)
    return StateTensor(w_data[idx])


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    axis = tuple(-i - 1 for i in range(len(normalized_shape)))
    mean = np.mean(x_data, axis=axis, keepdims=True)
    var = np.var(x_data, axis=axis, keepdims=True)
    x_norm = (x_data - mean) / np.sqrt(var + eps)
    if weight is not None:
        w_data = weight.data if isinstance(weight, RawTensor) else np.array(weight)
        x_norm = x_norm * w_data
    if bias is not None:
        b_data = bias.data if isinstance(bias, RawTensor) else np.array(bias)
        x_norm = x_norm + b_data
    return StateTensor(x_norm)


def binary_cross_entropy(input, target):
    inp = input.data if isinstance(input, RawTensor) else np.array(input)
    tgt = target.data if isinstance(target, RawTensor) else np.array(target)
    eps = 1e-15
    inp = np.clip(inp, eps, 1 - eps)
    return StateTensor(np.mean(-(tgt * np.log(inp) + (1 - tgt) * np.log(1 - inp))))


def mse_loss(input, target):
    inp = input.data if isinstance(input, RawTensor) else np.array(input)
    tgt = target.data if isinstance(target, RawTensor) else np.array(target)
    return StateTensor(np.mean((inp - tgt) ** 2))


def cross_entropy(input, target):
    inp = input.data if isinstance(input, RawTensor) else np.array(input)
    tgt = target.data if isinstance(target, RawTensor) else np.array(target)
    if tgt.ndim == 1:
        tgt = np.eye(inp.shape[-1])[tgt.astype(int)]
    log_probs = log_softmax(StateTensor(inp), dim=-1).data
    return StateTensor(np.mean(-np.sum(tgt * log_probs, axis=-1)))


def one_hot(indices, num_classes):
    idx = indices.data if isinstance(indices, RawTensor) else np.array(indices)
    return StateTensor(np.eye(num_classes)[idx.astype(int)])


def pad(x, pad_width, mode='constant', value=0):
    x_data = x.data if isinstance(x, RawTensor) else np.array(x)
    pad = [(w[0], w[1]) for w in pad_width]
    return StateTensor(np.pad(x_data, pad, mode=mode, constant_values=value))


def cosine_similarity(x1, x2, dim=-1):
    x1_data = x1.data if isinstance(x1, RawTensor) else np.array(x1)
    x2_data = x2.data if isinstance(x2, RawTensor) else np.array(x2)
    norm1 = np.linalg.norm(x1_data, axis=dim, keepdims=True)
    norm2 = np.linalg.norm(x2_data, axis=dim, keepdims=True)
    return StateTensor(np.sum(x1_data * x2_data, axis=dim, keepdims=True) / (norm1 * norm2 + 1e-15))
