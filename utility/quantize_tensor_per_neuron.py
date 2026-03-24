import numpy as np

from utility.qmax import qmax


def quantize_tensor_per_neuron(x, bits=8, eps=1e-12):
    q_max = qmax(bits)
    if x.ndim == 2:
        max_abs = np.max(np.abs(x), axis=0, keepdims=True)
        scale = np.where(max_abs < eps, 1.0, max_abs / q_max).astype(np.float32)
        q = np.round(x / scale)
        q = np.clip(q, -q_max, q_max).astype(np.int32)
        return q, scale.reshape(-1)
    max_abs = np.abs(x)
    scale = np.where(max_abs < eps, 1.0, max_abs / q_max).astype(np.float32)
    q = np.round(x / scale)
    q = np.clip(q, -q_max, q_max).astype(np.int32)
    return q, scale.reshape(-1)
