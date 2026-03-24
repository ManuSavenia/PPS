import numpy as np

from utility.qmax import qmax


def quantize_tensor_per_layer(x, bits=8, eps=1e-12):
    q_max = qmax(bits)
    max_abs = np.max(np.abs(x))
    scale = 1.0 if max_abs < eps else float(max_abs / q_max)
    q = np.round(x / scale) if scale != 0 else np.zeros_like(x)
    q = np.clip(q, -q_max, q_max).astype(np.int32)
    return q, scale
