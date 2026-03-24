import numpy as np


def quantize_symmetric_signed(x, bits=8, axis=None, eps=1e-12):
    qmax = (2 ** (bits - 1)) - 1
    if axis is None:
        max_abs = np.max(np.abs(x))
    else:
        max_abs = np.max(np.abs(x), axis=axis, keepdims=True)
    scale = np.where(max_abs < eps, 1.0, max_abs / qmax)
    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax).astype(np.int32)
    dq = (q.astype(np.float32) * scale).astype(np.float32)
    return q, dq, scale
