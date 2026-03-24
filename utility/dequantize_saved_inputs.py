import numpy as np


def dequantize_saved_inputs(X_q, scales):
    return X_q.astype(np.float32) * scales
