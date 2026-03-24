import numpy as np

from utility.dequantize_saved_inputs import dequantize_saved_inputs


def get_outputs_io_dequantized_from_csv(model_to_eval, X_q_eval, scales):
    X_dq_eval = dequantize_saved_inputs(X_q_eval, scales)
    return model_to_eval.predict(X_dq_eval.astype(np.float32), verbose=0)
