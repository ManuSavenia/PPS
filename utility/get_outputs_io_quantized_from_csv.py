import numpy as np


def get_outputs_io_quantized_from_csv(model_to_eval, X_q_eval):
    return model_to_eval.predict(X_q_eval.astype(np.float32), verbose=0)
