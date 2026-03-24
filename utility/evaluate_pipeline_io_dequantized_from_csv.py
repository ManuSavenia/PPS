import numpy as np
from sklearn import metrics

from utility.dequantize_saved_inputs import dequantize_saved_inputs


def evaluate_pipeline_io_dequantized_from_csv(model_to_eval, X_q_eval, Y_eval, scales):
    X_dq_eval = dequantize_saved_inputs(X_q_eval, scales)
    y_pred_float = model_to_eval.predict(X_dq_eval.astype(np.float32), verbose=0)
    y_true = np.argmax(Y_eval, axis=1)
    y_pred = np.argmax(y_pred_float, axis=1)
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "report": metrics.classification_report(y_true, y_pred, digits=4),
        "cm": metrics.confusion_matrix(y_true, y_pred),
    }
