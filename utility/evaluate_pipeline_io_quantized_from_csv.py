import numpy as np
from sklearn import metrics


def evaluate_pipeline_io_quantized_from_csv(model_to_eval, X_q_eval, Y_eval):
    y_pred_float = model_to_eval.predict(X_q_eval.astype(np.float32), verbose=0)
    y_true = np.argmax(Y_eval, axis=1)
    y_pred = np.argmax(y_pred_float, axis=1)
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "report": metrics.classification_report(y_true, y_pred, digits=4),
        "cm": metrics.confusion_matrix(y_true, y_pred),
    }
