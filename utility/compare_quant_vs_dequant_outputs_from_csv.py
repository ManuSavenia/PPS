import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utility.get_outputs_io_dequantized_from_csv import get_outputs_io_dequantized_from_csv
from utility.get_outputs_io_quantized_from_csv import get_outputs_io_quantized_from_csv


def compare_quant_vs_dequant_outputs_from_csv(model_to_eval, X_q_eval, Y_eval, model_name, split_name, scales):
    y_dq = get_outputs_io_dequantized_from_csv(model_to_eval, X_q_eval, scales)
    y_q = get_outputs_io_quantized_from_csv(model_to_eval, X_q_eval)
    y_true_onehot = Y_eval.astype(np.float32)

    abs_err_q_vs_dq = np.abs(y_dq - y_q)

    pred_dq = np.argmax(y_dq, axis=1)
    pred_q = np.argmax(y_q, axis=1)
    y_true = np.argmax(Y_eval, axis=1)

    return {
        "Modelo": model_name,
        "Particion": split_name,
        "MAE_q_vs_dq": mean_absolute_error(y_dq.reshape(-1), y_q.reshape(-1)),
        "MSE_q_vs_dq": mean_squared_error(y_dq.reshape(-1), y_q.reshape(-1)),
        "MaxAbsErr_q_vs_dq": abs_err_q_vs_dq.max(),
        "MAE_dq_vs_real": mean_absolute_error(y_true_onehot.reshape(-1), y_dq.reshape(-1)),
        "MSE_dq_vs_real": mean_squared_error(y_true_onehot.reshape(-1), y_dq.reshape(-1)),
        "MAE_q_vs_real": mean_absolute_error(y_true_onehot.reshape(-1), y_q.reshape(-1)),
        "MSE_q_vs_real": mean_squared_error(y_true_onehot.reshape(-1), y_q.reshape(-1)),
        "Top1_coincidencia_q_vs_dq": (pred_q == pred_dq).mean(),
        "Top1_diferencia_q_vs_dq": (pred_q != pred_dq).mean(),
        "Acc_decuantizado": (pred_dq == y_true).mean(),
        "Acc_cuantizado": (pred_q == y_true).mean(),
    }
