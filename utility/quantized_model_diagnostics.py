import numpy as np
import pandas as pd

from utility.evaluate_pipeline_io_dequantized_from_csv import evaluate_pipeline_io_dequantized_from_csv
from utility.evaluate_pipeline_io_quantized_from_csv import evaluate_pipeline_io_quantized_from_csv
from utility.quantize_symmetric_signed import quantize_symmetric_signed


def _model_accuracy_rows(model_name, model_obj, x_train_q, y_train_q, x_test_q, y_test_q, train_scales, test_scales):
    quant_train = evaluate_pipeline_io_quantized_from_csv(model_obj, x_train_q, y_train_q)
    quant_test = evaluate_pipeline_io_quantized_from_csv(model_obj, x_test_q, y_test_q)
    dequant_train = evaluate_pipeline_io_dequantized_from_csv(model_obj, x_train_q, y_train_q, train_scales)
    dequant_test = evaluate_pipeline_io_dequantized_from_csv(model_obj, x_test_q, y_test_q, test_scales)

    return [
        {
            "Modelo": model_name,
            "Flujo": "I/O quantizado",
            "Conjunto": "EntrenamientoQ",
            "Accuracy": quant_train["accuracy"],
            "F1_macro": quant_train["f1_macro"],
        },
        {
            "Modelo": model_name,
            "Flujo": "I/O quantizado",
            "Conjunto": "PruebaQ",
            "Accuracy": quant_test["accuracy"],
            "F1_macro": quant_test["f1_macro"],
        },
        {
            "Modelo": model_name,
            "Flujo": "I/O de-cuantizado",
            "Conjunto": "EntrenamientoQ",
            "Accuracy": dequant_train["accuracy"],
            "F1_macro": dequant_train["f1_macro"],
        },
        {
            "Modelo": model_name,
            "Flujo": "I/O de-cuantizado",
            "Conjunto": "PruebaQ",
            "Accuracy": dequant_test["accuracy"],
            "F1_macro": dequant_test["f1_macro"],
        },
    ]


def build_accuracy_comparison(model_q_layer, model_q_neuron, x_train_q, y_train_q, x_test_q, y_test_q, train_scales, test_scales):
    rows = []
    rows.extend(
        _model_accuracy_rows(
            "Q por capa",
            model_q_layer,
            x_train_q,
            y_train_q,
            x_test_q,
            y_test_q,
            train_scales,
            test_scales,
        )
    )
    rows.extend(
        _model_accuracy_rows(
            "Q por neurona",
            model_q_neuron,
            x_train_q,
            y_train_q,
            x_test_q,
            y_test_q,
            train_scales,
            test_scales,
        )
    )
    return pd.DataFrame(rows)


def _weight_error_row(layer_name, tensor_name, w_ref, axis_mode, bits):
    axis = None if axis_mode == "per_layer" else 0
    q, w_dq, scale = quantize_symmetric_signed(w_ref.astype(np.float32), bits=bits, axis=axis)
    err = w_dq - w_ref.astype(np.float32)
    qmax = (2 ** (bits - 1)) - 1

    row = {
        "Layer": layer_name,
        "Tensor": tensor_name,
        "Mode": axis_mode,
        "Shape": str(tuple(w_ref.shape)),
        "MAE_w": float(np.mean(np.abs(err))),
        "RMSE_w": float(np.sqrt(np.mean(err ** 2))),
        "MaxAbsErr_w": float(np.max(np.abs(err))),
        "LevelsUsed": int(np.unique(q).size),
        "SaturationCount": int(np.sum(np.abs(q) == qmax)),
    }

    if isinstance(scale, np.ndarray):
        row["ScaleMin"] = float(np.min(scale))
        row["ScaleMax"] = float(np.max(scale))
        row["ScaleRatioMaxMin"] = float(np.max(scale) / (np.min(scale) + 1e-12))
    else:
        row["ScaleMin"] = float(scale)
        row["ScaleMax"] = float(scale)
        row["ScaleRatioMaxMin"] = 1.0

    return row


def build_weight_quantization_error_report(base_model, bits=8):
    rows = []
    for layer in base_model.layers:
        if not hasattr(layer, "get_weights"):
            continue

        weights = layer.get_weights()
        if not weights:
            continue

        for idx, w in enumerate(weights):
            tensor_name = "kernel" if idx == 0 else "bias"
            rows.append(_weight_error_row(layer.name, tensor_name, w, "per_layer", bits))
            rows.append(_weight_error_row(layer.name, tensor_name, w, "per_neuron", bits))

    return pd.DataFrame(rows)


def _deployed_weight_levels(model_obj, mode, bits):
    axis = None if mode == "per_layer" else 0
    rows = []
    qmax = (2 ** (bits - 1)) - 1

    for layer in model_obj.layers:
        if not hasattr(layer, "get_weights"):
            continue

        weights = layer.get_weights()
        if not weights:
            continue

        for idx, w in enumerate(weights):
            tensor_name = "kernel" if idx == 0 else "bias"
            q, _, _ = quantize_symmetric_signed(w.astype(np.float32), bits=bits, axis=axis)
            rows.append(
                {
                    "Layer": layer.name,
                    "Tensor": tensor_name,
                    "Mode": mode,
                    "Shape": str(tuple(w.shape)),
                    "LevelsUsed": int(np.unique(q).size),
                    "SaturationCount": int(np.sum(np.abs(q) == qmax)),
                }
            )

    return pd.DataFrame(rows)


def build_deployed_levels_report(model_q_layer, model_q_neuron, bits=8):
    df_layer = _deployed_weight_levels(model_q_layer, "per_layer", bits)
    df_layer.insert(0, "Modelo", "Q por capa")

    df_neuron = _deployed_weight_levels(model_q_neuron, "per_neuron", bits)
    df_neuron.insert(0, "Modelo", "Q por neurona")

    return pd.concat([df_layer, df_neuron], ignore_index=True)
