import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score

from utility.dequantize_saved_inputs import dequantize_saved_inputs


def _as_output_list(predictions):
    if isinstance(predictions, list):
        return predictions
    return [predictions]


def _build_intermediate_model(model):
    layer_outputs = []
    layer_names = []

    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            layer_outputs.append(layer.output)
            layer_names.append(layer.name)
        except Exception:
            continue

    if not layer_outputs:
        raise ValueError("No se pudieron obtener salidas intermedias del modelo")

    if hasattr(model, "inputs") and model.inputs:
        model_inputs = model.inputs
        if len(model_inputs) == 1:
            model_inputs = model_inputs[0]
    else:
        model_inputs = model.layers[0].input

    intermediate_model = tf.keras.Model(inputs=model_inputs, outputs=layer_outputs)
    return intermediate_model, layer_names


def _layer_drift_metrics(reference_output, comparison_output):
    reference_flat = reference_output.reshape(reference_output.shape[0], -1).astype(np.float32)
    comparison_flat = comparison_output.reshape(comparison_output.shape[0], -1).astype(np.float32)
    diff = comparison_flat - reference_flat

    ref_norm = np.linalg.norm(reference_flat, axis=1)
    cmp_norm = np.linalg.norm(comparison_flat, axis=1)
    cosine = np.sum(reference_flat * comparison_flat, axis=1) / (ref_norm * cmp_norm + 1e-12)

    return {
        "MAE": float(np.mean(np.abs(diff))),
        "RMSE": float(np.sqrt(np.mean(diff ** 2))),
        "MaxAbsDiff": float(np.max(np.abs(diff))),
        "RelativeMAE": float(np.mean(np.abs(diff)) / (np.mean(np.abs(reference_flat)) + 1e-12)),
        "CosineSimilarity": float(np.mean(cosine)),
    }


def layerwise_quantized_vs_dequantized_report(model, x_quantized, y_eval, scales, dataset_name="dataset"):
    x_dequantized = dequantize_saved_inputs(x_quantized, scales)

    intermediate_model, layer_names = _build_intermediate_model(model)
    quantized_outputs = _as_output_list(intermediate_model.predict(x_quantized.astype(np.float32), verbose=0))
    dequantized_outputs = _as_output_list(intermediate_model.predict(x_dequantized.astype(np.float32), verbose=0))

    rows = []
    final_quantized = quantized_outputs[-1].reshape(quantized_outputs[-1].shape[0], -1)
    final_dequantized = dequantized_outputs[-1].reshape(dequantized_outputs[-1].shape[0], -1)
    y_true = np.argmax(y_eval, axis=1)
    y_pred_q = np.argmax(final_quantized, axis=1)
    y_pred_dq = np.argmax(final_dequantized, axis=1)

    final_accuracy_q = accuracy_score(y_true, y_pred_q)
    final_accuracy_dq = accuracy_score(y_true, y_pred_dq)
    final_agreement = float(np.mean(y_pred_q == y_pred_dq))

    for layer_name, q_out, dq_out in zip(layer_names, quantized_outputs, dequantized_outputs):
        metrics = _layer_drift_metrics(dq_out, q_out)
        rows.append(
            {
                "Dataset": dataset_name,
                "Layer": layer_name,
                "OutputShape": "x".join(str(dim) for dim in q_out.shape[1:]),
                **metrics,
                "FinalAccuracyQuantized": final_accuracy_q,
                "FinalAccuracyDequantized": final_accuracy_dq,
                "FinalPredictionAgreement": final_agreement,
            }
        )

    report = pd.DataFrame(rows)
    summary = {
        "dataset": dataset_name,
        "final_accuracy_quantized": final_accuracy_q,
        "final_accuracy_dequantized": final_accuracy_dq,
        "final_prediction_agreement": final_agreement,
    }
    return report, summary
