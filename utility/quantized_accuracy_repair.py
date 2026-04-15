import numpy as np
import pandas as pd

from sklearn import metrics

from utility.dequantize_saved_inputs import dequantize_saved_inputs
from utility.quantize_symmetric_signed import quantize_symmetric_signed


def _apply_activation(x, activation_name):
    if activation_name in (None, "linear"):
        return x
    if activation_name == "tanh":
        return np.tanh(x)
    if activation_name == "relu":
        return np.maximum(x, 0.0)
    if activation_name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if activation_name == "softmax":
        z = x - np.max(x, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    raise ValueError(f"Activación no soportada: {activation_name}")


def _fake_quant_symmetric(x, max_abs, bits=8):
    qmax = (2 ** (bits - 1)) - 1
    max_abs = max(float(max_abs), 1e-12)
    scale = max_abs / qmax
    q = np.round(x / scale)
    q = np.clip(q, -qmax, qmax)
    return (q * scale).astype(np.float32)


def _fake_quant_asymmetric(x, xmin, xmax, bits=8):
    qmin, qmax = 0.0, float((2 ** bits) - 1)
    xmin = float(xmin)
    xmax = float(xmax)
    if xmax - xmin < 1e-12:
        return x.astype(np.float32)

    scale = (xmax - xmin) / (qmax - qmin)
    zp = np.round(qmin - xmin / scale)
    q = np.round(x / scale + zp)
    q = np.clip(q, qmin, qmax)
    dq = (q - zp) * scale
    return dq.astype(np.float32)


def _activation_quantize(x, param, scheme, bits=8):
    if scheme == "symmetric":
        return _fake_quant_symmetric(x, param["max_abs"], bits=bits)
    if scheme == "asymmetric":
        return _fake_quant_asymmetric(x, param["xmin"], param["xmax"], bits=bits)
    raise ValueError("scheme debe ser 'symmetric' o 'asymmetric'")


def _collect_dense_layers(model_obj):
    dense_layers = []
    for layer in model_obj.layers:
        cls_name = layer.__class__.__name__.lower()
        if "dense" in cls_name:
            dense_layers.append(layer)
    if not dense_layers:
        raise ValueError("No se encontraron capas Dense en el modelo")
    return dense_layers


def selective_clip_and_requantize_dense_weights(model_src, mode="per_layer", clip_percentile=99.5, sat_threshold=1, bits=8):
    repaired = model_src.__class__.from_config(model_src.get_config())
    repaired.build(model_src.input_shape)

    repaired_weights = []
    for layer in model_src.layers:
        w_list = layer.get_weights()
        if not w_list:
            continue

        for idx, w in enumerate(w_list):
            w = w.astype(np.float32)
            is_kernel_2d = (idx == 0 and w.ndim == 2)

            if not is_kernel_2d:
                axis = None if mode == "per_layer" else 0
                _, w_dq, _ = quantize_symmetric_signed(w, bits=bits, axis=axis)
                repaired_weights.append(w_dq)
                continue

            if mode == "per_layer":
                q, _, _ = quantize_symmetric_signed(w, bits=bits, axis=None)
                sat_count = int(np.sum(np.abs(q) == ((2 ** (bits - 1)) - 1)))
                if sat_count > sat_threshold:
                    thr = float(np.percentile(np.abs(w), clip_percentile))
                    w = np.clip(w, -thr, thr)
                _, w_dq, _ = quantize_symmetric_signed(w, bits=bits, axis=None)
                repaired_weights.append(w_dq)
            else:
                w_clip = w.copy()
                for neuron_idx in range(w.shape[1]):
                    col = w[:, neuron_idx]
                    q_col, _, _ = quantize_symmetric_signed(col, bits=bits, axis=None)
                    sat_count = int(np.sum(np.abs(q_col) == ((2 ** (bits - 1)) - 1)))
                    if sat_count > sat_threshold:
                        thr = float(np.percentile(np.abs(col), clip_percentile))
                        w_clip[:, neuron_idx] = np.clip(col, -thr, thr)
                _, w_dq, _ = quantize_symmetric_signed(w_clip, bits=bits, axis=0)
                repaired_weights.append(w_dq)

    repaired.set_weights(repaired_weights)
    repaired.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return repaired


def calibrate_activation_params(model_obj, x_calib_dq, scheme="symmetric", percentile=99.5):
    dense_layers = _collect_dense_layers(model_obj)
    params = []

    a = x_calib_dq.astype(np.float32)
    for layer in dense_layers:
        if scheme == "symmetric":
            max_abs = float(np.percentile(np.abs(a), percentile))
            params.append({"max_abs": max(max_abs, 1e-12)})
        else:
            lo = float(np.percentile(a, (100.0 - percentile) / 2.0))
            hi = float(np.percentile(a, 100.0 - (100.0 - percentile) / 2.0))
            if hi - lo < 1e-12:
                hi = lo + 1e-6
            params.append({"xmin": lo, "xmax": hi})

        w = layer.get_weights()
        z = a @ w[0] + w[1]
        activation_name = getattr(layer.activation, "__name__", "linear")
        a = _apply_activation(z, activation_name)

    return params


def predict_with_activation_quantization(model_obj, x_dq, activation_params, scheme="symmetric", bits=8, quantize_output_layer=False):
    dense_layers = _collect_dense_layers(model_obj)
    a = x_dq.astype(np.float32)

    for idx, layer in enumerate(dense_layers):
        a_q = _activation_quantize(a, activation_params[idx], scheme=scheme, bits=bits)
        w = layer.get_weights()
        z = a_q @ w[0] + w[1]

        activation_name = getattr(layer.activation, "__name__", "linear")
        a = _apply_activation(z, activation_name)

        if idx < len(dense_layers) - 1 or quantize_output_layer:
            a = _activation_quantize(a, activation_params[idx], scheme=scheme, bits=bits)

    return a


def evaluate_activation_quantized_pipeline(model_obj, x_q_eval, y_eval, scales, activation_params, scheme="symmetric", bits=8):
    x_dq = dequantize_saved_inputs(x_q_eval, scales)
    y_pred_float = predict_with_activation_quantization(
        model_obj,
        x_dq,
        activation_params,
        scheme=scheme,
        bits=bits,
        quantize_output_layer=False,
    )
    y_true = np.argmax(y_eval, axis=1)
    y_pred = np.argmax(y_pred_float, axis=1)
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "report": metrics.classification_report(y_true, y_pred, digits=4),
        "cm": metrics.confusion_matrix(y_true, y_pred),
    }


def run_quantized_repair_search(
    model_name,
    model_obj,
    mode,
    x_train_q,
    y_train_q,
    x_test_q,
    y_test_q,
    train_scales,
    test_scales,
    clip_percentiles,
    activation_percentiles,
    schemes,
    sat_threshold=1,
    bits=8,
):
    rows = []
    best = None

    for clip_p in clip_percentiles:
        repaired = selective_clip_and_requantize_dense_weights(
            model_obj,
            mode=mode,
            clip_percentile=clip_p,
            sat_threshold=sat_threshold,
            bits=bits,
        )

        x_calib_dq = dequantize_saved_inputs(x_train_q, train_scales)

        for scheme in schemes:
            for act_p in activation_percentiles:
                act_params = calibrate_activation_params(
                    repaired,
                    x_calib_dq,
                    scheme=scheme,
                    percentile=act_p,
                )

                train_eval = evaluate_activation_quantized_pipeline(
                    repaired,
                    x_train_q,
                    y_train_q,
                    train_scales,
                    act_params,
                    scheme=scheme,
                    bits=bits,
                )
                test_eval = evaluate_activation_quantized_pipeline(
                    repaired,
                    x_test_q,
                    y_test_q,
                    test_scales,
                    act_params,
                    scheme=scheme,
                    bits=bits,
                )

                row = {
                    "Modelo": model_name,
                    "mode": mode,
                    "clip_percentile": clip_p,
                    "activation_scheme": scheme,
                    "activation_percentile": act_p,
                    "acc_train": train_eval["accuracy"],
                    "acc_test": test_eval["accuracy"],
                    "f1_test": test_eval["f1_macro"],
                    "model_obj": repaired,
                }
                rows.append(row)

                if best is None or row["acc_test"] > best["acc_test"]:
                    best = row

    search_df = pd.DataFrame([{k: v for k, v in r.items() if k != "model_obj"} for r in rows])
    return search_df, best
