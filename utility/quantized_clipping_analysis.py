import os

import numpy as np
import tensorflow as tf

from utility.quantize_symmetric_signed import quantize_symmetric_signed
from utility.evaluate_pipeline_io_quantized_from_csv import evaluate_pipeline_io_quantized_from_csv


def build_clipped_model_from_source(model_src, clip_percentile=99.5, bits=8, mode="per_layer"):
    clipped_model = tf.keras.models.clone_model(model_src)
    clipped_model.build(model_src.input_shape)

    axis = None if mode == "per_layer" else 0
    new_weights = []

    for weight in model_src.get_weights():
        weight = weight.astype(np.float32)
        threshold = float(np.percentile(np.abs(weight), clip_percentile))
        clipped_weight = weight if threshold <= 0.0 else np.clip(weight, -threshold, threshold)
        _, weight_dequantized, _ = quantize_symmetric_signed(clipped_weight, bits=bits, axis=axis)
        new_weights.append(weight_dequantized)

    clipped_model.set_weights(new_weights)
    clipped_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return clipped_model


def count_levels(model_obj, mode="per_layer", bits=8):
    axis = None if mode == "per_layer" else 0
    quantized_weights = []

    for weight in model_obj.get_weights():
        quantized, _, _ = quantize_symmetric_signed(weight.astype(np.float32), bits=bits, axis=axis)
        quantized_weights.append(quantized.reshape(-1))

    quantized_weights = np.concatenate(quantized_weights)
    qmax = (2 ** (bits - 1)) - 1
    return int(np.unique(quantized_weights).size), int(np.sum(np.abs(quantized_weights) == qmax))


def evaluate_clipping_sweep(model_name, model_src, mode, candidate_percentiles, x_train_q, y_train_q, x_test_q, y_test_q):
    base_train = evaluate_pipeline_io_quantized_from_csv(model_src, x_train_q, y_train_q)
    base_test = evaluate_pipeline_io_quantized_from_csv(model_src, x_test_q, y_test_q)
    base_levels, base_sat = count_levels(model_src, mode=mode, bits=8)

    rows = []
    best_row = None

    for percentile in candidate_percentiles:
        clipped_model = build_clipped_model_from_source(model_src, clip_percentile=percentile, bits=8, mode=mode)
        clipped_train = evaluate_pipeline_io_quantized_from_csv(clipped_model, x_train_q, y_train_q)
        clipped_test = evaluate_pipeline_io_quantized_from_csv(clipped_model, x_test_q, y_test_q)
        clipped_levels, clipped_sat = count_levels(clipped_model, mode=mode, bits=8)

        row = {
            "Modelo": model_name,
            "Percentil": percentile,
            "Accuracy base train": base_train["accuracy"],
            "Accuracy base test": base_test["accuracy"],
            "Accuracy clipped train": clipped_train["accuracy"],
            "Accuracy clipped test": clipped_test["accuracy"],
            "Delta test": clipped_test["accuracy"] - base_test["accuracy"],
            "Niveles base": base_levels,
            "Niveles clipped": clipped_levels,
            "Saturaciones base": base_sat,
            "Saturaciones clipped": clipped_sat,
            "Modelo clipped": clipped_model,
        }
        rows.append(row)

        if best_row is None or row["Accuracy clipped test"] > best_row["Accuracy clipped test"]:
            best_row = row

    return base_train, base_test, rows, best_row
