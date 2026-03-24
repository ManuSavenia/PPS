import json
import os
import shutil

import numpy as np
import tensorflow as tf

from utility.qmax import qmax
from utility.quantize_tensor_per_layer import quantize_tensor_per_layer
from utility.quantize_tensor_per_neuron import quantize_tensor_per_neuron


def export_c_assets(project_root, out_dir=None):
    datasets_dir = os.path.join(project_root, "Cuantization_Test", "Data_Sets")
    models_dir = os.path.join(project_root, "Cuantization_Test", "Models")
    if out_dir is None:
        out_dir = os.path.join(project_root, "C_model", "data")
    os.makedirs(out_dir, exist_ok=True)

    inputs_dir = os.path.join(out_dir, "inputs")
    weights_dir = os.path.join(out_dir, "weights")
    weights_per_layer_dir = os.path.join(weights_dir, "per_layer")
    weights_per_neuron_dir = os.path.join(weights_dir, "per_neuron")
    scales_dir = os.path.join(out_dir, "scales")
    metadata_dir = os.path.join(out_dir, "metadata")
    reports_dir = os.path.join(out_dir, "reports")

    os.makedirs(inputs_dir, exist_ok=True)
    os.makedirs(weights_per_layer_dir, exist_ok=True)
    os.makedirs(weights_per_neuron_dir, exist_ok=True)
    os.makedirs(scales_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    bits_w = 8
    qmax_w = qmax(bits_w)

    model = tf.keras.models.load_model(os.path.join(models_dir, "fingers_model_no_quantization.h5"))
    activations = [
        layer.get_config().get("activation", "linear")
        for layer in model.layers
        if hasattr(layer, "get_config")
    ]

    w0, b0, w1, b1 = [w.astype(np.float32) for w in model.get_weights()]

    w0_q_pl, w0_s_pl = quantize_tensor_per_layer(w0, bits=bits_w)
    b0_q_pl, b0_s_pl = quantize_tensor_per_layer(b0, bits=bits_w)
    w1_q_pl, w1_s_pl = quantize_tensor_per_layer(w1, bits=bits_w)
    b1_q_pl, b1_s_pl = quantize_tensor_per_layer(b1, bits=bits_w)

    w0_q_pn, w0_s_pn = quantize_tensor_per_neuron(w0, bits=bits_w)
    b0_q_pn, b0_s_pn = quantize_tensor_per_neuron(b0, bits=bits_w)
    w1_q_pn, w1_s_pn = quantize_tensor_per_neuron(w1, bits=bits_w)
    b1_q_pn, b1_s_pn = quantize_tensor_per_neuron(b1, bits=bits_w)

    np.savetxt(os.path.join(weights_per_layer_dir, "weights_q_per_layer_w0.csv"), w0_q_pl, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(weights_per_layer_dir, "weights_q_per_layer_b0.csv"), b0_q_pl.reshape(1, -1), fmt="%d", delimiter=",")
    np.savetxt(os.path.join(weights_per_layer_dir, "weights_q_per_layer_w1.csv"), w1_q_pl, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(weights_per_layer_dir, "weights_q_per_layer_b1.csv"), b1_q_pl.reshape(1, -1), fmt="%d", delimiter=",")

    np.savetxt(os.path.join(weights_per_neuron_dir, "weights_q_per_neuron_w0.csv"), w0_q_pn, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(weights_per_neuron_dir, "weights_q_per_neuron_b0.csv"), b0_q_pn.reshape(1, -1), fmt="%d", delimiter=",")
    np.savetxt(os.path.join(weights_per_neuron_dir, "weights_q_per_neuron_w1.csv"), w1_q_pn, fmt="%d", delimiter=",")
    np.savetxt(os.path.join(weights_per_neuron_dir, "weights_q_per_neuron_b1.csv"), b1_q_pn.reshape(1, -1), fmt="%d", delimiter=",")

    qmeta = np.load(os.path.join(datasets_dir, "quantization_metadata_signed_symmetric.npz"), allow_pickle=True)
    train_scale = qmeta["train_scale"].astype(np.float32).reshape(-1)
    test_scale = qmeta["test_scale"].astype(np.float32).reshape(-1)
    bits_io = int(qmeta["bits_io"][0])
    qmax_io = int(qmeta["qmax"][0])

    np.savetxt(os.path.join(inputs_dir, "input_scale_train.csv"), train_scale.reshape(1, -1), fmt="%.10g", delimiter=",")
    np.savetxt(os.path.join(inputs_dir, "input_scale_test.csv"), test_scale.reshape(1, -1), fmt="%.10g", delimiter=",")
    np.savetxt(
        os.path.join(scales_dir, "weight_scales_per_layer.csv"),
        np.array([w0_s_pl, b0_s_pl, w1_s_pl, b1_s_pl], dtype=np.float32).reshape(1, -1),
        fmt="%.10g",
        delimiter=",",
    )
    np.savetxt(os.path.join(scales_dir, "weight_scales_per_neuron_w0.csv"), w0_s_pn.reshape(1, -1), fmt="%.10g", delimiter=",")
    np.savetxt(os.path.join(scales_dir, "weight_scales_per_neuron_b0.csv"), b0_s_pn.reshape(1, -1), fmt="%.10g", delimiter=",")
    np.savetxt(os.path.join(scales_dir, "weight_scales_per_neuron_w1.csv"), w1_s_pn.reshape(1, -1), fmt="%.10g", delimiter=",")
    np.savetxt(os.path.join(scales_dir, "weight_scales_per_neuron_b1.csv"), b1_s_pn.reshape(1, -1), fmt="%.10g", delimiter=",")

    for name in [
        "fingers_train_quant8_signed_symmetric.csv",
        "fingers_test_quant8_signed_symmetric.csv",
        "quantization_comparison_signed_symmetric.csv",
    ]:
        src = os.path.join(datasets_dir, name)
        if os.path.exists(src):
            dst_dir = reports_dir if name == "quantization_comparison_signed_symmetric.csv" else inputs_dir
            shutil.copy2(src, os.path.join(dst_dir, name))

    c_meta = {
        "bits_io": bits_io,
        "qmax_io": qmax_io,
        "bits_w": bits_w,
        "qmax_w": qmax_w,
        "activations": activations,
        "shapes": {"w0": list(w0.shape), "b0": list(b0.shape), "w1": list(w1.shape), "b1": list(b1.shape)},
        "input_scales": {"train": train_scale.tolist(), "test": test_scale.tolist()},
        "weight_scales": {
            "per_layer": {
                "w0": [float(w0_s_pl)],
                "b0": [float(b0_s_pl)],
                "w1": [float(w1_s_pl)],
                "b1": [float(b1_s_pl)],
            },
            "per_neuron": {
                "w0": [float(v) for v in w0_s_pn],
                "b0": [float(v) for v in b0_s_pn],
                "w1": [float(v) for v in w1_s_pn],
                "b1": [float(v) for v in b1_s_pn],
            },
        },
    }

    with open(os.path.join(metadata_dir, "quantization_metadata_c.json"), "w", encoding="utf-8") as f:
        json.dump(c_meta, f, indent=2)

    return out_dir
