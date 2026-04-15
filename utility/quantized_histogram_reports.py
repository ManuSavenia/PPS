import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utility.quantize_symmetric_signed import quantize_symmetric_signed


def _collect_tensor_q(model_obj, mode, bits=8):
    axis = None if mode == "per_layer" else 0
    rows = []

    for layer in model_obj.layers:
        if not hasattr(layer, "get_weights"):
            continue

        weights = layer.get_weights()
        if not weights:
            continue

        for idx, w in enumerate(weights):
            tensor_name = "kernel" if idx == 0 else "bias"
            q, _, _ = quantize_symmetric_signed(w.astype(np.float32), bits=bits, axis=axis)
            rows.append((layer.name, tensor_name, w, q))

    return rows


def _save_hist(values, title, out_path):
    plt.figure(figsize=(10, 4))
    bins = np.arange(-128.5, 128.5, 1.0)
    plt.hist(values.reshape(-1), bins=bins)
    plt.title(title)
    plt.xlabel("Nivel int8")
    plt.ylabel("Frecuencia")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def generate_quantized_histogram_reports(model_name, model_obj, mode, out_dir, bits=8):
    os.makedirs(out_dir, exist_ok=True)
    tensor_data = _collect_tensor_q(model_obj, mode, bits=bits)

    summary_rows = []
    all_q = []

    for layer_name, tensor_name, w, q in tensor_data:
        all_q.append(q.reshape(-1))
        out_path = os.path.join(out_dir, f"{model_name}_{layer_name}_{tensor_name}_hist.png")
        _save_hist(q, f"{model_name} | {layer_name} | {tensor_name}", out_path)

        summary_rows.append(
            {
                "Modelo": model_name,
                "Layer": layer_name,
                "Tensor": tensor_name,
                "Shape": str(tuple(w.shape)),
                "LevelsUsed": int(np.unique(q).size),
                "HistogramPath": out_path,
            }
        )

        if tensor_name == "kernel" and w.ndim == 2:
            for neuron_idx in range(w.shape[1]):
                q_neuron = q[:, neuron_idx]
                neuron_path = os.path.join(
                    out_dir,
                    f"{model_name}_{layer_name}_neuron_{neuron_idx}_hist.png",
                )
                _save_hist(
                    q_neuron,
                    f"{model_name} | {layer_name} | neurona {neuron_idx}",
                    neuron_path,
                )
                summary_rows.append(
                    {
                        "Modelo": model_name,
                        "Layer": layer_name,
                        "Tensor": f"neuron_{neuron_idx}",
                        "Shape": str(tuple(q_neuron.shape)),
                        "LevelsUsed": int(np.unique(q_neuron).size),
                        "HistogramPath": neuron_path,
                    }
                )

    if all_q:
        q_global = np.concatenate(all_q)
        out_global = os.path.join(out_dir, f"{model_name}_all_weights_hist.png")
        _save_hist(q_global, f"{model_name} | todos los pesos", out_global)
        summary_rows.append(
            {
                "Modelo": model_name,
                "Layer": "ALL",
                "Tensor": "all_weights",
                "Shape": str(tuple(q_global.shape)),
                "LevelsUsed": int(np.unique(q_global).size),
                "HistogramPath": out_global,
            }
        )

    return pd.DataFrame(summary_rows)
