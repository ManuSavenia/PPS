import numpy as np
import tensorflow as tf

from sklearn import metrics

from utility.dequantize_saved_inputs import dequantize_saved_inputs


def _requantize_to_tflite_input(x_real, input_scale, input_zero_point):
    q = np.round(x_real / input_scale + input_zero_point)
    q = np.clip(q, -128, 127).astype(np.int8)
    return q


def _dequantize_tflite_output(y_q, output_scale, output_zero_point):
    return (y_q.astype(np.float32) - output_zero_point) * output_scale


def evaluate_tflite_int8_from_csv(tflite_model_path, x_q_eval, y_eval, scales):
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    if input_scale == 0:
        raise ValueError("Escala de entrada inválida en el modelo TFLite")
    if output_scale == 0:
        raise ValueError("Escala de salida inválida en el modelo TFLite")

    x_real = dequantize_saved_inputs(x_q_eval, scales)
    x_int8 = _requantize_to_tflite_input(x_real, input_scale, input_zero_point)

    preds = []
    for i in range(x_int8.shape[0]):
        sample = x_int8[i : i + 1]
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        y_q = interpreter.get_tensor(output_details["index"])
        y_f = _dequantize_tflite_output(y_q, output_scale, output_zero_point)
        preds.append(y_f.reshape(-1))

    y_pred_float = np.stack(preds, axis=0)
    y_true = np.argmax(y_eval, axis=1)
    y_pred = np.argmax(y_pred_float, axis=1)

    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "f1_macro": metrics.f1_score(y_true, y_pred, average="macro"),
        "report": metrics.classification_report(y_true, y_pred, digits=4),
        "cm": metrics.confusion_matrix(y_true, y_pred),
        "input_scale": float(input_scale),
        "input_zero_point": int(input_zero_point),
        "output_scale": float(output_scale),
        "output_zero_point": int(output_zero_point),
    }
