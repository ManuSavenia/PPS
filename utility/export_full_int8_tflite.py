import contextlib
import io
import os

import numpy as np
import tensorflow as tf


def export_full_int8_tflite(model_to_convert, output_path, x_reference, max_samples=500):
    converter = tf.lite.TFLiteConverter.from_keras_model(model_to_convert)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    x_reference = x_reference.astype(np.float32)

    def representative_dataset():
        n = min(max_samples, x_reference.shape[0])
        for i in range(n):
            yield [x_reference[i : i + 1]]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    stderr_fd = os.dup(2)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            tflite_model = converter.convert()
    finally:
        os.dup2(stderr_fd, 2)
        os.close(stderr_fd)
        os.close(devnull_fd)

    with open(output_path, "wb") as f:
        f.write(tflite_model)
