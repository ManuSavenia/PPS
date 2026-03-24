import tensorflow as tf

from utility.quantize_symmetric_signed import quantize_symmetric_signed


def quantize_dense_weights_symmetric(model, bits=8, mode="per_layer"):
    q_model = tf.keras.models.clone_model(model)
    q_model.build(model.input_shape)
    new_weights = []
    for w in model.get_weights():
        if mode == "per_layer":
            _, w_dq, _ = quantize_symmetric_signed(w, bits=bits, axis=None)
        elif mode == "per_neuron":
            _, w_dq, _ = quantize_symmetric_signed(w, bits=bits, axis=0)
        else:
            raise ValueError("mode debe ser 'per_layer' o 'per_neuron'")
        new_weights.append(w_dq)
    q_model.set_weights(new_weights)
    q_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return q_model
