#include <math.h>
#include <stdlib.h>

#include "neural_net.h"

/* ### FULL_QUANT8: compact int8 runtime used by the exported dense model. ### */

static inline quant8 clamp_q8(int32_t value)
{
    if (value > Q_MAX)
        return Q_MAX;
    if (value < Q_MIN)
        return Q_MIN;
    return (quant8)value;
}

static inline float qparam_to_float(qparam_t qp)
{
    return (float)qp.scale_q / (float)QUANT_SCALE_ONE;
}

void dense_layer(dense_layer_t *layer, data1d_t *input, data1d_t *output)
{
    output->length = layer->output_size;
    output->qparam = layer->output_qparam;
    output->data = (quant8 *)swap_alloc(sizeof(quant8) * output->length);

    const uint32_t input_size = input->length;
    const float input_scale = qparam_to_float(input->qparam);
    const float weight_scale = qparam_to_float(layer->weights_qparam);
    const float output_scale = qparam_to_float(layer->output_qparam);

    for (uint32_t i = 0; i < layer->output_size; i++)
    {
        float acc = 0.0f;
        const quant8 *weights = &layer->weights[i * input_size];

        for (uint32_t j = 0; j < input_size; j++)
        {
            const float input_value = ((int32_t)input->data[j] - input->qparam.zero_point) * input_scale;
            const float weight_value = ((int32_t)weights[j] - layer->weights_qparam.zero_point) * weight_scale;
            acc += input_value * weight_value;
        }

        acc += layer->biases[i];

        if (output_scale > 0.0f)
        {
            output->data[i] = clamp_q8((int32_t)lrintf(acc / output_scale) + layer->output_qparam.zero_point);
        }
        else
        {
            output->data[i] = 0;
        }
    }
}

void softmax_activation(quant8 *data, uint32_t length, qparam_t qp)
{
    if (length == 0)
        return;

    /* find maximum in dequantized domain */
    float max_value = DEQUANTIZE(data[0], qp);
    for (uint32_t i = 1; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], qp);
        if (v > max_value)
        {
            max_value = v;
        }
    }

    float *probabilities = (float *)malloc(sizeof(float) * length);
    if (!probabilities)
        return;
    float sum = 0.0f;
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], qp);
        probabilities[i] = expf(v - max_value);
        sum += probabilities[i];
    }

    if (sum <= 0.0f)
    {
        sum = 1.0f;
    }

    for (uint32_t i = 0; i < length; i++)
    {
        float p = probabilities[i] / sum;
        data[i] = (quant8)QUANTIZE(p, qp);
    }

    free(probabilities);
}

void relu_activation(quant8 *data, uint32_t length, qparam_t qp)
{
    const int32_t q_zero = qp.zero_point;
    for (uint32_t i = 0; i < length; i++)
    {
        if ((int32_t)data[i] < q_zero)
        {
            data[i] = (quant8)q_zero;
        }
    }
}

void relu6_activation(quant8 *data, uint32_t length, qparam_t qp)
{
    const int32_t q_zero = qp.zero_point;
    const int32_t q_six = QUANTIZE(6.0f, qp);
    for (uint32_t i = 0; i < length; i++)
    {
        int32_t v = (int32_t)data[i];
        if (v < q_zero)
        {
            data[i] = (quant8)q_zero;
        }
        else if (v > q_six)
        {
            data[i] = (quant8)q_six;
        }
    }
}

void leakyrelu_activation(quant8 *data, uint32_t length, qparam_t qp, float alpha)
{
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], qp);
        if (v < 0.0f)
        {
            v = v * alpha;
        }
        data[i] = (quant8)QUANTIZE(v, qp);
    }
}

void tanh_activation(quant8 *data, uint32_t length, qparam_t qp)
{
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], qp);
        float out = tanhf(v);
        data[i] = (quant8)QUANTIZE(out, qp);
    }
}

void sigmoid_activation(quant8 *data, uint32_t length, qparam_t qp)
{
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], qp);
        float out = 1.0f / (1.0f + expf(-v));
        data[i] = (quant8)QUANTIZE(out, qp);
    }
}

void softsign_activation(quant8 *data, uint32_t length, qparam_t qp)
{
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], qp);
        float out = v / (1.0f + fabsf(v));
        data[i] = (quant8)QUANTIZE(out, qp);
    }
}

void softplus_activation(quant8 *data, uint32_t length, qparam_t qp)
{
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], qp);
        float out = log1pf(expf(v));
        data[i] = (quant8)QUANTIZE(out, qp);
    }
}
