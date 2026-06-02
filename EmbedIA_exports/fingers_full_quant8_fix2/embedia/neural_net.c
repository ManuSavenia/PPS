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

    const int32_t in_zp = input->qparam.zero_point;
    const int32_t w_zp = layer->weights_qparam.zero_point;
    const int32_t out_zp = layer->output_qparam.zero_point;

    /* scale_q are stored in Q0.15 (QUANT_SCALE_ONE) */
    const int32_t in_scale_q = input->qparam.scale_q;
    const int32_t w_scale_q = layer->weights_qparam.scale_q;
    const int32_t out_scale_q = layer->output_qparam.scale_q;

    /* multiplier_fixed in Q0.15 units: (in_scale_q * w_scale_q) / out_scale_q */
    int64_t multiplier_fixed = 0;
    if (out_scale_q != 0)
    {
        multiplier_fixed = ((int64_t)in_scale_q * (int64_t)w_scale_q) / (int64_t)out_scale_q;
        if (multiplier_fixed == 0)
            multiplier_fixed = 1; /* safeguard */
    }

    for (uint32_t i = 0; i < layer->output_size; i++)
    {
        int32_t acc = 0;
        const quant8 *weights = &layer->weights[i * input_size];

        for (uint32_t j = 0; j < input_size; j++)
        {
            int32_t in_val = (int32_t)input->data[j] - in_zp;
            int32_t w_val = (int32_t)weights[j] - w_zp;
            acc += in_val * w_val;
        }

        /* biases are stored as int32 in accumulator units */
        if (layer->biases)
        {
            acc += layer->biases[i];
        }

        /* Requantize: out = round(acc * multiplier_fixed / QUANT_SCALE_ONE) + out_zp */
        int32_t out_q = out_zp;
        if (multiplier_fixed > 0)
        {
            int64_t tmp = (int64_t)acc * multiplier_fixed;
            if (tmp >= 0)
            {
                tmp += (int64_t)QUANT_SCALE_HALF;
            }
            else
            {
                tmp -= (int64_t)QUANT_SCALE_HALF;
            }
            int32_t scaled = (int32_t)(tmp / (int64_t)QUANT_SCALE_ONE);
            out_q = scaled + out_zp;
        }

        output->data[i] = clamp_q8(out_q);
    }
}

void softmax_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp)
{
    if (length == 0)
        return;
    /* capture input qparam and use output qparam for requantization */
    qparam_t in_qp = *buf_qp;

    /* find maximum in dequantized domain */
    float max_value = DEQUANTIZE(data[0], in_qp);
    for (uint32_t i = 1; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], in_qp);
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
        float v = DEQUANTIZE(data[i], in_qp);
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
        data[i] = (quant8)QUANTIZE(p, out_qp);
    }

    /* update buffer qparam metadata to the output qparam */
    *buf_qp = out_qp;

    free(probabilities);
}

void relu_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp)
{
    qparam_t in_qp = *buf_qp;
    const int32_t q_zero = in_qp.zero_point;
    for (uint32_t i = 0; i < length; i++)
    {
        if ((int32_t)data[i] < q_zero)
        {
            data[i] = (quant8)q_zero;
        }
    }
    /* relu doesn't change quant params by default, but update to out_qp */
    *buf_qp = out_qp;
}

void relu6_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp)
{
    qparam_t in_qp = *buf_qp;
    const int32_t q_zero = in_qp.zero_point;
    const int32_t q_six = QUANTIZE(6.0f, in_qp);
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
    *buf_qp = out_qp;
}

void leakyrelu_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp, float alpha)
{
    qparam_t in_qp = *buf_qp;
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], in_qp);
        if (v < 0.0f)
        {
            v = v * alpha;
        }
        data[i] = (quant8)QUANTIZE(v, out_qp);
    }
    *buf_qp = out_qp;
}

void tanh_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp)
{
    qparam_t in_qp = *buf_qp;
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], in_qp);
        float out = tanhf(v);
        data[i] = (quant8)QUANTIZE(out, out_qp);
    }
    *buf_qp = out_qp;
}

void sigmoid_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp)
{
    qparam_t in_qp = *buf_qp;
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], in_qp);
        float out = 1.0f / (1.0f + expf(-v));
        data[i] = (quant8)QUANTIZE(out, out_qp);
    }
    *buf_qp = out_qp;
}

void softsign_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp)
{
    qparam_t in_qp = *buf_qp;
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], in_qp);
        float out = v / (1.0f + fabsf(v));
        data[i] = (quant8)QUANTIZE(out, out_qp);
    }
    *buf_qp = out_qp;
}

void softplus_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp)
{
    qparam_t in_qp = *buf_qp;
    for (uint32_t i = 0; i < length; i++)
    {
        float v = DEQUANTIZE(data[i], in_qp);
        float out = log1pf(expf(v));
        data[i] = (quant8)QUANTIZE(out, out_qp);
    }
    *buf_qp = out_qp;
}
