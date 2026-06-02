/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 * GitHub: https://github.com/Embed-ML/EmbedIA
 */
#include <math.h>
#include "quant8.h"

/* ### FULL_QUANT8: integer-only quantization helpers. ### */

void quantize_param(float *values, int size, qparam_t *qp)
{
    float min_val = values[0];
    float max_val = values[0];

    for (int i = 1; i < size; ++i)
    {
        if (values[i] < min_val)
            min_val = values[i];
        if (values[i] > max_val)
            max_val = values[i];
    }

    float float_scale = (max_val - min_val) / 255.0f;
    if (float_scale < 1e-8f)
        float_scale = 1e-8f;

    qp->scale_q = (int32_t)roundf(float_scale * QUANT_SCALE_ONE);
    if (qp->scale_q <= 0)
        qp->scale_q = 1;

    qp->zero_point = (int8_t)roundf(Q_MIN - min_val / float_scale);
    if (qp->zero_point > Q_MAX)
        qp->zero_point = Q_MAX;
    if (qp->zero_point < Q_MIN)
        qp->zero_point = Q_MIN;
}

void quantize_vec(float values[], quant8 qvalues[], int size, qparam_t qp)
{
    const float scale = (float)qp.scale_q / QUANT_SCALE_ONE;
    for (int i = 0; i < size; ++i)
    {
        int32_t quantized = (int32_t)roundf(values[i] / scale) + qp.zero_point;
        qvalues[i] = Q_CLAMP(quantized);
    }
}

void dequantize_vec(quant8 qvalues[], float values[], int size, qparam_t qp)
{
    const float scale = (float)qp.scale_q / QUANT_SCALE_ONE;
    for (int i = 0; i < size; ++i)
    {
        values[i] = ((float)(qvalues[i] - qp.zero_point)) * scale;
    }
}

int32_t mul_add_vec(quant8 a[], qparam_t qa, quant8 b[], qparam_t qb, int size)
{
    int32_t sum = 0;
    for (int i = 0; i < size; ++i)
    {
        const int32_t da = (int32_t)a[i] - qa.zero_point;
        const int32_t db = (int32_t)b[i] - qb.zero_point;
        sum += da * db;
    }
    return sum;
}
