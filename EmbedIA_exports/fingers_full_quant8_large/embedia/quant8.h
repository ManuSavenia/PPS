#ifndef QUANT8_H
#define QUANT8_H
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

#include <stdint.h>
#include <math.h>

typedef int8_t quant8;

typedef struct {
    int32_t scale_q;
    int8_t zero_point;
} qparam_t;

#define Q_MIN (-128)
#define Q_MAX 127
#define QUANT_SCALE_FRAC_BITS 15
#define QUANT_SCALE_ONE (1 << QUANT_SCALE_FRAC_BITS)
#define QUANT_SCALE_HALF (1 << (QUANT_SCALE_FRAC_BITS - 1))
#define Q_SCALE QUANT_SCALE_ONE
#define Q_MIN_VAL Q_MIN
#define Q_MAX_VAL Q_MAX

#define Q_CLAMP(qv) ((quant8)((qv > Q_MAX) ? Q_MAX : ((qv < Q_MIN) ? Q_MIN : (qv))))

/* ### FULL_QUANT8: direct int8 quantization helpers, no fixed-point dependency. ### */

#define QUANTIZE(val, qp) \
    Q_CLAMP((int)roundf((val) / ((float)(qp).scale_q / QUANT_SCALE_ONE) + (qp).zero_point))

#define DEQUANTIZE(qval, qp) \
    (((float)((qval) - (qp).zero_point)) * ((float)(qp).scale_q / QUANT_SCALE_ONE))

#define QUANTIZE_FIXED(val_qm, qp) \
    Q_CLAMP((((int32_t)(val_qm) + ((qp).scale_q >> 1)) / (qp).scale_q + (qp).zero_point))

#define DEQUANTIZE_FIXED(qval, qp) \
    (((int32_t)((qval) - (qp).zero_point)) * (qp).scale_q)

static inline int32_t float_to_q(float f) {
    return (int32_t)(f * QUANT_SCALE_ONE);
}

static inline float q_to_float(int32_t q) {
    return (float)q / QUANT_SCALE_ONE;
}

static inline int32_t q_mul(int32_t a, int32_t b) {
    return ((int64_t)a * b) >> QUANT_SCALE_FRAC_BITS;
}

static inline int32_t q_add(int32_t a, int32_t b) {
    int64_t tmp = (int64_t)a + b;
    if (tmp > INT32_MAX) return INT32_MAX;
    if (tmp < INT32_MIN) return INT32_MIN;
    return (int32_t)tmp;
}


#ifdef __cplusplus
extern "C" {
#endif

    // Calcula parámetros de cuantización
    void quantize_param(float *values, int size, qparam_t *qp);

    void quantize_vec(float values[], quant8 qvalues[], int size, qparam_t qp);
    void dequantize_vec(quant8 qvalues[], float values[], int size, qparam_t qp);

    int32_t mul_add_vec(quant8 a[], qparam_t qa, quant8 b[], qparam_t qb, int size);

#ifdef __cplusplus
}
#endif

#endif