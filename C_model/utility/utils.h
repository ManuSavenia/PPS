#ifndef UTILS_H
#define UTILS_H

#include <stdint.h>

float act_tanh(float x);
void act_softmax(const float *in, int n, float *out);
float dequantize_value(int32_t q, float scale);
int argmaxf(const float *x, int n);

int read_csv_int_matrix(const char *path, int rows, int cols, int32_t *out);
int read_csv_float_vector(const char *path, int expected_len, float *out);
int load_quantized_dataset_csv(
    const char *path,
    int feature_cols,
    int32_t **features_out,
    int **labels_out,
    int *samples_out
);

#endif
