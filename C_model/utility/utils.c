#include "utils.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE_BUF 8192

float act_tanh(float x)
{
    return tanhf(x);
}

void act_softmax(const float *in, int n, float *out)
{
    float max_v = in[0];
    for (int i = 1; i < n; ++i)
    {
        if (in[i] > max_v)
        {
            max_v = in[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        out[i] = expf(in[i] - max_v);
        sum += out[i];
    }

    if (sum <= 0.0f)
    {
        const float uniform = 1.0f / (float)n;
        for (int i = 0; i < n; ++i)
        {
            out[i] = uniform;
        }
        return;
    }

    for (int i = 0; i < n; ++i)
    {
        out[i] /= sum;
    }
}

float dequantize_value(int32_t q, float scale)
{
    return (float)q * scale;
}

int argmaxf(const float *x, int n)
{
    int idx = 0;
    for (int i = 1; i < n; ++i)
    {
        if (x[i] > x[idx])
        {
            idx = i;
        }
    }
    return idx;
}

static int line_is_blank(const char *line)
{
    while (*line != '\0')
    {
        if (!isspace((unsigned char)*line))
        {
            return 0;
        }
        ++line;
    }
    return 1;
}

int read_csv_int_matrix(const char *path, int rows, int cols, int32_t *out)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        return -1;
    }

    char line[LINE_BUF];
    int r = 0;
    while (fgets(line, sizeof(line), fp) != NULL && r < rows)
    {
        if (line_is_blank(line))
        {
            continue;
        }

        int c = 0;
        char *token = strtok(line, ",\n\r");
        while (token != NULL && c < cols)
        {
            out[r * cols + c] = (int32_t)strtol(token, NULL, 10);
            c++;
            token = strtok(NULL, ",\n\r");
        }

        if (c != cols)
        {
            fclose(fp);
            return -2;
        }
        r++;
    }

    fclose(fp);
    return (r == rows) ? 0 : -3;
}

int read_csv_float_vector(const char *path, int expected_len, float *out)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        return -1;
    }

    char line[LINE_BUF];
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (line_is_blank(line))
        {
            continue;
        }

        int n = 0;
        char *token = strtok(line, ",\n\r");
        while (token != NULL && n < expected_len)
        {
            out[n++] = strtof(token, NULL);
            token = strtok(NULL, ",\n\r");
        }

        fclose(fp);
        return (n == expected_len) ? 0 : -2;
    }

    fclose(fp);
    return -3;
}

int load_quantized_dataset_csv(
    const char *path,
    int feature_cols,
    int32_t **features_out,
    int **labels_out,
    int *samples_out)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        return -1;
    }

    char line[LINE_BUF];

    if (fgets(line, sizeof(line), fp) == NULL)
    {
        fclose(fp);
        return -2;
    }

    int count = 0;
    while (fgets(line, sizeof(line), fp) != NULL)
    {
        if (!line_is_blank(line))
        {
            count++;
        }
    }

    if (count <= 0)
    {
        fclose(fp);
        return -3;
    }

    int32_t *features = (int32_t *)malloc((size_t)count * (size_t)feature_cols * sizeof(int32_t));
    int *labels = (int *)malloc((size_t)count * sizeof(int));
    if (!features || !labels)
    {
        free(features);
        free(labels);
        fclose(fp);
        return -4;
    }

    rewind(fp);
    fgets(line, sizeof(line), fp);

    int row = 0;
    while (fgets(line, sizeof(line), fp) != NULL && row < count)
    {
        if (line_is_blank(line))
        {
            continue;
        }

        int col = 0;
        char *token = strtok(line, ",\n\r");
        while (token != NULL && col < feature_cols)
        {
            features[row * feature_cols + col] = (int32_t)strtol(token, NULL, 10);
            col++;
            token = strtok(NULL, ",\n\r");
        }

        if (col != feature_cols || token == NULL)
        {
            free(features);
            free(labels);
            fclose(fp);
            return -5;
        }

        labels[row] = (int)strtol(token, NULL, 10);
        row++;
    }

    fclose(fp);

    if (row != count)
    {
        free(features);
        free(labels);
        return -6;
    }

    *features_out = features;
    *labels_out = labels;
    *samples_out = count;
    return 0;
}
