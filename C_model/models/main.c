#include "../utility/utils.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_DIM 6
#define HIDDEN_DIM 8
#define OUTPUT_DIM 6

typedef enum
{
    SCALE_PER_LAYER = 0,
    SCALE_PER_NEURON = 1
} ScaleMode;

typedef struct
{
    int32_t w0[INPUT_DIM * HIDDEN_DIM];
    int32_t b0[HIDDEN_DIM];
    int32_t w1[HIDDEN_DIM * OUTPUT_DIM];
    int32_t b1[OUTPUT_DIM];

    float s_w0[HIDDEN_DIM];
    float s_b0[HIDDEN_DIM];
    float s_w1[OUTPUT_DIM];
    float s_b1[OUTPUT_DIM];

    ScaleMode mode;
    const char *name;
} QModel;

typedef struct
{
    float train_layer;
    float test_layer;
    float train_neuron;
    float test_neuron;
    int found;
} PythonRefs;

static int load_model_per_layer(QModel *m, const char *data_dir)
{
    char path[512];
    float layer_scales[4];

    snprintf(path, sizeof(path), "%s/weights/per_layer/weights_q_per_layer_w0.csv", data_dir);
    if (read_csv_int_matrix(path, INPUT_DIM, HIDDEN_DIM, m->w0) != 0)
        return -1;

    snprintf(path, sizeof(path), "%s/weights/per_layer/weights_q_per_layer_b0.csv", data_dir);
    if (read_csv_int_matrix(path, 1, HIDDEN_DIM, m->b0) != 0)
        return -2;

    snprintf(path, sizeof(path), "%s/weights/per_layer/weights_q_per_layer_w1.csv", data_dir);
    if (read_csv_int_matrix(path, HIDDEN_DIM, OUTPUT_DIM, m->w1) != 0)
        return -3;

    snprintf(path, sizeof(path), "%s/weights/per_layer/weights_q_per_layer_b1.csv", data_dir);
    if (read_csv_int_matrix(path, 1, OUTPUT_DIM, m->b1) != 0)
        return -4;

    snprintf(path, sizeof(path), "%s/scales/weight_scales_per_layer.csv", data_dir);
    if (read_csv_float_vector(path, 4, layer_scales) != 0)
        return -5;

    for (int j = 0; j < HIDDEN_DIM; ++j)
    {
        m->s_w0[j] = layer_scales[0];
        m->s_b0[j] = layer_scales[1];
    }
    for (int k = 0; k < OUTPUT_DIM; ++k)
    {
        m->s_w1[k] = layer_scales[2];
        m->s_b1[k] = layer_scales[3];
    }

    m->mode = SCALE_PER_LAYER;
    m->name = "Q por capa";
    return 0;
}

static int load_model_per_neuron(QModel *m, const char *data_dir)
{
    char path[512];

    snprintf(path, sizeof(path), "%s/weights/per_neuron/weights_q_per_neuron_w0.csv", data_dir);
    if (read_csv_int_matrix(path, INPUT_DIM, HIDDEN_DIM, m->w0) != 0)
        return -1;

    snprintf(path, sizeof(path), "%s/weights/per_neuron/weights_q_per_neuron_b0.csv", data_dir);
    if (read_csv_int_matrix(path, 1, HIDDEN_DIM, m->b0) != 0)
        return -2;

    snprintf(path, sizeof(path), "%s/weights/per_neuron/weights_q_per_neuron_w1.csv", data_dir);
    if (read_csv_int_matrix(path, HIDDEN_DIM, OUTPUT_DIM, m->w1) != 0)
        return -3;

    snprintf(path, sizeof(path), "%s/weights/per_neuron/weights_q_per_neuron_b1.csv", data_dir);
    if (read_csv_int_matrix(path, 1, OUTPUT_DIM, m->b1) != 0)
        return -4;

    snprintf(path, sizeof(path), "%s/scales/weight_scales_per_neuron_w0.csv", data_dir);
    if (read_csv_float_vector(path, HIDDEN_DIM, m->s_w0) != 0)
        return -5;

    snprintf(path, sizeof(path), "%s/scales/weight_scales_per_neuron_b0.csv", data_dir);
    if (read_csv_float_vector(path, HIDDEN_DIM, m->s_b0) != 0)
        return -6;

    snprintf(path, sizeof(path), "%s/scales/weight_scales_per_neuron_w1.csv", data_dir);
    if (read_csv_float_vector(path, OUTPUT_DIM, m->s_w1) != 0)
        return -7;

    snprintf(path, sizeof(path), "%s/scales/weight_scales_per_neuron_b1.csv", data_dir);
    if (read_csv_float_vector(path, OUTPUT_DIM, m->s_b1) != 0)
        return -8;

    m->mode = SCALE_PER_NEURON;
    m->name = "Q por neurona";
    return 0;
}

 static void infer_one(
    const QModel *m,
    const int32_t *x_q,
    const float *input_scales,
    float *probs_out)
{
    float x[INPUT_DIM];
    float h[HIDDEN_DIM];
    float z2[OUTPUT_DIM];

    for (int i = 0; i < INPUT_DIM; ++i)
    {
        x[i] = dequantize_value(x_q[i], input_scales[i]);
    }

    for (int j = 0; j < HIDDEN_DIM; ++j)
    {
        float acc = 0.0f;
        for (int i = 0; i < INPUT_DIM; ++i)
        {
            const float w = dequantize_value(m->w0[i * HIDDEN_DIM + j], m->s_w0[j]);
            acc += x[i] * w;
        }
        acc += dequantize_value(m->b0[j], m->s_b0[j]);
        h[j] = act_tanh(acc);
    }

    for (int k = 0; k < OUTPUT_DIM; ++k)
    {
        float acc = 0.0f;
        for (int j = 0; j < HIDDEN_DIM; ++j)
        {
            const float w = dequantize_value(m->w1[j * OUTPUT_DIM + k], m->s_w1[k]);
            acc += h[j] * w;
        }
        acc += dequantize_value(m->b1[k], m->s_b1[k]);
        z2[k] = acc;
    }

    act_softmax(z2, OUTPUT_DIM, probs_out);
}

static float evaluate_accuracy(
    const QModel *m,
    const int32_t *x,
    const int *y,
    int samples,
    const float *input_scales)
{
    int correct = 0;
    float probs[OUTPUT_DIM];

    for (int n = 0; n < samples; ++n)
    {
        const int32_t *x_row = &x[n * INPUT_DIM];
        infer_one(m, x_row, input_scales, probs);
        const int pred = argmaxf(probs, OUTPUT_DIM);
        if (pred == y[n])
        {
            correct++;
        }
    }

    return (samples > 0) ? ((float)correct / (float)samples) : 0.0f;
}

static void trim(char *s)
{
    size_t len = strlen(s);
    while (len > 0 && (s[len - 1] == '\n' || s[len - 1] == '\r' || s[len - 1] == ' '))
    {
        s[len - 1] = '\0';
        len--;
    }
}

static int parse_python_refs(const char *path, PythonRefs *refs)
{
    FILE *fp = fopen(path, "r");
    if (!fp)
    {
        refs->found = 0;
        return -1;
    }

    char line[1024];
    if (fgets(line, sizeof(line), fp) == NULL)
    {
        fclose(fp);
        refs->found = 0;
        return -2;
    }

    refs->found = 1;
    refs->train_layer = -1.0f;
    refs->test_layer = -1.0f;
    refs->train_neuron = -1.0f;
    refs->test_neuron = -1.0f;

    while (fgets(line, sizeof(line), fp) != NULL)
    {
        trim(line);

        char *model = strtok(line, ",");
        char *flow = strtok(NULL, ",");
        char *partition = strtok(NULL, ",");
        char *acc = strtok(NULL, ",");

        if (!model || !flow || !partition || !acc)
        {
            continue;
        }

        if (strcmp(flow, "I/O dequant (desde CSV)") != 0)
        {
            continue;
        }

        const float v = strtof(acc, NULL);
        if (strcmp(model, "Q por capa") == 0 && strcmp(partition, "EntrenamientoQ") == 0)
            refs->train_layer = v;
        if (strcmp(model, "Q por capa") == 0 && strcmp(partition, "PruebaQ") == 0)
            refs->test_layer = v;
        if (strcmp(model, "Q por neurona") == 0 && strcmp(partition, "EntrenamientoQ") == 0)
            refs->train_neuron = v;
        if (strcmp(model, "Q por neurona") == 0 && strcmp(partition, "PruebaQ") == 0)
            refs->test_neuron = v;
    }

    fclose(fp);
    return 0;
}

static int write_results(
    const char *path,
    float c_train_layer,
    float c_test_layer,
    float c_train_neuron,
    float c_test_neuron,
    const PythonRefs *refs)
{
    FILE *fp = fopen(path, "w");
    if (!fp)
    {
        return -1;
    }

    fprintf(fp, "Modelo,Particion,Accuracy_C,Accuracy_Python,AbsDiff\\n");

    const float py_train_layer = refs->train_layer;
    const float py_test_layer = refs->test_layer;
    const float py_train_neuron = refs->train_neuron;
    const float py_test_neuron = refs->test_neuron;

    fprintf(fp, "Q por capa,EntrenamientoQ,%.8f,%.8f,%.8f\\n", c_train_layer, py_train_layer, fabsf(c_train_layer - py_train_layer));
    fprintf(fp, "Q por capa,PruebaQ,%.8f,%.8f,%.8f\\n", c_test_layer, py_test_layer, fabsf(c_test_layer - py_test_layer));
    fprintf(fp, "Q por neurona,EntrenamientoQ,%.8f,%.8f,%.8f\\n", c_train_neuron, py_train_neuron, fabsf(c_train_neuron - py_train_neuron));
    fprintf(fp, "Q por neurona,PruebaQ,%.8f,%.8f,%.8f\\n", c_test_neuron, py_test_neuron, fabsf(c_test_neuron - py_test_neuron));

    fclose(fp);
    return 0;
}

int main(void)
{
    const char *data_dir = "../data";

    float input_scale_train[INPUT_DIM];
    float input_scale_test[INPUT_DIM];

    char path[512];
    snprintf(path, sizeof(path), "%s/inputs/input_scale_train.csv", data_dir);
    if (read_csv_float_vector(path, INPUT_DIM, input_scale_train) != 0)
    {
        fprintf(stderr, "Error: no se pudo leer input_scale_train.csv\n");
        return 1;
    }

    snprintf(path, sizeof(path), "%s/inputs/input_scale_test.csv", data_dir);
    if (read_csv_float_vector(path, INPUT_DIM, input_scale_test) != 0)
    {
        fprintf(stderr, "Error: no se pudo leer input_scale_test.csv\n");
        return 1;
    }

    QModel model_layer;
    QModel model_neuron;

    if (load_model_per_layer(&model_layer, data_dir) != 0)
    {
        fprintf(stderr, "Error: no se pudo cargar modelo per-layer\n");
        return 1;
    }
    if (load_model_per_neuron(&model_neuron, data_dir) != 0)
    {
        fprintf(stderr, "Error: no se pudo cargar modelo per-neuron\n");
        return 1;
    }

    int32_t *x_train = NULL;
    int *y_train = NULL;
    int n_train = 0;

    int32_t *x_test = NULL;
    int *y_test = NULL;
    int n_test = 0;

    snprintf(path, sizeof(path), "%s/inputs/fingers_train_quant8_signed_symmetric.csv", data_dir);
    if (load_quantized_dataset_csv(path, INPUT_DIM, &x_train, &y_train, &n_train) != 0)
    {
        fprintf(stderr, "Error: no se pudo cargar dataset train cuantizado\n");
        return 1;
    }

    snprintf(path, sizeof(path), "%s/inputs/fingers_test_quant8_signed_symmetric.csv", data_dir);
    if (load_quantized_dataset_csv(path, INPUT_DIM, &x_test, &y_test, &n_test) != 0)
    {
        fprintf(stderr, "Error: no se pudo cargar dataset test cuantizado\n");
        free(x_train);
        free(y_train);
        return 1;
    }

    const float c_train_layer = evaluate_accuracy(&model_layer, x_train, y_train, n_train, input_scale_train);
    const float c_test_layer = evaluate_accuracy(&model_layer, x_test, y_test, n_test, input_scale_test);
    const float c_train_neuron = evaluate_accuracy(&model_neuron, x_train, y_train, n_train, input_scale_train);
    const float c_test_neuron = evaluate_accuracy(&model_neuron, x_test, y_test, n_test, input_scale_test);

    printf("==============================================\\n");
    printf("C Inference (dequantized pipeline)\\n");
    printf("==============================================\\n");
    printf("Q por capa    - Train: %.6f  Test: %.6f\\n", c_train_layer, c_test_layer);
    printf("Q por neurona - Train: %.6f  Test: %.6f\\n", c_train_neuron, c_test_neuron);

    PythonRefs refs = {
        .train_layer = -1.0f,
        .test_layer = -1.0f,
        .train_neuron = -1.0f,
        .test_neuron = -1.0f,
        .found = 0};
    snprintf(path, sizeof(path), "%s/reports/quantization_comparison_signed_symmetric.csv", data_dir);
    parse_python_refs(path, &refs);

    if (refs.found)
    {
        printf("\\nComparación contra referencia Python (I/O dequant desde CSV):\\n");
        printf("Q por capa    - Train Δ: %.8f  Test Δ: %.8f\\n", fabsf(c_train_layer - refs.train_layer), fabsf(c_test_layer - refs.test_layer));
        printf("Q por neurona - Train Δ: %.8f  Test Δ: %.8f\\n", fabsf(c_train_neuron - refs.train_neuron), fabsf(c_test_neuron - refs.test_neuron));
    }
    else
    {
        printf("\\nNo se encontró CSV de referencia Python para comparación.\\n");
    }

    snprintf(path, sizeof(path), "%s/reports/c_inference_comparison.csv", data_dir);
    if (write_results(path, c_train_layer, c_test_layer, c_train_neuron, c_test_neuron, &refs) == 0)
    {
        printf("\\nResultados guardados en: %s\\n", path);
    }
    else
    {
        printf("\\nNo se pudo guardar el CSV de resultados.\\n");
    }

    free(x_train);
    free(y_train);
    free(x_test);
    free(y_test);

    return 0;
}
