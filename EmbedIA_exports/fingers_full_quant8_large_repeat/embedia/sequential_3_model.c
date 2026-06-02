/* Ensure common definitions are available first */
#ifndef EMBEDIA_MODEL_STORAGE
#define EMBEDIA_MODEL_STORAGE const
#endif

#include "common.h"
#include "neural_net.h"
#include "realtype.h"
#include "sequential_3_model.h"


// Initialization function prototypes
dense_layer_t init_dense_6_data(void);
dense_layer_t init_dense_7_data(void);


// Global Variables
dense_layer_t dense_6_data;
dense_layer_t dense_7_data;


void model_init()
{
        dense_6_data = init_dense_6_data();
    dense_7_data = init_dense_7_data();

}

void model_predict(data1d_t input, data1d_t * output)
{
        prepare_buffers();
    
    //******************** LAYER 0 *******************//
    // Layer name: dense_6
    data1d_t output0;
    dense_layer(&dense_6_data, &input, &output0);
    
    
    //******************** LAYER 1 *******************//
    // Layer name: dense_61
    tanh_activation(output0.data, 8, &output0.qparam, (qparam_t){ 258, 0 });
    
    //******************** LAYER 2 *******************//
    // Layer name: dense_7
    input = output0;
    dense_layer(&dense_7_data, &input, &output0);
    
    
    //******************** LAYER 3 *******************//
    // Layer name: dense_71
    softmax_activation(output0.data, 6, &output0.qparam, (qparam_t){ 258, 0 });
    
    *output = output0;
}

int model_predict_class(data1d_t input, data1d_t * results)
{
    model_predict(input, results);

    return argmax(*results);
    // return argmax(data1d_t);
}

// Implementation of initialization functions


dense_layer_t init_dense_6_data(void) {
    // 6 inputs, 8 neurons

    static EMBEDIA_MODEL_STORAGE quant8 weights[] = {  // [8 x 6]
        118, 35, 0, 36, -128, 61,  /* N0   |  2.325447,  0.073572, -0.869453,  0.119891, -4.452960,  0.774831 */
        54, 68, -39, 13, 35, -46,  /* N1   |  0.589534,  0.990583, -1.930887, -0.503902,  0.082715, -2.116818 */
        127, 28, -8, 22, -49, 79,  /* N2   |  2.624834, -0.103800, -1.093771, -0.267888, -2.189008,  1.286873 */
        100, 57, 69, -14, -2, 91,  /* N3   |  1.843732,  0.669817,  0.998044, -1.255060, -0.936451,  1.594587 */
        -10, 60, 51, 52, 122, 68,  /* N4   | -1.137578,  0.768034,  0.509895,  0.551417,  2.453978,  0.987340 */
        123, 97, 92, -10, -26, 33,  /* N5   |  2.483794,  1.768396,  1.625483, -1.153162, -1.570430,  0.039392 */
        -113, -1, -78, 79, 39, 66,  /* N6   | -3.934033, -0.897717, -2.994352,  1.278223,  0.190811,  0.932230 */
        47, 54, -32, 40, 119, -5  /* N7   |  0.413617,  0.597390, -1.729591,  0.216208,  2.368924, -0.997604 */
    };

    static EMBEDIA_MODEL_STORAGE int32_t biases[] = {  // [8]
        -139, -4774, -1391, -1778, 4110, 3786, -3402, -812  /* -0.061524, -2.109306, -0.614724, -0.785531,  1.815804,  1.672584, -1.502995, -0.358822 */
    };

    static EMBEDIA_MODEL_STORAGE dense_layer_t layer = {
        6, 8,
        weights, biases, { 890, 32 }, { 4830, 0 }
    };

    return layer;
}
dense_layer_t init_dense_7_data(void) {
    // 8 inputs, 6 neurons

    static EMBEDIA_MODEL_STORAGE quant8 weights[] = {  // [6 x 8]
        -110, -5, 105, 127, -5, 28, -54, -29,  /* N0   | -8.217440,  0.431333,  9.475675,  11.753713,  0.405538,  3.085797, -3.597622, -1.589562 */
        -91, 44, -3, -45, -43, -52, -42, 72,  /* N1   | -6.682950,  4.453371,  0.585883, -2.846460, -2.702246, -3.487927, -2.641204,  6.726367 */
        33, 10, 27, -28, 15, -62, 15, -71,  /* N2   |  3.531294,  1.621170,  3.031334, -1.510779,  2.021592, -4.299030,  2.082888, -5.032841 */
        -8, -63, -18, -8, 1, 52, 51, -1,  /* N3   |  0.166572, -4.328139, -0.653987,  0.202571,  0.919709,  5.122739,  5.043220,  0.723668 */
        -31, -68, -57, -22, 3, 11, -69, 8,  /* N4   | -1.738256, -4.755411, -3.871255, -0.995563,  1.077618,  1.710858, -4.841161,  1.503719 */
        84, 31, -15, 72, -128, 15, -36, -74  /* N5   |  7.708205,  3.366445, -0.402561,  6.718951, -10.198814,  2.048516, -2.114292, -5.234674 */
    };

    static EMBEDIA_MODEL_STORAGE int32_t biases[] = {  // [6]
        -7674, 3191, 199, -2063, -960, 245  /* -4.963518,  2.064179,  0.128668, -1.334169, -0.621117,  0.158444 */
    };

    static EMBEDIA_MODEL_STORAGE dense_layer_t layer = {
        8, 6,
        weights, biases, { 2692, -10 }, { 10544, 0 }
    };

    return layer;
}
