/* Ensure common definitions are available first */
{
    includes
}

// Initialization function prototypes
{
    prototypes_init
}

// Global Variables
{
    var
}

void model_init()
{
    {
        init
    }
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

    {
        predict_class
    }
    // return argmax(data1d_t);
}

// Implementation of initialization functions

{
    functions_init
}
