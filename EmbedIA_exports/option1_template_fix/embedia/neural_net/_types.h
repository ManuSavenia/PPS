/* ### FULL_QUANT8: layer type definitions for int8 runtime. ### */

typedef struct
{
    const quant8 *weights;
    quant8 bias;
} filter_t;

typedef struct
{
    uint16_t n_filters;
    filter_t *filters;
    uint16_t channels;
    size2d_t kernel;
    uint8_t padding;
    size2d_t strides;
    qparam_t qparam;
} conv2d_layer_t;

typedef struct
{
    uint16_t n_filters;
    filter_t *filters;
    uint16_t channels;
    uint16_t kernel_size;
    uint8_t padding;
    uint16_t stride;
    qparam_t qparam;
} conv1d_layer_t;

typedef struct
{
    const quant8 *weights;
    const quant8 *bias;
    uint16_t channels;
    size2d_t kernel_sz;
    uint8_t padding;
    size2d_t strides;
    qparam_t w_qparam;
    qparam_t b_qparam;
} depthwise_conv2d_layer_t;

typedef struct
{
    uint16_t n_filters;
    filter_t *point_filters;
    uint16_t point_channels;
    size2d_t point_kernel_sz;
    filter_t depth_filter;
    uint16_t depth_channels;
    size2d_t depth_kernel_sz;
    uint8_t padding;
    size2d_t strides;
    qparam_t qparam;
} separable_conv2d_layer_t;

typedef struct
{
    uint16_t input_size;
    uint16_t output_size;
    const quant8 *weights;
    const int32_t *biases;
    qparam_t weights_qparam;
    qparam_t output_qparam;
} dense_layer_t;

typedef struct
{
    uint16_t size;
    uint16_t strides;
} pooling2d_layer_t;

typedef struct
{
    uint16_t size;
    uint16_t strides;
} pooling1d_layer_t;

typedef struct
{
    const float *sub_val;
    const float *inv_div_val;
} normalization_layer_t;

typedef struct
{
    uint32_t length;
    const quant8 *moving_inv_std_dev;
    const quant8 *std_beta;
    qparam_t mov_qparam;
    qparam_t std_qparam;
} batch_normalization_layer_t;