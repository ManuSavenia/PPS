#ifndef _NEURAL_NET_H
#define _NEURAL_NET_H
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
#include "common.h"
#include "quant8.h"
#include "neural_net/_types.h"

/* ### FULL_QUANT8: runtime operates entirely on int8/quant8 types. ### */

//{includes}

/* LIBRARY FUNCTIONS PROTOTYPES */

/*
 * prepare_buffers()
 *   This function should be invoked only at the beginning of the predict function of the model file.
 *   Its purpose is to align the exchange buffers used by the different functions of the model. Due to
 *   the allocation strategy that never frees the memory, it happens that if the swap_alloc function
 *   is invoked an odd number of times in the 2nd invocation the predict reserves more memory than
 *   necessary  (something that usually happens with convolutional layers)
 */
void prepare_buffers();

/*
 * conv2d_layer()
 *   Function in charge of applying the convolution of a filter layer (conv_layer_t) without padding and strides
 *   on a given input data set.
 * Parameters:
 *  - layer => convolutional layer with loaded filters.
 *  - input => input data of type data3d_t
 *  - *output => pointer to the data3d_t structure where the result will be saved.
 */
void conv2d_layer(conv2d_layer_t layer, data3d_t input, data3d_t *output);

/* variant function with padding and strides */
void conv2d_padding_layer(conv2d_layer_t layer, data3d_t input, data3d_t *output);

/* variant function with strides without padding*/
void conv2d_strides_layer(conv2d_layer_t layer, data3d_t input, data3d_t *output);

/*
 * separable_conv2d_layer()
 *   Function in charge of applying the convolution of a filter layer (conv_layer_t) on a given input data set.
 *
 * Parameters:
 *  - layer => convolutional layer with loaded filters.
 *  - input => input data of type data3d_t
 *  - *output => pointer to the data3d_t structure where the result will be saved.
 */
void separable_conv2d_layer(separable_conv2d_layer_t layer, data3d_t input, data3d_t *output);

/*
 * depthwise_conv2d_layer()
 *   Function in charge of applying the depthwise of a filter layer with bias (depthwise_conv2d_layer_t) on a given input data set.
 * Parameters:
 * - layer => depthwise layer with loaded filters.
 * - input => input data of type data3d_t
 * - *output => pointer to the data3d_t structure where the result will be saved.
 */

void depthwise_conv2d_layer(depthwise_conv2d_layer_t layer, data3d_t input, data3d_t *output);

/*
 * dense_layer()
 *   Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *  - dense_layer => structure with the weights of the neurons of the dense layer.
 *  - input       => structure data1d_t with the input data to process.
 *  - *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t *layer, data1d_t *input, data1d_t *output);

/*
 * max_pooling2d_layer()
 *   Maxpooling layer, for now supports square size and stride. No support for padding
 * Parameters:
 *  - pool_size => size for pooling
 *  - stride    => stride for pooling
 *  - input     => input data
 *  - output    => output data
 */
void max_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t *output);

/*
 * average_pooling_2d()
 *   Function that applies an average pooling to an input with a window size of received
 *   by parameter (uint16_t strides)
 * Parameters:
 *  - input => input data of type data3d_t.
 *  - *output => pointer to the data3d_t structure where the result will be stored.
 */
void average_pooling2d_layer(pooling2d_layer_t pool, data3d_t input, data3d_t *output);

/*
 * flatten3d_layer()
 * Performs a variable shape change.
 * Converts the data format from data3d_t array format to data1d_t vector.
 * (prepares data for input into a layer of type dense_layer_t).
 * Parameters:
 *  -  input => input data of type data3d_t.
 *  -  *output => pointer to the data1d_t structure where the result will be stored.
 */
void flatten3d_layer(data3d_t input, data1d_t *output);

/***************************************************************************************************************************/
/* Activation functions/layers */

/* FULL_QUANT8: activations operate on quant8 buffers and require qparams */
/* Activation API updated: pass pointer to buffer qparam so runtime can update
   the buffer's qparam to the post-activation quantization parameters. The
   functions receive both a pointer to the current buffer qparam (which will
   be overwritten) and the desired output qparam (either a literal or the
   same as the buffer qparam when no change is needed). */
void softmax_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp);

void relu_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp);

void relu6_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp);

void leakyrelu_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp, float alpha);

void tanh_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp);

void sigmoid_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp);

void softsign_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp);

void softplus_activation(quant8 *data, uint32_t length, qparam_t *buf_qp, qparam_t out_qp);

/***************************************************************************************************************************/
/* Normalization layers */

/* Normalization function for:
 *  standard normalization  : (x_i-mean_i)/std_dev_i
 *  min_max normalization   : (x_i-min_i) / (max_i-min_i)
 *  robust normalization    : (x_i-q2_i)  / (q3_i-q1_i)
 */
void normalization1(normalization_layer_t s, data1d_t input, data1d_t *output);

#define standard_norm_layer(norm, input, output) normalization1(norm, input, output)

#define min_max_norm_layer(norm, input, output) normalization1(norm, input, output)

#define robust_norm_layer(norm, input, output) normalization1(norm, input, output)

/* Normalization function for:
 *  abs_max_normalization   : (x_i)/(abs_max_xi)
 */
void normalization2(normalization_layer_t s, data1d_t input, data1d_t *output);

#define max_abs_norm_layer(norm, input, output) normalization2(norm, input, output)

/* Batch normalization */
void batch_normalization_layer(batch_normalization_layer_t norm, uint32_t length, float *data);

void batch_normalization3d_layer(batch_normalization_layer_t layer, data3d_t *data);

void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data);

/* Rashaping Layers */

/* void zero_padding2d_layer(uint8_t pad_h, uint8_t pad_w, data3d_t input, data3d_t *output)
 * Applies zero-padding to a 2D input data array.
 * Parameters:
 *  - pad_h: Number of zero-padding rows to add at the top and bottom.
 *  - pad_w: Number of zero-padding columns to add at the left and right.
 *  - input: 3D data structure representing the input data.
 *  - output: Pointer to a 3D data structure where the zero-padded output will be stored.
 * Description:
 *   This function performs zero-padding on a 2D input data array. It adds the specified
 *   number of zero rows at the top and bottom (pad_h) and zero columns at the left and right (pad_w).
 *   The result is stored in the output data structure.
 */
void zero_padding2d_layer(uint8_t pad_h, uint8_t pad_w, data3d_t input, data3d_t *output);

/* Tranformation Layers */

/*  Converts Tensorflow/Keras Image (Height, Width, Channel) to Embedia format (Channel, Height, Width).
   Usually required for first convolutional layer
*/
void channel_adapt_layer(data3d_t input, data3d_t *output);

#endif
