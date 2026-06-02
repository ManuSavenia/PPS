/* EmbedIA model definition file*/
#ifndef _SEQUENTIAL_3_MODEL_H_H
#define _SEQUENTIAL_3_MODEL_H_H

/*

EmbedIA Layer | Name | #Param(NT) | Shape | MACs | ACOPs | Buffer (KiB) | Params (KiB)
Dense | dense_6 | 56 | (8,) | 48 | 24 |    0.020 |    0.078
Activation(tanh) | dense_61 | 0 | (8,) | 0 | 40 |    0.000 |    0.000
Dense | dense_7 | 54 | (6,) | 48 | 18 |    0.020 |    0.070
Activation(softmax) | dense_71 | 0 | (6,) | 0 | 30 |    0.000 |    0.000
Data types:
  Compute : FULL_QUANT8 (1 bytes)
  Storage : FULL_QUANT8 (1 bytes) -> FLASH (const)

Total params (NT)....: 110
Total params (KiB)...: 0.148
Total MACs operations: 96
Total AC operations..: 112
Peak RAM (bytes).....: 28  <- dense_7 (inp=8 + out=6 + tmp=6)

*/

#include "common.h"

#define INPUT_LENGTH 6

#define INPUT_SIZE 6


void model_init();

void model_predict(data1d_t input, data1d_t * output);

int model_predict_class(data1d_t input, data1d_t * results);

#endif
