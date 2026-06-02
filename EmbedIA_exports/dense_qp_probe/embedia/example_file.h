#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "common.h"



// The sample array below may contain up to 11 elements. Ensure the macros FST_TEST_SAMPLE and LST_TEST_SAMPLE are 
// within the range [0, 11] and that FST_TEST_SAMPLE ≤ LST_TEST_SAMPLE.
#define FST_TEST_SAMPLE 0
#define LST_TEST_SAMPLE 11
// number of examples to test in main file
#define TEST_SAMPLES (LST_TEST_SAMPLE-FST_TEST_SAMPLE+1)


const qparam_t sample_data_qp = {
    (int32_t) (0.02136516384780407*Q_SCALE), // Escala
    17 // Punto cero
};

static quant8 sample_data[][6]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
    {   113, -74, 49, -10, 106, 92 }
#endif
#if (FST_TEST_SAMPLE <= 1) && (1 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 1)
    ,
    #endif
    {   57, -36, -3, 42, 56, 56 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
    {   46, 9, -33, 64, 41, 77 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
    {   -21, 54, -2, 41, -8, -6 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
    {   23, 19, 2, 38, 10, 2 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
    {   -33, 82, 55, -17, -44, -53 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
    {   98, -36, 127, -128, 87, 54 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
    {   45, -29, -27, 60, 55, 44 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
    {   -6, 21, -36, 66, 5, 32 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
    {   11, 42, -11, 49, 2, 32 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
    {   12, 34, 15, 26, 2, -8 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
    {   -45, 67, 68, -34, -53, -51 }
#endif

};

static int sample_data_ids[][6]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 1) && (1 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 1)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
        {   3 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
        {   4 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
        {   5 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
        {   3 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
        {   4 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
        {   5 }
#endif

};



#endif