#ifndef _EXAMPLE_FILE_H
#define _EXAMPLE_FILE_H

#include "common.h"



// The sample array below may contain up to 23 elements. Ensure the macros FST_TEST_SAMPLE and LST_TEST_SAMPLE are 
// within the range [0, 23] and that FST_TEST_SAMPLE ≤ LST_TEST_SAMPLE.
#define FST_TEST_SAMPLE 0
#define LST_TEST_SAMPLE 23
// number of examples to test in main file
#define TEST_SAMPLES (LST_TEST_SAMPLE-FST_TEST_SAMPLE+1)


const qparam_t sample_data_qp = {
    (int32_t) (0.018506858497858047*Q_SCALE), // Escala
    9 // Punto cero
};

static quant8 sample_data[][6]= {
#if (FST_TEST_SAMPLE <= 0) && (0 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 0)
    ,
    #endif
    {   120, -96, 47, -22, 112, 96 }
#endif
#if (FST_TEST_SAMPLE <= 1) && (1 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 1)
    ,
    #endif
    {   100, -95, 30, -4, 107, 96 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
    {   55, -52, -14, 38, 53, 54 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
    {   33, -50, -43, 60, 56, 14 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
    {   42, 0, -49, 64, 36, 79 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
    {   -34, 13, -49, 64, -16, 30 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
    {   -35, 52, -13, 37, -19, -17 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
    {   -35, 47, -2, 28, -24, -11 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
    {   16, 11, -8, 33, 1, -8 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
    {   -16, 51, 45, -20, -22, -35 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
    {   -48, 84, 53, -31, -62, -72 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
    {   -66, 75, 64, -45, -70, -73 }
#endif
#if (FST_TEST_SAMPLE <= 12) && (12 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 12)
    ,
    #endif
    {   103, -52, 127, -128, 90, 52 }
#endif
#if (FST_TEST_SAMPLE <= 13) && (13 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 13)
    ,
    #endif
    {   94, -44, 102, -102, 93, 74 }
#endif
#if (FST_TEST_SAMPLE <= 14) && (14 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 14)
    ,
    #endif
    {   41, -45, -42, 59, 53, 40 }
#endif
#if (FST_TEST_SAMPLE <= 15) && (15 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 15)
    ,
    #endif
    {   32, -45, -52, 66, 35, 3 }
#endif
#if (FST_TEST_SAMPLE <= 16) && (16 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 16)
    ,
    #endif
    {   -17, 13, -52, 66, -5, 27 }
#endif
#if (FST_TEST_SAMPLE <= 17) && (17 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 17)
    ,
    #endif
    {   10, 4, -66, 75, 4, 59 }
#endif
#if (FST_TEST_SAMPLE <= 18) && (18 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 18)
    ,
    #endif
    {   2, 38, -24, 46, -8, 26 }
#endif
#if (FST_TEST_SAMPLE <= 19) && (19 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 19)
    ,
    #endif
    {   1, 43, -34, 53, 2, 34 }
#endif
#if (FST_TEST_SAMPLE <= 20) && (20 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 20)
    ,
    #endif
    {   3, 28, 7, 19, -9, -20 }
#endif
#if (FST_TEST_SAMPLE <= 21) && (21 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 21)
    ,
    #endif
    {   -33, 66, 48, -24, -39, -39 }
#endif
#if (FST_TEST_SAMPLE <= 22) && (22 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 22)
    ,
    #endif
    {   -62, 67, 68, -50, -72, -69 }
#endif
#if (FST_TEST_SAMPLE <= 23) && (23 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 23)
    ,
    #endif
    {   -23, 56, 56, -34, -46, -60 }
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
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 2) && (2 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 2)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 3) && (3 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 3)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 4) && (4 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 4)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 5) && (5 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 5)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 6) && (6 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 6)
    ,
    #endif
        {   3 }
#endif
#if (FST_TEST_SAMPLE <= 7) && (7 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 7)
    ,
    #endif
        {   3 }
#endif
#if (FST_TEST_SAMPLE <= 8) && (8 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 8)
    ,
    #endif
        {   4 }
#endif
#if (FST_TEST_SAMPLE <= 9) && (9 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 9)
    ,
    #endif
        {   4 }
#endif
#if (FST_TEST_SAMPLE <= 10) && (10 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 10)
    ,
    #endif
        {   5 }
#endif
#if (FST_TEST_SAMPLE <= 11) && (11 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 11)
    ,
    #endif
        {   5 }
#endif
#if (FST_TEST_SAMPLE <= 12) && (12 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 12)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 13) && (13 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 13)
    ,
    #endif
        {   0 }
#endif
#if (FST_TEST_SAMPLE <= 14) && (14 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 14)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 15) && (15 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 15)
    ,
    #endif
        {   1 }
#endif
#if (FST_TEST_SAMPLE <= 16) && (16 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 16)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 17) && (17 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 17)
    ,
    #endif
        {   2 }
#endif
#if (FST_TEST_SAMPLE <= 18) && (18 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 18)
    ,
    #endif
        {   3 }
#endif
#if (FST_TEST_SAMPLE <= 19) && (19 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 19)
    ,
    #endif
        {   3 }
#endif
#if (FST_TEST_SAMPLE <= 20) && (20 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 20)
    ,
    #endif
        {   4 }
#endif
#if (FST_TEST_SAMPLE <= 21) && (21 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 21)
    ,
    #endif
        {   4 }
#endif
#if (FST_TEST_SAMPLE <= 22) && (22 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 22)
    ,
    #endif
        {   5 }
#endif
#if (FST_TEST_SAMPLE <= 23) && (23 <= LST_TEST_SAMPLE)
    #if (FST_TEST_SAMPLE != 23)
    ,
    #endif
        {   5 }
#endif

};



#endif