#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include "helpers.h"

#define BLOCK_WIDTH 32

void activationFuncBackward(float *h_Z, float *h_dervA, float *h_dervZ, int numRows, int numCols);
void activationFuncForward(float *h_Z, float *h_Y, int numRows, int numCols);
void dotProduct(float* d_M, float* d_N, float* d_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols);

#endif // KERNELS_H
