#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include "helpers.h"

#define BLOCK_WIDTH 32

void dotProduct(float* d_M, float* d_N, float* d_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols);

#endif // KERNELS_H
