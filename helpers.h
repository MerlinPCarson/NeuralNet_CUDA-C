#ifndef HELPERS_H
#define HELPERS_H

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>

#define TEST_DOT_PRODUCT

// Macro for checking for cuda errors
#define cudaCheckError(status) 									\
do {												\
  cudaError_t err = status;									\
  if(err!=cudaSuccess) {									\
    printf("Cuda failure %s in %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);	\
    exit(EXIT_FAILURE);										\
  }												\
 } while(0)					

// display all cuda device
int cudaDeviceProperties();
void hostDotProduct(float* M, float* N, float* P, int num_MRows, int num_MCols, int num_NRows, int num_NCols);
void hostActivationFunc(float *h_Z, float *h_Y, int numRows, int numCols);

#endif // HELPERS_H
