#ifndef HELPERS_H
#define HELPERS_H

#include <stdlib.h>
#include <cuda_runtime_api.h>


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

#endif // HELPERS_H
