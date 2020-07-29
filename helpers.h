#ifndef HELPERS_H
#define HELPERS_H

#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "data.h"
#include "neural_net.h"

// Macro for checking for cuda errors
#define cudaCheckError(status) 									\
do {												\
  cudaError_t err = status;									\
  if(err!=cudaSuccess) {									\
    printf("Cuda failure %s in %s line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);	\
    exit(EXIT_FAILURE);										\
  }												\
 } while(0)					


// various helper functions
void printMatrix(float *X, int numRows, int numCols);
void testDatasets(std::vector<Data> &trainSet, std::vector<Data> &valSet, std::vector<Data> &testData);
void saveHistory(History history, const char* fileName);
int cudaDeviceProperties();

// CPU based math functions
void hostBatchPreds(float* output_activations, int * batch_pred);
void hostElementMult(float *h_M, float *h_N, float *h_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols);
void hostDotProduct(float* M, float* N, float* P, int num_MRows, int num_MCols, int num_NRows, int num_NCols);
void hostTranspose(float* A, int rows, int cols, float* &B);
void hostActivationFuncForward(float *h_Z, float *h_Y, int numRows, int numCols);
void hostActivationFuncBackward(float *h_Z, float *h_dervA, float *h_dervZ, int numRows, int numCols);
void hostTranspose(float *h_M, float *h_N, int num_MRows, int num_MCols);
float hostMSE(float* h_T, float* h_O, int batchSize, int numLabels);

#endif // HELPERS_H
