#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>
#include "helpers.h"

#define BLOCK_WIDTH 32

void activationFuncBackward(float *h_Z, float *h_dervA, float *h_dervZ, int numRows, int numCols);
void activationFuncForward(float *h_Z, float *h_Y, int numRows, int numCols);
void dotProduct(float* d_M, float* d_N, float* d_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols);
void elementMult(float *d_M, float *d_N, float *d_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols);
void transpose(float *h_M, float *h_N, int num_MRows, int num_MCols);
float MSE(float *h_T, float *h_O, int batchSize, int numLabels);

// Backprop
void updateWeights(float* d_w, float eta, float* d_dotP, float alpha, int Rows, int Cols);
void outputError(float* d_error, float* target, float* out_layer, int Rows, int Cols);
void hiddenError(float* d_error, float* outputUnits, float* hidden_layer, int Rows, int Cols);

#endif // KERNELS_H
