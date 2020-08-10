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
float MSE(unsigned short *h_T, float *h_O, int batchSize, int numLabels);
void batchPreds(float * h_activations, unsigned short * h_backPreds, int activation_size, int b_size);

// Backprop
void error_function(unsigned short * t, float* z, float* h, float* output_weights, float* delta_k, float* delta_j);
void update_weights(float eta, float alpha, float* hidden_weights, int wRows, int wCols, float* dotP, float * delta_weights);

#endif // KERNELS_H
