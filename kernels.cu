#include "kernels.h"
#include <stdio.h>
#include <math.h>

__device__ float sigmoid(float z)
{
    float y = (float)1 / (1 + exp(-z));
    return y;
}

__global__ void elementMultDevice(float *d_M, float *d_N, float *d_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols)
{
    int num_PRows = num_MRows;
    int num_PCols = num_MCols;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_PRows && col < num_PCols) {
        int idx = col + row * num_PCols;
        d_P[idx] = d_M[idx] * d_N[idx];
    }
}

__global__ void batchPredsDevice(float * out_activations, unsigned short * batch, int output_size, int batch_size)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < batch_size && col == 0)
    {
        unsigned short counter = 0;
        float maxValue = out_activations[row * output_size];
        for(int i = 1; i < output_size; ++i)
        {
            int idx = i + row*output_size;
            if(out_activations[idx] > maxValue)
            {
                maxValue = out_activations[idx];
                counter = i;
            }
        }
        batch[row] = counter;
    }
}

__global__ void activationFuncForwardDevice(float *d_Z, float *d_Y, int numRows, int numCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int idx = col + row * numCols;
        float z = d_Z[idx];

        d_Y[idx] = sigmoid(z);
    }
}

__global__ void activationFuncBackwardDevice(float *d_Z, float *d_dervA, float *d_dervZ, int numRows, int numCols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int idx = col + row * numCols;
        float s = sigmoid(d_Z[idx]);
        
        d_dervZ[idx] =  d_dervA[idx] * s * (1  - s) ;
    }
}

__global__ void transposeDevice(float *d_M, float *d_N, int num_MRows, int num_MCols)
{
    int num_NRows = num_MCols;
    int num_NCols = num_MRows;

    // Row index of N
    int rowN = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Col index of N
    int colN = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Row index of M
    int rowM = colN;
    
    // Col index of M
    int colM = rowN;

    if (rowN < num_NRows && colN < num_NCols) {
        // Each thread computes one element of the block
        int n_idx = colN + rowN * num_NCols;
        int m_idx = colM + rowM * num_MCols;
        d_N[n_idx] = d_M[m_idx];
    }
}

// This is currently a non tiled version based on the text book implementation
__global__ void dotProductDevice(float *d_M, float *d_N, float *d_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols)
{
    int num_PRows = num_MRows;
    int num_PCols = num_NCols;

    // Row index of the P and M
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Col index of the P and N
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < num_PRows && col < num_NCols) {
        float pVal = 0.0;

        // Each thread computes one element of the block
        int i, j;
        for (i = 0, j = 0; i < num_NRows && j < num_MCols; i++, j++) {
            int m_idx = j + row * num_MCols;
            int n_idx = col + i * num_NCols;
            pVal += d_M[m_idx] * d_N[n_idx];
        }

        d_P[row * num_PCols + col] = pVal;
    }
}

// d_T is 1D (batchSize), d_O is 2D (batchSize, numLabels)
// numRows = batch size
// d_sampleSquareErr: array to store the square error for each sample
__global__ void mseDevice(
    unsigned short *d_T,
    float *d_O,
    float *d_sampleSquareErr,
    float *batchLoss,
    int batchSize,
    int numLabels
    )
{
    int batchId = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(batchId < batchSize){
      int t_idx = d_T[batchId];
      //printf("batch ID: %d, target: %f\n", batchId, d_T[batchId]);
  
      *batchLoss = 0;
      
      // Sanity check
      if ((t_idx < 0) || (t_idx >= numLabels)) {
          printf("t_idx (%d) >= numLabels (%d)\n", t_idx, numLabels);
          return;
      }
  
      // Now go through each of the output values and calculate the MSE
      float err = 0;
      for (int j = 0; j < numLabels; j++) {
          int o_idx = j + batchId * numLabels;
  
          if (t_idx == j) {
              // If this is the same as the expected output
              float diff = 1 - d_O[o_idx];
              err += diff * diff;
          }
          else {
              float diff = d_O[o_idx];
              err += diff * diff;
          }
      }
      d_sampleSquareErr[batchId] = err;
    } 

    __syncthreads();

    // Calculate the square error for the batch
    
    // Need only one thread to do this
    if (batchId == 0) {
        for (int i = 0; i < batchSize; i++) {
            *batchLoss += d_sampleSquareErr[i];
        }
        *batchLoss /= (float)2;
        *batchLoss /= (float)batchSize;
    }
}

// 
// Interface functions for the corresponding kernel functions.
//

// h_T is 1D (batchSize), h_O is 2D (batchSize, numLabels)
// numRows = batch size
float MSE(unsigned short *h_T, float *h_O, int batchSize, int numLabels)
{
    float h_batchLoss = 0;
    unsigned short *d_T;
    float *d_O, *d_sampleSquareErr, *d_batchLoss;
    cudaError_t cudaStatus;

    // Allocate memory for device variables
    cudaStatus = cudaMalloc((void**)&d_T, batchSize * sizeof(unsigned short));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_O, batchSize * numLabels * sizeof(float));
    cudaCheckError(cudaStatus);
    
    cudaStatus = cudaMalloc((void**)&d_sampleSquareErr, batchSize * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_batchLoss, sizeof(float));
    cudaCheckError(cudaStatus);
    
    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_T, h_T, batchSize * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_O, h_O, batchSize * numLabels * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);
    
    dim3 gridDim((int)ceil((float)batchSize / BLOCK_WIDTH), 1, 1);
    dim3 blockDim(BLOCK_WIDTH, 1, 1);

    // Call the kernel
    mseDevice<<<gridDim, blockDim>>>(d_T, d_O, d_sampleSquareErr, d_batchLoss, batchSize, numLabels);

    // Copy back to host
    cudaStatus = cudaMemcpy(&h_batchLoss, d_batchLoss, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus);

    // Free device memory
    cudaFree(d_T);
    cudaFree(d_O);
    cudaFree(d_sampleSquareErr);
    cudaFree(d_batchLoss);
    
    return h_batchLoss;
}

// h_Y will have the output
void activationFuncForward(float *h_Z, float *h_Y, int numRows, int numCols)
{
    float *d_Z, *d_Y;
    cudaError_t cudaStatus;

    // Allocate memory for device variables
    cudaStatus = cudaMalloc((void**)&d_Z, numRows * numCols * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_Y, numRows * numCols * sizeof(float));
    cudaCheckError(cudaStatus);
    
    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_Z, h_Z, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_Y, h_Y, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);
    
    dim3 gridDim((int)ceil((float)numCols / BLOCK_WIDTH), (int)ceil((float)numRows / BLOCK_WIDTH), 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // Call the kernel
    activationFuncForwardDevice<<<gridDim, blockDim>>>(d_Z, d_Y, numRows, numCols);

    // Copy back to host
    cudaStatus = cudaMemcpy(h_Y, d_Y, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus);

    // Free device memory
    cudaFree(d_Z);
    cudaFree(d_Y);
}

// h_dervZ will have the output
void activationFuncBackward(float *h_Z, float *h_dervA, float *h_dervZ, int numRows, int numCols)
{
    float *d_Z, *d_dervA, *d_dervZ;
    cudaError_t cudaStatus;

    // Allocate memory for device variables
    cudaStatus = cudaMalloc((void**)&d_Z, numRows * numCols * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_dervA, numRows * numCols * sizeof(float));
    cudaCheckError(cudaStatus);
    
    cudaStatus = cudaMalloc((void**)&d_dervZ, numRows * numCols * sizeof(float));
    cudaCheckError(cudaStatus);
    
    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_Z, h_Z, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_dervA, h_dervA, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);
    
    cudaStatus = cudaMemcpy(d_dervZ, h_dervZ, numRows * numCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);
    
    dim3 gridDim((int)ceil((float)numCols / BLOCK_WIDTH), (int)ceil((float)numRows / BLOCK_WIDTH), 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // Call the kernel
    activationFuncBackwardDevice<<<gridDim, blockDim>>>(d_Z, d_dervA, d_dervZ, numRows, numCols);

    // Copy back to host
    cudaStatus = cudaMemcpy(h_dervZ, d_dervZ, numRows * numCols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus);

    // Free device memory
    cudaFree(d_Z);
    cudaFree(d_dervA);
    cudaFree(d_dervZ);
}

void dotProduct(float *h_M, float *h_N, float *h_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols)
{
    float *d_M, *d_N, *d_P;
    cudaError_t cudaStatus;
    int num_PRows = num_MRows;
    int num_PCols = num_NCols;

    if (num_MCols != num_NRows) {
        printf("(device) num_MCols != num_NRows\n");
        exit(-1);
    }

    // Allocate memory for device variables
    cudaStatus = cudaMalloc((void**)&d_M, num_MRows * num_MCols * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_N, num_NRows * num_NCols * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_P, num_PRows * num_PCols * sizeof(float));
    cudaCheckError(cudaStatus);

    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_M, h_M, num_MRows * num_MCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_N, h_N, num_NRows * num_NCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_P, h_P, num_PRows * num_PCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    dim3 gridDim((int)ceil((float)num_PCols / BLOCK_WIDTH), (int)ceil((float)num_PRows / BLOCK_WIDTH), 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // Call the kernel
    dotProductDevice<<<gridDim, blockDim>>>(d_M, d_N, d_P, num_MRows, num_MCols, num_NRows, num_NCols);

    // Copy back to host
    cudaStatus = cudaMemcpy(h_P, d_P, num_PRows * num_PCols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

// The transposed matrix h_N will have rows = num_MCols, cols = num_MRows
void transpose(float *h_M, float *h_N, int num_MRows, int num_MCols)
{
    float *d_M, *d_N;
    cudaError_t cudaStatus;

    // Allocate memory for device variables
    cudaStatus = cudaMalloc((void**)&d_M, num_MRows * num_MCols * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_N, num_MRows * num_MCols * sizeof(float));
    cudaCheckError(cudaStatus);
    
    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_M, h_M, num_MRows * num_MCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_N, h_N, num_MRows * num_MCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);
    
    // The rows and cols are interchanged here because of the transpose
    dim3 gridDim((int)ceil((float)num_MRows / BLOCK_WIDTH), (int)ceil((float)num_MCols / BLOCK_WIDTH), 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // Call the kernel
    transposeDevice<<<gridDim, blockDim>>>(d_M, d_N, num_MRows, num_MCols);

    // Copy back to host
    cudaStatus = cudaMemcpy(h_N, d_N, num_MRows * num_MCols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
}

void batchPreds(float * h_activations, unsigned short * h_batchPreds, int activation_size, int b_size)
{
    float *d_activations;
    unsigned short *d_batchPreds;
    cudaError_t cudaStatus;

    // Allocate memory for device variables
    cudaStatus = cudaMalloc((void**)&d_activations, activation_size* b_size* sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_batchPreds, activation_size * sizeof(unsigned short));
    cudaCheckError(cudaStatus);

    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_activations, h_activations, activation_size * b_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_batchPreds, h_batchPreds, activation_size * sizeof(unsigned short), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    dim3 gridDim((int)ceil((float)activation_size / BLOCK_WIDTH), (int)ceil((float) b_size / BLOCK_WIDTH), 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    batchPredsDevice<<<gridDim, blockDim>>>(d_activations, d_batchPreds, activation_size, b_size);

    //copy back to host
    cudaStatus = cudaMemcpy(h_batchPreds, d_batchPreds, activation_size * sizeof(unsigned short), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus);

    cudaFree(d_activations);
    cudaFree(d_batchPreds);

}

void elementMult(float *h_M, float *h_N, float *h_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols)
{
    float *d_M, *d_N, *d_P;
    cudaError_t cudaStatus;
    int num_PRows = num_MRows;
    int num_PCols = num_MCols;

    if (num_MRows != num_NRows) {
        printf("(device) num_MRows!= num_NRows\n");
        exit(-1);
    }

    if (num_MCols != num_NCols) {
        printf("(device) num_MCols != num_NCols\n");
        exit(-1);
    }

    // Allocate memory for device variables
    cudaStatus = cudaMalloc((void**)&d_M, num_MRows * num_MCols * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_N, num_NRows * num_NCols * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_P, num_PRows * num_PCols * sizeof(float));
    cudaCheckError(cudaStatus);

    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_M, h_M, num_MRows * num_MCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_N, h_N, num_NRows * num_NCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_P, h_P, num_PRows * num_PCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    dim3 gridDim((int)ceil((float)num_PCols / BLOCK_WIDTH), (int)ceil((float)num_PRows / BLOCK_WIDTH), 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    // Call the kernel
    elementMultDevice<<<gridDim, blockDim>>>(d_M, d_N, d_P, num_MRows, num_MCols, num_NRows, num_NCols);

    // Copy back to host
    cudaStatus = cudaMemcpy(h_P, d_P, num_PRows * num_PCols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus);

    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

__global__ void scalarMultiplication(double scalar, double* M, int Rows, int Cols){
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x; 

    if(r < Rows && c < Cols)
        M[r*Cols + c] *= scalar;
}

__global__ void updateWeights(float eta, float alpha, float* d_dotP, int Rows, int Cols, float* d_w, float* d_delta_weights){
    /*
        w -- set of weights being updated
        error -- the error by which the weights need to be updated
        layer -- can be the output-to-hidden layer OR the hidden-to-input layer
        alpha -- momentum
    */

    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if(r < Rows && c < Cols){
        int index = r*Cols + c;
        d_delta_weights[index] = eta * d_dotP[index]/BATCH_SIZE + alpha * d_delta_weights[index];
        d_w[index] += d_delta_weights[index];
    }

}

__global__ void outputError(float* d_error, unsigned short* t, float* d_out_layer, int Rows, int Cols){
    /*
        d_error   -- delta_k
        targets    -- one hot encode 1D array containing 0.9 for target label
        d_out_layer -- the squashed activations for the output layer
        Rows      -- should be 1 as they are all 1D arrays
        Cols      -- should be the number of ouput nodes 
    */
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if(r < Rows && c < Cols){ 
        int index = r*Cols + c;
        if(t[r] == c)
            d_error[index] = d_out_layer[index] * (1 - d_out_layer[index]) * (1 - d_out_layer[index]);
        else 
            d_error[index] = d_out_layer[index] * (1 - d_out_layer[index]) * (0 - d_out_layer[index]);
    }
    
}


__global__ void hiddenError(float* d_error, float* d_dotP, float* d_hidden_layer, int Rows, int Cols){
    /*
    d_error         -- delta_j    
    d_dotP          -- the output error dot output weights
    d_hidden_layer  -- the hidden activations
    Rows            -- should be 1 as they are all 1D arrays
    Cols            -- should be the number of ouput nodes 
    */
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x; 
    
    if(r < Rows && c < Cols){
        int index = r*Cols + c;
        d_error[index] = d_hidden_layer[index] * (1 - d_hidden_layer[index]) * (d_dotP[index]);
    }

}




void error_function(unsigned short * t, float* z, float* h, float* output_weights, float* delta_k, float* delta_j){
    
    //--------------  DEEIVCE Prep ----------------------
  float *d_z, *d_h, *d_k, *d_j;
  unsigned short *d_t;
  
  cudaError_t cudaStatus;
  cudaStatus = cudaMalloc((void**)&d_t, BATCH_SIZE * sizeof(unsigned short));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_t, t, BATCH_SIZE * sizeof(unsigned short), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);
  
  cudaStatus = cudaMalloc((void**)&d_z, BATCH_SIZE * NUM_LABELS * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_z, z, BATCH_SIZE * NUM_LABELS * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);
  
  cudaStatus = cudaMalloc((void**)&d_h, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_h, h, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);
  
  cudaStatus = cudaMalloc((void**)&d_k, BATCH_SIZE * NUM_LABELS * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_k, delta_k, BATCH_SIZE * NUM_LABELS * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);
  
  cudaStatus = cudaMalloc((void**)&d_j, BATCH_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_j, delta_j, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);
    
  
  // call kernel for weight update for each thread to update a weight
  int blockX = ceil((float)BATCH_SIZE/BLOCK_WIDTH);
  int blockY = ceil((float)NUM_LABELS/BLOCK_WIDTH);
  int threadX = BLOCK_WIDTH;
  int threadY = BLOCK_WIDTH;
  dim3 dimGrid(blockX,   blockY,  1);
  dim3 dimBlock(threadX, threadY, 1);
  //--------------  END: DEEIVCE Prep  ----------------------

  outputError<<<dimGrid, dimBlock>>>(d_k, d_t, d_z, BATCH_SIZE, NUM_LABELS); 
  
  // copy back to the host because we need delta K for the dotP
  cudaStatus = cudaMemcpy(delta_k, d_k, BATCH_SIZE * NUM_LABELS * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError(cudaStatus);

  float* errorTransposed;
  errorTransposed = (float*)malloc(BATCH_SIZE*NUM_LABELS*sizeof(float));
  transpose(delta_k, errorTransposed, BATCH_SIZE, NUM_LABELS);
  
  float *dotP, *d_dotP; 
  dotP = (float*)malloc(HIDDEN_SIZE*BATCH_SIZE*sizeof(float));
  dotProduct((float*)output_weights, errorTransposed, dotP, HIDDEN_SIZE, NUM_LABELS, NUM_LABELS, BATCH_SIZE);
  
  // Prep for hidden error
  blockX = ceil((float)HIDDEN_SIZE/BLOCK_WIDTH);
  blockY = ceil((float)BATCH_SIZE/BLOCK_WIDTH);
  threadX = BLOCK_WIDTH;
  threadY = BLOCK_WIDTH;
  dim3 dimGrid2(blockX,   blockY,  1);
  dim3 dimBlock2(threadX, threadY, 1);
  
  // used for the dot product of output error and output weights
  cudaStatus = cudaMalloc((void**)&d_dotP, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_dotP, dotP, HIDDEN_SIZE * BATCH_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  hiddenError<<<dimGrid2, dimBlock2>>>(d_j, d_dotP, d_h, BATCH_SIZE, HIDDEN_SIZE );

  // copy back to the host variables
  cudaStatus = cudaMemcpy(delta_j, d_j, BATCH_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError(cudaStatus);
  
  // deallocate device memory
  cudaFree(d_z);
  cudaFree(d_h);
  cudaFree(d_k);
  cudaFree(d_j);
  cudaFree(d_t);
  cudaFree(d_dotP);
  
  free(errorTransposed);
  free(dotP);
}

void update_weights(float eta, float alpha, float* weights, int wRows, int wCols, float* dotP, float * delta_weights){
/*
    dotP -- layer activations transposed x layer errors
*/

  //--------------  DEEIVCE Prep ----------------------
  float *d_w,  *d_dotP, *d_delta_weights;

  cudaError_t cudaStatus;    
  cudaStatus = cudaMalloc((void**)&d_w, wRows * wCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_w, weights, wRows * wCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  cudaStatus = cudaMalloc((void**)&d_dotP, wRows * wCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy( d_dotP,  dotP, wRows * wCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  cudaStatus = cudaMalloc((void**)&d_delta_weights, wRows * wCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_delta_weights,  delta_weights, wRows * wCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  // call kernel for weight update for each thread to update a weight
  int blockX = ceil((float)wCols / BLOCK_WIDTH);
  int blockY = ceil((float)wRows / BLOCK_WIDTH);
  int threadX = BLOCK_WIDTH;
  int threadY = BLOCK_WIDTH;
  dim3 dimGrid(blockX,   blockY,  1);
  dim3 dimBlock(threadX, threadY, 1);
  //--------------  END: DEEIVCE Prep ----------------------

  updateWeights<<<dimGrid, dimBlock>>>(eta, alpha, d_dotP, wRows, wCols, d_w, d_delta_weights);

  
  // copy back to the host variables
  cudaStatus = cudaMemcpy(weights, d_w,  wRows * wCols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError(cudaStatus);
  
  // copy back to the host variables
  cudaStatus = cudaMemcpy(delta_weights, d_delta_weights,  wRows * wCols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError(cudaStatus);
  
  // deallocate device memory
  cudaFree(d_w);
  cudaFree(d_dotP);
  cudaFree(d_delta_weights);
}
