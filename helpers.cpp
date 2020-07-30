#include <stdio.h>
#include <math.h>
#include "helpers.h"
#include "data.h"
#include <algorithm>
#include <iterator>
#include "neural_net.h"


// verify digits were correctly loaded into dataset
void testDatasets(std::vector<Data> &trainSet, std::vector<Data> &valSet, std::vector<Data> &testData)
{
  printf("\nFirst digit in training set\n");
  print_digit(trainSet[0].value, trainSet[0].label);
  printf("\nLast digit in training set\n");
  print_digit(trainSet[trainSet.size()-1].value, trainSet[trainSet.size()-1].label);
  printf("\nFirst digit in validation set\n");
  print_digit(valSet[0].value, valSet[0].label);
  printf("\nLast digit in validation set\n");
  print_digit(valSet[valSet.size()-1].value, valSet[valSet.size()-1].label);
  printf("\nFirst digit in test set\n");
  print_digit(testData[0].value, testData[0].label);
  printf("\nLast digit in test set\n");
  print_digit(testData[testData.size()-1].value, testData[testData.size()-1].label);
}

void printMatrix(float *X, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int id = j + i * numCols;
            printf("%f ", X[id]);
        }
        printf("\n");
    }
}

void hostBatchPreds(float* output_activations, int * batch_pred, int output_size, int b_size)
{
    for(int i = 0; i < b_size; ++i)
    {
        int counter = 0;
        for(int j = 1; j < output_size; ++j)
        {
            int idx = j + i*output_size;
            if(output_activations[idx] > output_activations[idx-1])
            {
                counter = j;
            }
        }
        batch_pred[i] = counter;
    }
}

// h_T is 1D (batchSize), h_O is 2D (batchSize, numLabels)
// numRows = batch size
float hostMSE(float *h_T, float *h_O, int batchSize, int numLabels)
{
    float batchLoss = 0.0;

    // Update the error table
    for (int i = 0; i < batchSize; i++) {
        int t_idx = h_T[i];
    
        // Sanity check
        if (t_idx >= numLabels) {
            printf("t_idx (%d) >= numLabels (%d)\n", t_idx, numLabels);
            exit(-1);
        }

        // Now go through each of the output values and calculate the MSE
        float err = 0;
        for (int j = 0; j < numLabels; j++) {
            int o_idx = j + i * numLabels;
    
            if (t_idx == j) {
                // If this is the same as the expected output
                float diff = 1 - h_O[o_idx];
                err += diff * diff;
            }
            else {
                float diff = h_O[o_idx];
                err += diff * diff;
            }
        }

        err /= 2;
        batchLoss += err;
    }

    return batchLoss / (float)batchSize;
}

void hostElementMult(float *h_M, float *h_N, float *h_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols){
    
    int num_PRows = num_MRows;
    int num_PCols = num_MCols;

    if(num_MRows != num_NRows || num_MCols != num_NCols) {
        printf("matrix dimensions don't match!");
        exit(-1);
    }

    for (int i = 0; i < num_PRows; i++) {
        for (int j = 0; j < num_PCols; j++) {
            int idx = j + i * num_PCols;
            h_P[idx] = h_M[idx] * h_N[idx];
        }
    }
}

float hostSigmoid(float z)
{
    float y = (float)1 / (1 + exp(-z));
    return y;
}

void hostActivationFuncForward(float *h_Z, float *h_Y, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int idx = j + i * numCols;
            h_Y[idx] = hostSigmoid(h_Z[idx]);
        }
    }
}

void hostActivationFuncBackward(float *h_Z, float *h_dervA, float *h_dervZ, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int idx = j + i * numCols;
            float s = hostSigmoid(h_Z[idx]);
        
            h_dervZ[idx] =  h_dervA[idx] * s * (1  - s) ;
        }
    }
}

void hostTranspose(float *h_M, float *h_N, int num_MRows, int num_MCols)
{
    for (int i = 0; i < num_MRows; i++) {
        for (int j = 0; j < num_MCols; j++) {
            int m_idx = j + i * num_MCols;
            int n_idx = i + j * num_MRows;
            
            h_N[n_idx] = h_M[m_idx] ;
        }
    }
}

void hostDotProduct(float *h_M, float *h_N, float *h_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols)
{
    int num_PRows = num_MRows;
    int num_PCols = num_NCols;

    if (num_MCols != num_NRows) {
        printf("num_MCols != num_NRows\n");
        exit(-1);
    }

    for (int row = 0; row < num_PRows; row++) {
        for (int col = 0; col < num_PCols; col++) {
            float pVal = 0.0;
            int p_idx = col + row * num_PCols;

            // Go through the M, N elements
            int i, j;
            for (i = 0, j = 0; i < num_NRows && j < num_MCols; i++, j++) {
                int m_idx = j + row * num_MCols;
                int n_idx = col + i * num_NCols;
                pVal += h_M[m_idx] * h_N[n_idx];
            }
            h_P[p_idx] = pVal;
        }
    }
}

int cudaDeviceProperties(){
  // get number of cude devices
  int nDevices;

  cudaCheckError(cudaGetDeviceCount(&nDevices));

  // if no cuda devices found, return error code
  if (nDevices < 1){
    printf("No CUDA devices found!");
    return 0;
  }

  double bytesInGiB = 1 << 30;

  // print stats for each cuda device found
  for (int i = 0; i < nDevices; ++i){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("\nDevice Number: %d\n", i);
    printf("  Device Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory (GiB): %lf\n", prop.totalGlobalMem/bytesInGiB);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwith (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

  return 1;
}


