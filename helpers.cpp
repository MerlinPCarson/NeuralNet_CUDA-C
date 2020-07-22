#include <stdio.h>
#include "helpers.h"
#include <math.h>

float hostSigmoid(float z)
{
    float y = (float)1 / (1 + exp(-z));
    return y;
}

void hostActivationFunc(float *h_Z, float *h_Y, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int idx = j + i * numCols;
            h_Y[idx] = hostSigmoid(h_Z[idx]);
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


