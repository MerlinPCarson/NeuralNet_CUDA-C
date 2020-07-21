#include "kernels.h"
#include <stdio.h>
#include <math.h>


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

void dotProduct(float* d_M, float* d_N, float* d_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols)
{
    int num_PRows = num_MRows;
    int num_PCols = num_NCols;

    if (num_MCols != num_NRows) {
        printf("(device) num_MCols != num_NRows\n");
        exit(-1);
    }

    dim3 gridDim((int)ceil((float)num_PCols / BLOCK_WIDTH), (int)ceil((float)num_PRows / BLOCK_WIDTH), 1);
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);

    dotProductDevice << <gridDim, blockDim >> > (d_M, d_N, d_P, num_MRows, num_MCols, num_NRows, num_NCols);
}
