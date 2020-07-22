#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include "kernels.h"
#include "helpers.h"

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

int testDotProduct()
{
    float *h_M, *h_N, *h_P1, *h_P2;
    bool bPass = true;
    int MAX_NUM_ROWS = 10, MAX_NUM_COLS = 10;

    for (int num_MRows = 1; num_MRows <= MAX_NUM_ROWS; num_MRows++) {
        for (int num_MCols = 1; num_MCols <= MAX_NUM_COLS; num_MCols++) {
                int num_NRows = num_MCols;
                for (int num_NCols = 1; num_NCols <= MAX_NUM_COLS; num_NCols++) {
                        
                        // Allocate memory for host variables
                        h_M = (float *)malloc(num_MRows * num_MCols * sizeof(float));
                        h_N = (float *)malloc(num_NRows * num_NCols * sizeof(float));
                        h_P1 = (float *)malloc(num_MRows * num_NCols * sizeof(float));
                        h_P2 = (float *)malloc(num_MRows * num_NCols * sizeof(float));
                        if (h_M == NULL || h_N == NULL || h_P1 == NULL || h_P2 == NULL) {
                            printf("Host variable memory allocation failure\n");
                            exit(EXIT_FAILURE);
                        }

                        // load arrays with some numbers
                        for (int i = 0; i < num_MRows; i++) {
                            for (int j = 0; j < num_MCols; j++) {
                                int id = j + i * num_MCols;
                                h_M[id] = 1.0;
                            }
                        }
                        
                        for (int i = 0; i < num_NRows; i++) {
                            for (int j = 0; j < num_NCols; j++) {
                                int id = j + i * num_NCols;
                                h_N[id] = 1.0;
                            }
                        }
                        
                        // Execute on CPU
                        hostDotProduct(h_M, h_N, h_P1, num_MRows, num_MCols, num_NRows, num_NCols);
                        
                        // Execute on GPU
                        dotProduct(h_M, h_N, h_P2, num_MRows, num_MCols, num_NRows, num_NCols);
                        
                        // Compare the GPU results with the CPU results
                        for (int i = 0; bPass && i < num_MRows; i++) {
                            for (int j = 0; bPass && j < num_NCols; j++) {
                                int id = j + i * num_NCols;
                                if (h_P1[id] != h_P2[id]) {
                                    bPass = false;
                                }
                            }
                        }

                        if (!bPass) {
                            printf("FAIL: dot product (row/col): [%d, %d] dot [%d, %d]\n",
                                    num_MRows, num_MCols, num_NRows, num_NCols);
                            printf("---dot product on CPU:---\n");
                            printMatrix(h_P1, num_MRows, num_NCols);
                            printf("------\n\n");
                            printf("---dot product on GPU:---\n");
                            printMatrix(h_P2, num_MRows, num_NCols);
                            printf("------\n\n");
                            return -1;
                        }

                        free(h_M);
                        free(h_N);
                        free(h_P1);
                        free(h_P2);

                        printf("PASS: dot product (row/col): [%d, %d] dot [%d, %d]\n",
                                    num_MRows, num_MCols, num_NRows, num_NCols);
                }
            }
        }

    printf("*** All tests PASSED: dot product ***\n");
    return 0;
}

// Only called while testing activation func
int testActivationFunc()
{
    float *h_Z, *h_Y1, *h_Y2;
    bool bPass = true;
    int MAX_NUM_ROWS = 10, MAX_NUM_COLS = 10;

    for (int numRows = 1; numRows <= MAX_NUM_ROWS; numRows++) {
        for (int numCols = 1; numCols <= MAX_NUM_COLS; numCols++) {
                    
            // Allocate memory for host variables
            h_Z = (float *)malloc(numRows * numCols * sizeof(float));
            h_Y1 = (float *)malloc(numRows * numCols * sizeof(float));
            h_Y2 = (float *)malloc(numRows * numCols * sizeof(float));
            
            if (h_Z == NULL || h_Y1 == NULL || h_Y2 == NULL) {
                printf("Host variable memory allocation failure\n");
                exit(EXIT_FAILURE);
            }

            // load arrays with some numbers
            for (int i = 0; i < numRows; i++) {
                for (int j = 0; j < numCols; j++) {
                    int id = j + numCols * i;
                    h_Z[id] = 1;
                }
            }

            // Execute on CPU
            hostActivationFunc(h_Z, h_Y1, numRows, numCols);

            // Execute on GPU
            activationFunc(h_Z, h_Y2, numRows, numCols);

            // Compare the GPU results with the CPU results
            for (int i = 0; bPass && i < numRows; i++) {
                for (int j = 0; bPass && j < numCols; j++) {
                    int id = j + i * numCols;
                    if (h_Y1[id] != h_Y2[id]) {
                        bPass = false;
                    }
                }
            }

            if (!bPass) {
                printf("FAIL: activation func (row/col): [%d, %d]\n", numRows, numCols);
                printf("---activation func on CPU:---\n");
                printMatrix(h_Y1, numRows, numCols);
                printf("------\n\n");
                printf("---activation func on GPU:---\n");
                printMatrix(h_Y2, numRows, numCols);
                printf("------\n\n");
                return -1;
            }

            free(h_Z);
            free(h_Y1);
            free(h_Y2);
            
            printf("PASS: activation func (row/col): [%d, %d]\n", numRows, numCols);

        }
    }
    
    printf("*** All tests PASSED: activation func ***\n");
    return 0;
}

int main(int argc, char * argv[])
{
  // identify cuda devices
  if(!cudaDeviceProperties()){
    return 1;
  }

  testDotProduct();
  testActivationFunc();

  return 0;
}
