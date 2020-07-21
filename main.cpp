#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <chrono>
#include <vector>

#include "kernels.h"
#include "neural_net.h"
#include "data.h"
#include "helpers.h"

// If TESTING macro is uncommented, load a subset of dataset 
#define TESTING
#ifndef TESTING
  #define TRAIN_SIZE (NUM_TRAIN)
  #define TEST_SIZE (NUM_TEST)
#else
  #define TRAIN_SIZE (100)
  #define TEST_SIZE (10)
#endif // TESTING

// Only called while testing dot product
int testDotProduct()
{
    float *d_M, *d_N, *d_P;
    float *h_M, *h_N, *h_P;
    int ROWS = 10, COLS = 10;
    cudaError_t cudaStatus;

    // Allocate memory for host variables
    h_M = (float *)malloc(ROWS * COLS * sizeof(float));
    h_N = (float *)malloc(ROWS * COLS * sizeof(float));
    h_P = (float *)malloc(ROWS * ROWS * sizeof(float));
    if (h_M == NULL || h_N == NULL || h_P == NULL) {
        printf("Host variable memory allocation failure\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for device variables
    cudaStatus = cudaMalloc((void**)&d_M, ROWS * COLS * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_N, ROWS * COLS * sizeof(float));
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_P, ROWS * ROWS * sizeof(float));
    cudaCheckError(cudaStatus);

    // load arrays with some numbers
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int id = j + COLS * i;
            h_M[id] = 1;
        }
    }

    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++) {
            int id = j + ROWS * i;
            h_N[id] = 1;
        }
    }
    memset(h_P, 0, ROWS * ROWS * sizeof(float));

    // Execute on CPU
    hostDotProduct(h_M, h_N, h_P, ROWS, COLS, COLS, ROWS);

    // print out the results
    printf("(TEST) dot product on CPU:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < ROWS; j++) {
            int id = j + i * ROWS;
            printf("%f ", h_P[id]);
        }
        printf("\n");
    }

    // Copy data to GPU
    cudaStatus = cudaMemcpy(d_M, h_M, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_N, h_M, ROWS * COLS * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMemcpy(d_P, h_M, ROWS * ROWS * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    // Execute on GPU
    dotProduct(d_M, d_N, d_P, ROWS, COLS, COLS, ROWS);

    // Copy back to host
    cudaStatus = cudaMemcpy(h_P, d_P, ROWS * ROWS * sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError(cudaStatus);

    printf("(TEST) dot product on GPU:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < ROWS; j++) {
            int id = j + i * ROWS;
            printf("%f ", h_P[id]);
        }
        printf("\n");
    }

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);

    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}

int main(int argc, char * argv[])
{
  // time program started 
  auto start = std::chrono::steady_clock::now();

  // set random seed
  srand(time(NULL));

  // number of examples to load from each dataset
  int trainSize = TRAIN_SIZE;
  int testSize = TEST_SIZE;

  // identify cuda devices
  if(!cudaDeviceProperties()){
    return 1;
  }

#ifdef TEST_DOT_PRODUCT
  testDotProduct();
#endif

  	// init example data structs
  std::vector<Data> trainData(trainSize);
  std::vector<Data> testData(testSize);

	// load training and test data from csv file
	load_csv(trainData, TRAIN_DATA, trainSize);
	load_csv(testData, TEST_DATA, testSize);

  // split training data into training and validation sets
  std::vector<Data>  trainSet;
  std::vector<Data>  valSet;
  train_test_split(trainData, trainSet, valSet, (float)testSize/trainSize);

  // show data set info
  std::cout << "\nSize of training set: " << trainSet.size() << std::endl;
  std::cout << "Size of validation set: " << valSet.size() << std::endl;
  std::cout << "Size of test set: " << testData.size() << std::endl;

#ifdef TESTING
  // print random digit from each dataset
  print_digit(trainSet[0]);
  print_digit(trainSet[rand()%trainSet.size()]);
  print_digit(trainSet[trainSet.size()-1]);
  print_digit(valSet[0]);
  print_digit(valSet[rand()%valSet.size()]);
  print_digit(valSet[valSet.size()-1]);
  print_digit(testData[0]);
  print_digit(testData[rand()%testSize]);
  print_digit(testData[testSize-1]);
#endif // TESTING

  // instantiate neural network with learning rate
  NeuralNet model = NeuralNet(0.01);

  // total time to run program 
  std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() - start;
  std::cout << "\nExecution time: " <<  elapsedSeconds.count() << " seconds\n";

  return 0;
}
