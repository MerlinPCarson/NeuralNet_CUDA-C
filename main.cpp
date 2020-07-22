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
    float *h_M, *h_N, *h_P;
    int M_ROWS = 5, M_COLS = 10;
    int N_ROWS = M_COLS, N_COLS = 3;

    // Allocate memory for host variables
    h_M = (float *)malloc(M_ROWS * M_COLS * sizeof(float));
    h_N = (float *)malloc(N_ROWS * N_COLS * sizeof(float));
    h_P = (float *)malloc(M_ROWS * N_COLS * sizeof(float));
    if (h_M == NULL || h_N == NULL || h_P == NULL) {
        printf("Host variable memory allocation failure\n");
        exit(EXIT_FAILURE);
    }

    // load arrays with some numbers
    for (int i = 0; i < M_ROWS; i++) {
        for (int j = 0; j < M_COLS; j++) {
            int id = j + M_COLS * i;
            h_M[id] = 1;
        }
    }

    for (int i = 0; i < N_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            int id = j + N_COLS * i;
            h_N[id] = 1;
        }
    }
    memset(h_P, 0, M_ROWS * N_COLS * sizeof(float));

    // Execute on CPU
    hostDotProduct(h_M, h_N, h_P, M_ROWS, M_COLS, N_ROWS, N_COLS);

    // print out the results
    printf("(TEST) dot product on CPU:\n");
    for (int i = 0; i < M_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            int id = j + i * N_COLS;
            printf("%f ", h_P[id]);
        }
        printf("\n");
    }
	// Reset back to 0
	memset(h_P, 0, M_ROWS * N_COLS * sizeof(float));

    // Execute on GPU
    dotProduct(h_M, h_N, h_P, M_ROWS, M_COLS, N_ROWS, N_COLS);

    printf("(TEST) dot product on GPU:\n");
    for (int i = 0; i < M_ROWS; i++) {
        for (int j = 0; j < N_COLS; j++) {
            int id = j + i * N_COLS;
            printf("%f ", h_P[id]);
        }
        printf("\n");
    }

    free(h_M);
    free(h_N);
    free(h_P);

    return 0;
}

// Only called while testing activation func
int testActivationFunc()
{
    float *h_Z, *h_Y;
    int ROWS = 10, COLS = 5;
    
    // Allocate memory for host variables
    h_Z = (float *)malloc(ROWS * COLS * sizeof(float));
    h_Y = (float *)malloc(ROWS * COLS * sizeof(float));
    if (h_Z == NULL || h_Y == NULL) {
        printf("Host variable memory allocation failure\n");
        exit(EXIT_FAILURE);
    }

    // load arrays with some numbers
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int id = j + COLS * i;
            h_Z[id] = 1;
            h_Y[id] = 0;
        }
    }

    // Execute on CPU
    hostActivationFunc(h_Z, h_Y, ROWS, COLS);

    // print out the results
    printf("(TEST) activation func on CPU:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int id = j + i * COLS;
            printf("%f ", h_Y[id]);
        }
        printf("\n");
    }

    // Execute on GPU
    activationFunc(h_Z, h_Y, ROWS, COLS);

    printf("(TEST) activation func on GPU:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int id = j + i * COLS;
            printf("%f ", h_Y[id]);
        }
        printf("\n");
    }

    free(h_Z);
    free(h_Y);

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
  testActivationFunc();
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
  print_digit(trainSet[0].value, trainSet[0].label);
  print_digit(trainSet[trainSet.size()-1].value, trainSet[trainSet.size()-1].label);
  print_digit(valSet[0].value, valSet[0].label);
  print_digit(valSet[valSet.size()-1].value, valSet[valSet.size()-1].label);
  print_digit(testData[0].value, testData[0].label);
  print_digit(testData[testSize-1].value, testData[testSize-1].label);
#endif // TESTING

  // instantiate neural network with learning rate
  NeuralNet model = NeuralNet(0.01);

  // main training loop
  int numEpochs = 2;
  History history = model.fit(trainSet, valSet, numEpochs);

  // total time to run program 
  std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() - start;
  std::cout << "\nExecution time: " <<  elapsedSeconds.count() << " seconds\n";

  return 0;
}

