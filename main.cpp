#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <random>
#include <chrono>

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
    float *dev_m, *dev_n, *dev_p;
    float *h_m, *h_n, *h_p;
    int ROWS = 5, COLS = 3;
    cudaError_t err;

    // Allocate memory for host variables
    h_m = (float *)malloc(ROWS * COLS * sizeof(float));
    h_n = (float *)malloc(ROWS * COLS * sizeof(float));
    h_p = (float *)malloc(ROWS * ROWS * sizeof(float));
    if (h_m == NULL || h_n == NULL || h_p == NULL) {
        printf("Host variable memory allocation failure\n");
        exit(EXIT_FAILURE);
    }

    // Allocate memory for device variables
    err = cudaMalloc((void**)&dev_m, ROWS * COLS * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&dev_n, ROWS * COLS * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void**)&dev_p, ROWS * ROWS * sizeof(float));
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }

    // load arrays with some numbers
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            int id = j + COLS * i;
            h_m[id] = 1;
        }
    }

    for (int i = 0; i < COLS; i++) {
        for (int j = 0; j < ROWS; j++) {
            int id = j + ROWS * i;
            h_n[id] = 1;
        }
    }
    memset(h_p, 0, ROWS * ROWS * sizeof(float));

    // Execute on CPU
    hostDotProduct(h_m, h_n, h_p, ROWS, COLS, COLS, ROWS);

    // print out the result
    printf("(TEST) dot product on CPU:\n");
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < ROWS; j++) {
            int id = j + i * ROWS;
            printf("%f ", h_p[id]);
        }
        printf("\n");
    }

    // Execute on GPU

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
	Data * trainData = new Data[trainSize];
	Data * testData = new Data[testSize];

	// load training and test data from csv file
	load_csv(trainData, TRAIN_DATA, trainSize);
	load_csv(testData, TEST_DATA, testSize);

#ifdef TESTING
  // print random digit from each dataset
  print_digit(trainData[rand()%trainSize]);
  print_digit(testData[rand()%testSize]);
#endif // TESTING

  delete trainData;
  delete testData;
 
  // total time to run program 
  std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() - start;
  std::cout << "\nExecution time: " <<  elapsedSeconds.count() << " seconds\n";

  return 0;
}
