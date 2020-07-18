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
