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
  std::vector<Data> trainData(trainSize);
  std::vector<Data> testData(testSize);

    // load training and test data from csv file
    load_csv(trainData, TRAIN_DATA, trainSize);
    load_csv(testData, TEST_DATA, testSize);

  // split training data into training and validation sets
  std::vector<Data>  trainSet;
  std::vector<Data>  valSet;
  //train_test_split(trainData, trainSet, valSet, (float)testSize/trainSize);

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

