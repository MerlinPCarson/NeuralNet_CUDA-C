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
  train_test_split(trainData, trainSet, valSet, (float)testSize/trainSize);

  // show data set info
  std::cout << "\nSize of training set: " << trainSet.size() << std::endl;
  std::cout << "Size of validation set: " << valSet.size() << std::endl;
  std::cout << "Size of test set: " << testData.size() << std::endl;

#ifdef TESTING
  // print first and last digit from each dataset
  testDatasets(trainSet, valSet, testData);
#endif // TESTING

  // instantiate neural network with learning rate
  NeuralNet model = NeuralNet(0.01);

  // time program started 
  auto start = std::chrono::steady_clock::now();

  // main training loop
  int numEpochs = 2;
  std::cout << "\nBeginning Training\n";
  History history = model.fit(trainSet, valSet, numEpochs);

  // test model
  std::vector<int> preds;
  std::vector<int> targets;
  model.predict(testData, preds, targets);

  // total time to run program 
  std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() - start;
  std::cout << "\nExecution time: " <<  elapsedSeconds.count() << " seconds\n";

  saveHistory(history, "model_history.csv");

  return 0;
}

