#include <iostream>
#include <math.h>
#include <random>
#include "neural_net.h"
#include "kernels.h"
#include "helpers.h"


//#define DEBUG
#define USE_GPU

NeuralNet::NeuralNet(){

}

NeuralNet::NeuralNet(float learning_rate): eta(learning_rate){

  // initialize all layer weights
  init_weights();

#ifdef DEBUG
  // display all layer weights
  show_weights();
#endif // DEBUG

}

// Glorot uniform weight initialization
void NeuralNet::init_weights(){

  std::default_random_engine generator;

  // init hidden layer weights
  double limit = sqrt(6.0 / (NUM_FEATURES + HIDDEN_SIZE));
  std::uniform_real_distribution<double> dist_hidden(-limit, limit);
  for(int i = 0; i < NUM_FEATURES; ++i){
    for(int j = 0; j < HIDDEN_SIZE; ++j){
      hidden_weights[i][j] = dist_hidden(generator);
    }
  }

  // init bias weight to 0 for each neuron in hidden layer
//  for(int i = 0; i < HIDDEN_SIZE; ++i){
//      hidden_weights[NUM_FEATURES][i] = 0;
//  }

  // init output layer weights
  limit = sqrt(6.0 / (HIDDEN_SIZE + NUM_LABELS));
  std::uniform_real_distribution<double> dist_output(-limit, limit);

  for(int i = 0; i < HIDDEN_SIZE; ++i){
    for(int j = 0; j < NUM_LABELS; ++j){
      output_weights[i][j] = dist_output(generator);
    }
  }
  
  // init bias weight to 0 for each neuron in output layer
//  for(int i = 0; i < NUM_LABELS; ++i){
//      output_weights[HIDDEN_SIZE][i] = 0.0;
//  }

}

// training function
History NeuralNet::fit(std::vector<Data> &trainSet, std::vector<Data> &valSet, int num_epochs){

  int * order = new int[trainSet.size()];
  int * valOrder = new int[valSet.size()];

  // validation order is shuffled
  for(int i = 0; i < valSet.size(); ++i){
    valOrder[i] = i;
  }

  float batch[BATCH_SIZE][NUM_FEATURES];
  float target[BATCH_SIZE];

  int numTrainBatches = floor(trainSet.size()/BATCH_SIZE);
  int numValBatches = floor(valSet.size()/BATCH_SIZE);

  // losses
  float loss = 0.0;
  float valLoss = 0.0;

  int i, j;

  // data struct for model's losses during training
  History history;

  // main training loop
  for(i = 0; i < num_epochs; ++i){
    std::cout << "\nEpoch: " << i + 1 << '/' << num_epochs << std::endl << std::endl;

    // train
    for(j = 0; j < numTrainBatches; ++j){
  
      // shuffle order of data
      shuffle_idx(order, trainSet.size());
  
      // load batch of data from training set using shuffled order
      make_batch(batch, target, trainSet, order, j);

      // forward pass
      forward(batch);
      
      // add batch loss to epoch loss
      loss += hostMSE(target, (float *)output_activation, BATCH_SIZE, NUM_LABELS);
      
      // backward pass
      // backward(batch, target, output);
      
    }

    // validate
    for(j = 0; j < numValBatches; ++j){

      // load batch of data from validation set 
      make_batch(batch, target, valSet, valOrder, j);

      // forward pass
      forward(batch);

      // add batch loss to epoch validation loss
      valLoss += hostMSE(target, (float *)output_activation, BATCH_SIZE, NUM_LABELS);

    }

    // average losses per epoch
    loss /= numTrainBatches;
    valLoss /= numValBatches;

    std::cout << "loss: " << loss << ", validation loss: " << valLoss << std::endl;

    // add epoch losses to history
    history.loss.push_back(loss);
    history.valLoss.push_back(valLoss);

    // reset losses for next epoch
    loss = 0.0;
    valLoss = 0.0;
    
  }

  return history;
}

// load batch of data from a dataset from a shuffled dataset
void NeuralNet::make_batch(float batch[][NUM_FEATURES], float * target, std::vector<Data> &dataSet, int * order, int batchNum){

  // starting position in training set order for current batch
  int startIdx = batchNum * BATCH_SIZE;
  int endIdx = startIdx + BATCH_SIZE;
  int pos = 0; 

  // copy random order of training elements to contiguous batch
  for(int i = startIdx; i < endIdx; ++i){
    memcpy(batch[pos], &dataSet[order[i]].value[0], (NUM_FEATURES) * sizeof(dataSet[0].value[0]));
    target[pos] = dataSet[order[i]].label;
    ++pos;
  }

}

// forward propagation of a batch of examples
void NeuralNet::forward(float batch[][NUM_FEATURES]){

#ifdef USE_GPU
    dotProduct((float*)batch, (float*)hidden_weights, (float*)hidden_signal, BATCH_SIZE, NUM_FEATURES, NUM_FEATURES, HIDDEN_SIZE);
    activationFuncForward((float*)hidden_signal, (float*)hidden_activation, BATCH_SIZE, HIDDEN_SIZE);
    dotProduct((float*)hidden_activation, (float*)output_weights, (float*)output_signal, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, NUM_LABELS);
    activationFuncForward((float*)output_signal, (float*)output_activation, BATCH_SIZE, NUM_LABELS);
#else
    hostDotProduct((float*)batch, (float*)hidden_weights, (float*)hidden_signal, BATCH_SIZE, NUM_FEATURES, NUM_FEATURES, HIDDEN_SIZE);
    hostActivationFuncForward((float*)hidden_signal, (float*)hidden_activation, BATCH_SIZE, HIDDEN_SIZE);
    hostDotProduct((float*)hidden_activation, (float*)output_weights, (float*)output_signal, BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, NUM_LABELS);
    hostActivationFuncForward((float*)output_signal, (float*)output_activation, BATCH_SIZE, NUM_LABELS);
#endif

#ifdef DEBUG
    printf("Printing output activations:\n");
    printMatrix((float*)output_activation, BATCH_SIZE, NUM_LABELS);
    printf("\n");
#endif // DEBUG

}

void NeuralNet::show_weights(){

  // display weights in hidden layer
  std::cout << "HIDDEN WEIGHTS:" << std::endl;
  for(int i = 0; i < NUM_FEATURES; ++i){
    for(int j = 0; j < HIDDEN_SIZE; ++j){
      std::cout << hidden_weights[i][j] << ' ';
    }
    std::cout << std::endl;
  }

  // display weights in output layer
  std::cout << "OUTPUT WEIGHTS:" << std::endl;
  for(int i = 0; i < HIDDEN_SIZE; ++i){
    for(int j = 0; j < NUM_LABELS; ++j){
      std::cout << output_weights[i][j] << ' ';
    }
    std::cout << std::endl;
  }
}

