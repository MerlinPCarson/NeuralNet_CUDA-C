#include <iostream>
#include <math.h>
#include <random>
#include "neural_net.h"


NeuralNet::NeuralNet(){

}

NeuralNet::NeuralNet(float learning_rate): eta(learning_rate){

  // initialize all layer weights
  init_weights();

  // display all layer weights
  show_weights();

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
  for(int i = 0; i < HIDDEN_SIZE; ++i){
      hidden_weights[NUM_FEATURES][i] = 0;
  }

  // init output layer weights
  limit = sqrt(6.0 / (HIDDEN_SIZE + NUM_LABELS));
  std::uniform_real_distribution<double> dist_output(-limit, limit);

  for(int i = 0; i < HIDDEN_SIZE; ++i){
    for(int j = 0; j < NUM_LABELS; ++j){
      output_weights[i][j] = dist_output(generator);
    }
  }
  
  // init bias weight to 0 for each neuron in output layer
  for(int i = 0; i < NUM_LABELS; ++i){
      output_weights[HIDDEN_SIZE][i] = 0.0;
  }

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

  // model batch outputs
  float output[BATCH_SIZE][NUM_LABELS];

  // losses
  float loss = 0.0;
  float valLoss = 0.0;

  int i, j;

  History history;

  // main training loop
  for(i = 0; i < num_epochs; ++i){
    std::cout << "\nEpoch: " << i << std::endl << std::endl;

    // train
    for(j = 0; j < numTrainBatches; ++j){
  
      // shuffle order of data
      shuffle_idx(order, trainSet.size());
  
      // load batch of data from training set using shuffled order
      make_batch(batch, target, trainSet, order, j);
 
      // forward pass
      // forward(batch, output);
      
      // calculated errors
      // loss += error(target, output)/BATCH_SIZE;
      
      // backward pass
      // backward(batch, target, output);
      
    }

    // validate
    for(j = 0; j < numValBatches; ++j){

      // load batch of data from validation set 
      make_batch(batch, target, valSet, valOrder, j);

      // forward pass
      // forward(batch, output);

      // calculated errors
      // valLoss += error(target, output)/BATCH_SIZE;

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
    memcpy(batch[pos], &dataSet[order[i]].value[0], NUM_FEATURES * sizeof(dataSet[0].value[0]));
    target[pos] = dataSet[order[i]].label;
    ++pos;
  }

}

void NeuralNet::show_weights(){

  // display weights in hidden layer
  std::cout << "HIDDEN WEIGHTS:" << std::endl;
  for(int i = 0; i < NUM_FEATURES+1; ++i){
    for(int j = 0; j < HIDDEN_SIZE; ++j){
      std::cout << hidden_weights[i][j] << ' ';
    }
    std::cout << std::endl;
  }

  // display weights in output layer
  std::cout << "OUTPUT WEIGHTS:" << std::endl;
  for(int i = 0; i < HIDDEN_SIZE+1; ++i){
    for(int j = 0; j < NUM_LABELS; ++j){
      std::cout << output_weights[i][j] << ' ';
    }
    std::cout << std::endl;
  }
}

