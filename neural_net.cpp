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
      output_weights[HIDDEN_SIZE][i] = 0;
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

