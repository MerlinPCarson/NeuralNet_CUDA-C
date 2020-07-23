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

  void NeuralNet::loss_function(int t, double* o, double* h, double* &delta_k, double* &delta_j){
    /* t       -- target label 
       o       -- output activations
       h       -- hidden actications
       delta_k -- output error 
       delta_j -- hidden error
    */

    // need an element wise multiplication 
    // output_error = output_activations *  (1 - output_activations) * (target - output_activations)

    // TODO hidden_activations and the output_weights can not include the bias
    // hidden_error = hidden_activations * (1 - hidden_activations) * [outut_error DOT output_weights]

  }

  void NeuralNet::update_weights(double* error, double* layer, bool input){
    /*
    * error -- the error to update the weights
    * layer -- can be the output-to-hidden layer OR the hidden-to-input layer
    * input -- determines which error and layer is used to update the weights
    */
    double* w;
    float* error;
    int weightRows, weightCols, layerRows, layerCols, error_size;
    if(input){
      w = this.hidde_weights;       // 785x10
      error = this.hidde_error;     // 1x10 (HIDDEN SIZE)
      error_size = HIDDEN_SIZE;
      weightRows = NUM_FEATURES+1;  // 785
      weightCols = HIDDEN_SIZE;     // 10
      layerRows = 1;                // FIXME
      layerCols = HIDDEN_SIZE;     // TODO double check layer dimensions
    }else{
      w = this.output_weights;      // 11x10
      error = this.output_error;    // 1x10  (OUTPUT LABELS)
      error_size = NUM_LABELS;      // 10
      weightRows = HIDDEN_SIZE+1;   // 11
      weightCols = NUM_LABELS       // 10
      layerRows = 1;                // FIXME
      layerCols = NUM_FEATURES;     // TODO double check layer dimensions
    }

    float* errorTransposed;
    tranpose(error, 1, error_size, errorTransposed);


    //--------------  DEEIVCE Prep ----------------------
    double* d_w;
    float *d_error, *d_layer, *d_dotP;

    
    cudaStatus = cudaMalloc((void**)&d_w, weightRows * weightCols * sizeof(float));
    cudaCheckError(cudaStatus);
    cudaStatus = cudaMemcpy(d_w, w, weightRows * weightCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_layer, layerRows * lahyerCols * sizeof(float));
    cudaCheckError(cudaStatus);
    cudaStatus = cudaMemcpy(d_layer, layer, layerRows * lahyerCols * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);


    cudaStatus = cudaMalloc((void**)&d_error, error_size * sizeof(float));
    cudaCheckError(cudaStatus);
    cudaStatus = cudaMemcpy(d_error, errorTransposed, error_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError(cudaStatus);

    cudaStatus = cudaMalloc((void**)&d_dotP, error_size * layerCols sizeof(float));
    cudaCheckError(cudaStatus);


    // call kernel for weight update for each thread to update a weight
    int blockX = ceil(weightCols/2);
    int blockY = ceil(weightRows/2);
    int threadX = BLOCK_WIDTH;
    int threadY = BLOCK_WIDTH;
    dim3 dimGrid(blockX,   blockY,  1);
    dim3 dimBlock(threadX, threadY, 1);
    //--------------  END: DEEIVCE Prep ----------------------

    
    // d_dotP's dimeninsions will be     layerCols x error_size
    dotProduct(d_layer, d_error, d_dotP, layerRows, layerCols, 1, error_size);
                            
                            // output-hidden    (1x10) hidden activations  DOT  error(1x10)
                            // hidden-input     (1x785) inputs  DOT  error(1x10) 
    updateWeights <<< dimGrid, dimBlock >>>(d_w, eta, d_dotP, alpha, layerCols, error_size );


  }
}

