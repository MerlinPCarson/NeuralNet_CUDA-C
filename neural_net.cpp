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

  // data struct for model's losses during training
  History history;

  // main training loop
  for(int i = 0; i < num_epochs; ++i){
    std::cout << "\nEpoch: " << i + 1 << '/' << num_epochs << std::endl << std::endl;

    // train
    for(int j = 0; j < numTrainBatches; ++j){
  
      // shuffle order of data
      shuffle_idx(order, trainSet.size());
  
      // load batch of data from training set using shuffled order
      make_batch(batch, target, trainSet, order, j);

      // forward pass
      forward(batch);
      
      // add batch loss to epoch loss
#ifdef USE_GPU
      loss += MSE(target, (float *)output_activation, BATCH_SIZE, NUM_LABELS);
#else // USE_GPU
      loss += hostMSE(target, (float *)output_activation, BATCH_SIZE, NUM_LABELS);
#endif // USE_GPU
      
      // backward pass
      // backward(batch, target, output);
      
    }

    // validate
    for(int j = 0; j < numValBatches; ++j){

      // load batch of data from validation set 
      make_batch(batch, target, valSet, valOrder, j);

      // forward pass
      forward(batch);

      // add batch loss to epoch validation loss
#ifdef USE_GPU
      valLoss += MSE(target, (float *)output_activation, BATCH_SIZE, NUM_LABELS);
#else // USE_GPU
      valLoss += hostMSE(target, (float *)output_activation, BATCH_SIZE, NUM_LABELS);
#endif // USE_GPU

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

void NeuralNet::predict(std::vector<Data> &testData, std::vector<int> &preds, std::vector<int> &targets){

  int * testOrder = new int[testData.size()];

  // validation order is shuffled
  for(int i = 0; i < testData.size(); ++i){
    testOrder[i] = i;
  }

  float batch[BATCH_SIZE][NUM_FEATURES];
  float target[BATCH_SIZE];
  int batch_pred[BATCH_SIZE];

  int numTestBatches = floor(testData.size()/BATCH_SIZE);

  for(int i = 0; i < numTestBatches; ++i){
    // load batch of data from validation set 
    make_batch(batch, target, testData, testOrder, i);


    // forward pass
    forward(batch);

    // get predictions (fills batch_pred with argmax of each row of output_activations)
//#ifdef USE_GPU
//    batchPreds((float*)output_activations, batch_pred);
//#else // USE_GPU
    hostBatchPreds((float*)output_activation, batch_pred);
//#endif // USE_GPU

    // add predictions and targets to output vectors
      for(int j = 0; j < BATCH_SIZE; ++j){
        preds.push_back(batch_pred[j]);
        targets.push_back((int)target[j]);

#ifdef DEBUG
        print_digit(batch[j], target[j]);
        printf("prediction: %d \n", batch_pred[j]);
#endif // DEBUG

      }

  }

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

void NeuralNet::error_function(int t, float* z, float* h, float* &delta_k, float* &delta_j){
  /* t       -- target label 
      z       -- output activations
      h       -- hidden actications
      delta_k -- output error 
      delta_j -- hidden error
  */
  
  //--------------  DEEIVCE Prep ----------------------
  float *d_z, *d_h, *d_k, *d_j, *d_target;
  float *outputUnits; 
  int outRows    = 1,  outCols    = NUM_LABELS;
  int hiddenRows = 1,  hiddenCols = HIDDEN_SIZE;


  // one hot probabilistic  matrix
  float targets[NUM_LABELS];
  for(int i = 0; i < NUM_LABELS; ++i)
    targets[i] = (t==i) ? 0.9 : 0.1;
  
  cudaError_t cudaStatus;
  cudaStatus = cudaMalloc((void**)&d_z, outRows * outCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_z, z, outRows * outCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  cudaStatus = cudaMalloc((void**)&d_h, hiddenRows * hiddenCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_h, h, hiddenRows * hiddenCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);


  cudaStatus = cudaMalloc((void**)&d_k, outRows * outCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_k, delta_k, outRows * outCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  cudaStatus = cudaMalloc((void**)&d_j, hiddenRows * hiddenCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_j, delta_j, hiddenRows * hiddenCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  cudaStatus = cudaMalloc((void**)&d_target,  NUM_LABELS * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_target, targets,  NUM_LABELS * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  cudaStatus = cudaMalloc((void**)&outputUnits, outRows * hiddenCols * sizeof(float));
  cudaCheckError(cudaStatus);


  // call kernel for weight update for each thread to update a weight
  int blockX = ceil(outRows/2);
  int blockY = ceil(outCols/2);
  int threadX = BLOCK_WIDTH;
  int threadY = BLOCK_WIDTH;
  dim3 dimGrid(blockX,   blockY,  1);
  dim3 dimBlock(threadX, threadY, 1);
  //--------------  END: DEEIVCE Prep ----------------------

  
  outputError <<< dimGrid, dimBlock >>>(delta_k, targets, z, outRows, outCols ); 
  int dleta_kRows = outRows;
  int dleta_kCols = outCols;
  
  
  // Prep for hidden error
  blockX = ceil(hiddenCols/2);
  blockY = ceil(hiddenRows/2);
  threadX = BLOCK_WIDTH;
  threadY = BLOCK_WIDTH;
  dim3 dimGrid2(blockX,   blockY,  1);
  dim3 dimBlock2(threadX, threadY, 1);
  
  // output error dot output weights = outputUnits
  //    1x10 @ 10x10  = 1x10
  dotProduct(delta_k, *output_weights, outputUnits, dleta_kRows, dleta_kCols, HIDDEN_SIZE, NUM_LABELS);
  hiddenError <<< dimGrid2, dimBlock2 >>>(delta_j, outputUnits, h, hiddenRows, hiddenCols );

  // copy back to the host variables
  cudaStatus = cudaMemcpy(delta_j, d_j, hiddenRows * hiddenCols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(delta_k, d_k, outRows * outCols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError(cudaStatus);

  // deallocate device memory
  cudaFree(d_z);
  cudaFree(d_h);
  cudaFree(d_k);
  cudaFree(d_j);
  cudaFree(d_target);
  cudaFree(outputUnits);
}

void NeuralNet::update_weights(float* error, float* layer, bool input){
  /*
  * error -- the error to update the weights
  * layer -- can be the output-to-hidden layer OR the hidden-to-input layer
  * input -- determines which error and layer is used to update the weights
  */
  
  float *curr_error, *w;
  int weightRows, weightCols, layerRows, layerCols, error_size;
  if(input){
    w = *this->hidden_weights;      // 784x10
    curr_error = this->hidden_error;    // 1x10 (HIDDEN SIZE)
    error_size = HIDDEN_SIZE;
    weightRows = NUM_FEATURES;    // 784
    weightCols = HIDDEN_SIZE;     // 10
    layerRows = 1;                // FIXME
    layerCols = HIDDEN_SIZE;      // TODO double check layer dimensions
  }else{
    w = *this->output_weights;      // 10x10
    curr_error = this->output_error;    // 1x10  (OUTPUT LABELS)
    error_size = NUM_LABELS;      // 10
    weightRows = HIDDEN_SIZE;     // 10
    weightCols = NUM_LABELS;       // 10
    layerRows = 1;                // FIXME
    layerCols = NUM_FEATURES;     // TODO double check layer dimensions
  }

  float errorTransposed[error_size];
  hostTranspose(curr_error, errorTransposed, 1, error_size);


  //--------------  DEEIVCE Prep ----------------------
  float *d_w, *d_error, *d_layer, *d_dotP;

  cudaError_t cudaStatus;    
  cudaStatus = cudaMalloc((void**)&d_w, weightRows * weightCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_w, w, weightRows * weightCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  cudaStatus = cudaMalloc((void**)&d_layer, layerRows * layerCols * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_layer, layer, layerRows * layerCols * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);


  cudaStatus = cudaMalloc((void**)&d_error, error_size * sizeof(float));
  cudaCheckError(cudaStatus);
  cudaStatus = cudaMemcpy(d_error, errorTransposed, error_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckError(cudaStatus);

  cudaStatus = cudaMalloc((void**)&d_dotP, error_size * layerCols * sizeof(float));
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

  
    // copy back to the host variables
  cudaStatus = cudaMemcpy(w, d_w,  weightRows * weightCols * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckError(cudaStatus);
  
  
    
    // deallocate device memory
  cudaFree(d_w);
  cudaFree(d_error);
  cudaFree(d_layer);
  cudaFree(d_dotP);

}


