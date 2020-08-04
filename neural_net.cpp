#include <iostream>
#include <math.h>
#include <random>
#include "neural_net.h"
#include "kernels.h"
#include "helpers.h"
#include <string.h>

#define DEBUG
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
  generator.seed(time(NULL));

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
      backward(batch, target); // target just has 2 labelsa
      
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
#ifdef USE_GPU
    batchPreds((float*)output_activation, batch_pred, NUM_LABELS, BATCH_SIZE);
#else // USE_GPU
    hostBatchPreds((float*)output_activation, batch_pred, NUM_LABELS, BATCH_SIZE);
#endif // USE_GPU

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

void NeuralNet::backward(float train_batch[][NUM_FEATURES],  float* target){
// output activations
// hidden activations are part of the class.

  //TODO need to process all the batches
  // need to think about input layer vs. hidden layer

  for(int i = 0; i<BATCH_SIZE; ++i){
    error(target[i]);
    update_hidden_weights(); 
    update_input_weights(train_batch[i]); 
  }
}



void NeuralNet::error(float t){
  /*  t       -- target label 
      z       -- output activations
      h       -- hidden actications
      delta_k -- output error 
      delta_j -- hidden error
  */
  error_function(t, (float*)output_activation, (float*)hidden_activation, (float*)output_weights, output_error, hidden_error);
}



void NeuralNet::update_hidden_weights(){
  /*
  * error -- the error to update the weights
  * layer -- can be the output-to-hidden layer OR the hidden-to-input layer
  * input -- determines which error and layer is used to update the weights
  */
  
  float *curr_error, *w;
  int weightRows, weightCols, layerRows, layerCols, error_size;

  w = (float*)output_weights;   // 10x10
  curr_error = output_error;    // 1x10  (OUTPUT LABELS)
  error_size = NUM_LABELS;      // 10

  weightRows = HIDDEN_SIZE;     // 10
  weightCols = NUM_LABELS;      // 10
  
  layerRows = BATCH_SIZE;       // output activation Rows
  layerCols = NUM_LABELS;       //
  

  float* errorTransposed;
  errorTransposed = (float*)malloc(error_size*sizeof(float));
  hostTranspose(curr_error, errorTransposed, 1, error_size);


  float* dotP;
  dotP = (float*)malloc(layerCols*error_size*sizeof(float));
  // d_dotP's dimeninsions will be     layerCols x error_size
  dotProduct((float*)hidden_activation, errorTransposed, dotP, layerRows, layerCols, 1, error_size);

  update_weights(eta, alpha, w, weightRows, weightCols, dotP, layerCols, error_size);

}



void NeuralNet::update_input_weights(float* batch){
  /*
  * error -- the error to update the weights
  * layer -- can be the output-to-hidden layer OR the hidden-to-input layer
  * input -- determines which error and layer is used to update the weights
  */
  
  float *curr_error, *w;
  int weightRows, weightCols, layerRows, layerCols, error_size;
  w = (float*)hidden_weights;      // 784x10
  curr_error = this->hidden_error;    // 1x10 (HIDDEN SIZE)
  error_size = HIDDEN_SIZE;

  weightRows = NUM_FEATURES;    // 784
  weightCols = HIDDEN_SIZE;     // 10
  layerRows = BATCH_SIZE;       // hidden activation Rows
  layerCols = HIDDEN_SIZE;      //

  float* errorTransposed;
  errorTransposed = (float*)malloc(error_size*sizeof(float));
  hostTranspose(curr_error, errorTransposed, 1, error_size);



  float* dotP;
  dotP = (float*)malloc(layerCols*error_size*sizeof(float));
  // d_dotP's dimeninsions will be     layerCols x error_size
  dotProduct(batch, errorTransposed, dotP, layerRows, layerCols, 1, error_size);

  update_weights(eta, alpha, w, weightRows, weightCols, dotP, layerCols, error_size );

}


