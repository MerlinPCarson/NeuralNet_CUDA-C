#include <iostream>
#include <math.h>
#include <random>
#include <time.h>
#include <chrono>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include "neural_net.h"
#include "kernels.h"
#include "helpers.h"
#include <string.h>

//#define DEBUG 
//#define SHOW_PREDS 
//#define SHOW_BATCH
#define USE_GPU

NeuralNet::NeuralNet(){
  eta = alpha = 0;
}

NeuralNet::NeuralNet(float learning_rate): eta(learning_rate){
  alpha = 0;
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
  //generator.seed(time(NULL));
  generator.seed(42);

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

  // time at start epoch
  auto start = std::chrono::steady_clock::now();

  int * order = new int[trainSet.size()];
  int * valOrder = new int[valSet.size()];

  // validation order is shuffled
  for(int i = 0; i < valSet.size(); ++i){
    valOrder[i] = i;
  }

  float batch[BATCH_SIZE][NUM_FEATURES];
  unsigned short target[BATCH_SIZE];

  int numTrainBatches = floor(trainSet.size()/BATCH_SIZE);
  int numValBatches = floor(valSet.size()/BATCH_SIZE);

  // progress bar vars
  int numBarSegments = 80;
  int batchesPerBarUpdate = numTrainBatches/numBarSegments;
  if(batchesPerBarUpdate == 0){
    batchesPerBarUpdate = 1;
  }

  // losses
  float loss = 0.0;
  float valLoss = 0.0;

  // data struct for model's losses during training
  History history;

  // main training loop
  for(int i = 0; i < num_epochs; ++i){
    std::cout << "\nEpoch: " << i + 1 << '/' << num_epochs << std::endl;

    // time at start epoch
    start = std::chrono::steady_clock::now();

    // train
    for(int j = 0; j < numTrainBatches; ++j){
    //  if(j == 2) break;

      //printf("Batch %d\n", j);
      // update status bar
      if((j % batchesPerBarUpdate) == 0){
        std::cout << "#" << std::flush; 
      }

      // shuffle order of data
      shuffle_idx(order, trainSet.size());
  
      // load batch of data from training set using shuffled order
      make_batch(batch, target, trainSet, order, j);

#ifdef SHOW_BATCH 
      for(int k = 0; k < BATCH_SIZE; ++k){
          print_digit(batch[k], target[k]);
      }
#endif // SHOW_BATCH

      // forward pass
      forward(batch);
      
      // add batch loss to epoch loss
      //printf("target: %f\n", target[0]);
#ifdef USE_GPU
//      printf("Printing output activations:\n");
//      printMatrix((float*)output_activation, BATCH_SIZE, NUM_LABELS);
//      printf("\n");
      loss += MSE(target, (float *)output_activation, BATCH_SIZE, NUM_LABELS);
#else // USE_GPU
//      printf("Printing output activations:\n");
//      printMatrix((float*)output_activation, BATCH_SIZE, NUM_LABELS);
//      printf("\n");
      loss += hostMSE(target, (float *)output_activation, BATCH_SIZE, NUM_LABELS);
#endif // USE_GPU
      
      // backward pass
//      printf("Output weights pre-backward\n");
//      printMatrix((float*)output_weights, HIDDEN_SIZE, NUM_LABELS);
      backward(batch, target); // target just has 2 labelsa
//      printf("Output weights post-backward\n");
//      printMatrix((float*)output_weights, HIDDEN_SIZE, NUM_LABELS);
      
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

    // total epoch time 
    std::chrono::duration<double> elapsedSeconds = std::chrono::steady_clock::now() - start;
    std::cout << " [epoch time: " <<  elapsedSeconds.count() << " seconds]\n";

    // show epoch losses
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

void NeuralNet::predict(std::vector<Data> &testData, std::vector<unsigned short> &preds, std::vector<unsigned short> &targets){

  int * testOrder = new int[testData.size()];

  // validation order is shuffled
  for(int i = 0; i < testData.size(); ++i){
    testOrder[i] = i;
  }

  float batch[BATCH_SIZE][NUM_FEATURES];
  unsigned short target[BATCH_SIZE];
  unsigned short batch_pred[BATCH_SIZE];

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

#ifdef SHOW_PREDS 
    printf("Printing output activations:\n");
    printMatrix((float*)output_activation, BATCH_SIZE, NUM_LABELS);
    printf("\n");
#endif // SHOW_PREDS

    // add predictions and targets to output vectors
      for(int j = 0; j < BATCH_SIZE; ++j){
        preds.push_back(batch_pred[j]);
        targets.push_back(target[j]);

#ifdef SHOW_PREDS 
        print_digit(batch[j], target[j]);
        printf("prediction: %d \n", batch_pred[j]);
#endif // SHOW_PREDS 

      }

  }

}

// Calculates accuracy of the passed in set.
float NeuralNet::accuracy(std::vector<unsigned short> &pred, std::vector<unsigned short> &targets)
{
    float acc = 0;
    
    if (pred.size() != targets.size()) {
        std::cout << "Vector sizes do not match (accuracy)" << std::endl;
        exit(-1);
    }

    for (auto it1 = pred.begin(), it2 = targets.begin(); it1 != pred.end() && it2 != targets.end(); it1++, it2++) {
        if (*it1 == *it2) acc++;
    }
    
    return acc / pred.size();
}

// load batch of data from a dataset from a shuffled dataset
void NeuralNet::make_batch(float batch[][NUM_FEATURES], unsigned short * target, std::vector<Data> &dataSet, int * order, int batchNum){

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

#ifdef DEBUG
  show_weights();
#endif // DEBUG

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
//  std::cout << "HIDDEN WEIGHTS:" << std::endl;
//  for(int i = 0; i < NUM_FEATURES; ++i){
//    for(int j = 0; j < HIDDEN_SIZE; ++j){
//      std::cout << hidden_weights[i][j] << ' ';
//    }
//    std::cout << std::endl;
//  }
//
  // display weights in output layer
  std::cout << "OUTPUT WEIGHTS:" << std::endl;
  for(int i = 0; i < HIDDEN_SIZE; ++i){
    for(int j = 0; j < NUM_LABELS; ++j){
      std::cout << output_weights[i][j] << ' ';
    }
    std::cout << std::endl;
  }
}

void NeuralNet::backward(float train_batch[][NUM_FEATURES],  unsigned short * target){
//  for(int i = 0; i<BATCH_SIZE; ++i){
    error(target);
    update_hidden_weights(); 
    update_input_weights(train_batch); 
//  }
}


void NeuralNet::error(unsigned short * target){
  // t       -- target label

  error_function(target, (float*)output_activation, 
                    (float*)hidden_activation, 
                    (float*)output_weights, 
                    (float*)output_error, 
                    (float*)hidden_error
                );
}



void NeuralNet::update_hidden_weights(){
  /*
  * error -- the error to update the weights
  * error -- the output error to update the output-to-hidden weights
  * layer -- the output-to-hidden layer
  */
  
  float *curr_error, *w;
  int weightRows, weightCols, layerRows, layerCols, errorRows, errorCols;


  w = (float*)output_weights;   // 10x10
  curr_error = (float*)output_error;    // 1x10  (OUTPUT LABELS)
  errorRows = BATCH_SIZE;
  errorCols = NUM_LABELS;      // 10


  weightRows = HIDDEN_SIZE;     // 10
  weightCols = NUM_LABELS;      // 10
  
  layerRows = BATCH_SIZE;       // output activation Rows
  layerCols = HIDDEN_SIZE;       //
  
//  printf("Output Error\n");
//  printMatrix((float*)output_error, BATCH_SIZE, NUM_LABELS);
//  printf("\n");

//  float* errorTransposed;
//  errorTransposed = (float*)malloc(errorRows * errorCols * sizeof(float));
//  hostTranspose(curr_error, errorTransposed, errorRows, errorCols);
//
//  printf("Output Error Transposed\n");
//  printMatrix(errorTransposed, NUM_LABELS, BATCH_SIZE);
//  printf("\n");
//
//  int pRows = errorCols, pCols = layerCols;
//  float* dotP;
//  dotP = (float*)malloc(pRows*pCols*sizeof(float));
//  // d_dotP's dimeninsions will be     layerCols x error_size
//  //         error T          dot   hidden Activations
//  // NUM LABELS x BATCH SIZE   @    BATCH SIZE x HIDDEN_SIZE  
//  //                10 x 1     @     1 x 2   = 10 x 2
//  dotProduct(errorTransposed, (float*)hidden_activation,  dotP, errorCols, errorRows, layerRows, layerCols);
  float* hiddenActsT;
  hiddenActsT = (float*)malloc(HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  hostTranspose((float*)hidden_activation, hiddenActsT, BATCH_SIZE, HIDDEN_SIZE);

//  printf("Hidden activations Transposed\n");
//  printMatrix(hiddenActsT, HIDDEN_SIZE, BATCH_SIZE);
//  printf("\n");

  float* dotP;
  dotP = (float*)malloc(HIDDEN_SIZE*NUM_LABELS*sizeof(float));
  // 
  dotProduct(hiddenActsT, (float*)output_error, dotP, HIDDEN_SIZE, BATCH_SIZE, BATCH_SIZE, NUM_LABELS);

//  printf("delta output weights\n");
//  printMatrix((float*)dotP, HIDDEN_SIZE, NUM_LABELS);
//  printf("\n");

  //update_weights(eta, alpha, w, weightRows, weightCols, dotP, pRows, pCols);
  update_weights(eta, alpha, w, HIDDEN_SIZE, NUM_LABELS, dotP, HIDDEN_SIZE, NUM_LABELS);
  
  //free(errorTransposed);
  free(hiddenActsT);
  free(dotP);
}



void NeuralNet::update_input_weights(float batch[BATCH_SIZE][NUM_FEATURES]){
  /*
  * error -- the hidden error to update the hidden-to-input  weights
  * layer -- the hidden-to-input layer
  */
  
  float *curr_error, *w;
  int weightRows, weightCols, layerRows, layerCols, errorRows, errorCols;
  w = (float*)hidden_weights;        // 784x10
  curr_error = (float*)hidden_error; // 2x10 (HIDDEN SIZE)
  errorRows = BATCH_SIZE;
  errorCols = HIDDEN_SIZE;

  weightRows = NUM_FEATURES;    // 784
  weightCols = HIDDEN_SIZE;     // 10

  layerRows = BATCH_SIZE;       // hidden activation Rows
  layerCols = NUM_FEATURES;      //  input cols

//  float* errorTransposed;
//  errorTransposed = (float*)malloc(errorRows * errorCols * sizeof(float));
//  hostTranspose(curr_error, errorTransposed, errorRows, errorCols);

//  printf("Hidden Error\n");
//  printMatrix((float*)hidden_error, BATCH_SIZE, HIDDEN_SIZE);
//  printf("\n");

  //int pRows = NUM_FEATURES, pCols = HIDDEN_SIZE;
  float* dotP;
  dotP = (float*)malloc(NUM_FEATURES*HIDDEN_SIZE*sizeof(float));
  // d_dotP's dimeninsions will be     layerCols x errorCols
  //         error T          dot   Batch 
  // HIDDEN SIZE x BATCH SIZE   @    BATCH SIZE x 784  
  //                2 x 1     @     1 x 784   = 2 x 784
  //
  float* batchT;
  batchT = (float*)malloc(NUM_FEATURES * BATCH_SIZE * sizeof(float));
  hostTranspose((float*)batch, batchT, NUM_FEATURES, BATCH_SIZE);

//  printf("inputs Transposed\n");
//  printMatrix(batchT, NUM_FEATURES, BATCH_SIZE);
//  printf("\n");

  dotProduct(batchT, (float*)hidden_error, dotP, NUM_FEATURES, BATCH_SIZE, BATCH_SIZE, HIDDEN_SIZE);
  
//  printf("delta hidden weights\n");
//  printMatrix((float*)dotP, NUM_FEATURES, HIDDEN_SIZE);
//  printf("\n");

  //update_weights(eta, alpha, w, weightRows, weightCols, dotP, pRows, pCols );
  update_weights(eta, alpha, w, NUM_FEATURES, HIDDEN_SIZE, dotP, NUM_FEATURES, HIDDEN_SIZE );

  //free(errorTransposed);
  free(batchT);
  free(dotP);
}


