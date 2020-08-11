#include <stdio.h>
#include <math.h>
#include <fstream>
#include "helpers.h"
#include "data.h"


// verify digits were correctly loaded into dataset
void testDatasets(std::vector<Data> &trainSet, std::vector<Data> &valSet, std::vector<Data> &testData)
{
  printf("\nFirst digit in training set\n");
  print_digit(trainSet[0].value, trainSet[0].label);
  printf("\nLast digit in training set\n");
  print_digit(trainSet[trainSet.size()-1].value, trainSet[trainSet.size()-1].label);
  printf("\nFirst digit in validation set\n");
  print_digit(valSet[0].value, valSet[0].label);
  printf("\nLast digit in validation set\n");
  print_digit(valSet[valSet.size()-1].value, valSet[valSet.size()-1].label);
  printf("\nFirst digit in test set\n");
  print_digit(testData[0].value, testData[0].label);
  printf("\nLast digit in test set\n");
  print_digit(testData[testData.size()-1].value, testData[testData.size()-1].label);
}

void printConfusionMatrix(std::vector<unsigned short> &pred, std::vector<unsigned short> &target)
{
    int conMatrix[NUM_LABELS][NUM_LABELS];
    
    printf("CONFUSION MATRIX:\n");
    if(pred.size() != target.size())
    {
        printf("Vector sizes do not match\n");
        exit(-1);
    }

    for(int i = 0; i < NUM_LABELS; i++) 
    {
        for(int j = 0; j < NUM_LABELS; j++)
        {
            conMatrix[i][j] = 0;
        }
    }

    for (auto it1 = pred.begin(), it2 = target.begin(); it1 != pred.end() && it2 != target.end(); it1++, it2++) {
        ++conMatrix[*it1][*it2];
    }

    for (int i = 0; i < NUM_LABELS; i++) {
        for (int j = 0; j < NUM_LABELS; j++) {
            printf("%d ", conMatrix[i][j]);
        }
        printf("\n");
    }

}

void printMatrix(float *X, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int id = j + i * numCols;
            printf("%f ", X[id]);
        }
        printf("\n");
    }
}

void hostBatchPreds(float *output_activations, unsigned short *batch_pred, int output_size, int b_size)
{
    for(int i = 0; i < b_size; ++i)
    {
        unsigned short counter = 0;
        float maxValue = output_activations[i*output_size];
        for(unsigned short j = 1; j < output_size; ++j)
        {
            int idx = j + i*output_size;
            if(output_activations[idx] > maxValue)
            {
                counter = j;
                maxValue = output_activations[idx];
            }
        }
        batch_pred[i] = counter;
    }
}

template<class T>
void saveDataToFile(std::ofstream &file, std::vector<T> &data){
    file << data[0];
    for(int i = 1; i < data.size(); ++i){
      file << ',';
      file << data[i];
    }
    file << std::endl;
}

void saveHistory(History history, const char* fileName){

  // verify there is anything to write to file
  if(history.loss.size() < 1 || history.valLoss.size() < 1){
    printf("There is no history to write to file");
    return;
  }

  // open file
  std::ofstream file(fileName);

  // write comma seprated training losses to file with newline at end
  if(history.loss.size() > 0){
    saveDataToFile<float>(file, history.loss);
  }

  // write comma seprated validation losses to file with newline at end
  if(history.valLoss.size() > 0){
    saveDataToFile<float>(file, history.valLoss);
  }

  // write comma seprated test accuracies to file with newline at end
  if(history.testAcc.size() > 0){
    saveDataToFile<float>(file, history.testAcc);
  }

  // write comma seprated predictions on test set for best accuracy 
  if(history.bestPreds.size() > 0){
    saveDataToFile<unsigned short>(file, history.bestPreds);
  }


  // write comma seprated test set target values
  if(history.bestTargets.size() > 0){
    saveDataToFile<unsigned short>(file, history.bestTargets);
  }

  // close file
  file.close();

}

// h_T is 1D (batchSize), h_O is 2D (batchSize, numLabels)
// numRows = batch size
float hostMSE(unsigned short *h_T, float *h_O, int batchSize, int numLabels)
{
    float batchLoss = 0.0;

    // Update the error table
    for (int i = 0; i < batchSize; i++) {
        unsigned short t_idx = h_T[i];
     //   printf("target: %f\n", h_T[i]);
    
        // Sanity check
        if (t_idx >= numLabels) {
            printf("t_idx (%d) >= numLabels (%d)\n", t_idx, numLabels);
            exit(-1);
        }

        // Now go through each of the output values and calculate the MSE
        float err = 0;
        for (int j = 0; j < numLabels; j++) {
            int o_idx = j + i * numLabels;
    
            if (t_idx == j) {
                // If this is the same as the expected output
                float diff = 1 - h_O[o_idx];
                err += diff * diff;
            }
            else {
                float diff = h_O[o_idx];
                err += diff * diff;
            }
        }

    //    printf("err: %f\n", err);
        err /= 2;
        batchLoss += err;
    }

    //printf("batch err: %f\n", batchLoss/batchSize);
    return batchLoss / (float)batchSize;
}

void hostElementMult(float *h_M, float *h_N, float *h_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols){
    
    int num_PRows = num_MRows;
    int num_PCols = num_MCols;

    if(num_MRows != num_NRows || num_MCols != num_NCols) {
        printf("matrix dimensions don't match!");
        exit(-1);
    }

    for (int i = 0; i < num_PRows; i++) {
        for (int j = 0; j < num_PCols; j++) {
            int idx = j + i * num_PCols;
            h_P[idx] = h_M[idx] * h_N[idx];
        }
    }
}

float hostSigmoid(float z)
{
    float y = (float)1 / (1 + exp(-z));
    return y;
}

void hostActivationFuncForward(float *h_Z, float *h_Y, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int idx = j + i * numCols;
            h_Y[idx] = hostSigmoid(h_Z[idx]);
        }
    }
}

void hostActivationFuncBackward(float *h_Z, float *h_dervA, float *h_dervZ, int numRows, int numCols)
{
    for (int i = 0; i < numRows; i++) {
        for (int j = 0; j < numCols; j++) {
            int idx = j + i * numCols;
            float s = hostSigmoid(h_Z[idx]);
        
            h_dervZ[idx] =  h_dervA[idx] * s * (1  - s) ;
        }
    }
}

void hostTranspose(float *h_M, float *h_N, int num_MRows, int num_MCols)
{
    for (int i = 0; i < num_MRows; i++) {
        for (int j = 0; j < num_MCols; j++) {
            int m_idx = j + i * num_MCols;
            int n_idx = i + j * num_MRows;
            
            h_N[n_idx] = h_M[m_idx] ;
        }
    }
}

void hostDotProduct(float *h_M, float *h_N, float *h_P, int num_MRows, int num_MCols, int num_NRows, int num_NCols)
{
    int num_PRows = num_MRows;
    int num_PCols = num_NCols;

    if (num_MCols != num_NRows) {
        printf("num_MCols != num_NRows\n");
        exit(-1);
    }

    for (int row = 0; row < num_PRows; row++) {
        for (int col = 0; col < num_PCols; col++) {
            float pVal = 0.0;
            int p_idx = col + row * num_PCols;

            // Go through the M, N elements
            int i, j;
            for (i = 0, j = 0; i < num_NRows && j < num_MCols; i++, j++) {
                int m_idx = j + row * num_MCols;
                int n_idx = col + i * num_NCols;
     //           printf("%f * %f + ", h_M[m_idx], h_N[n_idx]);
                pVal += h_M[m_idx] * h_N[n_idx];
            }
            h_P[p_idx] = pVal;
    //        printf("\n%f\n\n", pVal);
        }
    }
}

int cudaDeviceProperties(){
  // get number of cude devices
  int nDevices;

  cudaCheckError(cudaGetDeviceCount(&nDevices));

  // if no cuda devices found, return error code
  if (nDevices < 1){
    printf("No CUDA devices found!");
    return 0;
  }

  double bytesInGiB = 1 << 30;

  // print stats for each cuda device found
  for (int i = 0; i < nDevices; ++i){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("\nDevice Number: %d\n", i);
    printf("  Device Name: %s\n", prop.name);
    printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("  Total Global Memory (GiB): %lf\n", prop.totalGlobalMem/bytesInGiB);
    printf("  Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwith (GB/s): %f\n\n",
           2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
  }

  return 1;
}

void host_outputError(float* error, unsigned short* t, float* out_layer, int Rows, int Cols){
    int index;
    for(int r=0; r < Rows; ++r){
        for(int c=0; c < Rows; ++c){
            index = r*Cols + c;
            if(t[r] == c)
                error[index] = out_layer[index] * (1 - out_layer[index]) * (1 - out_layer[index]);
            else 
                error[index] = out_layer[index] * (1 - out_layer[index]) * (0 - out_layer[index]);
        }
    }


}

void host_hiddenError(float* error, float* dotP, float* hidden_layer, int Rows, int Cols){
    int index;
    for(int r=0; r < Rows; ++r){
        for(int c=0; c < Rows; ++c){
            index = r*Cols + c;
            error[index] = hidden_layer[index] * (1 - hidden_layer[index]) * (dotP[index]);
        }
    }

}

void host_error_function(unsigned short * t, float* z, float* h, float* output_weights, float* delta_k, float* delta_j){
  int dkRows = BATCH_SIZE, dkCols = NUM_LABELS;
  int djRows = BATCH_SIZE, djCols = HIDDEN_SIZE;
  host_outputError(delta_k, t, z, dkRows, dkCols);


    
  float* errorTransposed;
  errorTransposed = (float*)malloc(dkCols*dkRows*sizeof(float));
  hostTranspose(delta_k, errorTransposed, dkRows, dkCols);
  
  float *dotP; 
  dotP = (float*)malloc(HIDDEN_SIZE*BATCH_SIZE*sizeof(float));
  hostDotProduct((float*)output_weights, errorTransposed, dotP, HIDDEN_SIZE, NUM_LABELS, dkCols, dkRows);
  
  host_hiddenError(delta_j, dotP, h, djRows, djCols);

  // free host memory
  free(errorTransposed);
  free(dotP);
}

void hostUpdateWeights(float eta, float alpha, float* d_dotP, int Rows, int Cols, float* w, float* delta_weights){
    int index;
    for(int r=0; r < Rows; ++r){
        for(int c=0; c < Rows; ++c){
            index = r*Cols + c;
            delta_weights[index] = eta * dotP[index]/BATCH_SIZE + alpha * delta_weights[index];
            w[index] += delta_weights[index];
        }
    }
}


void host_update_weights(float eta, float alpha, float* weights, int wRows, int wCols, float* dotP, float * delta_weights){
    hostUpdateWeights(eta, alpha, dotP, wRows, wCols, weights, delta_weights);
}
