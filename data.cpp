#include <iostream>
#include <fstream>
#include "data.h"


// loads MNIST data from csv file 
void load_csv(std::vector<Data> &data, std::string data_file, int size){

    // open the data file
  std::ifstream csv_file(data_file);

    if (!csv_file.is_open()){
        std::cout << "Error opening " << data_file << std::endl;

        exit(1);
    }

    std::cout << "Loading " << size << " examples from " << data_file << std::endl;
    for (int i = 0; i < size; ++i){
        // first element of row is the label
        csv_file >> data[i].label;
        csv_file.ignore(1);

        // load in the values for the data item
        for (int j = 0; j < NUM_FEATURES; ++j){
            csv_file >> data[i].value[j];
            data[i].value[j] /= MAX_VAL;  // normalize values
            csv_file.ignore(1);	          // ignore comma or end of line char
        }

    // bias term
    data[i].value[NUM_FEATURES] = 1;
    }

    csv_file.close();
}

void shuffle_idx(int * order, int size){

  int temp;
  int swap;

  // init data order
    for (int i = 0; i < size; ++i){
        order[i] = i;
    }

  // randomize order
    for (int i = 0; i < size; ++i){
    temp = order[i];
    swap = rand() % size;
    order[i] = order[swap];
    order[swap] = temp;
  }

}

void train_test_split(std::vector<Data> &dataSet, std::vector<Data> &trainSet, std::vector<Data>  &testSet, float testRatio){

  int dataSize = dataSet.size();
  int testSize = floor(((float)dataSize * testRatio));

  int * order = new int[dataSize];

  shuffle_idx(order, dataSize);

  // put randomly selected examples into training set
    for (int i = 0; i < dataSize-testSize; ++i){
    trainSet.push_back(dataSet[order[i]]);
  }

  // put randomly selected examples into validation set
    for (int i = dataSize-testSize; i < dataSize; ++i){
    testSet.push_back(dataSet[order[i]]);
  }

  delete order;
}

void print_digit(float * digit, int label){

    int count = 0;
    for (int i = 0; i < HEIGHT; ++i){
        for (int j = 0; j < WIDTH; ++j){
            if (digit[count++] == 0.0){
                std::cout << " ";
      }
            else if (digit[count] < 0.5){
                std::cout << "/";
            }
            else{
                std::cout << "#";
            }
        }
        std::cout << std::endl;
    }

  std::cout << "Digit: " << label << std::endl;

    return;
}
