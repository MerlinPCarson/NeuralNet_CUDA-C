#include <iostream>
#include <fstream>
#include "data.h"


// loads MNIST data from csv file 
void load_csv(Data * data, std::string data_file, int size){

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

		// allocate memory for the data item's values
		data[i].value = new double[NUM_FEATURES];

		// load in the values for the data item
		for (int j = 0; j < NUM_FEATURES; ++j){
			csv_file >> data[i].value[j];
			data[i].value[j] /= MAX_VAL;
			csv_file.ignore(1);	// ignore comma or end of line char
		}
	}

	csv_file.close();
}

void train_test_split(Data * inData, int dataSize, std::vector<Data> &trainSet, std::vector<Data>  &testSet, float testRatio){

  int temp;
  int swap;

  int testSize = floor(((float)dataSize * testRatio));

  int order[dataSize];

  // init data order
	for (int i = 0; i < dataSize; ++i){
		order[i] = i;
	}

  // randomize order
	for (int i = 0; i < dataSize; ++i){
    temp = order[i];
    swap = rand() % dataSize;
    order[i] = order[swap];
    order[swap] = temp;
  }

  // put randomly selected examples into training set
	for (int i = 0; i < dataSize-testSize; ++i){
    trainSet.push_back(inData[order[i]]);
  }

  // put randomly selected examples into validation set
	for (int i = dataSize-testSize; i < dataSize; ++i){
    testSet.push_back(inData[order[i]]);
  }

}

void print_digit(Data &digit){

	int count = 0;
	for (int i = 0; i < HEIGHT; ++i){
		for (int j = 0; j < WIDTH; ++j){


			if (digit.value[count++] == 0.0){

				std::cout << " ";
		}
			else if (digit.value[count] < 0.5){
				std::cout << "/";
			}
			else{
				std::cout << "#";
			}

		}
		std::cout << std::endl;
	}

  std::cout << "Digit: " << digit.label << std::endl;

	return;
}
