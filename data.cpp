#include <iostream>
#include <fstream>
#include "data.h"


// loads MNIST data from csv file 
void load_csv(Data * &data, std::string data_file, int size){

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

void print_digit(Data digit){

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
