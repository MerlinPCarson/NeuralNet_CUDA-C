#ifndef DATA_H
#define DATA_H

#include <string>
#include <vector>

// setup paths for OS
#if defined(__linux__)
  #define TRAIN_DATA "data/mnist_train.csv"   // training data file
  #define TEST_DATA "data/mnist_test.csv"     // test data file
#elif defined(_WIN64) || defined(_WIN32)
  #define TRAIN_DATA "data\\mnist_train.csv"   // training data file
  #define TEST_DATA "data\\mnist_test.csv"     // test data file
#endif // OS check

#define HEIGHT (28)                         // number of pixel rows for digit
#define WIDTH  (28)                         // number of pixel columns for digit
#define NUM_FEATURES (784)                  // total number of pixels (height x width)
#define NUM_LABELS (10)                     // number of digits (0..9)
#define NUM_TRAIN (60000)                   // number of training examples
#define NUM_TEST  (10000)                   // number of testing examples
#define MAX_VAL (255)                       // maximum pixel value, for normalization

// data structure for MNIST digits
struct Data{
	int label;
	double value[NUM_FEATURES];
};

void load_csv(Data * data, std::string data_file, int size);
void train_test_split(Data * inData, int dataSize, std::vector<Data> &trainSet, std::vector<Data>  &testSet, float testRatio);
void print_digit(Data &digit);


#endif // DATA_H
