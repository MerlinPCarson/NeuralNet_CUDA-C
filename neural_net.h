#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "data.h"


#define HIDDEN_SIZE (10)    // number of neurons in hidden layer
#define BATCH_SIZE (2)     // number of examples between weight updates

struct History{
  std::vector<float> loss;
  std::vector<float> valLoss;
};

class NeuralNet{

    public:

        NeuralNet();
        NeuralNet(float eta);

        History fit(std::vector<Data> &trainSet, std::vector<Data> &valSet, int num_epochs);
        void predict(std::vector<Data> &testData, std::vector<int> &pred, std::vector<int> &target);
        float accuracy(std::vector<int> &pred, std::vector<int> &target);

    private:

        void init_weights();
        void show_weights();
        void error_function(int t, float* z, float* h, float* &delta_k, float* &delta_j);
        void update_weights(float* error, float* layer, bool input);

        // need softmax function

        void make_batch(float batch[][NUM_FEATURES], float * target, std::vector<Data> &dataSet, int * order, int batchNum);

        void forward(float batch[][NUM_FEATURES]);

        float eta;            // learning rate
        float alpha;          // momentum
        float output_error[NUM_LABELS];   // output-to-hidden error 2 Layer NN
        float hidden_error[HIDDEN_SIZE];   // hidden-to-input error  2 Layer NN

        float hidden_weights[NUM_FEATURES][HIDDEN_SIZE];    // hidden layer weights
        float output_weights[HIDDEN_SIZE][NUM_LABELS];      // output layer weights

        float hidden_signal[BATCH_SIZE][HIDDEN_SIZE];
        float hidden_activation[BATCH_SIZE][HIDDEN_SIZE];
        float output_signal[BATCH_SIZE][NUM_LABELS];
        float output_activation[BATCH_SIZE][NUM_LABELS];

};
    

#endif // NEURAL_NET_H
