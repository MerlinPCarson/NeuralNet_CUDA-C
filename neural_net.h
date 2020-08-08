#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "data.h"


#define HIDDEN_SIZE (30)    // number of neurons in hidden layer
#define BATCH_SIZE (1)     // number of examples between weight updates

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
        float accuracy(std::vector<int> &pred, std::vector<int> &targets);

    private:

        void init_weights();
        void show_weights();
        void error(float t);
        void update_hidden_weights();
        void update_input_weights(float batch[][NUM_FEATURES]);

        // need softmax function

        void make_batch(float batch[][NUM_FEATURES], float * target, std::vector<Data> &dataSet, int * order, int batchNum);

        void forward(float batch[][NUM_FEATURES]);
        void backward(float batch[][NUM_FEATURES], float* t);

        float eta;            // learning rate
        float alpha;          // momentum
        float output_error[BATCH_SIZE][NUM_LABELS];   // output-to-hidden error 2 Layer NN
        float hidden_error[BATCH_SIZE][HIDDEN_SIZE];   // hidden-to-input error  2 Layer NN

        float hidden_weights[NUM_FEATURES][HIDDEN_SIZE];    // hidden layer weights
        float output_weights[HIDDEN_SIZE][NUM_LABELS];      // output layer weights

        float hidden_signal[BATCH_SIZE][HIDDEN_SIZE];
        float hidden_activation[BATCH_SIZE][HIDDEN_SIZE];
        float output_signal[BATCH_SIZE][NUM_LABELS];
        float output_activation[BATCH_SIZE][NUM_LABELS];

};
    

#endif // NEURAL_NET_H
