#ifndef NUERAL_NET_H
#define NEURAL_NET_H

#include "data.h"

#define HIDDEN_SIZE (10)    // number of neurons in hidden layer
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

    private:

        void init_weights();
        void show_weights();

        void make_batch(float batch[][NUM_FEATURES], float * target, std::vector<Data> &dataSet, int * order, int batchNum);

        void forward(float batch[][NUM_FEATURES]);

        float eta;            // learning rate

        float hidden_weights[NUM_FEATURES][HIDDEN_SIZE];    // hidden layer weights
        float output_weights[HIDDEN_SIZE][NUM_LABELS];      // output layer weights

        float hidden_signal[BATCH_SIZE][HIDDEN_SIZE];
        float hidden_activation[BATCH_SIZE][HIDDEN_SIZE];
        float output_signal[BATCH_SIZE][NUM_LABELS];
        float output_activation[BATCH_SIZE][NUM_LABELS];

};
    

#endif // NEURAL_NET_H
