#ifndef NUERAL_NET_H
#define NEURAL_NET_H

#include "data.h"
#include "helpers.h"


#define HIDDEN_SIZE (10)    // number of neurons in hidden layer
#define BATCH_SIZE (10)     // number of examples between weight updates

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
        void loss_function(int, double*, double*, double* &, double* &);
        void update_weights();

        // need softmax function

        void make_batch(float batch[][NUM_FEATURES], float * target, std::vector<Data> &dataSet, int * order, int batchNum);

        float eta;            // learning rate
        float alpha;          // momentum
        float output_error[NUM_LABELS];   // output-to-hidden error 2 Layer NN
        float hidden_error[HIDDEN_SIZE];   // hidden-to-input error  2 Layer NN

        float hidden_weights[NUM_FEATURES+1][HIDDEN_SIZE];    // hidden layer weights
        float output_weights[HIDDEN_SIZE+1][NUM_LABELS];      // output layer weights

};
    

#endif // NEURAL_NET_H
