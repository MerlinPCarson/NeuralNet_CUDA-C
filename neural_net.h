#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "data.h"


#define HIDDEN_SIZE (300)    // number of neurons in hidden layer
#define BATCH_SIZE (1)     // number of examples between weight updates

struct History{
  std::vector<float> loss;
  std::vector<float> valLoss;
  std::vector<float> testAcc;
  std::vector<unsigned short> bestPreds;
  std::vector<unsigned short> bestTargets;
};

class NeuralNet{

    public:

        NeuralNet();
        NeuralNet(float eta, float alpha);

        History fit(std::vector<Data> &trainSet, std::vector<Data> &valSet, std::vector<Data> &testSet, int num_epochs);
        void predict(std::vector<Data> &testData, std::vector<unsigned short> &pred, std::vector<unsigned short> &target);
        float accuracy(std::vector<unsigned short> &pred, std::vector<unsigned short> &targets);

    private:

        void init_weights();
        void show_weights();
        void calc_errors(unsigned short * target);
        void update_hidden_weights();
        void update_input_weights(float batch[][NUM_FEATURES]);

        void make_batch(float batch[][NUM_FEATURES], unsigned short * target, std::vector<Data> &dataSet, int * order, int batchNum);

        void forward(float batch[][NUM_FEATURES]);
        void backward(float batch[][NUM_FEATURES], unsigned short * t);

        float eta;            // learning rate
        float alpha;          // momentum coefficient

        float output_error[BATCH_SIZE][NUM_LABELS];   // output-to-hidden error 2 Layer NN
        float hidden_error[BATCH_SIZE][HIDDEN_SIZE];   // hidden-to-input error  2 Layer NN

        float hidden_weights[NUM_FEATURES][HIDDEN_SIZE];    // hidden layer weights
        float output_weights[HIDDEN_SIZE][NUM_LABELS];      // output layer weights

        // for momentum
        float delta_hidden_weights[NUM_FEATURES][HIDDEN_SIZE];    
        float delta_output_weights[HIDDEN_SIZE][NUM_LABELS];    

        // forward pass signals and activations
        float hidden_signal[BATCH_SIZE][HIDDEN_SIZE];
        float hidden_activation[BATCH_SIZE][HIDDEN_SIZE];
        float output_signal[BATCH_SIZE][NUM_LABELS];
        float output_activation[BATCH_SIZE][NUM_LABELS];

};
    

#endif // NEURAL_NET_H
