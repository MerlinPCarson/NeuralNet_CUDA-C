#ifndef NUERAL_NET_H
#define NEURAL_NET_H

#include "data.h"
#include "helpers.h"


#define HIDDEN_SIZE (10)    // number of neurons in hidden layer

class NeuralNet{

    public:

        NeuralNet();
        NeuralNet(float eta);


    private:

        void init_weights();
        void show_weights();
        void loss_function(int, double*, double*, double* &, double* &);
        void update_weights();

        // need softmax function

        float eta;            // learning rate
        float alpha;          // momentum

        double hidden_weights[NUM_FEATURES+1][HIDDEN_SIZE];    // hidden layer weights
        double output_weights[HIDDEN_SIZE+1][NUM_LABELS];   // output layer weights

};
    

#endif // NEURAL_NET_H
