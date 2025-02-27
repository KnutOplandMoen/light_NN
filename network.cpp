#include "matrix.h"
#include <vector>
#include "network.h"

/**
 * Initialise the weights for the neural network layers.
 * 
 * This function creates and initializes the weight matrices for the neural network.
 * The weights are initialized between the input layer and the first hidden layer,
 * between each pair of consecutive hidden layers, and between the last hidden layer
 * and the output layer.
 */
std::vector <Matrix> network::initialise_weights(Matrix input_layer, std::vector <Matrix> hidden_layers, Matrix output_layer) {
    std::vector <Matrix> weights; 
    weights.push_back(Matrix(hidden_layers[0].getRows(), input_layer.getRows())); 
    for (int i = 0; i < hidden_layers.size(); ++i) {
        weights.push_back(Matrix(hidden_layers[i+1].getRows(), hidden_layers[i].getRows()));
    }
    weights.push_back(Matrix(output_layer.getRows(), hidden_layers.back().getRows()));

    this -> weights = weights;
}

