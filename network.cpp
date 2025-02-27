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
std::vector <Matrix> network::initialise_weights() {

    Matrix matrix = Matrix(hidden_layers[0].getRows(), input_layer.getRows());
    matrix.setRandomValues(-0.05, 0.05);
    weights.push_back(matrix); 

    for (int i = 0; i < hidden_layers.size(); ++i) {
        Matrix matrix = Matrix(hidden_layers[i+1].getRows(), hidden_layers[i].getRows());
        matrix.setRandomValues(-0.05, 0.05);
        weights.push_back(matrix);

    }

    Matrix matrix = Matrix(output_layer.getRows(), hidden_layers.back().getRows());
    matrix.setRandomValues(-0.05, 0.05);
    weights.push_back(matrix);
}

/**
 *Initialise the hidden layers:
 *Making Nx1 size vectors depending on inputs given from user in hidden_layers_sizes param
 */
std::vector <Matrix> network::initialise_hidden_layers() {
    for (int i = 0; i < hidden_layers_sizes.size(); ++i) {
        hidden_layers.push_back(Matrix(hidden_layers_sizes[i], 1)); //making empty matrixes for the layers in network
    }
}

/**
 * Going forward in the network, computing the node values using matrix multiplication with the weigths
 * At last the output layer is computed
 */
Matrix network::feed_forward() {
    hidden_layers[0] = weights[0] * input_layer; //Computing first layer values
    for (int i = 1; i < hidden_layers.size(); ++i) {
        hidden_layers[i] = weights[i] * hidden_layers[i-1]; //To do: Add bias and activation function
    }
    return weights.back() * hidden_layers.back(); //To do: Add a output function option here on the output layer: for instance softmax
}
