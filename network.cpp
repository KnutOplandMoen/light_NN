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
void network::initialise_weights() {

    Matrix matrix1(hidden_layers[0].getRows(), input_layer.getRows());
    matrix1.setRandomValues(-0.05, 0.05);
    weights.push_back(matrix1); 

    for (int i = 0; i < hidden_layers.size() - 1; ++i) {
        Matrix matrix2(hidden_layers[i+1].getRows(), hidden_layers[i].getRows());
        matrix2.setRandomValues(-0.5, 0.5);
        weights.push_back(matrix2);

    }

    Matrix matrix3(output_layer.getRows(), hidden_layers.back().getRows());
    matrix3.setRandomValues(-0.5, 0.5);
    weights.push_back(matrix3);
}

/**
 *Initialise the hidden layers:
 *Making Nx1 size vectors depending on inputs given from user in hidden_layers_sizes param
 */
void network::initialise_hidden_layers() {
    for (int i = 0; i < hidden_layers_sizes.size(); ++i) {
        hidden_layers.push_back(Matrix(hidden_layers_sizes[i], 1)); //making empty matrixes for the layers in network
    }
}

void network::initialise_biases() {
    for (int i = 0; i < hidden_layers_sizes.size(); i++){
        biases.push_back(Matrix(hidden_layers_sizes[i], 1));
    }
    biases.push_back(Matrix(output_layer_size, 1));
}

/**
 * Going forward in the network, computing the node values using matrix multiplication with the weigths
 * At last the output layer is computed
 */
Matrix network::feed_forward() {
    hidden_layers[0] = (weights[0] * input_layer).applyActivationFunction(activationFuncions[0]); //Computing first layer values
    for (int i = 1; i < hidden_layers.size() ; ++i) {
        hidden_layers[i] = ((weights[i] * hidden_layers[i-1]) + biases[i]).applyActivationFunction(activationFuncions[i]); //To do: Add bias and activation function
    }
    output_layer = ((weights.back() * hidden_layers.back()) + biases.back()).applyActivationFunction(activationFuncions.back());
    return output_layer; //To do: Add a output function option here on the output layer: for instance softmax
}

void network::visualise_network(bool show_hidden) {
    // Print the results
    std::cout << "Input Layer: \n" << input_layer << std::endl;

    if (show_hidden) {
        std::cout << "Hidden layers in neural net with corresponding weights: \n" << std::endl;
        std::cout << "weigths Input -> first hidden:\n" << weights[0] << std::endl;
        for (int i = 0; i < hidden_layers.size(); i++) {
            std::cout << "Layer " << i + 1 << ": \n\n" << hidden_layers[i] << std::endl;
            if (i == hidden_layers.size() - 1) {
                std::cout << "Weights " << i+1 << ". -> Output layer \n" << weights[i+1] << std::endl;
            }
            else {
            std::cout << "Weights " << i+1 << ". -> " << i+2 << ". layer \n" << weights[i+1] << std::endl;
            }
        }
    }
    // Print the final output from the network
    std::cout << "Output Layer: \n" << output_layer << std::endl;
}