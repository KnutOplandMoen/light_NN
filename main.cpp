#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
#include <vector>

int main() {

    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {10, 10};  // One hidden layer with 3 neurons
    Matrix input_layer(3, 1);  // Input layer with 2 neurons (2 values)
    Matrix output_layer(20, 1); // Output layer with 1 neuron (1 value)
    std::vector <std::string> activation_functions = {"reLu", "reLu",  "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

    input_layer[0][0] = 5; // Example input value 1
    input_layer[1][0] = 5; // Example input value 2
    input_layer[2][0] = 2;

    // Initialize the network with the layers
    network nn(input_layer, hidden_layers_sizes, output_layer, activation_functions);


    // Perform the forward pass
    Matrix output = nn.feed_forward();

    //Visualise the network after the forward pass
    bool show_hidden_layers = true;
    nn.visualise_network(show_hidden_layers);

    return 0;
}