#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
#include <vector>

int main() {

    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {3};  // One hidden layer with 3 neurons
    Matrix input_layer(2, 1);  // Input layer with 2 neurons (2 values)
    Matrix output_layer(1, 1); // Output layer with 1 neuron (1 value)

    // Initialize the network with the layers
    network nn(input_layer, hidden_layers_sizes, output_layer);

    // Set some example values for the input layer
    input_layer[0][0] = 0.1; // Example input value 1
    input_layer[1][0] = 0.5; // Example input value 2

    // Perform the forward pass
    Matrix output = nn.feed_forward();

    // Print the results
    std::cout << "Input Layer: \n" << input_layer << std::endl;

    // Print the final output from the network
    std::cout << "Output Layer: \n" << output << std::endl;

    return 0;
}