#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>
#include "q_network.h"

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {10, 10};  // hidden layers and neurons in each layer
    int output_layer_size = 4; // Output layer with 11 neurons
    int input_layer_size = 4; // inout layer with 4 neurons

    std::vector <std::string> activation_functions = {"leakyReLu", "leakyReLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

    // Initialize the network with the layers
    q_network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions);

    int epochs = 10;
    double learning_rate = 0.01;
    double batch_size = 32;

    // Train the network
    nn.train(2, 100);

    return 0;
}