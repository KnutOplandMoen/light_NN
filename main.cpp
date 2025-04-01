#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>
#include "q_network.h"
#include "game.h"

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {64, 64};  // hidden layers and neurons in each layer
    int output_layer_size = 4; // Output layer with 11 neurons
    int input_layer_size = 12; // inout layer with 4 neurons

    std::vector <std::string> activation_functions = {"reLu", "reLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

    // Initialize the network with the layers
    q_network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions);

    int games = 5000;
    int batch_size = 100;
    int mini_batch_size = 32;
    double learning_rate = 0.01;
    // Train the network
    nn.train(games, batch_size, mini_batch_size, learning_rate);
    nn.save_state("64x64weight.txt");
    return 0;
}