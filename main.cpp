#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>
#include "q_network.h"
#include "game.h"

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {128, 64};  // hidden layers and neurons in each layer
    int output_layer_size = 4; // Output layer with 11 neurons
    int input_layer_size = 16; // inout layer with 4 neurons

    std::vector <std::string> activation_functions = {"reLu", "reLu", ""}; //activation and output functions, should match be of dim: (1 + number of hidden layers)
    // Initialize the network with the layers
    q_network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions);

    //nn.load_state("128x64weight_1.txt");
    nn.set_epsilon(0.95);
    for (auto& weight : nn.get_weights()) {
        std::cout << weight << std::endl;
    }

    int games = 200;

    int batch_size = 50000;
    int mini_batch_size = 32;
    double learning_rate = 0.001;
    // Train the network
    nn.train(games, batch_size, mini_batch_size, learning_rate);
    nn.save_state("128x64weight_1.txt");
    return 0;
}