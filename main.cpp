#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>
#include "game.h"
#include "q_network.h"

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {128, 64};  
    int output_layer_size = 4; // Output layer with 4 neurons
    int input_layer_size = 16; // Input layer with 16 neurons

    std::vector<std::string> activation_functions = {"reLu", "reLu", ""}; // Activation functions

    // Initialize the network
    q_network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions);

    nn.load_state("good_snake_128x64.txt"); // Load trained model

    int games = 5;  // Number of games to play

    nn.play(games); // Play the game using the trained model

    return 0;
}