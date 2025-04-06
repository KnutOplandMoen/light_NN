#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>
#include "game.h"
#include "q_network.h"
#include <chrono>

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {128, 64};  // Hidden layers and neurons in each layer
    int output_layer_size = 4; // Output layer with 4 neurons
    int input_layer_size = 225; // Input layer with 16 neurons

    std::vector<std::string> activation_functions = {"reLu", "reLu", ""}; // Activation functions

    // Initialize the network
    q_network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions);

    //nn.load_state("good_snake_128x64.txt"); // Load pre-trained model (if available)
    nn.set_epsilon(0.1); // Set exploration rate
    //nn.set_epsilon_min(0.01); // Set minimum epsilon

    int games = 200;  // Number of training episodes
    int batch_size = 50000;
    int mini_batch_size = 32;
    double learning_rate = 0.001;

    std::map<std::string, int> autosave_name_per_n_games = {{"autosave_snake.txt", 50}}; 
    // Autosave the model every 50 games

    nn.train(games, batch_size, mini_batch_size, learning_rate, autosave_name_per_n_games);
    nn.save_state("trained_q_network.txt"); // Save trained model

    return 0;
}