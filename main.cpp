#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>
#include "game.h"
#include "q_network.h"
#include <chrono>

int main() {
    auto start_time = std::chrono::high_resolution_clock::now(); // Start timing

    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {10, 10};  // hidden layers and neurons in each layer
    int output_layer_size = 11; // Output layer with 11 neurons
    int input_layer_size = 4; // Input layer with 4 neurons

    std::vector<std::string> activation_functions = {"leakyReLu", "leakyReLu", "softmax"}; // Activation and output functions
    std::string model_name = "abcx_model.txt"; //Model name that we are loading from, if no model, dont pass any name.

    network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions, model_name);  // Initialize the network with the layers
    
    // Get the data
    data_struct data = get_data(4, 11, "Data.txt"); 
    std::vector<Matrix> x_labels = data.x_labels;
    std::vector<Matrix> y_labels = data.y_labels;

    data_struct train_test_data = get_test_train_split(x_labels, y_labels, 0.75); // Splitting data into training and test sets
    std::vector<Matrix> x_labels_train = train_test_data.x_labels_train;
    std::vector<Matrix> y_labels_train = train_test_data.y_labels_train;
    std::vector<Matrix> x_labels_test = train_test_data.x_labels_test;
    std::vector<Matrix> y_labels_test = train_test_data.y_labels_test;

    // Set the training parameters
    int epochs = 10;
    double learning_rate = 0.01;
    double batch_size = 32;

    // Train the network
    nn.train(x_labels_train, y_labels_train, x_labels_test, y_labels_test, epochs, learning_rate, batch_size, true);

    nn.save_state("file_to_save.txt"); // Save model state to file

    auto end_time = std::chrono::high_resolution_clock::now(); // End timing
    std::chrono::duration<double> elapsed_time = end_time - start_time;
    std::cout << "Execution time: " << elapsed_time.count() << " seconds" << std::endl;

    return 0;
}