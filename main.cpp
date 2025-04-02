#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {10, 10};  // hidden layers and neurons in each layer
    int output_layer_size = 11; // Output layer with 11 neurons
    int input_layer_size = 4; // Input layer with 4 neurons

    std::vector<std::string> activation_functions = {"leakyReLu", "leakyReLu", "softmax"}; // Activation and output functions
    std::string model_name = "abcx_copy.txt"; //Model name that we are loading from, if no model, dont pass any name.

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
    
    // Test the network with multiple inputs (for visualization)
    feed_forward_visualise nn_vis(100, 100, 1000, 700, "Feed forward pass");

    for (double b = 0; b <= 7; ++b) { //Loop through some potential outputs
        for (double a = 0; a <= 3; ++a) {
            nn_vis.next_frame();
            Matrix input = input_to_matrix({a, b, 0, 1});
            std::vector<std::vector<Matrix>> prediction = nn.feed_forward_pass(input);
            nn_vis.visualize_feed_forward(prediction[0], input);
            usleep(2000000); // Wait for 2 seconds
        }
    }

    nn_vis.wait_for_close(); // Wait for the window to close
    nn.save_state("file_to_save.txt"); // Save model state to file
    return 0;
}