#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {10, 10};  // hidden layers and neurons in each layer
    int output_layer_size = 11; // Output layer with 11 neurons
    int input_layer_size = 4; // inout layer with 4 neurons

    std::vector <std::string> activation_functions = {"leakyReLu", "leakyReLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

    // Initialize the network with the layers
    network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions);
    nn.load_state("abcx_model.txt"); // Load the state (weights and biases) from file

    
    // Get the data
    data_struct data = get_data(4, 11, "Data.txt"); //Todo: Why is this taking such a long time?
    std::vector <Matrix> x_labels = data.x_labels;
    std::vector <Matrix> y_labels = data.y_labels;

    data_struct train_test_data = get_test_train_split(x_labels, y_labels, 0.75); //Splitting the data into training and test data
    std::vector <Matrix> x_labels_train = train_test_data.x_labels_train;
    std::vector <Matrix> y_labels_train = train_test_data.y_labels_train;
    std::vector <Matrix> x_labels_test = train_test_data.x_labels_test;
    std::vector <Matrix> y_labels_test = train_test_data.y_labels_test;

    // Set the training parameters
    int epochs = 5;
    double learning_rate = 0.001;
    double batch_size = 32;

    std::vector<Matrix> weights = nn.get_weights();

    // Train the network
    nn.train(x_labels_train, y_labels_train, x_labels_test, y_labels_test, epochs, learning_rate, batch_size, true);
    
    /* 
    Test the network on a mutiple inputs
    Should be commented out later!
    */
    feed_forward_visualise nn_vis(100, 100, 1000, 700, "Feed forward pass"); // Create a window for visualization

    for (double b = 0; b <= 7; ++b) {
        for (double a = 0; a <= 3; ++a) {
            nn_vis.next_frame(); // Show the window
            Matrix input = input_to_matrix({a, b, 0, 1}); // Input to the network
            std::vector<std::vector<Matrix>> prediction = nn.feed_forward_pass(input); // Feed forward pass
            nn_vis.visualize_feed_forward(prediction[0], input); // Visualize the feed forward pass with AnimationWindow
            nn.visualise_network_terminal(input, true);
            usleep(2000000); // Wait for 2 seconds
        }
    }

    nn_vis.wait_for_close(); // Wait for the window to close
     // Visualize the network
    nn.save_state("abcx_model_5_5.txt"); // Save the weights and viases to a file in binary format
    return 0;
}