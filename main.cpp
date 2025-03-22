#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {10, 10};  // hidden layers and neurons in each layer
    Matrix output_layer(11, 1); // Output layer with 11 neurons
    Matrix input_layer = input_to_matrix({0, 0, 0, 0}); // inout layer with 4 neurons

    std::vector <std::string> activation_functions = {"reLu", "reLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

    // Initialize the network with the layers
    network nn(input_layer, hidden_layers_sizes, output_layer, activation_functions);
    nn.load_state("abcx_model.txt"); // Load the state (weights and biases) from file

    /* 
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
    int epochs = 0;
    double learning_rate = 0.01;
    double batch_size = 32;

    std::vector<Matrix> weights = nn.get_weights();

    // Train the network
    nn.train(x_labels_train, y_labels_train, x_labels_test, y_labels_test, epochs, learning_rate, batch_size, true);
    */

    // Test the network on a single input
    Matrix input = input_to_matrix({0, 6, 3, 1}); // Input to the network
    std::vector<std::vector<Matrix>> prediction = nn.feed_forward_pass(input); // Feed forward pass
    std::cout << "Prediction: \n" << prediction[0].back() << std::endl; // Print the prediction
    visualize_feed_forward(prediction[0], input); // Visualize the feed forward pass with AnimationWindow

    nn.save_state("abcx_copy.txt"); // Save the weights and viases to a file in binary format
    return 0;
}