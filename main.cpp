#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
#include <vector>

int main() {

    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {10, 10};  // One hidden layer with 3 neurons
    Matrix output_layer(11, 1); // Output layer with 10 neuron (numbers 1-10) for classification
    std::vector <std::string> activation_functions = {"reLu", "reLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

    Matrix input_layer = input_to_matrix({0, 1, 1, 0});
    //should return 1 for the number 2

    // Initialize the network with the layers
    network nn(input_layer, hidden_layers_sizes, output_layer, activation_functions);


    // Perform the forward pass
    Matrix output = nn.feed_forward();

    //Visualise the network after the forward pass
    bool show_hidden_layers = true;
    nn.visualise_network(show_hidden_layers);
    int prediction = nn.get_prediction();
    std::cout << "Prediction: " << prediction << std::endl; 

    // Get the data
    std::vector <std::vector<Matrix>> data = get_data(4, 11);
    std::vector <Matrix> y_labels_train = data[1];
    std::vector <Matrix> x_labels_train = data[0];
    
    // Set the training parameters
    int epochs = 20;
    double learning_rate = 0.1;
    double batch_size = 32;

    // Train the network
    nn.train(x_labels_train, y_labels_train, epochs, learning_rate,  batch_size);

    // Perform the forward pass with same data and check performance
    for (double i = 0; i < 5; ++i) {
    Matrix test1_matrix = input_to_matrix({0, 1, i, 0});
    int correct_prediction = i;
    std::cout << "Correct: " << correct_prediction << std::endl;
    std::vector <std::vector<Matrix>> feed_forward = nn.feed_forward_batch(test1_matrix);
    Matrix output_layer_copy = feed_forward[0].back();
    int prediction_n = nn.get_prediction(output_layer_copy);
    std::cout << "Prediction: " << prediction_n << std::endl;
    std::cout << "Output: \n" << output_layer_copy << std::endl;
    }

    
    return 0;
}