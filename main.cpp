#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
#include <vector>

int main() {

    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {20, 20, 10, 15};  // One hidden layer with 3 neurons
    Matrix input_layer(4, 1);  // Input layer with 3 neurons corresponfing to ax^2 + bx + c, where a, b, c, x are the inputs
    Matrix output_layer(11, 1); // Output layer with 10 neuron (numbers 1-10) for classification
    std::vector <std::string> activation_functions = {"reLu", "reLu", "reLu", "reLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

    input_layer[0][0] = 1; // Example input value 1
    input_layer[1][0] = 0; // Example input value 2
    input_layer[2][0] = 4; // Example input value 3
    input_layer[3][0] = 1; // Example input value 4

    //should return 1 for the number 2

    // Initialize the network with the layers
    network nn(input_layer, hidden_layers_sizes, output_layer, activation_functions);


    // Perform the forward pass
    Matrix output = nn.feed_forward();

    //Visualise the network after the forward pass
    bool show_hidden_layers = true;
    nn.visualise_network(show_hidden_layers);

    // Get the data
    std::vector <std::vector<Matrix>> data = get_data(4, 11);
    std::vector <Matrix> y_labels_train = data[1];
    std::vector <Matrix> x_labels_train = data[0];
    
    // Set the training parameters
    int epochs = 5;
    double learning_rate = 0.7;
    double batch_size = 10;

    // Train the network
    nn.train(x_labels_train, y_labels_train, epochs, learning_rate,  batch_size);

    // Perform the forward pass with same data and check performance
    nn.feed_forward();
    nn.visualise_network(show_hidden_layers);
    
    return 0;
}