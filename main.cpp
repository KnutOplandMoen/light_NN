#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
#include <vector>

int f(int a, int b, int c, int x) {
    return ((a * (x*x)) + (b*x) + c);
}

int main() {

    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {5, 5};  // One hidden layer with 3 neurons
    Matrix output_layer(2, 1); // Output layer with 10 neuron (numbers 1-10) for classification
    std::vector <std::string> activation_functions = {"leakyReLu", "leakyReLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

    Matrix input_layer = input_to_matrix({0, 0});
    //should return 1 for the number 2

    // Initialize the network with the layers
    network nn(input_layer, hidden_layers_sizes, output_layer, activation_functions);


    //Visualise the network after the forward pass
    bool show_hidden_layers = false;
    
    // Get the data
    std::vector <std::vector<Matrix>> data = get_data(2, 2);
    std::vector <Matrix> y_labels_train = data[1];
    std::vector <Matrix> x_labels_train = data[0];

    // Set the training parameters
    int epochs = 20;
    double learning_rate = 0.01;
    double batch_size = 10;

    // Train the network
    nn.train(x_labels_train, y_labels_train, epochs, learning_rate,  batch_size);

    nn.visualise_network(show_hidden_layers);
    
    return 0;
}