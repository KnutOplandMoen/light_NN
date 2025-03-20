#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
#include <vector>
#include "models/SaveToTxt.h"

int main() {

    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {10, 10};  // hidden layers and neurons in each layer
    Matrix output_layer(11, 1); // Output layer with 11 neurons
    Matrix input_layer = input_to_matrix({0, 0, 0, 0}); // inout layer with 4 neurons

    save_to_txt("test.txt", {input_layer});
    std::vector <std::string> activation_functions = {"leakyReLu", "leakyReLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)


    // Initialize the network with the layers
    network nn(input_layer, hidden_layers_sizes, output_layer, activation_functions);
    
    // Get the data
    std::vector <std::vector<Matrix>> data = get_data(4, 11);
    std::vector <Matrix> y_labels = data[1];
    std::vector <Matrix> x_labels = data[0];

    std::vector <std::vector<Matrix>> train_test_data = get_test_train_split(x_labels, y_labels, 0.75); //Splitting the data into training and test data
    std::vector <Matrix> x_labels_train = train_test_data[0];
    std::vector <Matrix> y_labels_train = train_test_data[1];
    std::vector <Matrix> x_labels_test = train_test_data[2];
    std::vector <Matrix> y_labels_test = train_test_data[3];

    // Set the training parameters
    int epochs = 1;
    double learning_rate = 0.1;
    double batch_size = 32;

    // Train the network
    nn.train(x_labels_train, y_labels_train, x_labels_test, y_labels_test, epochs, learning_rate, batch_size);

    nn.visualise_network();
    
    return 0;
}