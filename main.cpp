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
    std::vector <std::string> activation_functions = {"reLu", "reLu", "softmax"}; //activation and output functions, should match be of dim: (1 + number of hidden layers)

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
    int epochs = 100;
    double learning_rate = 0.1;
    double batch_size = 32;

    // Train the network
    nn.train(x_labels_train, y_labels_train, epochs, learning_rate,  batch_size);

    nn.visualise_network(show_hidden_layers);

    int num_correct = 0;
    int num_total = 0;
    for (int j = 0; j < 1000; ++j) {
    for (int x1 = 0; x1 <= 1; ++x1) {
        for (int x2 = 0; x2 <= 1; ++x2) {
            int correct_prediction = (x1 | x2); // Perform bitwise OR operation
            if (correct_prediction <= 10) {
                Matrix test1_matrix = input_to_matrix({static_cast<double>(x1), static_cast<double>(x2)});
                std::vector<std::vector<Matrix>> feed_forward = nn.feed_forward_batch(test1_matrix);
                std::vector<Matrix> output_layer = feed_forward[0];
                Matrix output_layer_copy = feed_forward[0].back();
                int prediction_n = nn.get_prediction(output_layer_copy);
                if (prediction_n == correct_prediction) {
                    num_correct++;
                }
                std::cout << "Prediction: " << prediction_n << " Correct: " << correct_prediction << std::endl;

                num_total++;
            }
        }
    }
}
    
    std::cout << "Accuracy: " << static_cast<double>(num_correct) / num_total << std::endl;
    
    return 0;
}