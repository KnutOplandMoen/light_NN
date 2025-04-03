#include "std_lib_facilities.h"
#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <unistd.h>
#include "game.h"
#include "q_network.h"

int main() {
    // Define the network architecture (must match the training configuration)
    int input_layer_size = 4;           // Input layer with 4 neurons
    std::vector<int> hidden_layers_sizes = {10, 10};  // Hidden layers
    int output_layer_size = 11;         // Output layer with 11 neurons
    std::vector<std::string> activation_functions = {"leakyReLu", "leakyReLu", "softmax"};

    // Load the trained model by providing the filename
    std::string model_name = "abcx_model.txt";
    network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions, model_name);

    // Prepare a new input for prediction
    // For example, we use an input vector: {value1, value2, value3, value4}
    Matrix input = input_to_matrix({2.5, 3.0, 0.5, 1.0});

    // Perform a feed-forward pass to get the prediction
    std::vector<std::vector<Matrix>> prediction = nn.feed_forward_pass(input);

    //Visualising our feed forward pass
    feed_forward_visualise window(200, 200, 1000, 500, "feed_forward pass"); 
    std::vector<std::string> x_labels_names = {"a", "b", "c", "x"}; //declare our x labels
    std::vector<std::string> y_labels_names ={"0","1", "2","3","4","5","6","7","8","9","10"}; //declare our y_labels
    window.visualize_feed_forward(prediction[0], input, x_labels_names, y_labels_names); //visualise 

    // Display the prediction result (Index with highest value)
    std::cout << "Prediction: " << prediction[0].back().getMaxRow() << std::endl;
    window.wait_for_close();
    return 0;
}