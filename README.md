# light_NN

# Table of Contents

- [Project Setup](#project-setup)
  - [Option 1 (Using TDT4102 template)](#optinion-1-using-tdt4102-template)
  - [Option 2 (Not using template)](#option-2-not-using-template)
- [Network Methods](#network-methods)
  - [network::network → Constructor](#networknetwork---constructor)
  - [network::train → Training Function](#networktrain---training-function)
  - [network::feed_forward_pass](#networkfeed_forward_pass)
  - [network::load_state and network::save_state](#networkload_state-and-networksave_state)
- [Other Functionality](#other-functionality)
  - [feed_forward_visualise Class](#feed_forward_visualise-class)
- [Setup](#setup)
  - [Training the Neural Network](#training-the-neural-network)
  - [Q-Learning for Training and Playing](#q-learning-for-training-and-playing)
      - [Training the Q-Network](#training-the-q-network)
      - [Playing Using a Saved Model](#playing-using-a-saved-model)

## Project setup:

### Optinion 1 (Using TDT4102 template):
1. Download as zip -> VScode
2. Install TDT4102 extention
3. CTRL + Shift + p
4. TDT4102: Create Project from TDT4102 Template
5. Choose blank project
6. Skip overwrite of main.cpp, but overwrite all other files.

### Option 2 (Not using template):
Use [old_arcithecture](https://github.com/KnutOplandMoen/light_NN/tree/old_arcitecture) branch.

## Network methods:

### network::network -> Constructor
The constructor initializes weights, biases and hidden layers. 
- initialise_biases() -> initializes biases to 0
- initialise_hidden_layers() -> makes an vector with matrixes representing each hidden layer
- initialize_weights() -> Makes weigths matrixes, all using HE-initialization
    - [ ] TODO: Add functionality for different weight initialization methods :shipit:
### network::train -> training function
training the network based on folowing params: 
- std::vector Matrix train_x_labels
- std::vector Matrix train_y_labels
- std::vector Matrix test_x_labels
- std::vector Matrix test_y_labels
- int epochs
- double learning_rate
- int batch_size
- bool animation -> set to false by default. Lets user see training progress as animation.

### network::feed_forward_pass
Does a feed fowrad pass in the network.  
  
**Input** -> Input of class matrix.  
**returns** -> {all activated layer, all layers without activation function applied}   
  
**get output layer** -> To get prediction use feed_forward_pass(input)[0].back()

### network::load_state and network::save_state
Lets user load and save models (The weights and biases)    
  
- **load_state(std::string file)** -> Updates the network with weights and biases stored in file  
- **save_state(std::string file)** -> saves to models weights and biases to file
  
## Other functionality:
### feed_forward_visualise class
Given activated layers in the network and the input that gave those activated layers, feed_forward_visualise::visualize_feed_forward(activated_layers, input) will give a nice visualisation like this:
![image](https://github.com/user-attachments/assets/94a61829-f464-4bd0-8437-6961b240990e)


Here trying to predict the input [1, 0, 7, 3] corresponding to **x** = 1, **c** = 0, **b** = 7 and **a** = 3 in the function:  
**a**x^2 + **b**x + **c**. Here the network predict output neuron 9, which is correct :star_struck::star_struck: 
# Setup
## Training the Neural Network

To set up and train the network, use the following code:

<details>
  <summary>Click to expand</summary>

```cpp
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
    std::string model_name = "model_to_load.txt"; //Model name that we are loading from, if no model, dont pass any name.

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
```
</details>

## Q-Learning for Training and Playing

This section explains how to train and play using the Q-learning network.

### Training the Q-Network

<details>
  <summary>Click to expand</summary>

```cpp
#include "q_network.h"

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {128, 64};  // Hidden layers and neurons in each layer
    int output_layer_size = 4; // Output layer with 4 neurons
    int input_layer_size = 16; // Input layer with 16 neurons

    std::vector<std::string> activation_functions = {"reLu", "reLu", ""}; // Activation functions

    // Initialize the network
    q_network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions);

    nn.load_state("good_snake_128x64.txt"); // Load pre-trained model (if available)
    nn.set_epsilon(0.1); // Set exploration rate
    nn.set_epsilon_min(0.01); // Set minimum epsilon

    int games = 200;  // Number of training episodes
    int batch_size = 50000;
    int mini_batch_size = 32;
    double learning_rate = 0.001;

    std::map<std::string, int> autosave_name_per_n_games = {{"autosave_snake.txt", 50}}; 
    // Autosave the model every 50 games

    nn.train(games, batch_size, mini_batch_size, learning_rate, autosave_name_per_n_games);
    nn.save_state("trained_q_network.txt"); // Save trained model

    return 0;
}
```
</details>

### Playing using a saved model

<details>
<summary>Click to expand</summary>
#include "q_network.h"

int main() {
    // Define the sizes for input, hidden layers, and output layers
    std::vector<int> hidden_layers_sizes = {128, 64};  
    int output_layer_size = 4; // Output layer with 4 neurons
    int input_layer_size = 16; // Input layer with 16 neurons

    std::vector<std::string> activation_functions = {"reLu", "reLu", ""}; // Activation functions

    // Initialize the network
    q_network nn(input_layer_size, hidden_layers_sizes, output_layer_size, activation_functions);

    nn.load_state("trained_q_network.txt"); // Load trained model

    int games = 200;  // Number of games to play

    nn.play(games); // Play the game using the trained model

    return 0;
}
```
</details> 
