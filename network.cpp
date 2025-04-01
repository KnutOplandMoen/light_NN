#include "network.h"
#include <unistd.h>
/**
 * Initialise the weights for the neural network layers.
 * 
 * This function creates and initializes the weight matrices for the neural network.
 * The weights are initialized between the input layer and the first hidden layer,
 * between each pair of consecutive hidden layers, and between the last hidden layer
 * and the output layer.
 */
void network::initialise_weights() { //Initialising the weights for the network //TODO: this doenst need to run if weights are loaded...

    double limit = limit = sqrt(2.0 / input_layer_size);
    Matrix matrix1(hidden_layers[0].getRows(), input_layer_size);
    matrix1.setRandomValues(-limit, limit);
    weights.push_back(matrix1);

    // Apply similar logic for other layers
    for (int i = 0; i < hidden_layers.size() - 1; ++i) {
        int n_inputs = hidden_layers[i].getRows();
        limit = sqrt(2.0 / (hidden_layers_sizes[i] + hidden_layers_sizes[i+1]));
        Matrix matrix2(hidden_layers[i+1].getRows(), n_inputs);
        matrix2.setRandomValues(-limit, limit);
        weights.push_back(matrix2);
    }

    int last_hidden_size = hidden_layers.back().getRows();
    limit = sqrt(2.0 / last_hidden_size);
    Matrix matrix3(output_layer_size, last_hidden_size);
    matrix3.setRandomValues(-limit, limit);
    weights.push_back(matrix3);
}

std::vector<Matrix> network::get_weights() {
    return weights;
}
/**
 *Initialise the hidden layers:
 *Making Nx1 size vectors depending on inputs given from user in hidden_layers_sizes param
 */
void network::initialise_hidden_layers() { 
    for (int i = 0; i < hidden_layers_sizes.size(); ++i) {
        hidden_layers.push_back(Matrix(hidden_layers_sizes[i], 1)); //making empty matrixes for the layers in network
    }
}

void network::initialise_biases() { //TODO: this doenst need to run if biases are loaded from old model...
    for (int i = 0; i < hidden_layers_sizes.size(); i++){
        biases.push_back(Matrix(hidden_layers_sizes[i], 1));
    }
    biases.push_back(Matrix(output_layer_size, 1));
}

std::vector <std::vector<Matrix>> network::feed_forward_pass(const Matrix& x_labels) const{
    std::vector<Matrix> hidden_layers_copy = hidden_layers;
    std::vector<Matrix> activation;
    std::vector<Matrix> weigted_inputs;
    activation.reserve(hidden_layers.size() + 1); //Reserving space for the hidden layers sice we know the size
    weigted_inputs.reserve(hidden_layers.size() + 1);

    hidden_layers_copy[0] = ((weights[0] * x_labels) + biases[0]).applyActivationFunction(activationFuncions[0]); //Computing first layer values
    activation.push_back(hidden_layers_copy[0]);
    weigted_inputs.push_back((weights[0] * x_labels) + biases[0]);

    for (int i = 1; i < hidden_layers.size() ; ++i) {
        Matrix weighted_input = (weights[i] * activation[i - 1]) + biases[i];
        weigted_inputs.push_back(weighted_input);
        activation.push_back(weighted_input.applyActivationFunction(activationFuncions[i]));
    }
    activation.push_back(((weights.back() * activation.back()) + biases.back()).applyActivationFunction(activationFuncions.back()));
    weigted_inputs.push_back((weights.back() * activation[activation.size() - 2]) + biases.back());
    
    return {activation, weigted_inputs}; //To do: Add a output function option here on the output layer: for instance softmax
}

std::vector <Matrix> network::get_errors(Matrix& x_labels, Matrix& y_labels) const{ //Backpropagating through network to get errors for each layer
    //Making copy
    std::vector <std::vector<Matrix>> feed_forward = feed_forward_pass(x_labels);

    std::vector <Matrix> errors;

    Matrix error_prev = feed_forward[0].back() - y_labels;
    errors.push_back(error_prev);

    for (int i = hidden_layers.size() - 1; i >= 0; --i) {
        Matrix error = hadamard((weights[i+1].transposed() * error_prev), feed_forward[1][i].applyActivationFunction_derivative(activationFuncions[i]));
        error_prev = error;
        errors.push_back(error_prev);
    }
    return errors;
}

void network::update_loss(Matrix& predicted, Matrix& correct) {
    double batch_loss = 0;
    for (int i = 0; i < predicted.getRows(); ++i) {  // Iterate over samples
        for (int c = 0; c < predicted.getCols(); ++c) {  // Iterate over classes
            if (correct[i][c] > 0) {  // Only sum for the true class (one-hot)
                batch_loss += -log(predicted[i][c]) * correct[i][c];
            }
        }
    }
    loss += batch_loss;
}

// Gradient descent for weights
void network::gradient_descent_weights(std::vector<std::vector<Matrix>>& errors, double& learning_rate, Matrix& x_labels, std::vector<std::vector<Matrix>>& batch_activated_layer) {
    std::vector<Matrix> sum(weights.size());
    for (int i = 0; i < weights.size(); ++i) { // Initialize sum
        sum[i] = Matrix(weights[i].getRows(), weights[i].getCols()); //add the same size as the weights
    }
    for (int trening = 0; trening < errors.size(); ++trening) { // For each sample
        std::vector<Matrix> activated_layers = batch_activated_layer[trening]; // Get the activated layers for the sample
        for (int lag = 0; lag < errors[trening].size(); ++lag) { // Iterate over all layers
            int error_idx = errors[trening].size() - 1 - lag;
            Matrix gradient = errors[trening][error_idx] * activated_layers[lag].transposed(); // Compute the gradient
            sum[lag] = sum[lag] + gradient;
        }
    }
    for (int layer = 0; layer < weights.size(); ++layer) {
        weights[layer] = weights[layer] - sum[layer].divideByNumber(errors.size() / learning_rate);
    }
}

void network::gradient_descent_biases(std::vector<std::vector<Matrix>>& errors, double& learning_rate, Matrix& x_labels, std::vector<std::vector<Matrix>>& batch_activated_layers) {
    int L = biases.size(); // Number of layers with biases (hidden + output = 3)
    std::vector<Matrix> sum(L);
    for (int i = 0; i < L; ++i) {
        sum[i] = Matrix(biases[i].getRows(), biases[i].getCols()); // 10x1, 10x1, 11x1
    }
    for (int trening = 0; trening < errors.size(); ++trening) { // For each sample
        for (int lag = 0; lag < L; ++lag) { // Iterate over all layers
            int error_idx = L - 1 - lag; // Correct error index
            sum[lag] = sum[lag] + errors[trening][error_idx];
        }
    }
    for (int layer = 0; layer < L; ++layer) {
        biases[layer] = biases[layer] - sum[layer].divideByNumber(errors.size() / learning_rate);
    }
}


void network::train(std::vector<Matrix> train_x_labels, std::vector<Matrix> train_y_labels, std::vector <Matrix> test_x_labels, std::vector <Matrix> test_y_labels, int epochs, double learning_rate, int batch_size, bool animation) {
    std::cout << "\n----------------------------------\n" << std::endl;
    std::cout << "\033[1;36mInfo: \033[0m" << "Initializing training of network\n";
    std::cout << "\n-----------------------------------------" << std::endl;
    std::cout << "\033[1;30m    Parameters \033[0m\n";
    std::cout << "      Number of hidden layers     : " << hidden_layers.size() << std::endl;
    std::cout << "      Number of training samples  : " << train_x_labels.size() << std::endl;
    std::cout << "      Number of test samples      : " << test_x_labels.size() << std::endl;
    std::cout << "      Learning rate               : " << learning_rate << std::endl;
    std::cout << "      Batch size                  : " << batch_size << std::endl;
    std::cout << "      Epochs                      : " << epochs << std::endl;
    std::cout << "-------------------------------------------" << std::endl;

    double epochs_n[epochs];
    double loss_n[epochs];
    double accuracy_n[epochs];

    training_visualise window(100, 100, 1000, 500, "Training network");
    //TODO: find better way to do this -> atm creates window even when animation is false
    if (animation) {
        window.initialise();
    }
    else {
        window.close(); // Should close the window if animation is false -> not working!!!
    }
    
        
    for (int i = 0; i < epochs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        // Train the network with the training data
        for (int j = 0; j < train_x_labels.size(); j += batch_size) {
            std::vector<std::vector<Matrix>> batch_errors;
            std::vector<std::vector<Matrix>> batch_activated_layers;
            std::vector<std::vector<Matrix>> batch_weighted_inputs; 
            std::vector<Matrix> batch_predictions;
            for (int k = 0; k < batch_size && (j + k) < train_x_labels.size(); ++k) { // For each batch (in parallel)
                int index = j + k;
                std::vector<std::vector<Matrix>> feed_forward = feed_forward_pass(train_x_labels[index]);
                std::vector<Matrix> activated_layers = feed_forward[0];
                activated_layers.insert(activated_layers.begin(), train_x_labels[index]);
                std::vector<Matrix> weighted_inputs = feed_forward[1]; // Extract weighted inputs
                std::vector<Matrix> error = get_errors(train_x_labels[index], train_y_labels[index]);
                batch_errors.push_back(error);
                batch_activated_layers.push_back(activated_layers);
                batch_weighted_inputs.push_back(weighted_inputs); // Store weighted inputs
                batch_predictions.push_back(feed_forward[0].back());
            }
            gradient_descent_weights(batch_errors, learning_rate, train_x_labels[j], batch_activated_layers);
            gradient_descent_biases(batch_errors, learning_rate, train_x_labels[j], batch_activated_layers);
            for (int k = 0; k < batch_predictions.size(); ++k) {
                update_loss(batch_predictions[k], train_y_labels[j + k]);
            }
        }
        
        // Test the network with the test data
        std::vector <Matrix> predictions;
        for (int j = 0; j < test_x_labels.size(); ++j) {
            std::vector<std::vector<Matrix>> feed_forward = feed_forward_pass(test_x_labels[j]);
            predictions.push_back(feed_forward[0].back());
        }
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

        double current_accuracy = get_accuracy(predictions, test_y_labels);
        double current_loss = loss / train_x_labels.size();
        std::cout << "Epoch " << i + 1 << ": " << "\033[1;32mDone\033[0m\n";
        std::cout << "---------" << "\n";
        std::cout << "\033[1;30mAccuracy: \033[0m\n" << current_accuracy << "%\n";
        std::cout << "\033[1;30mLoss: \033[0m\n" << current_loss << "\n";
        std::cout << "\033[1;30mTime taken for epoch: \033[0m\n" << static_cast<double> (duration.count()) / 1000 << " s" << "\n";
        std::cout << "\033[1;30mEstimated time left: \033[0m\n" << static_cast<double> (duration.count()) / 1000 * (epochs - i - 1) << " s" << "\n";
        std::cout << "-----------------" << "\n";

        if (animation) { // Update the window
            epochs_n[i] = i;
            loss_n[i] = current_loss;
            accuracy_n[i] = current_accuracy;
            window.update(epochs_n, loss_n, accuracy_n, current_accuracy, current_loss, epochs, i+1);
            window.next_frame();
        }


        loss = 0; // Reset loss
    }

std::cout << "Training complete!" << std::endl;
usleep(1000000); // Wait for 2 seconds
if (animation) {
    window.update(epochs_n, loss_n, accuracy_n, accuracy_n[epochs-1], loss_n[epochs-1], epochs, epochs);
    window.finish();

}
}


void network::visualise_network_terminal(Matrix& input, bool show_hidden) {
    // Print the results
    std::cout << "Input Layer: \n" << input << std::endl;

    std::vector<std::vector<Matrix>> feed_forward = feed_forward_pass(input);
    std::vector<Matrix> activated_layers = feed_forward[0];

    if (show_hidden) {
        std::cout << "Hidden layers in neural net with corresponding weights: \n" << std::endl;
        std::cout << "weigths Input -> first hidden:\n" << weights[0] << std::endl;
        for (int i = 0; i < hidden_layers.size(); i++) {
            std::cout << "Layer " << i + 1 << " with " << activationFuncions[i] << " applied: \n\n" << activated_layers[i] << std::endl;
            if (i == hidden_layers.size() - 1) {
                std::cout << "Weights " << i+1 << ". -> Output layer \n" << weights[i+1] << std::endl;
            }
            else {
            std::cout << "Weights " << i+1 << ". -> " << i+2 << ". layer \n" << weights[i+1] << std::endl;
            }
        }
    }
    // Print the final output from the network
    std::cout << "Output Layer with " << activationFuncions.back() << " applied: \n" << activated_layers.back() << std::endl;
}


void network::check_params() {
    if (activationFuncions.size() != hidden_layers_sizes.size() + 1) {
        throw std::invalid_argument("The number of activation functions must match the number of layers in the network + 1.");
    }
}

void network::save_state(const std::string& filename) { //Saving the weights and biases to a file
    std::string path = "c:\\Users\\baklu\\Documents\\Kode\\Project_neural_network\\light_NN\\models\\";
    std::string file_n =  path + filename;
    std::cout << "\033[1;36mInfo: \033[0m" << "Saving model weights and biases to:\n" << path + filename << "...\n";
    if (std::filesystem::exists(file_n)) {
        std::cout << "\033[1;31mWarning: \033[0m" << filename <<" already exists!\nAre you sure you want to overwrite your previus model? [yes/no]\nAnswer: " << std::endl;
        std::string answer;
        std::cin >> answer;
        while (answer != "yes" && answer != "no") {
            std::cout << "Please enter yes or no\nAnswer: ";
            std::cin >> answer;
        }
        if (answer != "yes") {
            std::cout << "\033[1;36mInfo: \033[0m\n" << "Model not saved" << std::endl;
            return;
        }
        else {
            std::cout << "\033[1;36mInfo: \033[0m\n" << "Overwriting " << filename << "..." << std::endl;
        }
    };

    std::ofstream file(file_n, std::ios::binary);
    if (!file.is_open()) {
        throw std::invalid_argument("Could not open file: " + filename);
    }
    
    // Save weights
    int num_weights = weights.size();
    file.write(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
    for (auto& w : weights) {
        w.SaveToBin(file);
    }
    // Save biases
    int num_biases = biases.size();
    file.write(reinterpret_cast<char*>(&num_biases), sizeof(num_biases));
    for (auto& b : biases) {
        b.SaveToBin(file);
    }

    std::cout << "Network weights and biases saved: " << "\033[1;32mDone\033[0m\n";
    file.close();
}

void network::load_state(const std::string& filename) { //Loading the weights and biases from a file
    std::string path = "c:\\Users\\baklu\\Documents\\Kode\\Project_neural_network\\light_NN\\models\\";
    std::string file_n = path + filename;
    std::ifstream file(file_n, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "\033[1;31mError: \033[0m" << "Could not open the file:\n" << path + filename << std::endl;
        std::cout << "Please make sure the file is in the correct directory and that the name is correct" << std::endl;
        throw std::invalid_argument("Could not open file: " + filename);
    }

    // Load weights
    std::cout << "\033[1;36mInfo: \033[0m" << "Loading model weights and biases from: " << filename << "..." << std::endl;

    int num_weights;
    file.read(reinterpret_cast<char*>(&num_weights), sizeof(num_weights));
    weights.clear();
    for (int i = 0; i < num_weights; ++i) {
        Matrix m;
        m.LoadFromBin(file);
        weights.push_back(m);
    }
    // Load biases
    int num_biases;
    file.read(reinterpret_cast<char*>(&num_biases), sizeof(num_biases));
    biases.clear();
    for (int i = 0; i < num_biases; ++i) {
        Matrix m;
        m.LoadFromBin(file);
        biases.push_back(m);
    }
    std::cout << "Network weights and biases loaded: " << "\033[1;32mDone\033[0m\n";
    file.close();
}