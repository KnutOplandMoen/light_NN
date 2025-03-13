#include "matrix.h"
#include <vector>
#include "network.h"
#include "functions.h"

/**
 * Initialise the weights for the neural network layers.
 * 
 * This function creates and initializes the weight matrices for the neural network.
 * The weights are initialized between the input layer and the first hidden layer,
 * between each pair of consecutive hidden layers, and between the last hidden layer
 * and the output layer.
 */
void network::initialise_weights() {
    int input_size = input_layer.getRows();
    double limit = sqrt(2.0 / input_size);
    Matrix matrix1(hidden_layers[0].getRows(), input_size);
    matrix1.setRandomValues(-limit, limit);
    weights.push_back(matrix1);

    // Apply similar logic for other layers
    for (int i = 0; i < hidden_layers.size() - 1; ++i) {
        int n_inputs = hidden_layers[i].getRows();
        limit = sqrt(2.0 / n_inputs);
        Matrix matrix2(hidden_layers[i+1].getRows(), n_inputs);
        matrix2.setRandomValues(-limit, limit);
        weights.push_back(matrix2);
    }

    int last_hidden_size = hidden_layers.back().getRows();
    limit = sqrt(2.0 / last_hidden_size);
    Matrix matrix3(output_layer.getRows(), last_hidden_size);
    matrix3.setRandomValues(-limit, limit);
    weights.push_back(matrix3);
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

void network::initialise_biases() {
    for (int i = 0; i < hidden_layers_sizes.size(); i++){
        biases.push_back(Matrix(hidden_layers_sizes[i], 1));
    }
    biases.push_back(Matrix(output_layer_size, 1));
}

/**
 * Going forward in the network, computing the node values using matrix multiplication with the weigths
 * At last the output layer is computed
 */
Matrix network::feed_forward() {
    hidden_layers[0] = ((weights[0] * input_layer) + biases[0]).applyActivationFunction(activationFuncions[0]); //Computing first layer values
    for (int i = 1; i < hidden_layers.size() ; ++i) {
        hidden_layers[i] = ((weights[i] * hidden_layers[i-1]) + biases[i]).applyActivationFunction(activationFuncions[i]); 
    }
    output_layer = ((weights.back() * hidden_layers.back()) + biases.back()).applyActivationFunction(activationFuncions.back());
    return output_layer; //To do: Add a output function option here on the output layer: for instance softmax
}

std::vector <std::vector<Matrix>> network::feed_forward_batch(Matrix x_labels) const{
    std::vector<Matrix> hidden_layers_copy = hidden_layers;
    std::vector<Matrix> activation;
    std::vector<Matrix> weigted_inputs;

    Matrix output_layer_copy = output_layer;

    hidden_layers_copy[0] = ((weights[0] * x_labels) + biases[0]).applyActivationFunction(activationFuncions[0]); //Computing first layer values
    activation.push_back(hidden_layers_copy[0]);
    weigted_inputs.push_back((weights[0] * x_labels) + biases[0]);

    for (int i = 1; i < hidden_layers.size() ; ++i) {
        hidden_layers_copy[i] = ((weights[i] * hidden_layers_copy[i-1]) + biases[i]).applyActivationFunction(activationFuncions[i]); 
        activation.push_back(hidden_layers_copy[i]);
        weigted_inputs.push_back((weights[i] * hidden_layers_copy[i-1]) + biases[i]);
    }
    output_layer_copy = ((weights.back() * hidden_layers_copy.back()) + biases.back()).applyActivationFunction(activationFuncions.back());
    activation.push_back(output_layer_copy);
    weigted_inputs.push_back((weights.back() * hidden_layers_copy.back()) + biases.back());
    
    return {activation, weigted_inputs}; //To do: Add a output function option here on the output layer: for instance softmax
}

std::vector <Matrix> network::get_errors(Matrix x_labels, Matrix y_labels) const{ //Backpropagating through network to get errors for each layer
    //Making copy
    std::vector <std::vector<Matrix>> feed_forward = feed_forward_batch(x_labels);

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

void network::update_loss(Matrix predicted, Matrix correct) {
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
void network::gradient_descent_weights(std::vector <std::vector <Matrix>> errors, double learning_rate, Matrix x_labels, std::vector <std::vector<Matrix>> feed_forward) {
    std::vector <Matrix> sum(weights.size());

    //Making empty matrices for the sum of errors
    for (int i = 0; i < weights.size(); ++i) {
        sum[i] = Matrix(weights[i].getRows(), weights[i].getCols());
    }

    Matrix output_layer_copy = feed_forward[0].back(); //Getting the output layer
    std::vector<Matrix> activated_layers = feed_forward[0]; //Getting the activated layers

    activated_layers.insert(activated_layers.begin(), x_labels); //Inserting the input layer to the activated layers
    for (int trening = 0; trening < errors.size(); trening++) { //Going through the errors

        for (int lag = 0; lag < errors[trening].size(); lag++) {
            sum[lag] = sum[lag] + (errors[trening][errors[trening].size() - lag - 1]*activated_layers[lag].transposed()); //Adding the errors * the activated layers transposed to sum
        } 

    }

    for (int layer = 0; layer < weights.size(); ++layer) { //

        weights[layer] = weights[layer] - divideByNumber(sum[layer], errors.size()/learning_rate); //Updating the weights

    }

}

void network::gradient_descent_biases(std::vector <std::vector <Matrix>> errors, double learning_rate, Matrix x_labels, std::vector <std::vector<Matrix>> feed_forward) {
    std::vector <Matrix> sum(hidden_layers.size());

    //Making empty matrices for the sum of errors
    for (int i = 0; i < hidden_layers.size(); ++i) {
        sum[i] = Matrix(hidden_layers[i].getRows(), 1); 
    }
    sum.push_back(Matrix(output_layer_size, 1));

    Matrix output_layer_copy = feed_forward[0].back(); //Getting the output layer
    std::vector<Matrix> activated_layers = feed_forward[0]; //Getting the activated layers

    activated_layers.insert(activated_layers.begin(), x_labels); //Inserting the input layer to the activated layers
    for (int trening = 0; trening < errors.size(); trening++) { //Going through the errors

        for (int lag = 0; lag < errors[trening].size(); lag++) {
            sum[lag] = sum[lag] + (errors[trening][errors[trening].size() - lag - 1]); //Adding the errors * the activated layers transposed to sum
        } 
    }

    for (int layer = 0; layer < biases.size(); ++layer) { //Updating the biases

        biases[layer] = biases[layer] - divideByNumber(sum[layer], errors.size()/learning_rate); //Updating the biases

    }

}

void network::train(std::vector <Matrix> train_x_labels, std::vector <Matrix> train_y_labels, int epochs, double learning_rate, int batch_size) { //Training the network
    std::vector <std::vector <Matrix>> errors;
    for (int i = 0; i < epochs; ++i) {
        std::cout << "Epoch: " << i << std::endl;
        std::vector<std::vector<Matrix>> errors;
        for (int j = 0; j < train_x_labels.size(); j += 1) {
            for (int k = 0; k < batch_size && (j+k) < train_x_labels.size(); ++k) {
                std::vector<Matrix> error = get_errors(train_x_labels[j + k], train_y_labels[j + k]);
                errors.push_back(error);
            }
            std::vector <std::vector<Matrix>> feed_forward = feed_forward_batch(train_x_labels[j]);
            gradient_descent_weights(errors, learning_rate, train_x_labels[j], feed_forward);
            gradient_descent_biases(errors, learning_rate, train_x_labels[j], feed_forward);
            update_loss(feed_forward[0].back(), train_y_labels[j]); //Updating the loss
            errors.clear();
        }
        std::cout << "loss: " << loss/train_x_labels.size() << std::endl; 
        loss = 0; //Resetting the loss
    }
}

void network::visualise_network(bool show_hidden) {
    // Print the results
    std::cout << "Input Layer: \n" << input_layer << std::endl;

    if (show_hidden) {
        std::cout << "Hidden layers in neural net with corresponding weights: \n" << std::endl;
        std::cout << "weigths Input -> first hidden:\n" << weights[0] << std::endl;
        for (int i = 0; i < hidden_layers.size(); i++) {
            std::cout << "Layer " << i + 1 << " with " << activationFuncions[i] << " applied: \n\n" << hidden_layers[i] << std::endl;
            if (i == hidden_layers.size() - 1) {
                std::cout << "Weights " << i+1 << ". -> Output layer \n" << weights[i+1] << std::endl;
            }
            else {
            std::cout << "Weights " << i+1 << ". -> " << i+2 << ". layer \n" << weights[i+1] << std::endl;
            }
        }
    }
    // Print the final output from the network
    std::cout << "Output Layer with " << activationFuncions.back() << " applied: \n" << output_layer << std::endl;
}

int network::get_prediction(Matrix output_layer) {
    double max = 0;
    int max_index = 0;
    for (int i = 0; i < output_layer.getRows(); ++i) {
        if (output_layer[i][0] > max) {
            max = output_layer[i][0];
            max_index = i;
        }
    }
    return max_index;
}

int network::get_prediction() {
    double max = 0;
    int max_index = 0;
    for (int i = 0; i < output_layer.getRows(); ++i) {
        if (output_layer[i][0] > max) {
            max = output_layer[i][0];
            max_index = i;
        }
    }
    return max_index;
}

void network::check_params() {
    if (activationFuncions.size() != hidden_layers_sizes.size() + 1) {
        throw std::invalid_argument("The number of activation functions must match the number of layers in the network + 1.");
    }
}

