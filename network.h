#pragma once
#include "matrix.h"
#include <vector>
#include <unordered_map>
#include "functions.h"
#include <fstream>
#include <chrono>
#include <filesystem>
#include "AnimationWindow.h"
#include <unistd.h>

class network {
    protected:
    std::vector <Matrix> hidden_layers; //A vector containing the hidden layers
    std::vector <int> hidden_layers_sizes; //A vector with the sizes of each hiddenlayer
    int input_layer_size;
    int output_layer_size;
    std::vector <Matrix> weights;
    std::vector<Matrix> biases;
    std::vector <std::string> activationFuncions;
    double loss;


    public:
    
    network(int input_layer_size, std::vector <int> hidden_layers_sizes, int output_layer_size, std::vector <std::string> activationFuncions)
        : activationFuncions(activationFuncions), input_layer_size(input_layer_size), output_layer_size(output_layer_size), hidden_layers_sizes(hidden_layers_sizes) {
        initialise_biases();
        initialise_hidden_layers();
        initialise_weights();
        check_params();
    }
    std::vector<Matrix> get_weights();
    std::vector <std::string> getActivationFunctions() {return activationFuncions;}
    void initialise_hidden_layers();
    void initialise_weights();
    void initialise_biases();
    void check_params();

    std::vector <std::vector<Matrix>> feed_forward_pass(const Matrix& x_labels) const;
    std::vector <Matrix> get_errors(Matrix& x_labels, Matrix& y_labels) const;
    void gradient_descent_weights(std::vector<std::vector<Matrix>>& errors, double& learning_rate, Matrix& x_labels, std::vector<std::vector<Matrix>>& batch_activated_layers);
    void gradient_descent_biases(std::vector<std::vector<Matrix>>& errors, double& learning_rate, Matrix& x_labels, std::vector<std::vector<Matrix>>& batch_activated_layers);

    void visualise_network_terminal(Matrix& input, bool show_hidden = false);

    void update_loss(Matrix& predicted, Matrix& correct);
    void save_state(const std::string& filename);
    void load_state(const std::string& filename);

    void train(std::vector <Matrix> train_x_labels, std::vector <Matrix> train_y_labels, std::vector <Matrix> test_x_labels, std::vector <Matrix> test_y_labels, int epochs, double learning_rate, int batch_size, bool animation = false);

};