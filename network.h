#include "matrix.h"
#include <vector>
#include <unordered_map>

class network {
    private:
    Matrix input_layer; // A matrix with input layer values
    std::vector <Matrix> hidden_layers; //A vector containing the hidden layers
    std::vector <int> hidden_layers_sizes; //A vector with the sizes of each hiddenlayer
    Matrix output_layer; // A matrix with output layer values
    int input_layer_size;
    int output_layer_size;
    std::vector <Matrix> weights;
    std::vector<Matrix> biases;
    std::vector <std::string> activationFuncions;

    public:
    network(Matrix input_layer, std::vector <int> hidden_layers_sizes, Matrix output_layer, std::vector <std::string> activationFuncions)
        : activationFuncions(activationFuncions), input_layer(input_layer), output_layer(output_layer),
          input_layer_size(input_layer.getRows()), output_layer_size(output_layer.getRows()), hidden_layers_sizes(hidden_layers_sizes) {
        initialise_biases();
        initialise_hidden_layers();
        initialise_weights();
        check_params();
    }

    void initialise_hidden_layers();
    void initialise_weights();
    void initialise_biases();
    void check_params();

    Matrix feed_forward();
    Matrix feed_forward_batch(Matrix x_labels) const;
    std::vector <Matrix> get_errors(Matrix x_labels, Matrix y_labels) const;
    void gradient_descent_weights(std::vector <std::vector <Matrix>> errors, double learning_rate);

    void visualise_network(bool show_hidden);

    void train(std::vector <Matrix> train_x_labels, std::vector <Matrix> train_y_labels, int epochs, double learning_rate, int batch_size);

};