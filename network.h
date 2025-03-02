#include "matrix.h"
#include <vector>
#include <unordered_map>

class network {
    private:
    Matrix input_layer;
    std::vector <Matrix> hidden_layers;
    std::vector <int> hidden_layers_sizes;
    Matrix output_layer;
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
    }

    void initialise_hidden_layers();
    void initialise_weights();
    void initialise_biases();
    Matrix feed_forward();
    void visualise_network(bool show_hidden);
};