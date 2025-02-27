#include "matrix.h"
#include <vector>
#include <unordered_map>

class network {
    private:
    Matrix input_layer;
    std::vector <Matrix> hidden_layers;
    Matrix output_layer;
    int input_layer_size;
    int hidden_layers_size;
    int output_layer_size;
    std::vector <Matrix> weights;

    public:
    network(Matrix input_layer, std::vector <Matrix> hidden_layers, Matrix output_layer)
        : input_layer(input_layer), hidden_layers(hidden_layers), output_layer(output_layer),
          input_layer_size(input_layer.getRows()), hidden_layers_size(hidden_layers.size()), output_layer_size(output_layer.getRows()) {
    }

    std::vector <Matrix> initialise_hidden_layers(std::vector <int> hidden_layers_sizes);
    std::vector <Matrix> initialise_weights(Matrix input_layer, std::vector <Matrix> hidden_layers, Matrix output_layer);
    void feed_forward();
};