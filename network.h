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

    public:
    network(Matrix input_layer, std::vector <int> hidden_layers_sizes, Matrix output_layer)
        : input_layer(input_layer), output_layer(output_layer),
          input_layer_size(input_layer.getRows()), output_layer_size(output_layer.getRows()), hidden_layers_sizes(hidden_layers_sizes) {
    }

    void initialise_hidden_layers();
    void initialise_weights();
    Matrix feed_forward();
};