#include "functions.h"
#include "matrix.h"
#include "network.h"

double randDouble(double lowerBound, double upperBound){
    std::random_device rnd;
    std::default_random_engine generator(rnd());
    std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
    return distribution(generator);
}

double sigmoid(double n){
    return 1/(1 + exp(-n));
}

double reLu(double n){
    if (n < 0){
        return 0.0;
    }
    else{
        return n;
    }
}

double cost(std::vector <double> output_layer, std::vector <double> correct_output_layer) {
    if (output_layer.size() != correct_output_layer.size()) {
        throw std::invalid_argument("output_layer and correct output layer must have same dimensions");
    }
    else {
        double cost = 0;
        for (int i = 0; i < output_layer.size(); ++i) {
            cost += std::pow((output_layer[i] - correct_output_layer[i]), 2);
        }
        return cost; //return cost
    }
    
}

double d_sigmoid(double z) {
    return sigmoid(z) * (1 - sigmoid(z));
}

double d_ReLu(double z) {
    if (z > 0) {
        return 1;
    }
    else {
        return 0;
    }
}

Matrix hademan(Matrix a, Matrix b) {
    int rows = a.getRows();
    int cols = a.getCols();
    if (rows != b.getRows() || cols != b.getCols()){
        throw std::invalid_argument("Dimensions do not match for matrix adding.");
    }
    Matrix sum(rows, cols);
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            sum[i][j] = a[i][j] * b[i][j];
        }
    }
    return sum;
}

std::vector <Matrix> feed_forward_batch(Matrix input_layer, std::vector <Matrix> hidden_layers_batch, std::vector <Matrix> weights, std::vector <std::string> activationFuncions, std::vector <Matrix> biases) {
    hidden_layers_batch[0] = (weights[0] * input_layer).applyActivationFunction(activationFuncions[0]); //Computing first layer values
    for (int i = 1; i < hidden_layers_batch.size() ; ++i) {
        hidden_layers_batch[i] = ((weights[i] * hidden_layers_batch[i-1]) + biases[i]).applyActivationFunction(activationFuncions[i]); 
    }
    hidden_layers_batch.push_back(((weights.back() * hidden_layers_batch.back()) + biases.back()).applyActivationFunction(activationFuncions.back()));
    return hidden_layers_batch; //To do: Add a output function option here on the output layer: for instance softmax
}

Matrix sum_gradient_layer(std::vector <Matrix> errors, std::vector <Matrix> layers) {
    Matrix m1 = errors.back() * layers[layers.size() - 2].transposed();
    Matrix sum = m1;
    for (int i = layers.size() - 1; i >= 0; --i) {
        sum = (errors[i] * layers[i].transposed()) + m1;
        m1 = errors[i] * layers[i].transposed();
    }

    return sum;
}