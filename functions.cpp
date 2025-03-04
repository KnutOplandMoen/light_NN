#include "functions.h"
#include "matrix.h"

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

double Error_layer(Matrix train_y_labels, Matrix output_layer, std::vector <std::string> activation_functions, std::vector <Matrix> hidden_layers, std::vector <Matrix> weights) {
    Matrix current_error(output_layer.getRows(), 1);
    Matrix prev_error(output_layer.getRows(), 1);
    prev_error = output_layer - train_y_labels;

    for (int i = hidden_layers.size() - 1; i >= 0; --i) {
        if (activation_functions[i] == "ReLu") {
            current_error = hademan((weights[i+1].transposed() * prev_error), hidden_layers[i].applyActivationFunction_derivative("ReLu")); 
            prev_error = current_error;
                    }

        else if (activation_functions[i] == "sigmoid") {
            current_error = hademan((weights[i+1].transposed() * prev_error), hidden_layers[i].applyActivationFunction_derivative("sigmoid")); 
            prev_error = current_error;
        }
    } 
}