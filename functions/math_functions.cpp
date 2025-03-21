#include "math_functions.h"

double get_accuracy(std::vector <Matrix>& predictions, std::vector <Matrix>& correct) {
    if (predictions.size() != correct.size()) {
        throw std::invalid_argument("The number of predictions must match the number of correct labels");
    }
    else {
        double correct_predictions = 0;
        for (int i = 0; i < predictions.size(); ++i) {
            int predicted = predictions[i].getMaxRow();
            int correct_label = correct[i].getMaxRow();
            if (predicted == correct_label) {
                correct_predictions++;
            }
        }
        return (correct_predictions / predictions.size()) * 100;
    }
}

double randDouble(double lowerBound, double upperBound){
    std::random_device rnd;
    std::default_random_engine generator(rnd());
    std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
    return distribution(generator);
}

double cost(std::vector <double>& output_layer, std::vector <double>& correct_output_layer) {
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

Matrix hadamard(Matrix m1, Matrix m2) {
    int cols = m1.getCols();
    int rows = m1.getRows();
    if (rows != m2.getRows() || cols != m2.getCols()) {
        throw std::invalid_argument("Matrices must have the same dimensions for Hadamard product");
    }
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = m1[i][j] * m2[i][j];
        }
    }
    return result;
}
