#include "functions.h"
#include "matrix.h"
#include <fstream>
#include <string>

double randDouble(double lowerBound, double upperBound){
    std::random_device rnd;
    std::default_random_engine generator(rnd());
    std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
    return distribution(generator);
}

double sigmoid(double n){
    return 1/(1 + exp(-n));
}

double d_sigmoid(double n) {
    double sig = sigmoid(n);
    return sig * (1 - sig);
}

double reLu(double n){
    if (n < 0){
        return 0.0;
    }
    else{
        return n;
    }
}

double d_ReLu(double n) {
    if (n <= 0){
        return 0.0;
    }
    else{
        return 1.0;
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

Matrix divideByNumber(Matrix m, double number) {
    int cols = m.getCols();
    int rows = m.getRows();
    Matrix result(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result[i][j] = m[i][j] / number;
        }
    }
    return result;
}

std::vector <std::vector<Matrix>> get_data(int dim_x, int dim_y) {
    std::ifstream file("data.txt");
    std::vector <Matrix> y_labels;
    std::vector <Matrix> x_labels;

    if (!file) {
        std::cerr << "Could not open the file!" << std::endl;
    }
    else {
        std::string line;
        while (std::getline(file, line)) {
            Matrix y_vector(dim_y, 1);
            Matrix x_vector(dim_x, 1);
            for (int i = 0; i < dim_y; ++i) {
                y_vector[i][0] = line[i+dim_x] - '0';
            }
            for (int i = 0; i < dim_x; ++i) {
                char temp = line[i];
                x_vector[i][0] = temp - '0';
            }
            y_labels.push_back(y_vector);
            x_labels.push_back(x_vector);
        }
        return {x_labels, y_labels};
    }
}