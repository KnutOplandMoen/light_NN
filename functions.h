#pragma once
#include <iostream>
#include <cmath>
#include <random>
#include "matrix.h"
class Matrix;

double randDouble(double lowerBound, double upperBound);

double reLu(double n);

double d_ReLu(double n);

double sigmoid(double n);

double d_sigmoid(double n);

double leakyReLu(double n);
 
double d_leakyReLu(double n);

double cost(std::vector <double>& output_layer, std::vector <double>& correct_output_layer);

Matrix hadamard(Matrix m1, Matrix m2);

Matrix divideByNumber(Matrix& m, double number);

struct data_struct {
    std::vector <Matrix> x_labels;
    std::vector <Matrix> y_labels;
    std::vector <Matrix> x_labels_train;
    std::vector <Matrix> y_labels_train;
    std::vector <Matrix> x_labels_test;
    std::vector <Matrix> y_labels_test;
};

data_struct get_data(int dim_x, int dim_y, const std::string& filename);

Matrix input_to_matrix(std::vector <double> input);

double get_accuracy(std::vector <Matrix>& predictions, std::vector <Matrix>& correct);

data_struct get_test_train_split(std::vector <Matrix> x_labels, std::vector <Matrix> y_labels, double split);
