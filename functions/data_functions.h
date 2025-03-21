#pragma once
#include "../matrix.h"
#include <vector>
#include <string>
#include <fstream>

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
