#pragma once
#include "matrix.h"
#include "fstream"
#include <string>

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

data_struct get_test_train_split(std::vector <Matrix> x_labels, std::vector <Matrix> y_labels, double split);

std::string getModelPath();