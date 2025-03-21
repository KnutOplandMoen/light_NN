#include "data_functions.h"

data_struct get_data(int dim_x, int dim_y, const std::string& filename) {
    std::ifstream file("c:\\Users\\knuto\\Documents\\programering\\NN\\light_NN\\" + filename); //Change path to your own
    std::vector <Matrix> y_labels;
    std::vector <Matrix> x_labels;

    if (!file) {
       throw std::invalid_argument("Could not open the file!");
    }
    else {
        data_struct data;
        std::cout << "Loading data from: " << filename << "..." << std::endl;
        std::string line;
        while (std::getline(file, line)) {
            Matrix y_vector(dim_y, 1);
            Matrix x_vector(dim_x, 1);
            for (int i = 0; i < dim_y; ++i) {
                char temp = line[i + dim_x];
                if (!isdigit(temp)) {
                    throw std::invalid_argument("The data file must only contain digits");
                }
                y_vector[i][0] = line[i+dim_x] - '0'; // Convert characther to integr with - '0'
            }
            for (int i = 0; i < dim_x; ++i) {
                char temp = line[i];
                if (!isdigit(temp)) {
                    throw std::invalid_argument("The data file must only contain digits");
                }
                x_vector[i][0] = temp - '0';
            }
            y_labels.push_back(y_vector);
            x_labels.push_back(x_vector);
        }
        data.x_labels = x_labels;
        data.y_labels = y_labels;
        return {data};
    }
}

Matrix input_to_matrix(std::vector <double> input) {//transforming n sized vector to nx1 matrix
    Matrix m(input.size(), 1);
    for (int i = 0; i < input.size(); ++i) {
        m[i][0] = input[i];
    }
    return m;
}


data_struct get_test_train_split(std::vector <Matrix> x_labels, std::vector <Matrix> y_labels, double split) {
    
    if (x_labels.size() != y_labels.size()) {
        throw std::invalid_argument("The number of x_labels must match the number of y_labels");
    }

    else {
        data_struct split_data;
        int split_index = x_labels.size() * split;
        std::vector <Matrix> x_labels_train;
        std::vector <Matrix> y_labels_train;
        std::vector <Matrix> x_labels_test;
        std::vector <Matrix> y_labels_test;
        for (int i = 0; i < split_index; ++i) {
            x_labels_train.push_back(x_labels[i]);
            y_labels_train.push_back(y_labels[i]);
        }
        for (int i = split_index; i < x_labels.size(); ++i) {
            x_labels_test.push_back(x_labels[i]);
            y_labels_test.push_back(y_labels[i]);
        }
        split_data.x_labels_train = x_labels_train;
        split_data.y_labels_train = y_labels_train;
        split_data.x_labels_test = x_labels_test;
        split_data.y_labels_test = y_labels_test;

        return {split_data};
    }
}