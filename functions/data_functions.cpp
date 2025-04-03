#include "data_functions.h"
#include <filesystem>
namespace fs = std::filesystem;

/**
 * @brief Returns absolute path to a file
 * 
 * @param relativePath Relative path to file
 */
std::string getModelPath(std::string relativePath) {
    return (fs::current_path() / relativePath).string();
}

/**
 * @brief Returns x- and y-labels for neural net training, gathered from a txt file.
 * 
 * Datapoints are integers 0-9, non-seperated.
 * 
 * If input is 5 7 3 and output is 0 1, it should be saved in txt file as 57301
 * 
 * @param dim_x Dimension of input
 * 
 * @param dim_y Dimension of output
 */
data_struct get_data(int dim_x, int dim_y, const std::string& filename) {
    std::string path = getModelPath("Data/");
    std::ifstream file(path + filename); //Change path to your own
    std::vector <Matrix> y_labels;
    std::vector <Matrix> x_labels;

    if (!file) {
        std::cerr << "\033[1;31mError: \033[0m" << "Could not open the file:\n" << path + filename << std::endl;
        std::cout << "Please make sure the file is in the correct directory and that the name is correct" << std::endl;
        throw std::invalid_argument("Could not open the file!");
    }
    else {
        data_struct data;
        std::cout << "\033[1;36mInfo: \033[0m" << "Loading data from: " << filename << "..." << std::endl;
        std::string line;
        while (std::getline(file, line)) {
            Matrix y_vector(dim_y, 1);
            Matrix x_vector(dim_x, 1);
            for (int i = 0; i < dim_y; ++i) {
                char temp = line[i + dim_x];
                try{
                    if (!isdigit(temp)) {
                        std::cerr << "\033[1;31mWarning: \033[0m" << "Potential invalid character detected: " << temp << " (ASCII: " << int(temp) << ")" << std::endl;
                        throw std::invalid_argument("The data file should only contain digits");
                    }}
                    catch (std::invalid_argument& e) {
                        std::cout << e.what() << std::endl;
                    }
                y_vector[i][0] = line[i+dim_x] - '0'; // Convert characther to integr with - '0'
            }
             //TODO: idk why this doesnt throw error with text in termianl...
            for (int i = 0; i < dim_x; ++i) {
                char temp = line[i];
                try{
                if (!isdigit(temp)) {
                    std::cerr << "\033[1;31mWarning: \033[0m" << "Potential invalid character detected: " << temp << " (ASCII: " << int(temp) << ")" << std::endl;
                    throw std::invalid_argument("The data file should only contain digits");
                }}
                catch (std::invalid_argument& e) {
                    std::cout << e.what() << std::endl;
                }
                x_vector[i][0] = temp - '0';
                }
            y_labels.push_back(y_vector);
            x_labels.push_back(x_vector);
        }
        data.x_labels = x_labels;
        data.y_labels = y_labels;
        std::cout << "Data loading: " << "\033[1;32mDone \033[0m\n";
        return {data};
    }
}

/**
 * @brief Converts an n sized vector to a nx1 sized matrix.
 * 
 * Matrix type is needed to use in Network class.
 */
Matrix input_to_matrix(std::vector <double> input) {//transforming n sized vector to nx1 matrix
    Matrix m(input.size(), 1);
    for (int i = 0; i < input.size(); ++i) {
        m[i][0] = input[i];
    }
    return m;
}

/**
 * @brief Splits dataset between training and test set 
 * 
 * Returns data_struct with members x/y_labels_train and x/y_labels_test
 * 
 * @param x_labels Entire set of x-labels(network inputs)
 * @param y_labels Entire set of y-labels(Desired network outputs)
 * @param split "Percentage" of data to use as training set. 0.7 = 70%
 */
data_struct get_test_train_split(std::vector <Matrix> x_labels, std::vector <Matrix> y_labels, double split) {
    if(split > 1.0 || split < 0.0){
        throw std::invalid_argument("Data must be split between 0 and 1")
    }
    if (x_labels.size() != y_labels.size()) {
        throw std::invalid_argument("The number of x_labels must match the number of y_labels");
    }

    else {
        std::cout << "\033[1;36mInfo: \033[0m" << "Splitting data into train/test train and test set..." << std::endl;
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

        std::cout << "Train-test splitting: " << "\033[1;32mDone\033[0m\n";
        return {split_data};
    }
}