#include "matrix.h"

//"default" constructor
Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols){
    data.resize(static_cast<size_t>(rows));
    for (size_t i = 0; i < data.size(); i++){
        data.at(i).resize(cols, 0.0);
    }
}

//Deepcopy constructor
Matrix::Matrix(const Matrix& m) : rows(m.getRows()), cols(m.getCols()){
    data.resize(rows);
    for (size_t i = 0; i < data.size(); i++){
        data.at(i) = m.data.at(i);
    }
}

//assign operation utilizing copy-swap
Matrix& Matrix::operator=(Matrix rhs){
    std::swap(data, rhs.data);
    std::swap(rows, rhs.rows);
    std::swap(cols, rhs.cols);
    return *this;
}


Matrix Matrix::operator*(const Matrix &rhs) const {
    if (cols != rhs.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }
    Matrix product(rows, rhs.cols);
    for (size_t i = 0; i < rows; i++){
<<<<<<< HEAD
        for (size_t j = 0; j < rhs.cols; j++){ // Corrected: iterate over the correct number of columns
=======
        for (size_t j = 0; j < rhs.cols; j++){
>>>>>>> 259841872b533bef323f4553b3d36fa5b0b8c540
            for (size_t k = 0; k < cols; k++){
                product[i][j] += data[i][k] * rhs.data[k][j];
            }
        }
    }
    return product;
}



void Matrix::setRandomValues(double lowerBound, double upperBound){
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            data[i][j] = randDouble(lowerBound, upperBound);
        }
    }
}


std::ostream &operator<<(std::ostream &os, const Matrix &m){
    for (size_t i = 0; i < m.rows; i++){
        os << "[";
        for (size_t j = 0; j < m.cols; j++){
            os << m.data.at(i).at(j) << ", ";
        }
        os << "]\n";
    }
    return os;
}


Matrix Matrix::applyActivationFunction(std::string func){
    Matrix activatedMatrix(rows, cols);
    if (func == "sigmoid"){
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                activatedMatrix[i][j] = sigmoid(data[i][j]);
            }
        }
    }
    else if (func == "reLu"){
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                activatedMatrix[i][j] = reLu(data[i][j]);
            }
        }
    }
    else if (func == "tanH"){
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                activatedMatrix[i][j] = tanh(data[i][j]);
            }
        }
    }
    else {
        throw std::invalid_argument("Unknown activation function: " + func);
    }
    return activatedMatrix;

}