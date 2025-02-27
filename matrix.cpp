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

Matrix& Matrix::operator=(Matrix rhs){
    std::swap(data, rhs.data);
    std::swap(rows, rhs.rows);
    std::swap(cols, rhs.cols);
    return *this;
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
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            activatedMatrix[i][j] = sigmoid(data[i][j]);
        }
    }
    return activatedMatrix;

}