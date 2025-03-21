#include "matrix.h"
#include "functions/functions.h"
#include <cmath>
#include <fstream>
#include <omp.h>

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
    #pragma omp parallel for collapse(2) //Parallelizing the loop
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < rhs.cols; j++){
            for (size_t k = 0; k < cols; k++){
                product[i][j] += data[i][k] * rhs.data[k][j];
            }
        }
    }
    return product;
}


Matrix Matrix::operator+(const Matrix &rhs) const{
    if (rows != rhs.rows || cols != rhs.cols){
        std::cout << "Rows: " << rows << " Cols: " << cols << std::endl;
        std::cout << "Rows: " << rhs.rows << " Cols: " << rhs.cols << std::endl;
        throw std::invalid_argument("Dimensions do not match for matrix adding.");
    }
    Matrix sum(rows, cols);
    #pragma omp parallel for collapse(2) //Parallelizing the loop
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            sum[i][j] = data[i][j] + rhs.data[i][j];
        }
    }
    return sum;
}

Matrix Matrix::operator-(const Matrix &rhs) const{
    if (rows != rhs.rows || cols != rhs.cols){
        throw std::invalid_argument("Dimensions do not match for matrix adding.");
    }
    Matrix sum(rows, cols);
    #pragma omp parallel for collapse(2) //Parallelizing both loop
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            sum[i][j] = data[i][j] - rhs.data[i][j];
        }
    }
    return sum;
}


Matrix Matrix::transposed() const{
    Matrix transposed(cols, rows);
    #pragma omp parallel for collapse(2) // Parallelizing both loops
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            transposed[j][i] = data[i][j];
        }
    }
    return transposed;
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
            if (j == m.cols - 1) {
                os << m.data.at(i).at(j);
            }
            else {
            os << m.data.at(i).at(j) << ", ";
            }
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
    else if (func == "softmax") {
        // Find the maximum value in the matrix to avoid overflow
        double max_val = data[0][0];
        #pragma omp parallel for collapse(2) reduction(max:max_val) // Parallelizing both loops and getting max value
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < this->getCols(); ++j) {
                if (data[i][j] > max_val) {
                    max_val = data[i][j];
                }
            }
        }
    
        double exponential_sum = 0.0;
        for (size_t i = 0; i < rows; ++i) { //we dont use paralell here becouse of small size (TODO: test with paralell and see if it is faster!!)
            for (size_t j = 0; j < cols; ++j) {
                double exponential = std::exp(data[i][j] - max_val);
                exponential_sum += exponential;
                activatedMatrix[i][j] = exponential;
            }
        }
    
        // Normalize by the sum of exponentials
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                activatedMatrix[i][j] /= exponential_sum;
            }
        }
    }
    else if (func == "leakyReLu") {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                activatedMatrix[i][j] = leakyReLu(data[i][j]);
            }
        }
    }
    else if (func == ""){
        activatedMatrix = Matrix(*this);
    }
    else {
        throw std::invalid_argument("Unknown activation function: " + func);
    }
    return activatedMatrix;

}

Matrix Matrix::applyActivationFunction_derivative(std::string func) {
    Matrix activatedMatrix(rows, cols);
    if (func == "sigmoid"){
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                activatedMatrix[i][j] = d_sigmoid(data[i][j]);
            }
        }
    }
    else if(func == "reLu"){
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                activatedMatrix[i][j] = d_ReLu(data[i][j]);
            }
        }
    }
    else if (func == "leakyReLu") {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                activatedMatrix[i][j] = d_leakyReLu(data[i][j]);
            }
        }
    }
    return activatedMatrix;

}

double Matrix::getMaxRow() const {
    double max = 0;
    int max_idx = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (data[i][j] > max) {
                max = data[i][j];
                max_idx = i;
            }
        }
    }
    return max_idx;
}

void Matrix::SaveToBin(std::ofstream& file){
    file.write(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.write(reinterpret_cast<char*>(&cols), sizeof(cols));
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            file.write(reinterpret_cast<char*>(&data[i][j]), sizeof(data[i][j]));
        }
    }
}

void Matrix::LoadFromBin(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    data.resize(rows);
    for (size_t i = 0; i < rows; ++i) {
        data[i].resize(cols);
        for (size_t j = 0; j < cols; ++j) {
            file.read(reinterpret_cast<char*>(&data[i][j]), sizeof(data[i][j]));
        }
    }
}