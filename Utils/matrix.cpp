#include "matrix.h"
#include "functions.h"
#include <cmath>
#include <fstream>
#include <stdexcept> // Required for std::runtime_error
#include <limits>    // Required for std::numeric_limits
#include <string>    // Required for std::to_string
#include <iostream> // Make sure iostream is included

Matrix::Matrix(long rows, long cols) : rows(rows), cols(cols) {
    // Log immediately upon entry, forcing flush
    std::cerr << "Matrix::Matrix called with rows=" << rows << ", cols=" << cols << std::endl;

    // Add your dimension checks here (using std::cerr for output)
    if (rows <= 0 || cols <= 0) {
         std::cerr << "ERROR: Matrix dimensions must be positive. Got rows=" << rows << ", cols=" << cols << std::endl;
         throw std::runtime_error("Invalid matrix dimensions"); // Or handle differently
    }
    unsigned long long total_elements = static_cast<unsigned long long>(rows) * cols;
     std::cerr << "Matrix::Matrix attempting to allocate " << total_elements << " elements." << std::endl;

    // ... rest of the constructor (resize/allocation) ...
    data.resize(total_elements, 0.0); // Using resize after checks
}

//Deepcopy constructor
Matrix::Matrix(const Matrix& m) : rows(m.getRows()), cols(m.getCols()), data(m.data){}

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
        for (size_t k = 0; k < cols; k++){
            double data_copy = data[i * cols + k];
            for (size_t j = 0; j < rhs.cols; j++){
                product[i][j] +=  data_copy * rhs.data[k * rhs.cols + j];
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
   
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            sum[i][j] = data[i * cols + j] + rhs.data[i * cols + j];
        }
    }
    return sum;
}

Matrix Matrix::operator-(const Matrix &rhs) const{
    if (rows != rhs.rows || cols != rhs.cols){
        throw std::invalid_argument("Dimensions do not match for matrix adding.");
    }
    Matrix sum(rows, cols);
   
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            sum[i][j] = data[i * cols + j] - rhs.data[i * cols + j];
        }
    }
    return sum;
}


Matrix Matrix::transposed() const{
    Matrix transposed(cols, rows);

    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            transposed[j][i] = data[i * cols + j];
        }
    }
    return transposed;
}


void Matrix::setRandomValues(double lowerBound, double upperBound){
    for (size_t i = 0; i < rows; i++){
        for (size_t j = 0; j < cols; j++){
            data[i * cols + j] = randDouble(lowerBound, upperBound);
        }
    }
}


std::ostream &operator<<(std::ostream &os, const Matrix &m){
    for (size_t i = 0; i < m.rows; i++){
        os << "[";
        for (size_t j = 0; j < m.cols; j++){
            if (j == m.cols - 1) {
                os << m.data[i * m.cols + j];
            }
            else {
            os << m.data[i * m.cols + j] << ", ";
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
                activatedMatrix[i][j] = sigmoid(data[i * cols + j]);
            }
        }
    }
    else if (func == "reLu"){
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                activatedMatrix[i][j] = reLu(data[i * cols + j]);
            }
        }
    }
    else if (func == "tanH"){
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                activatedMatrix[i][j] = tanh(data[i * cols + j]);
            }
        }
    }
    else if (func == "softmax") {
        double max_val = data[0];
      
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < this->getCols(); ++j) {
                if (data[i * cols + j] > max_val) {
                    max_val = data[i * cols + j];
                }
            }
        }
    
        double exponential_sum = 0.0;
        for (size_t i = 0; i < rows; ++i) { //we dont use paralell here becouse of small size (TODO: test with paralell and see if it is faster!!)
            for (size_t j = 0; j < cols; ++j) {
                double exponential = std::exp(data[i * cols + j] - max_val);
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
                activatedMatrix[i][j] = leakyReLu(data[i * cols + j]);
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
                activatedMatrix[i][j] = d_sigmoid(data[i * cols + j]);
            }
        }
    }
    else if(func == "reLu"){
        for (size_t i = 0; i < rows; i++){
            for (size_t j = 0; j < cols; j++){
                activatedMatrix[i][j] = d_ReLu(data[i * cols + j]);
            }
        }
    }
    else if (func == "leakyReLu") {
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                activatedMatrix[i][j] = d_leakyReLu(data[i * cols + j]);
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
            if (data[i * cols + j] > max) {
                max = data[i * cols + j];
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
            file.write(reinterpret_cast<char*>(&data[i * cols + j]), sizeof(data[i * cols + j]));
        }
    }
}

void Matrix::LoadFromBin(std::ifstream& file) {
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows)); // Read the number of rows
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols)); // Read the number of cols
    data.resize(rows * cols); // Resize to total size
    file.read(reinterpret_cast<char*>(data.data()), sizeof(double) * rows * cols); // Read all elements at once
}

Matrix Matrix::divideByNumber(double number) {
    Matrix result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = data[i * cols + j] / number;
        }
    }
    return result;
}