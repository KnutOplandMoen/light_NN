#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>



class Matrix{
private:
    std::vector<double> data;
    int rows;
    int cols;
public:
    Matrix(int rows, int cols);
    Matrix() = default;
    Matrix(const Matrix& m); 
    int getRows() const {return rows;}
    int getCols() const {return cols;}
    double getMaxRow() const;

    void setRandomValues(double lowerBound, double upperBound);
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
    Matrix& operator=(Matrix rhs);
    double* operator[](const size_t index) {return &data[index * cols];}
    const double* operator[](size_t i) const {return &data[i * cols];}
    Matrix operator*(const Matrix& rhs) const;
    Matrix operator+(const Matrix& rhs) const;
    Matrix operator-(const Matrix &rhs) const;
    Matrix divideByNumber(double number);

    Matrix transposed() const;
    Matrix applyActivationFunction(std::string func);
    Matrix applyActivationFunction_derivative(std::string func);

    void SaveToBin(std::ofstream& file);
    void LoadFromBin(std::ifstream& file);
};