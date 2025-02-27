#pragma once
#include "iostream"
#include "vector"
#include "memory"
#include "algorithm"


class Matrix{
private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;
public:
    Matrix(int rows, int cols);
    Matrix(const Matrix& m);
    int getRows() const {return rows;}
    int getCols() const {return cols;}
    Matrix& operator=(const Matrix rhs);
    Matrix operator*(const Matrix& rhs);
    Matrix transpose();
    Matrix applySigmoid();

};

