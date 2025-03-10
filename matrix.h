#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include "functions.h"



class Matrix{
private:
    std::vector<std::vector<double>> data;
    int rows;
    int cols;
public:
    Matrix(int rows, int cols);
    Matrix(const Matrix& m); 
    Matrix() = delete; //no default constructor
    int getRows() const {return rows;}
    int getCols() const {return cols;}
    void setRandomValues(double lowerBound, double upperBound);
    friend std::ostream& operator<<(std::ostream& os, const Matrix& m);
    Matrix& operator=(Matrix rhs);
    std::vector<double>& operator[](size_t index) {return data.at(index);}
    Matrix operator*(const Matrix& rhs) const;
    Matrix operator+(const Matrix& rhs) const;
    Matrix transposed();
    Matrix applyActivationFunction(std::string func);

};


