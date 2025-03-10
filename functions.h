#pragma once
#include <iostream>
#include <cmath>
#include <random>
#include "matrix.h"
class Matrix;

double randDouble(double lowerBound, double upperBound);

double reLu(double n);

double d_ReLu(double n);

double sigmoid(double n);

double d_sigmoid(double n);

double cost(std::vector <double> output_layer, std::vector <double> correct_output_layer);

Matrix hadamard(Matrix m1, Matrix m2);

Matrix divideByNumber(Matrix m, double number);

std::vector <std::vector<Matrix>> get_data(int dim_x, int dim_y);