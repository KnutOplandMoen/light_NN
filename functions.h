#pragma once
#include <iostream>
#include <cmath>
#include <random>

double randDouble(double lowerBound, double upperBound);

double reLu(double n);

double d_ReLu(double z);

double sigmoid(double n);

double d_sigmoid(double z);

double cost(std::vector <double> output_layer, std::vector <double> correct_output_layer);

double dCda(Matrix train_y_labels, Matrix output_layer);