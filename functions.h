#pragma once
#include <iostream>
#include <cmath>
#include <random>
#include "matrix.h"

double randDouble(double lowerBound, double upperBound);

double reLu(double n);

double d_ReLu(double z);

double sigmoid(double n);

double d_sigmoid(double z);

double cost(std::vector <double> output_layer, std::vector <double> correct_output_layer);

double dCda(Matrix train_y_labels, Matrix output_layer);

Matrix hademan(Matrix a, Matrix b);

std::vector <Matrix> feed_forward_batch(Matrix input_layer, std::vector <Matrix> hidden_layers, std::vector <Matrix> weights, std::vector <std::string> activationFuncions, std::vector <Matrix> biases);

Matrix sum_gradient_layer(std::vector <Matrix> errors,std::vector <Matrix> hidden_layers);