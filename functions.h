#pragma once
#include <iostream>
#include <cmath>
#include <random>

double randDouble(double lowerBound, double upperBound);

double reLu(double n);

double sigmoid(double n);

double tanH(double n);

double cost(std::vector <double> output_layer, std::vector <double> correct_output_layer);
