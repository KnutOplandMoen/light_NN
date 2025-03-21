#include "../matrix.h"
#include <vector>
#include <random>

double get_accuracy(std::vector <Matrix>& predictions, std::vector <Matrix>& correct);

double randDouble(double lowerBound, double upperBound);

double cost(std::vector <double>& output_layer, std::vector <double>& correct_output_layer);

Matrix hadamard(Matrix m1, Matrix m2);
