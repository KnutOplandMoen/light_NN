#include "activation_functions.h"

double sigmoid(double n){
    return 1/(1 + exp(-n));
}

double d_sigmoid(double n) {
    double sig = sigmoid(n);
    return sig * (1 - sig);
}

double reLu(double n){
    if (n < 0){
        return 0.0;
    }
    else{
        return n;
    }
}

double d_ReLu(double n) {
    if (n <= 0){
        return 0.0;
    }
    else{
        return 1.0;
    }
}

double leakyReLu(double n) {
    return (n > 0) ? n : 0.01 * n;
}

double d_leakyReLu(double n) {
    return (n > 0) ? 1.0 : 0.01;
}