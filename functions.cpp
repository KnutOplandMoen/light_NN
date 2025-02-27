#include "functions.h"

double randDouble(double lowerBound, double upperBound){
    std::random_device rnd;
    std::default_random_engine generator(rnd());
    std::uniform_real_distribution<double> distribution(lowerBound, upperBound);
    return distribution(generator);
}

double sigmoid(double n){
    return 1/(1 + exp(-n));
}

double reLu(double n){
    if (n < 0){
        return 0.0;
    }
    else{
        return n;
    }
}
