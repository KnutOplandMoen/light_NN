#include "functions.h"
#include "matrix.h"
#include <iostream>


int main() {
    Matrix m(5,5);
    std::cout << m;
    m.setRandomValues(-5.0, 5.0);
    std::cout << m;
    std::cout << m.applyActivationFunction("sigmoid");
    return 0;
}