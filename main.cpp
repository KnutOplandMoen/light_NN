#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
#include <vector>

int main() {

    Matrix input(4, 1);
    input.setRandomValues(-1,1);
    std::cout << input;
    return 0;
}