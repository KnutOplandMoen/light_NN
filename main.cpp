#include "functions.h"
#include "matrix.h"
#include "network.h"
#include <iostream>
#include <vector>

int main() {
    Matrix m(5,5);
    std::cout << m;
    m.setRandomValues(-5.0, 5.0);
    std::cout << m;
    std::cout << m.applyActivationFunction("sigmoid");

    Matrix input(4, 1);
    std::vector <int> hidden_layers = {2, 2};
    Matrix output(4, 1);

    network my_net(input, hidden_layers, output);
    my_net.initialise_hidden_layers();
    my_net.initialise_weights();
    Matrix output_layer = my_net.feed_forward();
    return 0;
}