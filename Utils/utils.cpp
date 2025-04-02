#include "utils.h"

int randIntBetween(int lowLim, int upLim){
    std::random_device rd;
    std::default_random_engine generator(rd());
    std::uniform_int_distribution<int> dist(lowLim, upLim);
    return dist(generator);

}