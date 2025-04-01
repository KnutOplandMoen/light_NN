#pragma once
#include "matrix.h"
#include "board.h"

const unordered_map<int, std::string> intToDirection = {
    {0, "UP"},
    {1, "RIGHT"},
    {2, "DOWN"},
    {3, "LEFT"}
};


class Game : public Board { //each new game class, for instance pong or snake should inherit this class in some way, or use the same functions..
private:
    Board board;
    Matrix state;
    int game_over = 0;
public:
    void take_action(int action);
    int is_over();
    double getReward(bool grow, bool collision, TDT4102::Point lastPos);
};