#pragma once
#include "matrix.h"
#include "board.h"


const unordered_map<int, std::string> intToDirection = {
    {0, "UP"},
    {1, "RIGHT"},
    {2, "DOWN"},
    {3, "LEFT"}
};

/** 
    @brief 
    The `Game` class is designed to be a base class for different game implementations, 
    such as Pong or Snake. Each specific game should either inherit from this class or 
    use the same core functions for consistency.  

    @note Each new game class, for instance pong or snake should inherit this class in some way, or use the same functions..

    @warning  now it inherits from `Board`, which is a specific class for game functionality for snake. The most 
    important thing is that `take_action`, `is_over` and `getReward` member functions are working.
*/
class Game : public Board { 
private:
    Matrix state;
    int game_over = 0;
public:
    void take_action(int action);
    int is_over();
    double getReward(bool grow, bool collision, TDT4102::Point lastPos);
};