#include "game.h"
#include <iostream>
#include <thread>
#include <chrono>


void Game::take_action(int action)
{
    TDT4102::Point newDir = directionMap.at(intToDirection.at(action));

    if (newDir == snake.direction * TDT4102::Point{-1, -1}) {
        return;
    }
    snake.direction = directionMap.at(intToDirection.at(action));
}

int Game::is_over(){
    if(snake.collision()){
        return 1;
    }
    return 0;
}

double Game::getReward(bool grow, bool collision, TDT4102::Point lastPos)
{
    if (grow){
        return 10;
    }
    else if (collision){
        return -10;
    }

    double currentDist = distanceToFood(snake.getSnakeHead());
    double lastDist = distanceToFood(lastPos);
    double distChange = lastDist - currentDist;

    if (distChange > 0) {
        return 1 * distChange + 1.0; // Reward proportional to distance reduction + survival bonus
    }
    else if (distChange < 0) {
        return 1 * distChange - 1.0; // Penalty proportional to distance increase
    }
    else {
        return 0.1; // Small survival bonus for neutral move
    }
}
