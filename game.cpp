#include "game.h"
#include <iostream>
#include <thread>
#include <chrono>


void Game::take_action(int action)
{
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
        return 100;
    }
    else if (collision){
        return -200;
    }
    else if(distanceToFood(snake.getSnakeHead()) < distanceToFood(lastPos)){
        return 10;
    }
    else{
        return -20;
    }
}
