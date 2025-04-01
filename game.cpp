#include "game.h"
#include <iostream>
#include <thread>
#include <chrono>

Game::Game(){
    board = Board();

}

void Game::take_action(int action)
{
    directionChange(intToDirection.at(action));
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
        return -100;
    }
    else if(distanceToFood(snake.getSnakeHead()) < distanceToFood(lastPos)){
        return 1;
    }
    else{
        return -1;
    }
}
