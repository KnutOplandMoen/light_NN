#include "snake.h"


TDT4102::Point operator*(const TDT4102::Point& p1, const TDT4102::Point& p2){
    return TDT4102::Point{p1.x*p2.x, p1.y*p2.y};
}

void operator+=(TDT4102::Point& p1, const TDT4102::Point& p2){
    p1.x += p2.x;
    p1.y += p2.y;
}

TDT4102::Point operator+(const TDT4102::Point &p1, const TDT4102::Point &p2)
{
    return {p1.x+p2.x, p1.y+p2.y};
}

Snake::Snake(int blockSize, int boardW, int boardH) : blockSize(blockSize), boardW(boardW), boardH(boardH){
    //starting snake centered
    snakeHead = {((boardW/blockSize)/2)*blockSize, ((boardH/blockSize)/2)*blockSize};
}

//the board class will give a direction, 
void Snake::changeDirection(std::string key){
    direction = directionMap.at(key);

}

void Snake::move(bool grow){
    TDT4102::Point oldHead = snakeHead;        
    snakeHead += direction * moveIncrement;      

    //old headposition is where the first element of the body will be
    snakeBody.push_front(oldHead);

    // if the snake should not grow, remove last
    //this implementation is more effective than moving every part of the snakebody, 
    //the only visual movements are at the fron and back
    if (!grow) {
        snakeBody.pop_back();
    }
}
bool Snake::collision()
{
    if(!(snakeHead.x >= 0 && snakeHead.x < boardW)){
        return true;
    }
    if(!(snakeHead.y >= 0 && snakeHead.y < boardH)){
        return true;
    }
    for(size_t i = 0; i < snakeBody.size(); ++i){
        if (snakeBody.at(i) == snakeHead){
            return true;
        }
    }
    return false;
}


//only need to check snakehead, all other pieces move in its path
int Snake::collisionFood(const std::vector<TDT4102::Point>& foodVec){
    for (size_t i = 0; i < foodVec.size(); i++){
        if (snakeHead == foodVec.at(i)){
            return static_cast<int>(i);
        }
    }
    return -1; //returns index of food, incase we want multiple food items at once.
}


