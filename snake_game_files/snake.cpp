#include "snake.h"

/**
 * @brief Overloaded multiplication operator for Point struct
 * 
 * Product of two points is computed as {x*x, y*y} and returns a new point
 */
TDT4102::Point operator*(const TDT4102::Point& p1, const TDT4102::Point& p2){
    return TDT4102::Point{p1.x*p2.x, p1.y*p2.y};
}

/**
 * @brief overloaded += operator for Point struct
 * 
 * Sum of points is computed as {x+x, y+y} and returns a point
 */
void operator+=(TDT4102::Point& p1, const TDT4102::Point& p2){
    p1.x += p2.x;
    p1.y += p2.y;
}

/**
 * @brief overloaded += operator for Point struct
 * 
 * Sum of points is computed as {x+x, y+y} and returns a point
 */
TDT4102::Point operator+(const TDT4102::Point &p1, const TDT4102::Point &p2)
{
    return {p1.x+p2.x, p1.y+p2.y};
}

/**
 * @brief Snake constructor initializing its members
 * 
 * @param boardW, boardH Dimension of game-scene(pixels), must be multiplum of blockSize
 * 
 * @param blockSize Side-length of square cell(pixels)
 */
Snake::Snake(int blockSize, int boardW, int boardH) : blockSize(blockSize), boardW(boardW), boardH(boardH){
    //starting snake centered
    snakeHead = {((boardW/blockSize)/2)*blockSize, ((boardH/blockSize)/2)*blockSize};
}

/**
 * @brief changing class member direction(point struct)
 * 
 * Utilizing map<string, point> directionMap
 * 
 * @param key string representation of direction, e.g "RIGHT", "UP" etc.
 */
void Snake::changeDirection(std::string key){
    direction = directionMap.at(key);

}

/**
 * @brief moves entire Snake object one cell
 * 
 * @param grow Is true if Snake has gotten food
 */
void Snake::move(bool grow){
    TDT4102::Point oldHead = snakeHead;        
    snakeHead += direction * moveIncrement;      

    //old headposition is where the first element of the body will be
    snakeBody.push_front(oldHead);

    // if the snake should not grow, remove last
    //this implementation is more effective than moving every part of the snakebody, 
    //the only visual movements are at the front and back
    if (!grow) {
        snakeBody.pop_back();
    }
}

/**
 * @brief returns true if snakeHead position is out of bounds or same as a snakeBody object
 */
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


/**
 * @brief Checks if snakeHead is at the position as any food objects in foodVec
 * 
 * Opens possibility for multiple food items at once
 * 
 * Returns -1 if snakeHead is not on location of food
 * 
 * @param foodVec Vector holding food objects currently in play(point struct)
 */
int Snake::collisionFood(const std::vector<TDT4102::Point>& foodVec){
    for (size_t i = 0; i < foodVec.size(); i++){
        if (snakeHead == foodVec.at(i)){
            return static_cast<int>(i);
        }
    }
    return -1; //returns index of food, incase we want multiple food items at once.
}


