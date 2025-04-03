#pragma once
#include "std_lib_facilities.h"
#include "AnimationWindow.h"
#include "algorithm"
#include "deque"
#include <functional> 
#include "data_functions.h"

//needed hash function for a TDT point, used to check if the random location of new food
//is at the same place as the snakebody
namespace std {
    template <>
    struct hash<TDT4102::Point> {
        size_t operator()(const TDT4102::Point& p) const {
            return hash<int>()(p.x) ^ (hash<int>()(p.y) << 1); // XOR for hashing
        }
    };
}

//also used for speedy lookup in hash table when comparing food to snakebody
namespace TDT4102 {
    inline bool operator==(const Point& p1, const Point& p2) {
        return (p1.x == p2.x && p1.y == p2.y);
    }
}


//possible directions, when you are going right in the screen you are only moving in the positive x direction etc.
const unordered_map<std::string, TDT4102::Point> directionMap{
    {"RIGHT", {1,0}},
    {"DOWN", {0,1}},
    {"LEFT", {-1,0}},
    {"UP", {0,-1}}
};




class Snake{
private:
    int blockSize;
    int boardW;
    int boardH;
    TDT4102::Point moveIncrement = {blockSize, blockSize};
    TDT4102::Point snakeHead;
    std::deque<TDT4102::Point> snakeBody; //needed a deque for push_front
public:
    Snake(int blockSize, int boardW, int boardH);
    void changeDirection(std::string key);
    void move(bool grow);
    bool collision();
    int collisionFood(const std::vector<TDT4102::Point>& foodVec);
    std::deque<TDT4102::Point> getSnakeBody() {return snakeBody;}
    TDT4102::Point getSnakeHead() {return snakeHead;}
    TDT4102::Point getDirection() {return direction;}
    TDT4102::Point direction;
    TDT4102::Image image{getModelPath() + "snake_game_files/" + "head.png"};
    TDT4102::Image apple{getModelPath() + "snake_game_files/" + "apple.jpg"};
};



//operators needed for the move logic
TDT4102::Point operator*(const TDT4102::Point& p1, const TDT4102::Point& p2);
void operator+=(TDT4102::Point& p1, const TDT4102::Point& p2);

TDT4102::Point operator+(const TDT4102::Point& p1, const TDT4102::Point& p2);