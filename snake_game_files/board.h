#pragma once
#include "AnimationWindow.h"
#include "std_lib_facilities.h"
#include "snake.h"
#include "utils.h"
#include "unordered_set"
#include <thread>
#include <atomic>
#include "Matrix.h"

extern "C" __declspec(dllimport) void __stdcall Sleep(unsigned long dwMilliseconds);

class Board : public AnimationWindow{
protected:
    std::atomic<bool> running;
    std::thread inputThread;//start a seperate thread for continuous input handling
    static constexpr int blockSize = 40;
    static constexpr int boardW = 600;
    static constexpr int boardH = 600;
    static constexpr int steps = 5;
    static constexpr int verticalBlocks = boardH/blockSize;
    static constexpr int horizontalBlocks = boardW/blockSize;
    
    void directionChange();
public:
    std::vector<TDT4102::Point> foodVec;
    Snake snake;
    Board(int x = 300, int y = 100);
    ~Board();//need destructor for proper handling of the inputThread
    Matrix getState();
    Matrix getState_full_board();
    void handleInput();
    void playSnake();
    int distanceToFood(TDT4102::Point p);
    void drawBoard();
    void newFood();
    int getHeight() {return boardH;}
    int getWidth() {return boardW;}
};