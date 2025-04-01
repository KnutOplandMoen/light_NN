#pragma once
#include "AnimationWindow.h"
#include "std_lib_facilities.h"
#include "snake.h"
#include "utils.h"
#include "unordered_set"
#include <thread>
#include <atomic>
#include "C://Users//baklu//Documents//Kode//Project_neural_network//light_nn//Matrix.h"

extern "C" __declspec(dllimport) void __stdcall Sleep(unsigned long dwMilliseconds);

class Board : public AnimationWindow{
protected:
    std::atomic<bool> running;
    std::thread inputThread;//start a seperate thread for continuous input handling
    static constexpr int blockSize = 30;
    static constexpr int boardW = 600;
    static constexpr int boardH = 600;
    static constexpr int steps = 5;
    static constexpr int verticalBlocks = boardH/blockSize;
    static constexpr int horizontalBlocks = boardW/blockSize;
    
    std::vector<TDT4102::Point> foodVec;
    void drawBoard();
    void newFood();
    void directionChange();
public:
    Snake snake;
    Board();
    ~Board();//need destructor for proper handling of the inputThread
    Matrix getState();
    void handleInput();
    void playSnake();
    int distanceToFood(TDT4102::Point p);
};