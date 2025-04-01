#include "board.h"

void Board::newFood(){
    std::unordered_set<TDT4102::Point> occupiedPositions;
    occupiedPositions.insert(snake.getSnakeHead());
    for (const auto& part : snake.getSnakeBody()) {
        occupiedPositions.insert(part);
    }

    TDT4102::Point foodPos;
    do {
        foodPos.x = randIntBetween(0, horizontalBlocks - 1) * blockSize;
        foodPos.y = randIntBetween(0, verticalBlocks - 1) * blockSize;
    } while (occupiedPositions.count(foodPos) > 0);
    //Use hashset because when the snake gets to a certain size,
    //it could take a while to check the entire body multiple times
    //This will for the most part not help, 
    //but hinders random spikes in frame time when snake is big.
    foodVec.push_back(foodPos);
}
Board::Board() : 
    AnimationWindow{300, 100, boardW, boardH, "SnakeGame"},
    snake(blockSize, boardW, boardH),
    running(true)
{
    
    snake.changeDirection("RIGHT");
    newFood();
}

Board::~Board(){
    running = false;
    if(inputThread.joinable()){
        inputThread.join();
    }
    //properly joining the second thread
}

void Board::handleInput(){
    //this ensures input is being checked 200 times a second and
    //reduces risk of ghosting
    while(running){
        directionChange();
        Sleep(5);//must have some delay to dampen cpu usage
    }
}

void Board::directionChange(){
    if (is_key_down(KeyboardKey::RIGHT)){
        snake.changeDirection("RIGHT");
    }
    else if (is_key_down(KeyboardKey::UP)){
        snake.changeDirection("UP");
    }
    else if (is_key_down(KeyboardKey::DOWN)){
        snake.changeDirection("DOWN");
    }
    else if (is_key_down(KeyboardKey::LEFT)){
        snake.changeDirection("LEFT");
    }
}


void Board::drawBoard(){
    //Draw grid lines
    for (int i = 1; i < horizontalBlocks; i++){
        draw_line({i*blockSize, 0}, {i*blockSize, boardW}, TDT4102::Color::light_gray);
    }
    for (int i = 1; i < verticalBlocks; i++){
        draw_line({0, i*blockSize}, {boardH, i*blockSize}, TDT4102::Color::light_gray);
    }

    
    //Draw food items
    for (size_t i = 0; i < foodVec.size(); i++){
        draw_rectangle({foodVec.at(i)}, blockSize, blockSize, TDT4102::Color::red);
    }

    //Draw snake head and body
    draw_rectangle({snake.getSnakeHead()}, blockSize, blockSize, TDT4102::Color::black);
    for (const TDT4102::Point& bodyPiece : snake.getSnakeBody()){
        draw_rectangle({bodyPiece}, blockSize, blockSize, TDT4102::Color::green);
    }
    
}

Matrix Board::getState(){
    /*
    [
    danger_up, right, down, left,  // [0, 0, 1, 0, #one-hot
    food_up, right, down, left,    //  0, 1, 0, 0, #one-hot
    dir_up, right, down, left      //  0, 0, 1, 0] #one-hot
    ]
    */
    Matrix state(12, 1);
    std::unordered_set<TDT4102::Point> occupiedPositions;
    TDT4102::Point pos = snake.getSnakeHead();
    TDT4102::Point moveIncrement = {blockSize, blockSize};
    TDT4102::Point dir = snake.getDirection();
    
    for (const TDT4102::Point& part : snake.getSnakeBody()) {
        occupiedPositions.insert(part);
    }
    if(occupiedPositions.contains(pos + directionMap.at("UP")*moveIncrement) || pos.y == 0){
        state[0][0] = 1;
    }else if(pos + directionMap.at("UP")*moveIncrement == foodVec[0]){
        state[4][0] = 1;
    }
    if(occupiedPositions.contains(pos + directionMap.at("RIGHT")*moveIncrement) || pos.x + blockSize == boardW){
        state[1][0] = 1;
    }else if(pos + directionMap.at("RIGHT")*moveIncrement == foodVec[0]){
        state[5][0] = 1;
    }
    if(occupiedPositions.contains(pos + directionMap.at("DOWN")*moveIncrement) || pos.y + blockSize == boardH){
        state[2][0] = 1;
    }else if(pos + directionMap.at("DOWN")*moveIncrement == foodVec[0]){
        state[6][0] = 1;
    }
    if(occupiedPositions.contains(pos + directionMap.at("LEFT")*moveIncrement) || pos.x == 0){
        state[3][0] = 1;
    }else if(pos + directionMap.at("LEFT")*moveIncrement == foodVec[0]){
        state[7][0] = 1;
    }
    if (dir == directionMap.at("UP")){
        state[8][0] = 1;
    }else if (dir == directionMap.at("RIGHT")){
        state[9][0] = 1;
    }else if (dir == directionMap.at("DOWN")){
        state[10][0] = 1;
    }else {
        state[11][0] = 1;
    }
    
    return state;
    
    
}

void Board::playSnake(){
    inputThread = std::thread(&Board::handleInput, this);
    int foodCollision = -1;
    bool grow;
    const int frameDuration = 130;
    while (!should_close()){
        grow = false;
        drawBoard();
        
        if (snake.collision()){
            close();
            std::cout << "Score: " << snake.getSnakeBody().size() << endl;
        }
        foodCollision = snake.collisionFood(foodVec);
        if (foodCollision != -1){
            foodVec.erase(foodVec.begin() + foodCollision);
            newFood();
            grow = true;
        }
        
        snake.move(grow);
        Sleep(frameDuration);
        next_frame();

    }
}


int Board::distanceToFood(TDT4102::Point p){
    TDT4102::Point f = foodVec[0];

    return std::abs(p.x - f.x) + std::abs(p.y - f.y);
}