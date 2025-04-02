#include "q_network.h"
#include "game.h"
#include "map"
#include "math_functions.h"
#include <deque>
#include <unistd.h> 
#include "animation_functions.h"

/**
 * @brief Select an action from one of the following:
 * 
 * 1. Random -> if epsilon is close to 1 this is likely
 * 
 * 2. Network -> if epsilon is close to 0 this is more likely
 * 
 * 3. User -> from_user needs to be togled manually to true
 * 
 * @param state Current state of the game.
 * @param game_play Game class with game information.
 */
int q_network::select_action(Matrix& state, Game& game_play) {
    double number = randDouble(0, 10000);

    bool from_user = false;
    
    if (!from_user) {
    if (number / 10000 < epsilon) {  
        return static_cast<int>(number) % output_layer_size;  // Random action (explore)
    } else {
        Matrix q_values = feed_forward_pass(state)[0].back();
        return q_values.getMaxRow();  // Best action (exploit)
    }
    }
    else { //Take move from user
        int move;
        std::string move_l;
        std::cout <<"move:\n:";
        
        cin >> move_l;
        if (move_l == "w") {
            move = 0;
        }
        else if (move_l == "d") {
            move = 1;
        }
        else if (move_l == "a") {
            move = 3;
        }
        else if (move_l == "s") {
            move = 2;
        }
        else {
            move = 0;
        }
        std::cout << "chose move: " << move << std::endl;
        return move;
        }
    }

/**
 * @brief Gets information by taking in the state, doing a action in this state with select action, 
 * then adding information needed for later training to an information struct
 * 
 * @param state Current state of game.
 * @param game_play Game class with game information.
 * 
 * @note This is a game specific function, for instance this is one for snake, make new one or change this one for specific game
 * 
 * @return A information struct with:
 * `q_values`, `q_value`, `reward`, `done`, `q_target_value`, `q_target` and `state`
 */
information q_network::get_information(Matrix& state, Game& game_play) {
    bool grow = false;
    bool collision = false;
    TDT4102::Point lastPos = game_play.snake.getSnakeHead();
    Matrix q_values = feed_forward_pass(state)[0].back();

    int action = select_action(state, game_play);
    double q_value = q_values[action][0];
    Matrix prev_state = game_play.getState();

    game_play.take_action(action); //TODO: Here next state needs to be made.. in environment

    if (game_play.snake.collisionFood(game_play.foodVec) != -1){
        grow = true;
        game_play.foodVec.clear();
        game_play.newFood();
    }

    game_play.snake.move(grow);
    
    if (game_play.snake.collisionFood(game_play.foodVec) != -1){
        grow = true;
    }

    std::deque<TDT4102::Point> body = game_play.snake.getSnakeBody();

    for (TDT4102::Point& part : body) {
        if ((part.x == game_play.snake.getSnakeHead().x) && (part.y == game_play.snake.getSnakeHead().y)) {
            collision = true;
            std::cout << "crashed into itself! " << std::endl;
        }
    }

    if(!(game_play.snake.getSnakeHead().x >= 0 && game_play.snake.getSnakeHead().x < game_play.getWidth())){
        std::cout << "crashed into wall! " << std::endl;
    }
    else if(!(game_play.snake.getSnakeHead().y >= 0 && game_play.snake.getSnakeHead().y < game_play.getHeight())){
        std::cout << "crashed into wall! " << std::endl;
    }

    
    collision = game_play.snake.collision() || collision;
    //std::cout << "collision: " << collision << std::endl;
    
    double reward = game_play.getReward(grow, collision, lastPos);
    total_reward += reward;

    int done = game_play.is_over();

    Matrix new_state = game_play.getState();

    Matrix next_action = feed_forward_pass(new_state)[0].back();
    double max_next_q_value = next_action[next_action.getMaxRow()][0];

    double q_target_value = reward + (1 - done) * gamma * max_next_q_value;

    Matrix q_target = q_values;
    q_target[action][0] = q_target_value;
    information info(q_values, q_value, reward, done, q_target_value, q_target, state);

    return info;

}


/**
 * @brief Updates network using normal backpropagation
 * 
 * @param Learning_rate Learning rate (0-1).
 * @param mini_batch Deque with information structs to train on.
 * @param mini_batch_size The size of the mini-batch used for training the network.
 */
void q_network::update_net(double learning_rate, int mini_batch_size, std::deque<information> mini_batch) {
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<Matrix>> batch_errors;
    std::vector<std::vector<Matrix>> batch_activated_layers;

    for (int k = 0; k < mini_batch.size(); ++k) {
        information& info = mini_batch[k];
        std::vector<std::vector<Matrix>> ff = feed_forward_pass(info.state);
        std::vector<Matrix> activated_layers = ff[0];
        activated_layers.insert(activated_layers.begin(), info.state); 
        std::vector<Matrix> error = get_errors(info.state, info.q_target); 
        batch_errors.push_back(error);
        batch_activated_layers.push_back(activated_layers);
    }

    gradient_descent_weights(batch_errors, learning_rate, mini_batch[0].state, batch_activated_layers);
    gradient_descent_biases(batch_errors, learning_rate, mini_batch[0].state, batch_activated_layers);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
}


/**
 * @brief Trains the Q-network using the reinforcement learning paradigm.
 * 
 * This function simulates a specified number of games, collects experiences 
 * during gameplay, and updates the Q-network using mini-batches of experiences. 
 * It also applies epsilon-greedy exploration, decays epsilon over time, and 
 * optionally saves the network state at specified intervals.
 * 
 * @param games The number of games to simulate for training.
 * @param batch_size The maximum size of the experience replay buffer.
 * @param mini_batch_size The size of the mini-batch used for training the network.
 * @param learning_rate The learning rate for updating the network.
 * @param autosave_file A map containing file paths as keys and save intervals 
 *                      (in terms of games) as values. If provided, the network 
 *                      state is saved to the specified files at the given intervals.
 */
void q_network::train(int games, int batch_size, int mini_batch_size, double learning_rate, std::map<std::string, int> autosave_file) {
    std::cout << "starting training" << std::endl;
    std::deque<information> experiences;
    for (int game = 0; game < games; ++game) {
        std::cout << "Game: " << game+1 << std::endl;
        Game game_play;
        int move = 0;
        total_reward = 0;
        while (!game_play.is_over()) {
            game_play.next_frame();
            Matrix state = game_play.getState();
            information info = get_information(state, game_play);
            game_play.drawBoard();

            if (experiences.size() >= batch_size) {
                experiences.pop_front();
            }

            if (experiences.size() >= mini_batch_size) {
                std::vector<int> indices(experiences.size());
                std::iota(indices.begin(), indices.end(), 0);
                std::shuffle(indices.begin(), indices.end(), std::mt19937{std::random_device{}()});
                
                std::deque<information> mini_batch;
                for (int i = 0; i < mini_batch_size && i < experiences.size(); ++i) {
                    mini_batch.push_back(experiences[indices[i]]);
                }
                update_net(learning_rate, mini_batch_size, mini_batch);  // Train on a single mini-batch
            }
            

            experiences.push_back(info);
        }

        epsilon = std::max(min_epsilon, epsilon * epsilon_decay); // Decay epsilon over time, with a minimum value of 0.1
        std::cout << "game: " << game << "/ " << games << " finished" << std::endl; 
        std::cout << "snake size: " << game_play.snake.getSnakeBody().size() << std::endl;
        std::cout << "total reward: " << total_reward << std::endl;
        std::cout << "epsilon: " << epsilon << std::endl;

        if (!autosave_file.empty()) {
            for (const auto& [key, value] : autosave_file) {
            if (game % value == 0) {
                save_state(key, true);
            }
            }
        }

        game_play.close();

    }
}

/**
 * @brief Test a network and play the game snake with model
 * 
 * @param games The number of games to simulate for training.
 */
void q_network::play(int games) { 
    
    feed_forward_visualise nn_vis(50, 50, 1000, 700, "Feed forward pass"); //Initialize visualization

    for (int game = 0; game < games; ++ game) {
        Game game_play;
        int move = 0;
        total_reward = 0;
        while (!game_play.is_over()) {
            nn_vis.next_frame();
            game_play.next_frame();

            Matrix state = game_play.getState();

            std::vector<std::vector<Matrix>> ff = feed_forward_pass(state);
            std::vector<Matrix> activated_layers = ff[0];
            Matrix output = activated_layers.back();
            
            bool grow = false;
            bool collision = false;
            int action = output.getMaxRow();
            game_play.take_action(action);
            if (game_play.snake.collisionFood(game_play.foodVec) != -1){
                grow = true;
                game_play.foodVec.clear();
                game_play.newFood();
            }
        
            game_play.snake.move(grow);
            
            if (game_play.snake.collisionFood(game_play.foodVec) != -1){
                grow = true;
            }
        
            std::deque<TDT4102::Point> body = game_play.snake.getSnakeBody();
        
            for (TDT4102::Point& part : body) {
                if ((part.x == game_play.snake.getSnakeHead().x) && (part.y == game_play.snake.getSnakeHead().y)) {
                    collision = true;
                    std::cout << "crashed into itself! " << std::endl;
                }
            }
        
            if(!(game_play.snake.getSnakeHead().x >= 0 && game_play.snake.getSnakeHead().x < game_play.getWidth())){
                std::cout << "crashed into wall! " << std::endl;
            }
            else if(!(game_play.snake.getSnakeHead().y >= 0 && game_play.snake.getSnakeHead().y < game_play.getHeight())){
                std::cout << "crashed into wall! " << std::endl;
            }
        
            game_play.drawBoard();
            nn_vis.visualize_feed_forward(activated_layers, state); //Vis feed forward     
        }
        std::cout << "game: " << game << "/ " << games << " finished" << std::endl; 
        std::cout << "snake size: " << game_play.snake.getSnakeBody().size() << std::endl;
    }
}